# demo inspired by https://huggingface.co/spaces/lambdalabs/image-mixer-demo
import argparse
import copy
import gradio as gr
import torch
from functools import partial
from itertools import chain
from torch import autocast
from pytorch_lightning import seed_everything

from basicsr.utils import tensor2img
from ldm.inference_base import DEFAULT_NEGATIVE_PROMPT, diffusion_inference, get_adapters, get_sd_models
from ldm.modules.extra_condition import api
from ldm.modules.extra_condition.api import ExtraCondition, get_cond_model
from ldm.modules.encoders.adapter import CoAdapterFuser
import os
from huggingface_hub import hf_hub_url
import subprocess
import shlex
import cv2

torch.set_grad_enabled(False)

urls = {
    'TencentARC/T2I-Adapter':[
        'models/coadapter-canny-sd15v1.pth',
        'models/coadapter-style-sd15v1.pth',
        'models/coadapter-fuser-sd15v1.pth',

    ],
    'runwayml/stable-diffusion-v1-5': ['v1-5-pruned-emaonly.ckpt'],
}

if os.path.exists('models') == False:
    os.mkdir('models')
for repo in urls:
    files = urls[repo]
    for file in files:
        url = hf_hub_url(repo, file)
        name_ckp = url.split('/')[-1]
        save_path = os.path.join('models',name_ckp)
        if os.path.exists(save_path) == False:
            subprocess.run(shlex.split(f'wget {url} -O {save_path}'))

supported_cond = ['style', 'canny']

# config
parser = argparse.ArgumentParser()
parser.add_argument(
    '--sd_ckpt',
    type=str,
    default='models/v1-5-pruned-emaonly.ckpt',
    help='path to checkpoint of stable diffusion model, both .ckpt and .safetensor are supported',
)
parser.add_argument(
    '--vae_ckpt',
    type=str,
    default=None,
    help='vae checkpoint, anime SD models usually have seperate vae ckpt that need to be loaded',
)
global_opt = parser.parse_args()
global_opt.config = 'configs/stable-diffusion/sd-v1-inference.yaml'
for cond_name in supported_cond:
    setattr(global_opt, f'{cond_name}_adapter_ckpt', f'models/coadapter-{cond_name}-sd15v1.pth')
global_opt.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
global_opt.max_resolution = 512 * 512
global_opt.sampler = 'ddim'
global_opt.cond_weight = 1.0
global_opt.C = 4
global_opt.f = 8
#TODO: expose style_cond_tau to users
global_opt.style_cond_tau = 1.0

# stable-diffusion model
sd_model, sampler = get_sd_models(global_opt)
# adapters and models to processing condition inputs
adapters = {}
cond_models = {}

torch.cuda.empty_cache()

# fuser is indispensable
coadapter_fuser = CoAdapterFuser(unet_channels=[320, 640, 1280, 1280], width=768, num_head=8, n_layes=3)
coadapter_fuser.load_state_dict(torch.load(f'models/coadapter-fuser-sd15v1.pth'))
coadapter_fuser = coadapter_fuser.to(global_opt.device)


def run(*args):
    with torch.inference_mode(), \
            sd_model.ema_scope(), \
            autocast('cuda'):

        inps = []
        for i in range(0, len(args) - 8, len(supported_cond)):
            inps.append(args[i:i + len(supported_cond)])

        opt = copy.deepcopy(global_opt)
        opt.prompt, opt.neg_prompt, opt.scale, opt.n_samples, opt.seed, opt.steps, opt.resize_short_edge, opt.cond_tau \
            = args[-8:]

        ims1 = []
        ims2 = []
        for idx, (b, im1, im2, cond_weight) in enumerate(zip(*inps)):
            if idx > 0:
                if b != 'Nothing' and (im1 is not None or im2 is not None):
                    if im1 is not None:
                        h, w, _ = im1.shape
                    else:
                        h, w, _ = im2.shape
                # break
        # resize all the images to the same size
        for idx, (b, im1, im2, cond_weight) in enumerate(zip(*inps)):
            if idx == 0:
                ims1.append(im1)
                ims2.append(im2)
                continue
            if b != 'Nothing':
                if im1 is not None:
                    im1 = cv2.resize(im1, (w, h), interpolation=cv2.INTER_CUBIC)
                if im2 is not None:
                    im2 = cv2.resize(im2, (w, h), interpolation=cv2.INTER_CUBIC)
            ims1.append(im1)
            ims2.append(im2)

        conds = []
        activated_conds = []
        for idx, (b, im1, im2, cond_weight) in enumerate(zip(*inps)):
            cond_name = supported_cond[idx]
            if b == 'Nothing':
                if cond_name in adapters:
                    adapters[cond_name]['model'] = adapters[cond_name]['model'].cpu()
            else:
                activated_conds.append(cond_name)
                if cond_name in adapters:
                    adapters[cond_name]['model'] = adapters[cond_name]['model'].to(opt.device)
                else:
                    adapters[cond_name] = get_adapters(opt, getattr(ExtraCondition, cond_name))
                adapters[cond_name]['cond_weight'] = cond_weight

                process_cond_module = getattr(api, f'get_cond_{cond_name}')

                if b == 'Image':
                    if cond_name not in cond_models:
                        cond_models[cond_name] = get_cond_model(opt, getattr(ExtraCondition, cond_name))
                    conds.append(process_cond_module(opt, ims1[idx], 'image', cond_models[cond_name]))
                else:
                    conds.append(process_cond_module(opt, ims2[idx], cond_name, None))

        features = dict()
        for idx, cond_name in enumerate(activated_conds):
            cur_feats = adapters[cond_name]['model'](conds[idx])
            if isinstance(cur_feats, list):
                for i in range(len(cur_feats)):
                    cur_feats[i] *= adapters[cond_name]['cond_weight']
            else:
                cur_feats *= adapters[cond_name]['cond_weight']
            features[cond_name] = cur_feats

        adapter_features, append_to_context = coadapter_fuser(features)

        output_conds = []
        for cond in conds:
            output_conds.append(tensor2img(cond, rgb2bgr=False))

        ims = []
        seed_everything(opt.seed)
        for _ in range(opt.n_samples):
            result = diffusion_inference(opt, sd_model, sampler, adapter_features, append_to_context)
            ims.append(tensor2img(result, rgb2bgr=False))

        # Clear GPU memory cache so less likely to OOM
        torch.cuda.empty_cache()
        return ims


def change_visible(im1, im2, val):
    outputs = {}
    if val == "Image":
        outputs[im1] = gr.update(visible=True)
        outputs[im2] = gr.update(visible=False)
    elif val == "Nothing":
        outputs[im1] = gr.update(visible=False)
        outputs[im2] = gr.update(visible=False)
    else:
        outputs[im1] = gr.update(visible=False)
        outputs[im2] = gr.update(visible=True)
    return outputs

# with gr.Blocks(title="CoAdapter", css=".gr-box {border-color: #8136e2}") as demo:
with gr.Blocks(css='style.css') as demo:

    btns = []
    ims1 = []
    ims2 = []
    cond_weights = []

    with gr.Row():
        for cond_name in supported_cond:
            with gr.Group():
                with gr.Column():
                    if cond_name == 'style':
                        btn1 = gr.Radio(
                        choices=["Image", "Nothing"],
                        label=f"Input type for {cond_name}",
                        interactive=True,
                        value="Nothing",
                    )
                    else:
                        btn1 = gr.Radio(
                            choices=["Image", cond_name, "Nothing"],
                            label=f"Input type for {cond_name}",
                            interactive=True,
                            value="Nothing",
                        )
                    im1 = gr.Image(label="Image", interactive=True, visible=False, type="numpy")
                    im2 = gr.Image(label=cond_name, interactive=True, visible=False, type="numpy")
                    cond_weight = gr.Slider(
                        label="Condition weight", minimum=0, maximum=5, step=0.05, value=1, interactive=True)

                    fn = partial(change_visible, im1, im2)
                    btn1.change(fn=fn, inputs=[btn1], outputs=[im1, im2], queue=False)

                    btns.append(btn1)
                    ims1.append(im1)
                    ims2.append(im2)
                    cond_weights.append(cond_weight)

    with gr.Column():
        prompt = gr.Textbox(label="Prompt", visible=False)
        neg_prompt = gr.Textbox(visible=False, label="Negative Prompt", value=DEFAULT_NEGATIVE_PROMPT)
        scale = gr.Slider(label="Guidance Scale (Classifier free guidance)", value=7.5, minimum=1, maximum=20, step=0.1)
        n_samples = gr.Slider(label="Num samples", value=1, minimum=1, maximum=3, step=1)
        seed = gr.Slider(label="Seed", value=42, minimum=0, maximum=10000, step=1)
        steps = gr.Slider(label="Steps", value=50, minimum=10, maximum=100, step=1)
        resize_short_edge = gr.Slider(label="Image resolution", value=512, minimum=320, maximum=1024, step=1)
        cond_tau = gr.Slider(
            label="timestamp parameter that determines until which step the adapter is applied",
            value=1.0,
            minimum=0.1,
            maximum=1.0,
            step=0.05)

    with gr.Row():
        submit = gr.Button("Generate")
    output = gr.Gallery(rows=2, height='auto')
    # cond = gr.Gallery(rows=2, height='auto')

    inps = list(chain(btns, ims1, ims2, cond_weights))
    inps.extend([prompt, neg_prompt, scale, n_samples, seed, steps, resize_short_edge, cond_tau])
    submit.click(fn=run, inputs=inps, outputs=output)
# demo.launch()
demo.launch(debug=True, share=True)