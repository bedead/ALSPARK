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
from ldm.inference_base import (
    DEFAULT_NEGATIVE_PROMPT,
    diffusion_inference,
    get_adapters,
    get_sd_models,
)
from ldm.modules.extra_condition import api
from ldm.modules.extra_condition.api import ExtraCondition, get_cond_model
from ldm.modules.encoders.adapter import CoAdapterFuser
import os
from huggingface_hub import hf_hub_url
import subprocess
import shlex
import cv2

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
urls = {
    "TencentARC/T2I-Adapter": [
        "models/coadapter-canny-sd15v1.pth",
        "models/coadapter-style-sd15v1.pth",
        "models/coadapter-fuser-sd15v1.pth",
    ],
    "runwayml/stable-diffusion-v1-5": ["v1-5-pruned-emaonly.ckpt"],
}

if os.path.exists("models") == False:
    os.mkdir("models")
for repo in urls:
    files = urls[repo]
    for file in files:
        url = hf_hub_url(repo, file)
        name_ckp = url.split("/")[-1]
        save_path = os.path.join("models", name_ckp)
        if os.path.exists(save_path) == False:
            subprocess.run(shlex.split(f"wget {url} -O {save_path}"))

supported_cond = ["style", "canny"]

# config
parser = argparse.ArgumentParser()
parser.add_argument(
    "--sd_ckpt",
    type=str,
    default="models/v1-5-pruned-emaonly.ckpt",
    help="path to checkpoint of stable diffusion model, both .ckpt and .safetensor are supported",
)
parser.add_argument(
    "--vae_ckpt",
    type=str,
    default=None,
    help="vae checkpoint, anime SD models usually have seperate vae ckpt that need to be loaded",
)
global_opt = parser.parse_args()
global_opt.config = "configs/stable-diffusion/sd-v1-inference.yaml"
for cond_name in supported_cond:
    setattr(
        global_opt,
        f"{cond_name}_adapter_ckpt",
        f"models/coadapter-{cond_name}-sd15v1.pth",
    )
global_opt.device = device
global_opt.max_resolution = 512 * 512
global_opt.sampler = "ddim"
global_opt.cond_weight = 1.0
global_opt.C = 4
global_opt.f = 8
# TODO: expose style_cond_tau to users
global_opt.style_cond_tau = 1.0

# stable-diffusion model
sd_model, sampler = get_sd_models(global_opt)
# adapters and models to processing condition inputs
adapters = {}
cond_models = {}

torch.cuda.empty_cache()

# fuser is indispensable
coadapter_fuser = CoAdapterFuser(
    unet_channels=[320, 640, 1280, 1280], width=768, num_head=8, n_layes=3
)
coadapter_fuser.load_state_dict(torch.load(f"models/coadapter-fuser-sd15v1.pth"))
coadapter_fuser = coadapter_fuser.to(global_opt.device)


def run(
    im1_canny,
    im1_style,
    cond_weight_style,
    cond_weight_canny,
    prompt,
    neg_prompt,
    scale,
    n_samples,
    seed,
    steps,
    resize_short_edge,
    cond_tau,
):
    with torch.inference_mode(), sd_model.ema_scope(), autocast(device):

        opt = copy.deepcopy(global_opt)
        (
            opt.prompt,
            opt.neg_prompt,
            opt.scale,
            opt.n_samples,
            opt.seed,
            opt.steps,
            opt.resize_short_edge,
            opt.cond_tau,
        ) = (
            prompt,
            neg_prompt,
            scale,
            n_samples,
            seed,
            steps,
            resize_short_edge,
            cond_tau,
        )

        print(f"{neg_prompt}")

        # Resize the input images to match sizes
        ims1, ims2 = [], []
        h, w = None, None
        if im1_canny is not None:
            h, w, _ = im1_canny.shape
        elif im1_style is not None:
            h, w, _ = im1_style.shape

        if h and w:
            if im1_canny is not None:
                im1_canny = cv2.resize(im1_canny, (w, h), interpolation=cv2.INTER_CUBIC)
            if im1_style is not None:
                im1_style = cv2.resize(im1_style, (w, h), interpolation=cv2.INTER_CUBIC)

        ims1.append(im1_canny)
        ims2.append(im1_style)

        # Prepare condition inputs
        conds = []
        activated_conds = []
        cond_weights = [cond_weight_canny, cond_weight_style]
        input_images = [im1_canny, im1_style]
        cond_names = ["canny", "style"]

        for idx, (input_image, cond_weight, cond_name) in enumerate(
            zip(input_images, cond_weights, cond_names)
        ):
            if input_image is not None:
                activated_conds.append(cond_name)
                if cond_name in adapters:
                    adapters[cond_name]["model"] = adapters[cond_name]["model"].to(
                        opt.device
                    )
                else:
                    adapters[cond_name] = get_adapters(
                        opt, getattr(ExtraCondition, cond_name)
                    )
                adapters[cond_name]["cond_weight"] = cond_weight

                process_cond_module = getattr(api, f"get_cond_{cond_name}")
                if cond_name not in cond_models:
                    cond_models[cond_name] = get_cond_model(
                        opt, getattr(ExtraCondition, cond_name)
                    )
                conds.append(
                    process_cond_module(
                        opt, input_image, "image", cond_models[cond_name]
                    )
                )
            else:
                if cond_name in adapters:
                    adapters[cond_name]["model"] = adapters[cond_name]["model"].cpu()

        # Process the features
        features = dict()
        for idx, cond_name in enumerate(activated_conds):
            cur_feats = adapters[cond_name]["model"](conds[idx])
            if isinstance(cur_feats, list):
                for i in range(len(cur_feats)):
                    cur_feats[i] *= adapters[cond_name]["cond_weight"]
            else:
                cur_feats *= adapters[cond_name]["cond_weight"]
            features[cond_name] = cur_feats

        adapter_features, append_to_context = coadapter_fuser(features)

        # Generate output images
        output_conds = []
        for cond in conds:
            output_conds.append(tensor2img(cond, rgb2bgr=False))

        ims = []
        seed_everything(opt.seed)
        for _ in range(opt.n_samples):
            result = diffusion_inference(
                opt, sd_model, sampler, adapter_features, append_to_context
            )
            ims.append(tensor2img(result, rgb2bgr=False))

        # Clear GPU memory cache
        torch.cuda.empty_cache()
        return ims


# with gr.Blocks(title="CoAdapter", css=".gr-box {border-color: #8136e2}") as demo:
with gr.Blocks(css="style.css") as demo:
    with gr.Row(equal_height=True):
        with gr.Column(scale=7):
            output = gr.Gallery(height="auto")

        with gr.Column(scale=3):
            # For "canny"
            with gr.Group():
                with gr.Column():
                    im1_canny = gr.Image(
                        label="Image",
                        interactive=True,
                        visible=True,
                        type="numpy",
                    )
                cond_weight_canny = gr.Slider(
                    label="Condition weight for Image",
                    minimum=0,
                    maximum=5,
                    step=0.05,
                    value=1,
                    interactive=True,
                )

            # For "style"
            with gr.Group():
                with gr.Column():
                    im1_style = gr.Image(
                        label="Style Image",
                        interactive=True,
                        visible=True,
                        type="numpy",
                    )
                    cond_weight_style = gr.Slider(
                        label="Condition weight for style",
                        minimum=0,
                        maximum=5,
                        step=0.05,
                        value=1,
                        interactive=True,
                    )

            # Common parameters
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", visible=False)
                neg_prompt = gr.Textbox(
                    visible=False,
                    label="Negative Prompt",
                    value=DEFAULT_NEGATIVE_PROMPT,
                )
                scale = gr.Slider(
                    label="Guidance Scale (Classifier free guidance)",
                    value=7.5,
                    minimum=1,
                    maximum=15,
                    step=0.1,
                )
                n_samples = gr.Slider(
                    label="Num samples",
                    value=1,
                    minimum=1,
                    maximum=3,
                    step=1,
                    visible=False,
                )
                seed = gr.Slider(
                    label="Seed",
                    value=42,
                    minimum=0,
                    maximum=10000,
                    step=1,
                    visible=False,
                )
                steps = gr.Slider(
                    label="Steps", value=50, minimum=10, maximum=100, step=1
                )
                resize_short_edge = gr.Slider(
                    label="Image resolution",
                    value=512,
                    minimum=320,
                    maximum=1024,
                    step=1,
                    visible=False,
                )
                cond_tau = gr.Slider(
                    label="Timestamp parameter for adapter application",
                    value=1.0,
                    minimum=0.1,
                    maximum=1.0,
                    step=0.05,
                )

            with gr.Row():
                submit = gr.Button("Transfer Style")

    # Collect inputs

    inps = [
        im1_canny,
        im1_style,
        cond_weight_style,
        cond_weight_canny,
        prompt,
        neg_prompt,
        scale,
        n_samples,
        seed,
        steps,
        resize_short_edge,
        cond_tau,
    ]
    submit.click(fn=run, inputs=inps, outputs=output)
# demo.launch()

demo.launch(debug=True, share=True)
