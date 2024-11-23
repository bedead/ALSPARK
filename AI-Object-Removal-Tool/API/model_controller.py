import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from diffusers.utils.loading_utils import load_image
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.schedulers.scheduling_dpmsolver_multistep import (
    DPMSolverMultistepScheduler,
)

from pipelines.pipeline_PowerPaint import StableDiffusionInpaintPipeline as Pipeline
from pipelines.power_paint_tokenizer import PowerPaintTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

model = "andregn/Realistic_Vision_V3.0-inpainting"
model1 = "Sanster/PowerPaint-V1-stable-diffusion-inpainting"


def initialize_model():
    pipe = Pipeline.from_pretrained(
        model1,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        variant="fp16" if device == "cuda" else None,
    )
    pipe.tokenizer = PowerPaintTokenizer(pipe.tokenizer)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if device == "cuda":
        pipe.enable_model_cpu_offload()
    # pipe = pipe.to(device)

    return pipe


def add_task_to_prompt(prompt, neg_prompt):
    promptA = prompt + " P_ctxt"
    promptB = prompt + "P_ctxt"
    negative_promptA = neg_prompt + "P_obj"
    negative_promptB = neg_prompt + "P_obj"

    return promptA, promptB, negative_promptA, negative_promptB


@torch.inference_mode()
def predict(
    pipe,
    input_image,
    prompt,
    fitting_degree,
    ddim_steps,
    scale,
    negative_prompt,
):
    width, height = input_image["image"].convert("RGB").size

    if width < height:
        input_image["image"] = (
            input_image["image"].convert("RGB").resize((640, int(height / width * 640)))
        )
    else:
        input_image["image"] = (
            input_image["image"].convert("RGB").resize((int(width / height * 640), 640))
        )

    promptA, promptB, negative_promptA, negative_promptB = add_task_to_prompt(
        prompt, negative_prompt
    )
    print(promptA, promptB, negative_promptA, negative_promptB)
    img = np.array(input_image["image"].convert("RGB"))

    W = int(np.shape(img)[0] - np.shape(img)[0] % 8)
    H = int(np.shape(img)[1] - np.shape(img)[1] % 8)
    input_image["image"] = input_image["image"].resize((H, W))
    input_image["mask"] = input_image["mask"].resize((H, W))
    result = pipe(
        promptA=promptA,
        promptB=promptB,
        tradoff=fitting_degree,
        tradoff_nag=fitting_degree,
        negative_promptA=negative_promptA,
        negative_promptB=negative_promptB,
        image=input_image["image"].convert("RGB"),
        mask_image=input_image["mask"].convert("RGB"),
        width=H,
        height=W,
        guidance_scale=scale,
        num_inference_steps=ddim_steps,
    ).images[0]
    mask_np = np.array(input_image["mask"].convert("RGB"))
    red = np.array(result).astype("float") * 1
    red[:, :, 0] = 180.0
    red[:, :, 2] = 0
    red[:, :, 1] = 0
    result_m = np.array(result)
    result_m = Image.fromarray(
        (
            result_m.astype("float") * (1 - mask_np.astype("float") / 512.0)
            + mask_np.astype("float") / 512.0 * red
        ).astype("uint8")
    )
    m_img = (
        input_image["mask"].convert("RGB").filter(ImageFilter.GaussianBlur(radius=3))
    )
    m_img = np.asarray(m_img) / 255.0
    img_np = np.asarray(input_image["image"].convert("RGB")) / 255.0
    ours_np = np.asarray(result) / 255.0
    ours_np = ours_np * m_img + (1 - m_img) * img_np
    result_paste = Image.fromarray(np.uint8(ours_np * 255))

    dict_res = [input_image["mask"].convert("RGB"), result_m]

    dict_out = [input_image["image"].convert("RGB"), result_paste]

    return dict_out, dict_res


# Example usage

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# pipe = Pipeline.from_pretrained(
#     "Sanster/PowerPaint-V1-stable-diffusion-inpainting",
#     torch_dtype=torch.float16 if device == "cuda" else torch.float32,
#     use_auth_token=True,
#     revision="fp16" if device == 'cuda' else None,
#     safety_checker=None,
#     variant="fp16" if device == 'cuda' else None,
# )
# pipe.tokenizer = PowerPaintTokenizer(pipe.tokenizer)
# pipe = pipe.to(device)

# if device == "cuda":
#     pipe.enable_model_cpu_offload()


# img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
# mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
# image = load_image(img_url).convert("RGB")
# mask = load_image(mask_url).convert("RGB")

# input_image = {"image": image, "mask": mask}
# prompt = ""
# negative_prompt = ""
# fitting_degree = 1
# ddim_steps = 30
# guidance_scale = 12

# dict_out, dict_res = predict(
#     pipe,
#     input_image,
#     prompt,
#     fitting_degree,
#     ddim_steps,
#     guidance_scale,
#     negative_prompt,
# )


# result_image = dict_out[1]
# result_image.save(f"remove_result.png")
