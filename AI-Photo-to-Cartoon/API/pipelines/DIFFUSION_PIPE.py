from PIL import Image
import torch
from diffusers.pipelines.auto_pipeline import AutoPipelineForImage2Image
from diffusers.schedulers.scheduling_dpmsolver_multistep import (
    DPMSolverMultistepScheduler,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "Yntec/CartoonStyleClassic"
torch.set_grad_enabled(False)
base_prompt = ", high-quality cartoon with bold outlines, simplified colors, and minimal shading, digital cartoon look, 2d animation style, cell shading, flat color illustration, clean lines"
neg_prompt = "realistic textures, grainness, blurriness, harsh shadows, detail, complex backgrounds, muted colors, and lifeless expressions, deformed hands"


def initialize_model():
    pipeline = AutoPipelineForImage2Image.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        # safety_checker=None,
        use_safetensors=True,
    )
    pipeline = pipeline.to(device)
    if device == "cuda":
        pipeline.enable_model_cpu_offload()
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config
    )
    return pipeline


def photo_to_cartoon(image: Image.Image, prompt: str, model) -> Image.Image:
    prompt = prompt + base_prompt
    print(f"Prompt : {prompt}")
    with torch.inference_mode():
        gen_image = model(
            prompt=prompt,
            # num_inference_steps=10,
            image=image,
            strength=0.5,  # image weighting
            guidance_scale=7.5,  # prompt weighting
            # negative_prompt=neg_prompt,  # neg prompt
        ).images[0]

    torch.cuda.empty_cache()

    return gen_image
