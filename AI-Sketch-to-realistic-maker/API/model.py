from PIL import Image
import torch
import random
from diffusers.pipelines.auto_pipeline import AutoPipelineForImage2Image
from diffusers.schedulers.scheduling_dpmsolver_multistep import (
    DPMSolverMultistepScheduler,
)
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)


device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "emilianJR/epiCRealism"

torch.set_grad_enabled(False)
if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True

fixed_negative_prompt = """
(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, geometric shape, geometric), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck
"""

def initialize_model():
    pipe = AutoPipelineForImage2Image.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        # use_safetensors=True,
        # safety_checker=StableDiffusionSafetyChecker.from_pretrained(
        #     "CompVis/stable-diffusion-safety-checker"
        # ),
        # safety_checker=None,
    ).to(device)

    if device == "cuda":
        pipe.enable_model_cpu_offload()

    # Uncomment to use the custom scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe


def sketch_to_realistic(
    sketch_image: Image.Image, prompt: str, model, seed: int = None
) -> Image.Image:
    seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    # seed = 513307103
    print(f"Using seed: {seed}")  # Print the seed for reference

    prompt = f"Raw photo, {prompt}, photo-realistic, 8k uhd, dslr, soft lighting, high quality"
    print(f"Prompt : {prompt}")
    generator = torch.Generator(device=device).manual_seed(seed)

    with torch.no_grad():
        gen_image = model(
            prompt=prompt,
            image=sketch_image,
            strength=0.6,
            negative_prompt=fixed_negative_prompt,
            generator=generator,
            # num_inference_steps=30,
        ).images[0]

    return gen_image