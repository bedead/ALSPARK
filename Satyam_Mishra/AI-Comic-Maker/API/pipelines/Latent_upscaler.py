from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_latent_upscale import (
    StableDiffusionLatentUpscalePipeline,
)
import torch
from PIL import Image


class LatentUpscaler:
    def __init__(self, model_id, device="cuda"):
        """
        Initializes the Image Refiner model pipeline.
        """
        print(f"Initializing Image Refiner model: {model_id} on {device}")
        self.pipe = StableDiffusionLatentUpscalePipeline.from_pretrained(
            model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        if device == "cuda":
            self.pipe.enable_model_cpu_offload()

        self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe = self.pipe.to(device)

    def refine_image(
        self,
        image: Image.Image,
        prompt,
        negative_prompt=None,
        num_inference_steps=25,
        guidance_scale=5.0,
        strength=0.75,
    ):
        """
        Refines an image using the Image Refiner model.
        """
        print("Refining image with Image Refiner.")
        refined_image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]
        return refined_image
