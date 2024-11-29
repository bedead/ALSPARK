from pipelines.SD_PIPE import StableDiffusionWrapper
from pipelines.Latent_upscaler import LatentUpscaler
import torch


class ModelController:
    def __init__(
        self,
        sd_model_id="sd-legacy/stable-diffusion-v1-5",
        refiner_model_id="stabilityai/sd-x2-latent-upscaler",
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initializes Stable Diffusion and Latent Refiner pipelines.
        """
        print("Initializing Model Controller...")
        self.sd_wrapper = StableDiffusionWrapper(model_id=sd_model_id, device=device)
        self.image_refiner = LatentUpscaler(model_id=refiner_model_id, device=device)
        print("Model Controller initialized.")

    def run_pipeline(
        self,
        prompt,
        negative_prompt=None,
        num_inference_steps=50,
        guidance_scale=7.5,
        refiner_num_steps=25,
        refiner_guidance_scale=5.0,
        height=512,
        width=512,
        init_strength=0.75,
    ):
        """
        Runs the Stable Diffusion to Image Refiner pipeline.
        """
        print("Starting Stable Diffusion to Image Refiner pipeline...")

        # Step 1: Generate image with Stable Diffusion
        generated_image = self.sd_wrapper.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
        )

        # Step 2: Refine the generated image
        refined_image = self.image_refiner.refine_image(
            image=generated_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=refiner_num_steps,
            guidance_scale=refiner_guidance_scale,
            strength=init_strength,
        )

        print("Pipeline execution complete.")
        return refined_image
