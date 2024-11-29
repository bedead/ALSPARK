from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
import torch


class StableDiffusionWrapper:
    def __init__(self, model_id, device="cuda"):
        """
        Initializes the Stable Diffusion model pipeline.

        Parameters:
            model_id (str): The identifier of the pre-trained model.
            device (str): The device to run the model on ("cuda" or "cpu").
        """
        print(f"Initializing model: {model_id} on {device}")
        self.device = device  # Save device information for further use
        try:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
            self.pipe = self.pipe.to(device)
            if device == "cuda":
                self.pipe.enable_model_cpu_offload()
            print("Model initialized successfully.")
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise

    def generate_image(
        self,
        prompt,
        negative_prompt=None,
        num_inference_steps=50,
        guidance_scale=7.5,
        height=512,
        width=512,
    ):
        """
        Generates an image based on the provided prompt and hyperparameters.

        Parameters:
            prompt (str): The text prompt for the image generation.
            negative_prompt (str): The negative text prompt to suppress specific content.
            num_inference_steps (int): Number of denoising steps.
            guidance_scale (float): Scale for classifier-free guidance.
            height (int): Height of the generated image.
            width (int): Width of the generated image.

        Returns:
            PIL.Image.Image: The generated image.
        """
        if self.pipe is None:
            raise ValueError("Model pipeline is not initialized. Call __init__ first.")

        if not prompt:
            raise ValueError("Prompt cannot be empty.")

        print(f"Generating image with prompt: '{prompt}'")
        print(
            f"Steps: {num_inference_steps}, Scale: {guidance_scale}, Size: {width}x{height}"
        )

        try:
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
            ).images[0]
            print("Image generation successful.")
            return image
        except Exception as e:
            print(f"Error during image generation: {e}")
            raise
