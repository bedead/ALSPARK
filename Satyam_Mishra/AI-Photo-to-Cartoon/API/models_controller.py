from pipelines.DIFFUSION_PIPE import (
    initialize_model as initialize_diffusion,
    photo_to_cartoon as to_cartoon,
)
from pipelines.BLIP_PIPE import initialize_blip, get_prompt as get_blip_prompt
from pipelines.PROMPT_SIMILARITY_PIPE import (
    load_model as load_semantic_model,
    get_better_prompt as get_prompt,
)

from pipelines.UPSCALE_PIPE import initialize_upscaler, upscale
from pipelines.IMG_INFO_PIPE import is_high_resolution


class ModelController:
    def __init__(self):
        print("Initializing Real Diffusion model...")
        self.cartoon_diffusion_model = initialize_diffusion()

        print("Initializing BLIP model...")
        self.blip_processor, self.blip_model = initialize_blip()

        print("Initializing Upscaler model...")
        self.upscaler_model = initialize_upscaler()

        print("Initializing Semantic Similarity model...")
        self.semantic_model = load_semantic_model()

    def get_blip_prompt(self, sketch_image):
        """
        Generate a descriptive prompt using the BLIP model.
        """
        return get_blip_prompt(
            original_image=sketch_image,
            processor=self.blip_processor,
            model=self.blip_model,
        )

    def get_better_prompt(self, blip_prompt, user_prompt, threshold=0.65):
        """
        Compare BLIP and user prompts to select the best one based on similarity.
        """
        return get_prompt(
            blip_prompt=blip_prompt,
            user_prompt=user_prompt,
            model=self.semantic_model,
            threshold=threshold,
        )

    def photo_to_cartoon(self, photo, prompt):
        """
        Generate a realistic image from the sketch using the diffusion model.
        """
        return to_cartoon(
            image=photo, prompt=prompt, model=self.cartoon_diffusion_model
        )

    def process_photo(self, photo, user_prompt=None):
        """
        Main function to process the photo and generate a cartoon image.
        """

        high = is_high_resolution(photo)

        print(f"High Resolution : {high}")
        if high == 0:
            photo = upscale(photo, self.upscaler_model)
            print(f"Image upscaled.")

        blip_prompt = self.get_blip_prompt(photo)
        print(f"BLIP prompt: {blip_prompt}")

        if user_prompt:
            prompt = self.get_better_prompt(blip_prompt, user_prompt)
        else:
            prompt = blip_prompt
        print(f"Selected prompt: {prompt}")

        generated_image = self.photo_to_cartoon(photo, prompt)
        return generated_image
