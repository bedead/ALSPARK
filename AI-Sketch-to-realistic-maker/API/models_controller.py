from pipelines.REAL_DIFFUSION_PIPE import (
    initialize_model as initialize_diffusion,
    sketch_to_realistic as get_realistic,
)
from pipelines.BLIP_PIPE import initialize_blip, get_blip_prompt
from pipelines.PROMPT_SIMILARITY_PIPE import (
    load_model as load_semantic_model,
    get_better_prompt as get_prompt,
)


class ModelController:
    def __init__(self):
        print("Initializing Real Diffusion model...")
        self.real_diffusion_model = initialize_diffusion()

        print("Initializing BLIP model...")
        self.blip_processor, self.blip_model = initialize_blip()

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

    def sketch_to_realistic(self, sketch_image, prompt):
        """
        Generate a realistic image from the sketch using the diffusion model.
        """
        return get_realistic(
            sketch_image=sketch_image, prompt=prompt, model=self.real_diffusion_model
        )

    def process_sketch(self, sketch_image, user_prompt=None):
        """
        Main function to process the sketch image and generate a realistic image.
        """
        blip_prompt = self.get_blip_prompt(sketch_image)
        print(f"BLIP prompt: {blip_prompt}")

        if user_prompt:
            prompt = self.get_better_prompt(blip_prompt, user_prompt)
        else:
            prompt = blip_prompt
        print(f"Selected prompt: {prompt}")

        generated_image = self.sketch_to_realistic(sketch_image, prompt)
        return generated_image
