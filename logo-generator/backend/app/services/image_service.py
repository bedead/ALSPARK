from .sdxl_model import sdxl_model

class ImageService:
    @staticmethod
    def generate_and_save_image(PROMPT, NUM_IMAGES, NEGATIVE_PROMPT):
        # Generate the image
        images = sdxl_model.generate_image(PROMPT, NUM_IMAGES, NEGATIVE_PROMPT)

        return images
