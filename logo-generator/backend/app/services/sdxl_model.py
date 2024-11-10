import torch
from ..config import Config

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"{device} compute is available.")

class SDXLModel:
    def __init__(self):
        self.pipeline = DiffusionPipeline.from_pretrained(
            Config.BASE_MODEL, 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            ).to(device)
        self.pipeline.load_lora_weights(Config.LoRA_WEIGHTS, adapter_name=Config.ADAPTOR_NAME)

        if device == "cuda":
            self.pipeline.enable_model_cpu_offload()


    def generate_image(self, PROMPT, NUM_IMAGES, NEGATIVE_PROMPT):
        result = self.pipeline(
            prompt = PROMPT, 
            num_inference_steps=25, 
            num_images_per_prompt = NUM_IMAGES,
            negative_prompt = NEGATIVE_PROMPT,
        )

        return result

sdxl_model = SDXLModel()
