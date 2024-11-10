# model.py

from PIL import Image
import torch
from diffusers.pipelines.auto_pipeline import AutoPipelineForImage2Image
from styles import styles

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "segmind/small-sd"
model_id1 = "nota-ai/bk-sdm-small"
torch.set_grad_enabled(False)


def initialize_model():
    pipeline = AutoPipelineForImage2Image.from_pretrained(
        model_id1,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    pipeline = pipeline.to(device)
    if device == "cuda":
        pipeline.enable_model_cpu_offload()
    return pipeline


def stylize_image(
    input_image: Image.Image, style: str, model, blip_prompt: str
) -> Image.Image:

    style_prompt = f"{blip_prompt}, {styles.get(style, 'Van Gogh')}."
    print(f"Prompt : {style_prompt}")

    with torch.inference_mode():
        stylized_image = model(
            prompt=style_prompt,
            image=input_image,
            strength=0.55,
            guidance_scale=10,
        ).images[0]

    torch.cuda.empty_cache()

    return stylized_image
