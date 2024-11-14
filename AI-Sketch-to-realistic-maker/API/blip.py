import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"


def initialize_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    return processor, model


def get_blip_prompt(original_image: Image.Image, processor, model) -> str:
    inputs = processor(
        images=original_image, text="a photo of ", return_tensors="pt"
    ).to(device, torch.float16 if device == "cuda" else torch.float32)

    out = model.generate(**inputs)
    text = processor.decode(out[0], skip_special_tokens=True)

    torch.cuda.empty_cache()

    return text
