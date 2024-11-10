import gradio as gr
import requests
from PIL import Image
from io import BytesIO

API_URL = "http://127.0.0.1:5000/generate"  # Flask runs on port 5000 by default


def generate_realistic_image(sketch):
    buffered = BytesIO()
    sketch.save(buffered, format="PNG")
    img_data = buffered.getvalue()

    response = requests.post(
        API_URL,
        files={"sketch": ("sketch.png", img_data, "image/png")},
    )

    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        return img
    else:
        return "Error generating image!"


sketch_input = gr.components.Image(type="pil", label="Upload Sketch")
output_image = gr.components.Image(label="Generated Realistic Image")

gr.Interface(
    fn=generate_realistic_image,
    inputs=sketch_input,
    outputs=output_image,
    title="Sketch-to-Realistic Image Generator",
).launch(share=True)
