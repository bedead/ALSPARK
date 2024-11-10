import gradio as gr
import requests
from PIL import Image
from io import BytesIO

API_URL = "http://127.0.0.1:5000/stylize"


def stylize_image(image, style):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_data = buffered.getvalue()

    response = requests.post(
        API_URL,
        files={"image": ("image.png", img_data, "image/png")},
        data={"style": style},
    )

    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        return img
    else:
        return "Error generating stylized image!"


image_input = gr.components.Image(type="pil", label="Upload Image")
style_input = gr.components.Dropdown(
    choices=[
        "Van Gogh",
        "Pixel Art",
        "Manga Style",
    ],
    label="Select Style",
)
output_image = gr.components.Image(label="Stylized Image")

gr.Interface(
    fn=stylize_image,
    inputs=[image_input, style_input],
    outputs=output_image,
    title="AI Image Style Transfer ",
).launch(share=True)
