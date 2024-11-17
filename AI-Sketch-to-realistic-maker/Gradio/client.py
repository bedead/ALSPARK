import gradio as gr
import requests
from PIL import Image
from io import BytesIO
import numpy as np

API_URL = "http://127.0.0.1:5000/generate"


def generate_realistic_image(image, prompt):
    if image is None:
        return gr.Error("Either upload an image or draw.")

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_data = buffered.getvalue()

    data = {"prompt": prompt} if prompt else {}

    response = requests.post(
        API_URL, files={"sketch": ("sketch.png", img_data, "image/png")}, data=data
    )

    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        return img
    else:
        return gr.Error("Error generating image!")


def predict(im):
    return im["composite"]


def clear_all():
    return (
        None,
        None,
        "",
        None,
    )


with gr.Blocks() as demo:
    gr.Markdown("# Sketch-to-Realistic Image Generator")

    with gr.Row(variant="panel", equal_height=True):
        with gr.Column(scale=6):
            im = gr.ImageEditor(
                type="pil",
                label="Edit Image",
                image_mode="RGBA",
                # canvas_size=(900, 500),
                # crop_size="16:9",
                # format="png",
                height=700,
            )

        with gr.Column(scale=4):
            im_preview = gr.Image(interactive=False, label="Preview")

            # with gr.Row():
            prompt = gr.Textbox(
                label="Optional Prompt",
                placeholder="Write a descriptive prompt of sketch.",
            )

            with gr.Row():
                generate_button = gr.Button("Generate Image", variant="primary")
                clear_btn = gr.Button("Clear", variant="secondary")
            output_image = gr.Image(height=500, label="Generated Realistic Image")

    generate_button.click(
        generate_realistic_image,
        inputs=[im_preview, prompt],
        outputs=output_image,
    )

    clear_btn.click(
        fn=clear_all,
        inputs=None,
        outputs=[im, im_preview, prompt, output_image],
    )
    im.change(predict, inputs=im, outputs=im_preview, show_progress="hidden")


demo.launch(share=True)
