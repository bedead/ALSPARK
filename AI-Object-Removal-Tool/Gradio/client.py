import gradio as gr
import requests
import numpy as np
from PIL import Image
import io
import time


def process_image(img):
    if img is None:
        return gr.Warning("No image selected. Please upload an image to proceed.")

    alpha_channel = img["layers"][0][:, :, 3]
    mask = np.where(alpha_channel == 0, 0, 255).astype(np.uint8)
    mask_image = Image.fromarray(mask)
    background = Image.fromarray(img["background"])

    if np.all(mask == 0):
        return gr.Warning("No mask drawn. Please draw a mask to proceed.")

    yield None, background, mask_image

    image_bytes = io.BytesIO()
    background.save(image_bytes, format="PNG")
    mask_bytes = io.BytesIO()
    mask_image.save(mask_bytes, format="PNG")

    response = requests.post(
        "http://127.0.0.1:5000/remove_object",
        files={"image": image_bytes.getvalue(), "mask": mask_bytes.getvalue()},
    )

    if response.status_code == 200:
        result = Image.open(io.BytesIO(response.content))
        yield result, background, mask_image
    else:
        return gr.Warning("Error: API Call Failed | Check backend")


def clear_image():
    return None, None, None


with gr.Blocks() as demo:
    gr.Markdown("# AI Object removal Tool")
    with gr.Row():
        with gr.Column():
            img = gr.ImageMask(
                sources=["upload"],
                layers=False,
                label="Base Image",
                show_label=True,
                height=512,  # Set a fixed height for the displayed image
                width=512,  # Set a fixed width for the displayed image
                canvas_size=(512, 512),  # Defines the actual drawing area
                show_fullscreen_button=True,  # Allows viewing in fullscreen if needed
            )
            with gr.Row():
                btn = gr.Button("Remove Object", variant="primary")
                clear = gr.Button("Clear", variant="secondary")

        with gr.Column():
            mask_output = gr.Image(label="Mask Image", show_label=True)
            output_image = gr.Image(label="Inpainted Image", show_label=True)

    btn.click(
        fn=process_image,
        inputs=img,
        outputs=[output_image, img, mask_output],
    )

    clear.click(fn=clear_image, inputs=[], outputs=[img, mask_output, output_image])

demo.launch(share=True, debug=True)