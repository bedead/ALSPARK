import gradio as gr
import requests
import tempfile
from io import BytesIO

API_URL = "http://127.0.0.1:5000/generate_model"


def gen_model(input_image, remove_background):
    if not input_image:
        return gr.Error("Error: Please provide an image!")

    buffered = BytesIO()
    input_image.save(buffered, format="PNG")
    img_data = buffered.getvalue()

    files = {"image": ("image.png", img_data)}
    data = {"rm_bg": remove_background, "foreground_ratio": 0.85, "mc_resolution": 256}

    response = requests.post(API_URL, files=files, data=data)

    if response.status_code == 200:
        response_data = response.json()

        obj_file_path, glb_file_path = None, None

        # Download and save the obj model
        obj_model_url = response_data["models"][0]["url"]
        obj_response = requests.get(obj_model_url)
        if obj_response.status_code == 200:
            with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as obj_file:
                obj_file.write(obj_response.content)
                obj_file_path = obj_file.name

        # Download and save the glb model
        glb_model_url = response_data["models"][1]["url"]
        glb_response = requests.get(glb_model_url)
        if glb_response.status_code == 200:
            with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as glb_file:
                glb_file.write(glb_response.content)
                glb_file_path = glb_file.name

        # print("OBJ file saved at:", obj_file_path)
        # print("GLB file saved at:", glb_file_path)

        return obj_file_path, glb_file_path
    else:
        raise gr.Error(
            "Failed to generate 3D models. Please check the API and try again."
        )


with gr.Blocks() as demo:
    gr.Markdown("# Image to 3D Model Converter")

    with gr.Row(variant="panel"):
        with gr.Column():
            with gr.Row():
                input_image = gr.Image(
                    label="Upload Image",
                    type="pil",
                    image_mode="RGBA",
                    sources="upload",
                    elem_id="content_image",
                )
            remove_background = gr.Checkbox(
                label="Does image has background?", value=False
            )
            with gr.Row():
                generate_btn = gr.Button("Generate 3D Model", variant="primary")
                clear_btn = gr.Button("Clear", variant="secondary")
        with gr.Column():
            with gr.Tab("OBJ"):
                output_model_obj = gr.Model3D(
                    label="Output Model (OBJ Format)",
                    interactive=False,
                    height=500,
                )
                gr.Markdown(
                    "Note: Downloaded object will be flipped in case of .obj export. Export .glb instead or manually flip it before usage."
                )
            with gr.Tab("GLB"):
                output_model_glb = gr.Model3D(
                    label="Output Model (GLB Format)",
                    interactive=False,
                    height=500,
                )
                gr.Markdown(
                    "Note: The model shown here has a darker appearance. Download to get correct results."
                )

    generate_btn.click(
        fn=gen_model,
        inputs=[input_image, remove_background],
        outputs=[output_model_obj, output_model_glb],
    )

    clear_btn.click(
        fn=lambda: (None, None, None),
        inputs=[],
        outputs=[input_image, output_model_obj, output_model_glb],
    )

demo.launch(share=True, debug=True)
