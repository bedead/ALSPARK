import gradio as gr
import requests
from PIL import Image
from io import BytesIO

API_URL = "http://127.0.0.1:5000/transfer-style"
DEFAULT_NEGATIVE_PROMPT = (
    "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, "
    "fewer digits, cropped, worst quality, low quality"
)


def run(
    im1_canny,
    im1_style,
    cond_weight_style,
    cond_weight_canny,
    prompt,
    neg_prompt,
    scale,
    n_samples,
    steps,
    resize_short_edge,
    cond_tau,
):
    # Prepare image data
    img_data_canny = None
    img_data_style = None

    if im1_canny is not None:
        buffered_canny = BytesIO()
        im1_canny.save(buffered_canny, format="PNG")
        img_data_canny = buffered_canny.getvalue()
    else:
        return gr.Warning("Please upload Image.")

    if im1_style is not None:
        buffered_style = BytesIO()
        im1_style.save(buffered_style, format="PNG")
        img_data_style = buffered_style.getvalue()
    else:
        return gr.Warning("Please upload Style Image.")

    payload = {
        "prompt": prompt,
        "neg_prompt": neg_prompt,
        "scale": scale,
        "n_samples": n_samples,
        "steps": steps,
        "resize_short_edge": resize_short_edge,
        "cond_tau": cond_tau,
        "cond_weight_style": cond_weight_style,
        "cond_weight_canny": cond_weight_canny,
    }

    files = {}
    if img_data_canny:
        files["image_canny"] = ("canny.png", img_data_canny, "image/png")
    if img_data_style:
        files["image_style"] = ("style.png", img_data_style, "image/png")

    response = requests.post(API_URL, files=files, data=payload)

    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        return img
    else:
        return f"Error generating stylized image! Status code: {response.status_code}, Response: {response.text}"


with gr.Blocks() as demo:
    gr.Markdown("# Image Style Transfer Tool")
    with gr.Row(equal_height=True):
        with gr.Column(scale=7):
            output = gr.Image(height="auto")

        with gr.Column(scale=3):
            # For "canny"
            with gr.Group():
                with gr.Column():
                    im1_canny = gr.Image(
                        label="Image",
                        interactive=True,
                        visible=True,
                        type="pil",
                    )
                cond_weight_canny = gr.Slider(
                    label="Condition weight for Image",
                    minimum=0,
                    maximum=5,
                    step=0.05,
                    value=1,
                    interactive=True,
                    visible=False,
                )

            # For "style"
            with gr.Group():
                with gr.Column():
                    im1_style = gr.Image(
                        label="Style Image",
                        interactive=True,
                        visible=True,
                        type="pil",
                    )
                    cond_weight_style = gr.Slider(
                        label="Condition weight for style",
                        minimum=0,
                        maximum=5,
                        step=0.05,
                        value=1,
                        interactive=True,
                        visible=False,
                    )

            # Common parameters
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", visible=False)
                neg_prompt = gr.Textbox(
                    visible=False,
                    label="Negative Prompt",
                    value=DEFAULT_NEGATIVE_PROMPT,
                )
                scale = gr.Slider(
                    label="Guidance Scale (Classifier free guidance)",
                    value=7.5,
                    minimum=1,
                    maximum=15,
                    step=0.1,
                    visible=False,
                )
                n_samples = gr.Slider(
                    label="Num samples",
                    value=1,
                    minimum=1,
                    maximum=3,
                    step=1,
                    visible=False,
                )
                steps = gr.Slider(
                    label="Steps",
                    value=35,
                    minimum=10,
                    maximum=100,
                    step=1,
                    visible=False,
                )
                resize_short_edge = gr.Slider(
                    label="Image resolution",
                    value=512,
                    minimum=320,
                    maximum=1024,
                    step=1,
                    visible=False,
                )
                cond_tau = gr.Slider(
                    label="Timestamp parameter for adapter application",
                    value=1.0,
                    minimum=0.1,
                    maximum=1.0,
                    step=0.05,
                    visible=False,
                )

            with gr.Row():
                submit = gr.Button("Transfer Style")

    # Collect inputs

    inps = [
        im1_canny,
        im1_style,
        cond_weight_style,
        cond_weight_canny,
        prompt,
        neg_prompt,
        scale,
        n_samples,
        steps,
        resize_short_edge,
        cond_tau,
    ]
    submit.click(fn=run, inputs=inps, outputs=output)
# demo.launch()

demo.launch(debug=True, share=True)
