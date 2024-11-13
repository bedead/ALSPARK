import gradio as gr
import requests


def upscale_video(video, scaler_type, face_enhancer: bool):
    # Map the user-friendly scaler names to the actual model names
    scaler_mapping = {
        "Fast Upscaler": "realesr-animevideov3",
        "High Quality Upscaler": "RealESRGAN_x2plus",
        "Best Quality Upscaler": "RealESRGAN_x4plus",
    }

    selected_model = scaler_mapping[scaler_type]

    # Send the video, selected scaler type, and face enhancement option to the API
    url = "http://127.0.0.1:5001/upscale-video"
    files = {"video": open(video, "rb")}
    data = {"model": selected_model, "face_enhancer": face_enhancer}

    response = requests.post(url, files=files, data=data)

    if response.status_code == 200:
        output_video_path = "upscaled_video.mp4"
        with open(output_video_path, "wb") as f:
            f.write(response.content)

        return output_video_path
    else:
        return "Error: " + response.json().get("error", "Unknown error occurred.")


# Define the Gradio interface with video input, scaler selection, and face enhancer option
gr.Interface(
    fn=upscale_video,
    inputs=[
        gr.Video(label="Upload MP4 Video"),
        gr.Dropdown(
            ["Fast Upscaler", "High Quality Upscaler", "Best Quality Upscaler"],
            label="Select Upscaler",
            value="Fast Upscaler",
        ),
        gr.Checkbox(label="Enable Face Enhancer (Optional)", value=False),
    ],
    outputs=gr.Video(label="Upscaled Video"),
    title="AI Video Upscaler",
    description="Upload an MP4 video to upscale with different scaling options and optional face enhancement.",
).launch(share=True)
