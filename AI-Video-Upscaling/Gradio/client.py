import gradio as gr
import requests


def upscale_video(video, scaler_type, face_enhancer):
    # Map the user-friendly scaler names to the actual model names
    scaler_mapping = {
        "Upscaler 4x": "RealESRGAN_x4plus",
        "Cartoon/Anime Upscaler": "realesr-animevideov3",
        "Upscaler 2x": "RealESRGAN_x2plus",
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
            ["Upscaler 2x", "Cartoon/Anime Upscaler", "Upscaler 4x"], label="Select Upscaler"
        ),
        gr.Checkbox(label="Enable Face Enhancer (Optional)"),
    ],
    outputs=gr.Video(label="Upscaled Video"),
    title="AI Video Upscaler",
    description="Upload an MP4 video to upscale with different scaling options and optional face enhancement.",
).launch(share=True)
