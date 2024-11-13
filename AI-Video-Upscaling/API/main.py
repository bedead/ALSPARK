from flask import Flask, request, send_file
from model import RealESRGANProcessor
import tempfile
import mimetypes
import os
import torch

app = Flask(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"


@app.route("/upscale-video", methods=["POST"])
def upscale_video():
    if "video" not in request.files:
        return {"error": "No video file provided."}, 400

    video_file = request.files["video"]
    model_name = request.form.get("model", "realesr-animevideov3")
    do_face_enhance = bool(request.form.get("face_enhancer", False))

    mime_type, _ = mimetypes.guess_type(video_file.filename)
    # print(mime_type)
    if mime_type != "video/mp4":
        return {"error": "Invalid file format. Please upload an MP4 video."}, 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input_file:
        input_video_path = temp_input_file.name
        video_file.save(input_video_path)

    if os.path.getsize(input_video_path) == 0:
        return {"error": "Uploaded file is empty."}, 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output_file:
        output_video_path = temp_output_file.name

    esrgan = RealESRGANProcessor(
        model_name=model_name,
        device=device,
        face_enhance=do_face_enhance,
    )
    esrgan.process_video(input_video_path, output_video_path)
    torch.cuda.empty_cache()

    # video_upscaler.upscale_video(input_video_path, output_video_path)
    response = send_file(output_video_path, as_attachment=True, mimetype="video/mp4")

    os.remove(input_video_path)
    os.remove(output_video_path)

    return response


if __name__ == "__main__":
    os.makedirs("weights", exist_ok=True)
    app.run(port=5001, debug=True)
