# main.py
import os
from flask import Flask, request, url_for, jsonify, send_file
from model import preprocess_image, generate_3d_models, initialize_model
from PIL import Image

app = Flask(__name__)

model = initialize_model()


@app.route("/generate_model", methods=["POST"])
def generate_model():
    if "image" not in request.files:
        return jsonify({"error": "Image file and remove_background flag required"}), 400

    image_file = request.files.get("image")
    remove_background = request.form["rm_bg"].lower() == "true"
    foreground_ratio = float(request.form.get("foreground_ratio", 0.85))
    mc_resolution = int(request.form.get("mc_resolution", 256))

    input_image = Image.open(image_file)

    processed_image = preprocess_image(input_image, remove_background, foreground_ratio)

    obj_file_path, glb_file_path = generate_3d_models(
        processed_image, mc_resolution, model
    )

    obj_file_url = url_for("static", filename=f"models/model.obj", _external=True)
    glb_file_url = url_for("static", filename=f"models/model.glb", _external=True)

    return jsonify(
        {
            "models": [
                {"type": "obj", "url": obj_file_url},
                {"type": "glb", "url": glb_file_url},
            ]
        }
    )


@app.route("/<path:filename>")
def download_file(filename):
    return send_file(filename, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
