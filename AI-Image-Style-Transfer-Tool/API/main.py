# main.py

from flask import Flask, request, send_file
from model import stylize_image
from PIL import Image
import tempfile
from model import initialize_model, stylize_image
from blip import initialize_blip, get_prompt

app = Flask(__name__)
blip_processor, blip_model = initialize_blip()
model = initialize_model()


@app.route("/stylize", methods=["POST"])
def generate_image():
    image_file = request.files.get("image")
    style = request.form.get("style")

    # Check if image and style are provided
    if not image_file or not style:
        return {"error": "Image and style are required"}, 400

    image = Image.open(image_file)
    blip_prompt = get_prompt(image, blip_processor, blip_model)
    stylized_image = stylize_image(image, style, model, blip_prompt)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        stylized_image.save(tmp, format="PNG")
        tmp_path = tmp.name

    return send_file(tmp_path, mimetype="image/png", as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
