from flask import Flask, request, send_file
from PIL import Image
import tempfile
from model import initialize_model, photo_to_cartoon
from blip import initialize_blip, get_prompt
from check_img_quality import is_high_resolution
from upscale import initialize_upscaler, upscale

app = Flask(__name__)

model = initialize_model()
blip_processor, blip_model = initialize_blip()
model = initialize_model()
upscaler_model = initialize_upscaler()


@app.route("/photo_to_cartoon", methods=["POST"])
def generate_image():
    image = request.files.get("image")

    if not image:
        return {"error": "Input Image are required"}, 400

    image = Image.open(image)

    high = is_high_resolution(image)
    print(f"High Resolution : {high}")
    if high == 0:
        image = upscale(image, upscaler_model)
        print(f"Image upscaled.")

    blip_prompt = get_prompt(image, blip_processor, blip_model)
    generated_image = photo_to_cartoon(image, blip_prompt, model)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        generated_image.save(tmp, format="PNG")
        tmp_path = tmp.name

    return send_file(tmp_path, mimetype="image/png", as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
