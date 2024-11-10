from flask import Flask, request, send_file
from PIL import Image
import tempfile
from model import (
    initialize_model,
    sketch_to_realistic,
)
from blip import initialize_blip, get_prompt

app = Flask(__name__)

model = initialize_model()
blip_processor, blip_model = initialize_blip()


@app.route("/generate", methods=["POST"])
def generate_image():
    sketch = request.files.get("sketch")

    if not sketch:
        return {"error": "Sketch is required"}, 400

    sketch_image = Image.open(sketch)
    prompt = get_prompt(sketch_image, processor=blip_processor, model=blip_model)

    generated_image = sketch_to_realistic(sketch_image, prompt, model)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        generated_image.save(tmp, format="PNG")
        tmp_path = tmp.name

    return send_file(tmp_path, mimetype="image/png", as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
