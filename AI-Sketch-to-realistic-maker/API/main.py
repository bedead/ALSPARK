from flask import Flask, request, send_file
from PIL import Image
import tempfile
from models_controller import ModelController

app = Flask(__name__)

model_controller = ModelController()


@app.route("/generate", methods=["POST"])
def generate_image():
    sketch_file = request.files.get("sketch")
    user_prompt = request.form.get("prompt", None)

    if not sketch_file:
        return {"error": "Sketch is required"}, 400

    sketch_image = Image.open(sketch_file).convert("RGB")
    result_image = model_controller.process_sketch(
        sketch_image, user_prompt=user_prompt
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        result_image.save(tmp, format="PNG")
        tmp_path = tmp.name

    return send_file(tmp_path, mimetype="image/png", as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
