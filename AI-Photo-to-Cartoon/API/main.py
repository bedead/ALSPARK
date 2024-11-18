from flask import Flask, request, send_file
from PIL import Image
import tempfile
from models_controller import ModelController

app = Flask(__name__)
pipeline = ModelController()


@app.route("/photo_to_cartoon", methods=["POST"])
def generate_image():
    image = request.files.get("image")
    user_prompt = request.form.get("prompt", None)

    if not image:
        return {"error": "Input Image are required"}, 400

    image = Image.open(image)

    generated_image = pipeline.process_photo(photo=image, user_prompt=user_prompt)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        generated_image.save(tmp, format="PNG")
        tmp_path = tmp.name

    return send_file(tmp_path, mimetype="image/png", as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
