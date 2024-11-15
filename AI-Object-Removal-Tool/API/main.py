from flask import Flask, request, jsonify, send_file
from simple_lama_inpainting import SimpleLama
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
simple_lama = SimpleLama()


@app.route("/remove_object", methods=["POST"])
def inpaint():
    image_file = request.files["image"]
    mask_file = request.files["mask"]

    print("Request recieved")

    image = Image.open(image_file).convert("RGB")
    mask = Image.open(mask_file).convert("L")

    result = simple_lama(image, mask)

    result_io = io.BytesIO()
    result.save(result_io, format="PNG")
    result_io.seek(0)
    print(f"Image send")
    return send_file(result_io, mimetype="image/png")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
