from flask import Flask, request, send_file
from PIL import Image
import io
import numpy as np
from model_controller import initialize_model, predict

app = Flask(__name__)

# Initialize the model pipeline
pipeline = initialize_model()


@app.route("/remove_object", methods=["POST"])
def inpaint():
    try:
        # Get input files from the request
        image_file = request.files["image"]
        mask_file = request.files["mask"]

        print("Request received")

        image = Image.open(image_file).convert("RGB")
        mask = Image.open(mask_file).convert("RGB")

        input_data = {
            "image": image,
            "mask": mask,
        }

        dict_out, dict_res = predict(
            pipe=pipeline,
            input_image=input_data,
            prompt="",
            fitting_degree=1,
            ddim_steps=15,
            scale=12,
            negative_prompt="",
        )

        result_image = dict_out[1]

        result_io = io.BytesIO()
        result_image.save(result_io, format="PNG")
        result_io.seek(0)

        print("Image sent")
        return send_file(result_io, mimetype="image/png")

    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}, 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
