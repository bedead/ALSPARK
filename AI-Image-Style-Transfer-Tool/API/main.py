# main.py

from flask import Flask, request, send_file
from PIL import Image
import tempfile
import torch
from io import BytesIO
from model_controller import create_pipeline
import argparse
import os
from huggingface_hub import hf_hub_url
import subprocess
import shlex

app = Flask(__name__)

urls = {
    "TencentARC/T2I-Adapter": [
        "models/coadapter-canny-sd15v1.pth",
        "models/coadapter-style-sd15v1.pth",
        "models/coadapter-fuser-sd15v1.pth",
    ],
    "runwayml/stable-diffusion-v1-5": ["v1-5-pruned-emaonly.ckpt"],
}

if os.path.exists("models") == False:
    os.mkdir("models")
    for repo in urls:
        files = urls[repo]
        for file in files:
            url = hf_hub_url(repo, file)
            name_ckp = url.split("/")[-1]
            save_path = os.path.join("models", name_ckp)
            if os.path.exists(save_path) == False:
                subprocess.run(shlex.split(f"wget {url} -O {save_path}"))

parser = argparse.ArgumentParser()
parser.add_argument(
    "--sd_ckpt",
    type=str,
    default="models/v1-5-pruned-emaonly.ckpt",
    help="path to checkpoint of stable diffusion model, both .ckpt and .safetensor are supported",
)
parser.add_argument(
    "--vae_ckpt",
    type=str,
    default=None,
    help="vae checkpoint, anime SD models usually have seperate vae ckpt that need to be loaded",
)

supported_cond = ["style", "canny"]
global_opt = parser.parse_args()
global_opt.device = "cuda" if torch.cuda.is_available() else "cpu"
global_opt.config = "configs/stable-diffusion/sd-v1-inference.yaml"
for cond_name in supported_cond:
    setattr(
        global_opt,
        f"{cond_name}_adapter_ckpt",
        f"models/coadapter-{cond_name}-sd15v1.pth",
    )
global_opt.max_resolution = 512 * 512
global_opt.sampler = "ddim"
global_opt.cond_weight = 1.0
global_opt.C = 4
global_opt.f = 8
global_opt.style_cond_tau = 1

pipeline = create_pipeline(global_opt)
print("Model and API ready.")


@app.route("/transfer-style", methods=["POST"])
def generate_image():
    image_canny_file = request.files.get("image_canny")
    image_style_file = request.files.get("image_style")

    prompt = request.form.get("prompt")
    neg_prompt = request.form.get("neg_prompt")
    scale = float(request.form.get("scale", 7.5))
    n_samples = int(request.form.get("n_samples", 1))
    steps = int(request.form.get("steps", 35))
    resize_short_edge = int(request.form.get("resize_short_edge", 512))
    cond_tau = float(request.form.get("cond_tau", 1.0))
    cond_weight_style = float(request.form.get("cond_weight_style", 1.0))
    cond_weight_canny = float(request.form.get("cond_weight_canny", 1.0))

    image_canny = None
    image_style = None

    if image_canny_file:
        image_canny = Image.open(BytesIO(image_canny_file.read()))

    if image_style_file:
        image_style = Image.open(BytesIO(image_style_file.read()))

    # Log received data for debugging
    print("Prompt:", prompt)
    print("Negative Prompt:", neg_prompt)
    print("Scale:", scale)
    print("Number of Samples:", n_samples)
    print("Steps:", steps)
    print("Resize Short Edge:", resize_short_edge)
    print("Cond Tau:", cond_tau)
    print("Cond Weight Style:", cond_weight_style)
    print("Cond Weight Canny:", cond_weight_canny)
    print("Canny Image Received:", bool(image_canny))
    print("Style Image Received:", bool(image_style))

    stylized_image = pipeline.run(
        im1_canny=image_canny,
        im1_style=image_style,
        cond_weight_style=cond_weight_style,
        cond_weight_canny=cond_weight_canny,
        prompt=prompt,
        seed=42,
        neg_prompt=neg_prompt,
        scale=scale,
        n_samples=n_samples,
        steps=steps,
        resize_short_edge=resize_short_edge,
        cond_tau=cond_tau,
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        stylized_image.save(tmp, format="PNG")
        tmp_path = tmp.name

    return send_file(tmp_path, mimetype="image/png", as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
