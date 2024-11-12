import tempfile
import torch
import numpy as np
from PIL import Image
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation
import rembg
import shlex
import subprocess

device = "cuda" if torch.cuda.is_available() else "cpu"


def initialize_model():
    subprocess.run(
        shlex.split("pip install wheel/torchmcubes-0.1.0-cp310-cp310-linux_x86_64.whl")
    )

    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(131072)
    model.to(device)

    return model


rembg_session = rembg.new_session()


def preprocess_image(input_image, do_remove_background, foreground_ratio):
    """Preprocesses the image by removing the background and resizing foreground."""

    def fill_background(image):
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        return Image.fromarray((image * 255.0).astype(np.uint8))

    if do_remove_background:
        image = input_image.convert("RGB")
        image = remove_background(image, rembg_session)
        image = resize_foreground(image, foreground_ratio)
        image = fill_background(image)
    else:
        image = input_image
        if image.mode == "RGBA":
            image = fill_background(image)
    return image


def generate_3d_models(image, mc_resolution, model):
    """Generates 3D models in OBJ and GLB formats."""
    scene_codes = model(image, device=device)
    mesh = model.extract_mesh(scene_codes, resolution=mc_resolution)[0]
    mesh = to_gradio_3d_orientation(mesh)

    with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as obj_file:
        mesh.apply_scale([-1, 1, 1])
        mesh.export(obj_file.name)
        obj_file_path = obj_file.name

    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as glb_file:
        mesh.export(glb_file.name)
        glb_file_path = glb_file.name

    return obj_file_path, glb_file_path
