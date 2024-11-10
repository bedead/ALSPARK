from flask import Flask, request, send_file
import tempfile
import trimesh
import numpy as np
from PIL import Image
from model import initialize_model, gen_model  # Import initialize_model to load the model on startup

app = Flask(__name__)

model = initialize_model()

def to_glb(ply_path: str) -> str:
    mesh = trimesh.load(ply_path)
    rot = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    mesh = mesh.apply_transform(rot)
    rot = trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0])
    mesh = mesh.apply_transform(rot)
    mesh_path = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
    mesh.export(mesh_path.name, file_type="glb")
    return mesh_path.name

@app.route("/gen_model", methods=["POST"])
def generate_model():
    prompt = request.form.get("prompt")

    if not prompt:
        return {"error": "Prompt is required"}, 400

    ply_path = gen_model(prompt=prompt, model=model)
    tmp_path = to_glb(ply_path=ply_path)

    return send_file(tmp_path, mimetype="model/gltf-binary", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)