import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
import tempfile
from diffusers.utils.export_utils import export_to_ply

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
guidance_scale = 15.0
model_id1 = "openai/shap-e"

torch.set_grad_enabled(False)

def initialize_model():
    pipe = DiffusionPipeline.from_pretrained(
        model_id1,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
    )
    pipe = pipe.to(device)
    if device == 'cuda':
        pipe.enable_model_cpu_offload()

    return pipe

def gen_model(prompt, model):
    with torch.inference_mode():
        output = model(
            prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=64,
            frame_size=256,
            output_type="mesh"
        ).images[0]

    torch.cuda.empty_cache()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ply") as tmp:
        ply_path = export_to_ply(output, tmp.name)

    return ply_path
