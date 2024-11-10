import gradio as gr
import requests
import tempfile

API_URL = "http://127.0.0.1:5000/gen_model"


def gen_model(prompt_input):
    if not prompt_input:
        return "Error: Please provide an prompt!"
    datas = {"prompt": prompt_input}

    response = requests.post(API_URL, data=datas)

    if response.status_code == 200:
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name

        return temp_file_path
    else:
        print(f"Error: {response.text}")
        return "Error generating 3D model!"


# Gradio components
prompt_input = gr.components.Textbox(
    label="Enter Prompt", placeholder="Example: a shark"
)
output_glb = gr.components.Model3D(
    clear_color=[0.0, 0.0, 0.0, 0.0], label="Generated 3D Model (.glb)"
)

# Gradio Interface
gr.Interface(
    fn=gen_model, inputs=prompt_input, outputs=output_glb, title="AI 3D Model Generator"
).launch(share=True)
