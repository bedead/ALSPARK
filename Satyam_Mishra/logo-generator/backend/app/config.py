import os

class Config:
    # MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/your_db')
    # SDXL_MODEL_PATH = os.getenv('SDXL_MODEL_PATH', '/path/to/finetuned/LoRa/model')
    BASE_MODEL = "runwayml/stable-diffusion-v1-5"
    LoRA_WEIGHTS = "artificialguybr/logo-redmond-1-5v-logo-lora-for-liberteredmond-sd-1-5"
    ADAPTOR_NAME = "SD Logo LoRA"

    # BASE_MODEL1 = "stabilityai/stable-diffusion-xl-base-1.0"
    # LoRA_WEIGHTS1 = "artificialguybr/LogoRedmond-LogoLoraForSDXL-V2"
    # ADAPTOR_NAME1 = "SDXL Logo LoRA"

