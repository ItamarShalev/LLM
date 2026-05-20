import os
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoConfig, AutoModel



models = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct",
    "allenai/OLMo-2-1124-7B-Instruct",
    "ibm-granite/granite-3.3-8b-instruct",
    "deepseek-ai/DeepSeek-V3",  
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "microsoft/Phi-4-mini-instruct",
    "tiiuae/Falcon3-7B-Instruct",
    "dicta-il/dictalm2.0-instruct"
]

root = Path(__file__).parent
load_dotenv(root.parent / ".env")
data_path = root / "data"
data_path.mkdir(exist_ok=True)

architecture_file = data_path / "architecture.txt"
hf_token = os.getenv("HF_TOKEN")

def extract_architectures():
    with open(architecture_file, "w") as f:
        for model_name in models:
            try:
                config = AutoConfig.from_pretrained(model_name, token=hf_token)
            except Exception as e:
                print(f"Error loading {model_name}: {e}")
                continue
            config_dict = config.to_dict()

            f.write(f"{model_name}: ")
            for key, value in config_dict.items():
                f.write(f"{key}={value}\n")

            f.write("\n\n\n")



if __name__ == "__main__":
    extract_architectures()