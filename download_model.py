# download_model.py
import os
from transformers import AutoTokenizer, AutoModel

# Get the Hugging Face token from an environment variable
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set!")

print("Starting model download...")
# The model identifier from your code
model_name = 'Salesforce/codet5p-110m-embedding'

# Download tokenizer and model. This will save them to the default cache directory.
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    model = AutoModel.from_pretrained(model_name, token=HF_TOKEN, trust_remote_code=True)
    print("Model and tokenizer downloaded and cached successfully.")
except Exception as e:
    print(f"An error occurred during download: {e}")