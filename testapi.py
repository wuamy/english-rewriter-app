# from huggingface_hub import InferenceClient

# client = InferenceClient(token="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") # Replace with your token
# response = client.text_generation(model="distilbert/distilgpt2", prompt="Hello, how are you?")
# print(response)

import os
from huggingface_hub import InferenceClient

# Recommended: Get your token from https://huggingface.co/settings/tokens
# os.environ["HUGGINGFACE_TOKEN"] = "hf_..."

# Use a model known to have an inference provider
client = InferenceClient(model="meta-llama/Llama-3.1-8B-Instruct")
response = client.text_generation(prompt="Hello, how are you?")
print(response)
