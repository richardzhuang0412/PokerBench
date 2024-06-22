import torch
from transformers import AutoModel

# huggingface-cli login
print("Start Loading")
state_dict = torch.load("out/llama-2-base-pt-hf/model.pth")
model = AutoModel.from_pretrained(
    "out/llama-2-base-pt-hf/", local_files_only=True, state_dict=state_dict
)
print("Finish Loading")

print("Start Pushing")
model.push_to_hub("RZ412/poker-llama-2-7b-base-pt")
print("Finish Pushing")