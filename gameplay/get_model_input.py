import torch
from transformers import AutoModel, AutoTokenizer


state_dict = torch.load("../out/llama-2-base-ft-20k-hf/step-002500/model.pth")
model = AutoModel.from_pretrained(
    "../out/llama-2-base-ft-20k-hf/step-002500/", local_files_only=True, state_dict=state_dict
)

# Set the model to evaluation mode
model.eval()

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("../out/llama-2-base-ft-20k-hf/step-002500/", local_files_only=True)

# Function to make inference
def make_inference(model, input_text):
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Process the outputs (depends on the model and task)
    # For example, if it's a language model, you might want to generate text
    return outputs

input_text = "Hello, how are you?"
outputs = make_inference(model, input_text)

print(outputs)
