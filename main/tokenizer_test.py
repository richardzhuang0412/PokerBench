from transformers import AutoTokenizer

# Load the Llama tokenizer (replace with the specific Llama model if needed)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")

# Test tokenization of "CALL" vs " CALL"
text_1 = "ALL IN"
text_2 = "all in"

# Tokenize the texts
tokens_1 = tokenizer.tokenize(text_1)
tokens_2 = tokenizer.tokenize(text_2)

# Print the tokens
print(f"Tokens for '{text_1}': {tokens_1}")
print(f"Tokens for '{text_2}': {tokens_2}")

# Optional: Print the corresponding token IDs
token_ids_1 = tokenizer.convert_tokens_to_ids(tokens_1)
token_ids_2 = tokenizer.convert_tokens_to_ids(tokens_2)

print(f"Token IDs for '{text_1}': {token_ids_1}")
print(f"Token IDs for '{text_2}': {token_ids_2}")
