import vllm, requests

def host_vllm_model(gpu_id:int, checkpoint_path:str, port:str):
    pass

def query_vllm_host(prompt:str, port:int):
    pass

# url = "http://localhost:6000/v1/completions"
# data = {
#         "model": "facebook/opt-13b",
#         "prompt": "The law of gravity was discovered by",
#         "temperature": 0.8,
#         "top_p":1,
#         "max_tokens":200
#         }

# response = requests.post(url, json=data)

# print(response.text)

from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)

completion = client.chat.completions.create(
  model="meta-llama/Meta-Llama-3-70B-Instruct",
  messages=[
    {"role": "user", "content": "Hello!"}
  ],
  temperature=0.1,
  max_tokens=20
)

print(completion.choices[0].message.content)