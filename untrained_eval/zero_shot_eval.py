from openai import OpenAI
from tqdm import tqdm
import requests, os, ast, json, re, random, time
from together import Together
import pandas as pd
from tqdm import tqdm
import numpy as np

# TODO: Change the path!
PREFLOP_EVAL_SET_PATH = "../SFT/sft_preflop_test_eval.json"
POSTFLOP_EVAL_SET_PATH = "../SFT/sft_postflop_test_eval.json"
with open(PREFLOP_EVAL_SET_PATH, 'r') as file:
    preflop_eval_set = json.load(file)
with open(POSTFLOP_EVAL_SET_PATH, 'r') as file:
    postflop_eval_set = json.load(file)
# print(preflop_eval_set[0:3])
# print(postflop_eval_set[0:3])
# print(len(preflop_eval_set))
# print(len(postflop_eval_set))

# TODO: Change the API Keys!
openai_api_key = ""
together_api_key = ""

# model_list = ["gpt-3.5-turbo", "gpt-4-turbo", 
#               "meta-llama/Llama-3-8b-chat-hf", "meta-llama/Llama-3-70b-chat-hf", 
#               "meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-13b-chat-hf", "meta-llama/Llama-2-70b-chat-hf"]
model_list = ["gpt-3.5-turbo"]

openai_client = OpenAI(api_key=openai_api_key)
together_client = Together(api_key=together_api_key)

for model_name in model_list:
    print(f"Evaluating {model_name}")
    model_responses = []
    for index, eval_data in enumerate([preflop_eval_set, postflop_eval_set]):
        # print(index)
        for item in tqdm(eval_data):
            prompt = item['instruction']
            # options = item['options'] # Uncomment if evaluating with options
            answer = item['output']

            if "gpt" in model_name:
                # Uncomment this if using OpenAI Client
                completion = openai_client.chat.completions.create(
                    temperature=0.1,
                    model= model_name,
                    messages=[
                    {"role": "system", "content": ""},
                    {"role": "user", "content": prompt}
                    ]
                )
                response = completion.choices[0].message.content
            
            elif "llama" in model_name:
                # Uncomment this if using Together Client Chat Model
                completion = together_client.chat.completions.create(
                    temperature=0.05,
                    model= model_name,
                    messages=[{"role": "user", "content": prompt}],
                )
                response = completion.choices[0].message.content

            # # Uncomment this if using Together Client Language Model
            # completion = together_client.completions.create(
            #     model= model_name,
            #     prompt=prompt,
            # )
            # response = completion.choices[0].text

            # print(f"Model Response: {response}")
            # print(f"Correct Answer: {answer}")

            model_responses.append({"source": "preflop" if index == 0 else "postflop", 
                                "prompt": prompt, "response": response, "answer": answer})

    eval_result_output_path = f"{model_name.split('/')[-1].replace('-', '_')}_result_all.json"
    print(f"Result for {model_name} stored in {eval_result_output_path}")
    with open(eval_result_output_path, 'w') as json_file:
        json.dump(model_responses, json_file, indent=2)

#     # Evaluate Model Accuracy
#     response_df = pd.DataFrame(responses)
#     model_acc = round(sum([response.replace(" ", "").lower() == answer.replace(' ', '').lower() for response, answer in zip(list(response_df['response']), 
#                                                         list(response_df['answer']))])/response_df.shape[0], 3)
#     print(f"Accuracy for {model_name} in {data_name}: {model_acc}")

#     # TODO: Change the path!
#     accuracy_output_path = f"zero_shot_eval/model_accuracies.txt"
#     with open(accuracy_output_path, 'a') as file:
#         file.write(f"Accuracy for {model_name} in {data_name}: {model_acc}\n")
