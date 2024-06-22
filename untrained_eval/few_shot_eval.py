from openai import OpenAI
from tqdm import tqdm
import requests, os, ast, json, re, random, time
from together import Together
import pandas as pd
from tqdm import tqdm
import numpy as np

# Create a few-shot-example set to sample from
def create_example_bank(full_data:list, preflop:bool, sample_per_option:int):
    random.seed(42)
    if preflop:
        all_possible_options = ['raise', 'fold', 'call', 'all in', 'check']
    else:
        all_possible_options = ['raise', 'fold', 'call', 'check', 'bet']
    bank = {}
    for option in all_possible_options:
        option_subset = [entry for entry in full_data if option in entry['output']]
        bank[option] = random.sample(option_subset, min(sample_per_option, len(option_subset)))
    return bank

# Function to sample k entries and construct prompt
def construct_k_shot_prompt(test_prompt:str, example_bank:list, k:int=5):
    random.seed(42)
    examples = []
    for option in example_bank.keys():
        examples.extend(random.sample(example_bank[option], 1))
    random.shuffle(examples)
    examples = [sample['instruction'] + " " + sample['output'] + "\n" for sample in examples]
    examples_str = "\n".join(examples)
    # print(examples_str + "\n" + test_prompt)
    return examples_str + "\n" + test_prompt[:-2]

# Inference Function
def eval_model(model_name:str, openai_api_key:str, together_api_key:str, output_path:str,
               preflop_eval_set:list, postflop_eval_set:list, preflop_bank:dict, postflop_bank:dict,
               few_shot:bool):

    openai_client = OpenAI(api_key=openai_api_key)
    together_client = Together(api_key=together_api_key)

    print(f"Evaluating {model_name}")
    model_responses = []
    for index, bank, eval_set in [(0, preflop_bank, preflop_eval_set), (1, postflop_bank, postflop_eval_set)]:
        # print(index)
        for item in tqdm(eval_set):
            if few_shot:
                prompt = construct_k_shot_prompt(item['instruction'], bank)
            else:
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
                    {"role": "user", "content": prompt},
                    ],
                    max_tokens=20,
                )
                response = completion.choices[0].message.content
            
            elif "llama" in model_name and "chat" in model_name:
                # Uncomment this if using Together Client Chat Model
                completion = together_client.chat.completions.create(
                    temperature=0.1,
                    model= model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=20,
                )
                response = completion.choices[0].message.content
            elif "llama" in model_name:
                # Uncomment this if using Together Client Language Model
                completion = together_client.completions.create(
                    model= model_name,
                    temperature=0.1,
                    prompt=prompt,
                    max_tokens=20
                )
                response = completion.choices[0].text

            print(prompt)
            print(f"Model Response: {response}")
            print(f"Correct Answer: {answer}")

            model_responses.append({"source": "preflop" if index == 0 else "postflop", 
                                "prompt": prompt, "response": response, "answer": answer})

    # eval_result_output_path = f"{model_name.split('/')[-1].replace('-', '_')}_few_shot_result_all.json"
    with open(output_path, 'w') as json_file:
        json.dump(model_responses, json_file, indent=2)
    print(f"Result for {model_name} stored in {output_path}")
        
if __name__ == "__main__":
    PREFLOP_FULL_PATH = "../SFT/sft_preflop_full.json"
    POSTFLOP_FULL_PATH = "../SFT/sft_postflop_full.json"
    PREFLOP_EVAL_SET_PATH = "../SFT/sft_preflop_test_eval.json"
    POSTFLOP_EVAL_SET_PATH = "../SFT/sft_postflop_test_eval.json"
    FEW_SHOT = True
    with open(PREFLOP_FULL_PATH, 'r') as file:
        preflop_full = json.load(file)
    with open(POSTFLOP_FULL_PATH, 'r') as file:
        postflop_full = json.load(file)
    with open(PREFLOP_EVAL_SET_PATH, 'r') as file:
        preflop_eval_set = json.load(file)
    with open(POSTFLOP_EVAL_SET_PATH, 'r') as file:
        postflop_eval_set = json.load(file)
    
    preflop_bank = create_example_bank(preflop_full, preflop=True, sample_per_option=50)
    postflop_bank = create_example_bank(postflop_full, preflop=False, sample_per_option=50)

    openai_api_key = "openai_api_key"
    together_api_key = "together_api_key"

    # model_list = ["gpt-3.5-turbo", "gpt-4-turbo", 
    #           "meta-llama/Llama-3-8b-chat-hf", "meta-llama/Llama-3-70b-chat-hf", 
    #           "meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-13b-chat-hf", "meta-llama/Llama-2-70b-chat-hf"]
    model_list = ["meta-llama/Llama-2-70b-chat-hf", "meta-llama/Llama-3-70b-chat-hf"]
    
    for model in model_list:
        OUTPUT_PATH = f"{model.split('/')[-1].replace('-', '_')}_few_shot_result_redo.json"
        eval_model(model, openai_api_key, together_api_key, OUTPUT_PATH,
               preflop_eval_set, postflop_eval_set, preflop_bank, postflop_bank,
               FEW_SHOT)
