import pandas as pd
import json, math, re, time

# Determine Match
def check_match(model_response:str, label:str):
    # print(f"Model Response: {model_response}")
    # print(f"Label: {label}")
    if (label.lower() in model_response.lower()) or (
    "call" in label.lower() and "call" in model_response.lower()) or (
    "all in" in label.lower() and "all in" in model_response.lower()) or (
    "allin" in label.lower() and "allin" in model_response.lower()) or (
    "fold" in label.lower() and "fold" in model_response.lower()) or (
    "check" in label.lower() and "check" in model_response.lower()):
        return 1
    if (("raise" in model_response.lower()) and (
        "raise" in label.lower())) or ((
        "bet" in model_response.lower()) and (
        "bet" in label.lower())):
        pattern = r'\d+\.\d+|\d+'
        model_response_match = re.search(pattern, model_response)
        label_match = re.search(pattern, label)
        if model_response_match:
            model_size = float(model_response_match.group(0))
            label_size = float(label_match.group(0))
            # print(f"Model Size: {model_size}")
            # print(f"Label Size: {label_size}")
            if model_size == label_size:
                return 1
            else:
                return 0.5
        else:
            return 0.5
    return 0

# Evaluate Model Accuracy
def summarize_result(result:list):
    preflop_num_exact_match = 0
    postflop_num_exact_match = 0
    preflop_num_action_match = 0
    postflop_num_action_match = 0
    preflop_total_num = 0
    postflop_total_num = 0
    for entry in result:
        match = check_match(entry['response'], entry['answer'])
        # print(f"Match: {match}")
        # if (entry['answer'].lower() in entry['response'].lower()) or (
        #     "call" in entry['answer'].lower() and "call" in entry['response'].lower()) or (
        #     "all in" in entry['answer'].lower() and "all in" in entry['response'].lower()) or (
        #     "allin" in entry['answer'].lower() and "allin" in entry['response'].lower()) or (
        #     "fold" in entry['answer'].lower() and "fold" in entry['response'].lower()) or (
        #     "check" in entry['answer'].lower() and "check" in entry['response'].lower()):
        #     match = 1
        # elif ("raise" in entry['answer'].lower() and "raise" in entry['response'].lower()) or (
        #     "bet" in entry['answer'].lower() and "bet" in entry['response'].lower()):
        #     match = 0.5
        # else:
        #     match = 0

        if entry['source'] == "preflop":
            preflop_total_num += 1
            preflop_num_exact_match += math.floor(match)
            preflop_num_action_match += math.ceil(match)
        elif entry['source'] == "postflop":
            postflop_total_num += 1
            postflop_num_exact_match += math.floor(match)
            postflop_num_action_match += math.ceil(match)

    return (preflop_num_exact_match, preflop_num_action_match, preflop_total_num, 
            postflop_num_exact_match, postflop_num_action_match, postflop_total_num)

if __name__ == "__main__":
    
    # MODEL_LIST = ["gpt-3.5-turbo", "gpt-4-turbo", 
    #           "Llama-3-8b-chat-hf", "Llama-3-70b-chat-hf", "Llama-2-70b-chat-hf"]
    MODEL_LIST = ["Llama-2-70b-chat-hf"]
    # MODEL_LIST = ["gpt-3.5-turbo", "gpt-4-turbo"]

    for model in MODEL_LIST:
        RESULT_PATH = f"{model.replace('-', '_')}_few_shot_result_redo.json"

        with open(RESULT_PATH, 'r') as file:
            model_result = json.load(file)

        (preflop_num_exact_match, preflop_num_action_match, 
        preflop_total_num, postflop_num_exact_match, 
        postflop_num_action_match, postflop_total_num) = summarize_result(model_result)
        
        print(f"Results for {model}:")
        print(f"Preflop Exact Match Accuracy: {round(preflop_num_exact_match/preflop_total_num,4)}")
        print(f"Preflop Action Match Accuracy: {round(preflop_num_action_match/preflop_total_num,4)}")
        print(f"Total Exact Match Accuracy: {round((preflop_num_exact_match + postflop_num_exact_match)/(preflop_total_num + postflop_total_num),4)}")
        print(f"Postflop Exact Match Accuracy: {round(postflop_num_exact_match/postflop_total_num,4)}")
        print(f"Postflop Action Match Accuracy: {round(postflop_num_action_match/postflop_total_num,4)}")
        print(f"Total Action Match Accuracy: {round((preflop_num_action_match + postflop_num_action_match)/(preflop_total_num + postflop_total_num),4)}")
    
