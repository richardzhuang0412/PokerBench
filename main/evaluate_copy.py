import requests, json, random, os, signal, subprocess, time, argparse
from tqdm import tqdm
from untrained_eval.few_shot_eval import *

random.seed(42)
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str)
parser.add_argument("--log_path", type=str)
# parser.add_argument("--few_shot", type=bool)
parser.add_argument("--port", type=int)
args = parser.parse_args()
# print(args.model_path)
PREFLOP_TRAIN_EVAL_FILENAME = "SFT/sft_large_preflop_train_eval.json"
POSTFLOP_TRAIN_EVAL_FILENAME = "SFT/sft_large_postflop_train_eval.json"
PREFLOP_TEST_EVAL_FILENAME = "SFT/sft_preflop_test_eval.json"
POSTFLOP_TEST_EVAL_FILENAME = "SFT/sft_postflop_test_eval.json"
PREFLOP_FULL_PATH = "SFT/sft_preflop_full.json"
POSTFLOP_FULL_PATH = "SFT/sft_postflop_full.json"
EVAL_RESULT_OUTPUT_FOLDER = "eval_result_llama_3_8b_110k"
EVAL_LOG_OUTPUT_PATH = args.log_path
PORT = args.port
FEW_SHOT = False


model_responses = []
if FEW_SHOT:
    with open(PREFLOP_FULL_PATH, 'r') as file:
        preflop_full = json.load(file)
    with open(POSTFLOP_FULL_PATH, 'r') as file:
        postflop_full = json.load(file)
    preflop_bank = create_example_bank(preflop_full, preflop=True, sample_per_option=50)
    postflop_bank = create_example_bank(postflop_full, preflop=False, sample_per_option=50)
    eval_ls = [(PREFLOP_TEST_EVAL_FILENAME, preflop_bank), (POSTFLOP_TEST_EVAL_FILENAME, postflop_bank)]
else:
    eval_ls = [(PREFLOP_TRAIN_EVAL_FILENAME, None), (POSTFLOP_TRAIN_EVAL_FILENAME, None), 
                (PREFLOP_TEST_EVAL_FILENAME, None), (POSTFLOP_TEST_EVAL_FILENAME, None)]

try:
    for file, bank in eval_ls:
        with open(file, 'r') as json_file:
            eval_data = json.load(json_file)
            random.shuffle(eval_data)

        count = 0
        EVAL_SIZE = len(eval_data)
        for i in tqdm(range(EVAL_SIZE)):
            if FEW_SHOT:
                prompt = construct_k_shot_prompt(eval_data[i]['instruction'], bank)
                # print(prompt)
            else:
                prompt = eval_data[i]['instruction']
            response = requests.post(
                f"http://127.0.0.1:{PORT}/predict", 
                json={"prompt": prompt}
            )

            # print(response.json()['output'].split("Response:")[0])
            # print(response.json()['output'])
            model_response = response.json()['output'].rsplit("Your optimal action is:")[-1]
            # if args.few_shot:
                # print(f"Model Response: {model_response}")
            # print(eval_data[i]['output'])
            # print(eval_data[i]['label'].lower() in model_response.lower())

            if (eval_data[i]['output'].lower() in model_response.lower()) or (
                "call" in eval_data[i]['output'].lower() and "call" in model_response.lower()) or (
                "all in" in eval_data[i]['output'].lower() and "all in" in model_response.lower()) or (
                "allin" in eval_data[i]['output'].lower() and "allin" in model_response.lower()) or (
                "fold" in eval_data[i]['output'].lower() and "fold" in model_response.lower()) or (
                "check" in eval_data[i]['output'].lower() and "check" in model_response.lower()):
                match = 1
            elif ("raise" in eval_data[i]['output'].lower() and "raise" in model_response.lower()) or (
                "bet" in eval_data[i]['output'].lower() and "bet" in model_response.lower()):
                match = 0.5
            else:
                match = 0

            count += match
            # if i % 100 == 0 and i != 0:
            #     print(count/i)
            model_responses.append({"source": file, "prompt": prompt, "response": model_response, 
                                    "label": eval_data[i]['output'], "match": match})

        print(f"Accuracy of {args.model_path} on {file}: {round(count/EVAL_SIZE,3)}")

    with open(f"SFT/{EVAL_RESULT_OUTPUT_FOLDER}/{args.model_path}_zero_shot_eval_result.json", 'w') as json_file:
        json.dump(model_responses, json_file, indent=2)
    print("Responses Saved Successfully")
except:
    with open(EVAL_LOG_OUTPUT_PATH, 'a') as f:
            f.write(f"Error in Evaluating {args.model_path}\n")
# with open("eval_small_processed_no_option.json", 'r') as json_file:
#     eval_data = json.load(json_file)
#     random.shuffle(eval_data)

# count = 0
# SIZE = len(eval_data)
# model_responses = []
# for i in tqdm(range(SIZE)):
#     prompt = eval_data[i]['prompt']
#     response = requests.post(
#         "http://127.0.0.1:8000/predict", 
#         json={"prompt": prompt}
#     )

#     # print(response.json()['output'].split("Response:")[0])
#     # print(response.json()['output'].split("optimal action is:")[1])
#     # model_response = response.json()['output'].split("Response:")[1]
#     model_response = response.json()['output'].split("Your optimal action is:")[1]
#     # print(model_response)
#     # print(eval_data[i]['label'])
#     # print(eval_data[i]['label'].lower() in model_response.lower())
#     if (eval_data[i]['label'].lower() in model_response.lower()) or (
#         "call" in eval_data[i]['label'].lower() and "call" in model_response.lower()) or (
#         "all in" in eval_data[i]['label'].lower() and "all in" in model_response.lower()) or (
#         "allin" in eval_data[i]['label'].lower() and "allin" in model_response.lower()) or (
#         "fold" in eval_data[i]['label'].lower() and "fold" in model_response.lower()) or (
#         "check" in eval_data[i]['label'].lower() and "check" in model_response.lower()):
#         match = 1
#     elif ("raise" in eval_data[i]['label'].lower() and "raise" in model_response.lower()) or (
#         "bet" in eval_data[i]['label'].lower() and "bet" in model_response.lower()):
#         match = 0.5
#     else:
#         match = 0

#     count += match
#     # if i % 100 == 0 and i != 0:
#     #     print(count/i)
#     model_responses.append({"prompt": prompt, "response": model_response, "label": eval_data[i]['label'], "match": match})

# print(f"Accuracy of {args.model_path} on Eval Dataset: {round(count/SIZE,3)}")

# with open(f"SFT/eval_result/{args.model_path}_result_test.json", 'w') as json_file:
#     json.dump(model_responses, json_file, indent=2)
# print("Responses Saved Successfully")