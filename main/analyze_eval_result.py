import os, json, math

EVAL_RESULT_FOLDER = "SFT/eval_result_110k/"
OUTPUT_FILENAME = "llama_2_7b_eval_log_cleaned_110k.txt"
FILE_KEYWORD = "result_all"
# FILE_KEYWORD = "zero_shot"

accs = []
for checkpoint in sorted(os.listdir(EVAL_RESULT_FOLDER)):
    if FILE_KEYWORD not in checkpoint:
        continue
    checkpoint_name = checkpoint.split("_")[0]
    # print(checkpoint)
    # print("SFT/eval_result/" + checkpoint)
    try:
        with open(EVAL_RESULT_FOLDER + checkpoint, 'r') as json_file:
            responses = json.load(json_file)
    except:
        print(f"Error in loading eval results for {checkpoint_name}")
        continue

    result_train_preflop_exact_match = [math.floor(entry['match']) for entry in responses if "preflop_train_eval" in entry['source']]
    result_train_postflop_exact_match = [math.floor(entry['match']) for entry in responses if "postflop_train_eval" in entry['source']]
    result_train_preflop_action_match = [math.ceil(entry['match']) for entry in responses if "preflop_train_eval" in entry['source']]
    result_train_postflop_action_match = [math.ceil(entry['match']) for entry in responses if "postflop_train_eval" in entry['source']]
    result_test_preflop_exact_match = [math.floor(entry['match']) for entry in responses if "preflop_test_eval" in entry['source']]
    result_test_postflop_exact_match = [math.floor(entry['match']) for entry in responses if "postflop_test_eval" in entry['source']]
    result_test_preflop_action_match = [math.ceil(entry['match']) for entry in responses if "preflop_test_eval" in entry['source']]
    result_test_postflop_action_match = [math.ceil(entry['match']) for entry in responses if "postflop_test_eval" in entry['source']]

    acc_train_preflop_exact_match = round(sum(result_train_preflop_exact_match)/len(result_train_preflop_exact_match),4)
    acc_train_postflop_exact_match = round(sum(result_train_postflop_exact_match)/len(result_train_postflop_exact_match),4)
    acc_train_preflop_action_match = round(sum(result_train_preflop_action_match)/len(result_train_preflop_action_match),4)
    acc_train_postflop_action_match = round(sum(result_train_postflop_action_match)/len(result_train_postflop_action_match),4)
    acc_test_preflop_exact_match = round(sum(result_test_preflop_exact_match)/len(result_test_preflop_exact_match),4)
    acc_test_postflop_exact_match = round(sum(result_test_postflop_exact_match)/len(result_test_postflop_exact_match),4)
    acc_test_preflop_action_match = round(sum(result_test_preflop_action_match)/len(result_test_preflop_action_match),4)
    acc_test_postflop_action_match = round(sum(result_test_postflop_action_match)/len(result_test_postflop_action_match),4)

    accs.append(f"Preflop Train Exact Match Accuracy for {checkpoint_name}: {acc_train_preflop_exact_match}\n") 
    accs.append(f"Preflop Train Action Match Accuracy for {checkpoint_name}: {acc_train_preflop_action_match}\n")
    accs.append(f"Postflop Train Exact Match Accuracy for {checkpoint_name}: {acc_train_postflop_exact_match}\n") 
    accs.append(f"Postflop Train Action Match Accuracy for {checkpoint_name}: {acc_train_postflop_action_match}\n")
    accs.append(f"Preflop Test Exact Match Accuracy for {checkpoint_name}: {acc_test_preflop_exact_match}\n") 
    accs.append(f"Preflop Test Action Match Accuracy for {checkpoint_name}: {acc_test_preflop_action_match}\n")
    accs.append(f"Postflop Test Exact Match Accuracy for {checkpoint_name}: {acc_test_postflop_exact_match}\n") 
    accs.append(f"Postflop Test Action Match Accuracy for {checkpoint_name}: {acc_test_postflop_action_match}\n")
    accs.append(f"\n")
    
with open(OUTPUT_FILENAME, 'w') as file:
    file.writelines(accs)