import json,random

random.seed(42)
SFT_DATASET_FILENAME = "sft_combined_small.json"
PREFLOP_TRAIN_EVAL_DATASET_FILENAME = "sft_small_preflop_train_eval.json"
POSTFLOP_TRAIN_EVAL_DATASET_FILENAME = "sft_small_postflop_train_eval.json"

with open(SFT_DATASET_FILENAME, 'r') as json_file:
    combined_json = json.load(json_file)

preflop_subset = [entry for entry in combined_json if "The flop comes" not in entry['instruction']]
random.shuffle(preflop_subset)
raise_subset = [entry for entry in preflop_subset if ("raise" in entry['output'] or "all in" in entry['output'])]
call_subset = [entry for entry in preflop_subset if "call" in entry['output']]
check_subset = [entry for entry in preflop_subset if "check" in entry['output']]
fold_subset = [entry for entry in preflop_subset if "fold" in entry['output']]
print(len(raise_subset), len(call_subset), len(check_subset), len(fold_subset), )
preflop_train_eval_set = raise_subset[0:250] + call_subset[0:250] + check_subset + fold_subset[0:(1000-500-len(check_subset))]
random.shuffle(preflop_train_eval_set)
postflop_subset = [entry for entry in combined_json if "The flop comes" in entry['instruction']]
random.shuffle(postflop_subset)
postflop_train_eval_set = postflop_subset[0:1000]
random.shuffle(postflop_train_eval_set)
print(len(preflop_subset))
print(len(postflop_subset))

with open(PREFLOP_TRAIN_EVAL_DATASET_FILENAME, 'w') as json_file:
    json.dump(preflop_train_eval_set, json_file, indent=2)

with open(POSTFLOP_TRAIN_EVAL_DATASET_FILENAME, 'w') as json_file:
    json.dump(postflop_train_eval_set, json_file, indent=2)

