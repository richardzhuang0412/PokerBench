import json

EVAL_DATASET_PATH = "preflop_1k.json"
OUTPUT_PATH = "sft_preflop_test_eval.json"

with open(EVAL_DATASET_PATH, 'r') as json_file:
    eval_data = json.load(json_file)

for i in range(len(eval_data)):
    eval_data[i]['prompt'] = eval_data[i]['prompt'][0:268] + "\nThe player positions involved in this game are UTG, HJ, CO, BTN, SB, BB.\n" + eval_data[i]['prompt'][269:]

    REMOVE_OPTIONS = True

    if REMOVE_OPTIONS:
        eval_data[i]['prompt'] = eval_data[i]['prompt'][0:268] + "\nThe player positions involved in this game are UTG, HJ, CO, BTN, SB, BB.\n" + eval_data[i]['prompt'][269:]
        eval_data[i]['prompt'] = eval_data[i]['prompt'].split("your available moves are:")[0] + \
            "and your holding is" + \
            eval_data[i]['prompt'].rsplit("and your holding is")[2]
        eval_data[i]['prompt'] = eval_data[i]['prompt'].split(
                "Only give your decision in the format of the available moves above.")[0] + \
            eval_data[i]['prompt'].split(
                "Only give your decision in the format of the available moves above.")[1]

# print(eval_data[0]['prompt'])

def replace_keywords(data):
    if isinstance(data, dict):
        return {k: replace_keywords(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [replace_keywords(item) for item in data]
    elif isinstance(data, str):
        data = data.replace("ALLIN", "all in")
        data = data.replace("CALL", "call")
        data = data.replace("RAISE", "raise")
        data = data.replace("CHECK", "check")
        data = data.replace("BET", "bet")
        data = data.replace("FOLD", "fold")
        data = data.replace("UNDER_THE_GUN", "UTG")
        data = data.replace("HIJACK", "HJ")
        data = data.replace("CUTOFF", "CO")
        data = data.replace("BUTTON", "BTN")
        data = data.replace("SMALL_BLIND", "SB")
        data = data.replace("BIG_BLIND", "BB")
        data = data.replace("BIG_BLIND", "BB")
        return data
    else:
        return data

eval_data = replace_keywords(eval_data)

with open(OUTPUT_PATH, 'w') as json_file:
    json.dump(eval_data, json_file, indent=2)

