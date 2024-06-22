import json, random

random.seed(42)

PREFLOP_JSON_FILENAME = "sft_preflop_large.json"
POSTFLOP_JSON_FILENAME = "sft_postflop_large.json"
COMBINED_JSON_FILENAME = "sft_combined_large.json"

with open(PREFLOP_JSON_FILENAME, 'r') as json_file:
    preflop_json = json.load(json_file)
with open(POSTFLOP_JSON_FILENAME, 'r') as json_file:
    postflop_json = json.load(json_file)

combined_json = []
combined_json.extend(preflop_json)
combined_json.extend(postflop_json)
random.shuffle(combined_json)
with open(COMBINED_JSON_FILENAME, 'w') as json_file:
    json.dump(combined_json, json_file, indent=2)
