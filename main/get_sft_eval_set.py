import os,json

with open("sft_preflop_test_eval.json", 'r') as json_file:
    responses = json.load(json_file)

parsed_responses = [{"instruction":entry['prompt'], "output":entry['label']} for entry in responses]

with open(f"SFT/sft_preflop_test_eval.json", 'w') as json_file:
    json.dump(parsed_responses, json_file, indent=2)