import pandas as pd
import json

# result = pd.read_csv("history/110k_step_3400_vs_110k_step_3800_result.csv")
# print(result.groupby('game_stage').count()['game_number'].sort_values(ascending=False)/6)
# for stage in ['preflop', 'flop', 'turn', 'river']:
#     print(stage)
#     subset = result[result['game_stage']==stage].copy()
#     subset['winning'] = subset['ending_chip'] - 200
#     # print(sum(result['winning']))
#     winnings = {}
#     # print((subset.groupby('model_name').sum()['winning']) / max(subset['game_number']))
#     print((subset.groupby('game_stage').count()/6))

with open("../SFT/eval_result_110k/step-005000_eval_result_all.json", "r") as json_flle:
    responses = json.load(json_flle)

print(len([entry for entry in responses if (("raise" in entry['label']) or ("bet" in entry['label']))
           and ("preflop_test" in entry['source'])]))
print(len([entry for entry in responses if (("raise" in entry['label']) or ("bet" in entry['label']))
           and (entry['match'] >= 0.5) and ("preflop_test" in entry['source'])]))