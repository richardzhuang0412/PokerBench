import pandas as pd
import random, json
import matplotlib.pyplot as plt

# DATA_PATH = "SFT/sft_preflop_full.json"
# DATA_PATH = "SFT/sft_postflop_full.json"
# DATA_PATH = "SFT/sft_preflop_test_eval.json"
DATA_PATH = "SFT/sft_postflop_test_eval.json"
PREFLOP = False
PLOT_TITLE = 'Distribution of Decision Frequency in Preflop Training Set'
PLOT_OUTPUT_PATH = "plots/postflop_test_set_decision_distribution.png"

with open(DATA_PATH, 'r') as file:
    data = json.load(file)

if PREFLOP:
    all_possible_options = ['raise', 'fold', 'call', 'check', 'all in']
    raise_subset = [entry for entry in data if "raise" in entry['output']]
    fold_subset = [entry for entry in data if "fold" in entry['output']]
    call_subset = [entry for entry in data if "call" in entry['output']]
    check_subset = [entry for entry in data if "check" in entry['output']]
    allin_subset = [entry for entry in data if "all in" in entry['output']]
    option_subsets = [raise_subset, fold_subset, call_subset, check_subset, allin_subset]
else:
    all_possible_options = ['raise', 'fold', 'call', 'check', 'bet']
    raise_subset = [entry for entry in data if "raise" in entry['output']]
    fold_subset = [entry for entry in data if "fold" in entry['output']]
    call_subset = [entry for entry in data if "call" in entry['output']]
    check_subset = [entry for entry in data if "check" in entry['output']]
    bet_subset = [entry for entry in data if "bet" in entry['output']]
    option_subsets = [raise_subset, fold_subset, call_subset, check_subset, bet_subset]

counts = [len(subset)/len(data) for subset in option_subsets]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

plt.figure(figsize=(10, 6))
bars = plt.bar(all_possible_options, counts, color=colors)
LABEL_FONTSIZE = 40
XTICK_FONTSIZE = 35
YTICK_FONTSIZE = 35
TITLE_FONTSIZE = 22
plt.xlabel('Actions', fontsize=LABEL_FONTSIZE)
plt.ylabel('Ratio', fontsize=LABEL_FONTSIZE)
plt.xticks(fontsize=XTICK_FONTSIZE)
plt.yticks(fontsize=YTICK_FONTSIZE)
# plt.title(PLOT_TITLE, fontsize=TITLE_FONTSIZE)
plt.ylim(0,1)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, round(yval, 3), ha='center', va='bottom', fontsize=25)
    
plt.tight_layout()
plt.show()

plt.savefig(PLOT_OUTPUT_PATH)