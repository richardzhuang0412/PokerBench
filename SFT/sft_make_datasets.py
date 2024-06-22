import pandas as pd
import random, json

random.seed(42)
# Create Dataset of Size 
# 20k (preflop 10k + postflop 10k), 60k (preflop 30k + postflop 30k), and 110k (preflop 30k + postflop 80k)

RANDOM_SEED = 42
PREFLOP_SIZE = 30000
POSTFLOP_SIZE = 80000

PREFLOP_FILEPATH = "sft_preflop_large.csv"
POSTFLOP_FILEPATH = "sft_postflop_large.csv"

preflop_full = pd.read_csv("preflop_filtered.csv")
postflop_full = pd.read_csv("postflop_eval_dataset_filtered.csv", index_col=0)
# print(preflop_full.head(5))
# print(postflop_full.head(5))
preflop_test = pd.read_csv("preflop_eval_1k.csv", index_col=0)
postflop_test = pd.read_csv("postflop_eval_1k_final.csv", index_col=0)
# print(preflop_test.head(5))
# print(postflop_test.head(5))
print("Finish Reading Dataset")

raise_subset = preflop_full[(preflop_full['correct_decision'] == "allin") | (preflop_full['correct_decision'].str.contains('bb'))]
call_subset = preflop_full[preflop_full['correct_decision'] == "call"]
check_subset = preflop_full[preflop_full['correct_decision'] == "check"]
fold_subset = preflop_full[preflop_full['correct_decision'] == "fold"]
preflop_sample = pd.concat([
    raise_subset.sample(1934, random_state=42), call_subset.sample(2289, random_state=42),
    check_subset.sample(69, random_state=42), fold_subset.sample(PREFLOP_SIZE-1934-2289-69+1000, random_state=42)
    ]).reset_index(drop=True)
postflop_sample = postflop_full.sample(POSTFLOP_SIZE + postflop_test.shape[0], random_state=RANDOM_SEED)
print("Finish Sampling")

# Filtering out rows in the SFT dataset that are in the test set
def row_in_subset(row, subset):
    return subset.apply(lambda x: row.equals(x), axis=1).any()

preflop_sample = preflop_sample[~preflop_sample.apply(lambda row: row_in_subset(row, preflop_test), 
                                                      axis=1)].reset_index(drop=True).iloc[0:PREFLOP_SIZE]
postflop_sample = postflop_sample[~postflop_sample.apply(lambda row: row_in_subset(row, preflop_test), 
                                                         axis=1)].reset_index(drop=True).iloc[0:POSTFLOP_SIZE]

print(preflop_sample.shape)
print(postflop_sample.shape)

preflop_sample.to_csv(PREFLOP_FILEPATH)
postflop_sample.to_csv(POSTFLOP_FILEPATH)

