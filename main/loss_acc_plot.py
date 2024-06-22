import re
import matplotlib.pyplot as plt

def parse_training_log(training_log_path:str, per_every_step:int):
    data = {}
    with open(training_log_path, 'r') as f:
        for line in f:
            match = re.match(r'Epoch \d+ \| iter \d+ step (\d+) \| loss train: ([\d.]+), val: ([\d.]+)', line)
            if match:
                step = int(match.group(1))
                if step % per_every_step == 0:
                    train_loss = float(match.group(2))
                    val_loss = float(match.group(3))
                    data[step] = [train_loss, val_loss, 
                                  None, None, None, None, None, None, None, None]  # Initialize Train and Test accuracies as None
    return data

def parse_evaluation_log(evaluation_log_path, data):
    with open(evaluation_log_path, 'r') as f:
        for line in f:
            preflop_train_exact_match = re.match(r'Preflop Train Exact Match Accuracy for step-(\d+): ([\d.]+)', line)
            preflop_train_action_match = re.match(r'Preflop Train Action Match Accuracy for step-(\d+): ([\d.]+)', line)
            postflop_train_exact_match = re.match(r'Postflop Train Exact Match Accuracy for step-(\d+): ([\d.]+)', line)
            postflop_train_action_match = re.match(r'Postflop Train Action Match Accuracy for step-(\d+): ([\d.]+)', line)
            preflop_test_exact_match = re.match(r'Preflop Test Exact Match Accuracy for step-(\d+): ([\d.]+)', line)
            preflop_test_action_match = re.match(r'Preflop Test Action Match Accuracy for step-(\d+): ([\d.]+)', line)
            postflop_test_exact_match = re.match(r'Postflop Test Exact Match Accuracy for step-(\d+): ([\d.]+)', line)
            postflop_test_action_match = re.match(r'Postflop Test Action Match Accuracy for step-(\d+): ([\d.]+)', line)
            matches = [(preflop_train_exact_match, 2), (preflop_train_action_match, 3), (postflop_train_exact_match, 4),
                       (postflop_train_action_match, 5), (preflop_test_exact_match, 6), (preflop_test_action_match, 7),
                       (postflop_test_exact_match, 8), (postflop_test_action_match, 9)]
            for match, index in matches:
                if match:
                    step = int(match.group(1))
                    if step in data:
                        data[step][index] = float(match.group(2))

    return data

def plot_data(data, output_path, title):
    steps = sorted(data.keys())
    train_losses = [data[step][0] for step in steps]
    val_losses = [data[step][1] for step in steps]
    preflop_train_exact_match_acc = [data[step][2] for step in steps]
    preflop_train_action_match_acc = [data[step][3] for step in steps]
    postflop_train_exact_match_acc = [data[step][4] for step in steps]
    postflop_train_action_match_acc = [data[step][5] for step in steps]
    preflop_test_exact_match_acc = [data[step][6] for step in steps]
    preflop_test_action_match_acc = [data[step][7] for step in steps]
    postflop_test_exact_match_acc = [data[step][8] for step in steps]
    postflop_test_action_match_acc = [data[step][9] for step in steps]

    fig, ax1 = plt.subplots(figsize=(14,8))

    ax1.set_xlabel('Number of Optimizer Step', fontsize = 35)
    ax1.set_ylabel('Loss', fontsize = 35)
    ax1.set_ylim(0.05, 0.15)
    ax1.plot(steps, train_losses, label='Training Loss', color="#a6d75b", linestyle='--', linewidth=3, marker=".", markersize=15)
    ax1.plot(steps, val_losses, label='Validation Loss', color="#a6d75b", linestyle='-', linewidth=5, marker=".", markersize=15)
    ax1.tick_params(axis='both', which='major', labelsize=35)
    ax1.tick_params(axis='both', which='minor', labelsize=35)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', fontsize = 35)
    ax2.set_ylim(0, 0.9)
    ax2.plot(steps, preflop_train_exact_match_acc, label='Preflop Train EM Accuracy', color="red", linestyle='--', linewidth=3, marker=".", markersize=15)
    ax2.plot(steps, preflop_test_exact_match_acc, label='Preflop Test EM Accuracy', color="red", linestyle='-', linewidth=5, marker=".", markersize=15)
    ax2.plot(steps, postflop_train_exact_match_acc, label='Postflop Train EM Accuracy', color="blue", linestyle='--', linewidth=3, marker=".", markersize=15)
    ax2.plot(steps, postflop_test_exact_match_acc, label='Postflop Test EM Accuracy', color="blue", linestyle='-', linewidth=5, marker=".", markersize=15)
    # ax2.plot(steps, preflop_train_action_match_acc, label='Preflop Train Action Match Accuracy', linestyle='-.', marker='d')
    # ax2.plot(steps, postflop_train_action_match_acc, label='Postflop Train Action Match Accuracy', linestyle='-.', marker='8')
    # ax2.plot(steps, preflop_test_action_match_acc, label='Preflop Test Action Match Accuracy', linestyle='-.', marker='*')
    # ax2.plot(steps, postflop_test_action_match_acc, label='Postflop Test Action Match Accuracy', linestyle='-.', marker='.')
    ax2.tick_params(axis='both', which='major', labelsize=35)
    ax2.tick_params(axis='both', which='minor', labelsize=35)

    fig.tight_layout(rect=[0, 0, 1, 0.9])
    # plt.title(title, fontsize=20)
    fig.legend(loc='upper left', bbox_to_anchor=(0.14, 1), fontsize=20, framealpha=0.5)
    plt.show()
    plt.savefig(output_path)

if __name__ == "__main__":
    TRAINING_LOG_PATH = 'training_log_gemma_2b_110k_log.txt'
    EVALUATION_LOG_PATH = 'gemma_2b_eval_log_cleaned_110k.txt'
    OUTPUT_PLOT_PATH = "plots/loss_acc_plot_gemma_2b_110k.png"
    TITLE = 'Exact Match Accuracies for Gemma-2b on SFT Dataset of Size 110K'
    # TRAINING_LOG_PATH = 'sft_110k_log_cleaned.txt'
    # EVALUATION_LOG_PATH = 'llama_2_7b_eval_log_cleaned_110k.txt'
    # OUTPUT_PLOT_PATH = "plots/loss_acc_plot_llama_2_7b_110k_2.png"
    # TITLE = 'Exact Match Accuracies for Llama-2-7b on SFT Dataset of Size 110K'
    # TRAINING_LOG_PATH = 'training_log_llama_3_8b_110k_log.txt'
    # EVALUATION_LOG_PATH = 'llama_3_8b_eval_log_cleaned_110k.txt'
    # OUTPUT_PLOT_PATH = "plots/loss_acc_plot_llama_3_8b_110k_2.png"
    # TITLE = 'Exact Match Accuracies for Llama-3-8b on SFT Dataset of Size 110K'
    PER_EVERY_STEP = 200

    data = parse_training_log(TRAINING_LOG_PATH, per_every_step=PER_EVERY_STEP)
    # print(data)
    data = parse_evaluation_log(EVALUATION_LOG_PATH, data)
    # print(data)

    # Filter out steps where accuracy is None (in case there are missing entries)
    data = {step: values for step, values in data.items() if values[2] is not None}
    # data[50] = [1.378, 1.923, 0, 0, 0, 0] # For 20k Only
    # print(data)

    plot_data(data, OUTPUT_PLOT_PATH, TITLE)
