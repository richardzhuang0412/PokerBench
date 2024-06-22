import subprocess

def run_command_and_save_output(command, output_file):
    with open(output_file, 'w') as f:
        pass

    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
        with open(output_file, 'a') as f:
            f.write(line)
    
    process.stdout.close()
    process.wait()
    
    if process.returncode != 0:
        print(f"Command exited with non-zero status {process.returncode}")

if __name__ == "__main__":
    command = """export NCCL_P2P_DISABLE=1 && export CUDA_VISIBLE_DEVICES=4,5,6,7 && litgpt finetune full \
  --checkpoint_dir checkpoints/meta-llama/Meta-Llama-3-8B \
  --data JSON \
  --data.json_path SFT/sft_combined_large.json \
  --data.val_split_fraction 0.01 \
  --out_dir out/llama-3-8b-110k \
  --train.save_interval 200 \
  --train.log_interval 50 \
  --train.epochs 8 \
  --train.global_batch_size 128 \
  --train.micro_batch_size 16 \
  --train.learning_rate 1e-6 \
  --eval.interval 200 \
  --eval.initial_validation false \
  --devices 4
"""

    output_file = "sft_llama_3_8b_110k_log.txt"
    run_command_and_save_output(command, output_file)
