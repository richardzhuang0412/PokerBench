def filter_lines_with_epoch(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    filtered_lines = [line for line in lines if 'Epoch' in line]

    with open(output_file, 'w') as file:
        file.writelines(filtered_lines)

if __name__ == "__main__":
    input_file = 'sft_gemma_2b_110k_log.txt'
    output_file = 'training_log_gemma_2b_110k_log.txt'
    
    filter_lines_with_epoch(input_file, output_file)
    print(f"Filtered lines written to {output_file}")