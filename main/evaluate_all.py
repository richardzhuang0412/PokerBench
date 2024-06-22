import requests, json, random, os, signal, subprocess, time
from tqdm import tqdm

random.seed(42)

# print(os.listdir("out/llama-2-base-ft"))
model_paths = ["step-000" + str(num) for num in range(200,1000,200)] + ["step-00" + str(num) for num in range(1000,5001,200)]
# model_paths = ["step-000" + str(num) for num in range(100,1000,50)] + ["step-00" + str(num) for num in range(1000,3100,50)]
CHECKPOINT_FOLDER_PATH = "out/gemma-2b-110k/"
EVAL_LOG_OUTPUT_PATH = "eval_gemma_2b_110k_error_log.txt"
GPU_ID = 7
PORT = 9000
FEW_SHOT = False
logs = []
# model_paths = ["step-000600", "step-001200", "step-002200", 
#                "step-003400", "step-003800", "step-005000"]
print(model_paths)

with open(EVAL_LOG_OUTPUT_PATH, 'w') as file:
    pass

for model_path in model_paths:
    if "step" not in model_path:
        continue
    print(f"Evaluating {model_path}")
    hosting_command = f"export CUDA_VISIBLE_DEVICES={GPU_ID} && litgpt serve --checkpoint_dir {CHECKPOINT_FOLDER_PATH + model_path} --temperature 0.1 --max_new_tokens 50 --port {PORT}"
    host_process = subprocess.Popen(hosting_command, shell=True) # stdout=subprocess.PIPE, stderr=subprocess.PIPE
    # print(process.pid)
    time.sleep(30)

    eval_process = subprocess.Popen(f"python evaluate.py --model_path {model_path} --log_path {EVAL_LOG_OUTPUT_PATH} --port {PORT}", shell=True)
    eval_process.wait()

    def get_user_of_pid(pid):
        try:
            result = subprocess.check_output(['ps', '-o', 'user=', '-p', str(pid)]).decode().strip()
            return result
        except subprocess.CalledProcessError:
            return None

    def kill_processes_on_gpu(gpu_id, username='richard'):
        # Get the list of processes running on the specified GPU
        try:
            result = subprocess.check_output(['nvidia-smi', f'--id={gpu_id}', '--query-compute-apps=pid', '--format=csv,noheader']).decode().strip()
        except subprocess.CalledProcessError as e:
            print(f"Failed to query GPU {gpu_id}: {e}")
            return

        # Parse the PIDs
        gpu_processes = result.split('\n')
        gpu_processes = [int(pid) for pid in gpu_processes if pid]

        if not gpu_processes:
            print(f"No processes found on GPU {gpu_id}")
            return

        # print(f"Processes on GPU {gpu_id}: {gpu_processes}")

        # Kill each process that belongs to the specified user
        for pid in gpu_processes:
            process_user = get_user_of_pid(pid)
            if process_user == username:
                try:
                    subprocess.Popen(f"kill -9 {pid}", shell=True)
                    # print(f"Terminated process with PID: {pid} owned by {username}")
                except ProcessLookupError:
                    print(f"Process with PID {pid} not found")
                except Exception as e:
                    print(f"Failed to kill process with PID {pid}: {e}")
            else:
                print(f"Skipped terminating process with PID: {pid} (owned by {process_user})")

    kill_processes_on_gpu(GPU_ID, username='richard')
    result = subprocess.run(f"lsof -i :{PORT}", shell=True, capture_output=True, text=True)
    # print(result.stdout)
    if result.stdout:
        lines = result.stdout.splitlines()
        for k in range(1, len(lines)):
            # if any(["LISTEN" in word for word in lines[k].split()]):
            pid = lines[k].split()[1]
            subprocess.run(f"kill -9 {pid}", shell=True)
            # print(f"Port Process {pid} Killed")
    result = subprocess.run(f"lsof -i :{PORT}", shell=True, capture_output=True, text=True)
    while result.stdout:
        # print(result.stdout)
        time.sleep(5)
        result = subprocess.run(f"lsof -i :{PORT}", shell=True, capture_output=True, text=True)
    time.sleep(10)