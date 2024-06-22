import os
import signal
import subprocess
import time
import psutil
import requests

def start_server_and_check_status():
    command = "export CUDA_VISIBLE_DEVICES=7 && litgpt serve --checkpoint_dir out/llama-2-base-ft/step-000050 --temperature 0.2 --max_new_tokens 50"
    process = subprocess.Popen(command, shell=True)
    return process

def check_server_status():
    try:
        response = requests.get("http://127.0.0.1:8000/status")
        if response.status_code == 200:
            print("Server is running and responded with:", response.json())
        else:
            print("Server is not responding as expected. Status code:", response.status_code)
    except Exception as e:
        print(f"Failed to connect to server: {e}")

def terminate_process_tree(pid):
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            child.terminate()
        parent.terminate()

        _, still_alive = psutil.wait_procs(children, timeout=5)
        if still_alive:
            for p in still_alive:
                p.kill()

        parent.wait(timeout=5)
        print(f"Terminated process tree for PID: {pid}")
    except Exception as e:
        print(f"Error terminating process tree for PID {pid}: {e}")

def terminate_processes_on_gpu(gpu_id, username):
    try:
        result = subprocess.check_output(['nvidia-smi', f'--id={gpu_id}', '--query-compute-apps=pid,username', '--format=csv,noheader']).decode().strip()
        processes = [line.split(', ') for line in result.split('\n') if line]
        for pid, user in processes:
            if user == username:
                os.kill(int(pid), signal.SIGTERM)
                print(f"Terminated process {pid} on GPU {gpu_id} for user {username}")
    except Exception as e:
        print(f"Error terminating processes on GPU {gpu_id} for user {username}: {e}")

if __name__ == "__main__":
    for i in range(10):
        print(f"Iteration {i+1}/10")
        
        # Start the server
        process = start_server_and_check_status()
        print(f"Started server with PID: {process.pid}")
        
        # Wait a bit for the server to start
        time.sleep(10)
        
        # Check the server status
        check_server_status()
        
        # Terminate the server and related processes
        terminate_process_tree(process.pid)
        
        # Terminate any process on GPU ID 7 with user "richard"
        terminate_processes_on_gpu(7, "richard")
        
        # Wait before next iteration
        time.sleep(10)
