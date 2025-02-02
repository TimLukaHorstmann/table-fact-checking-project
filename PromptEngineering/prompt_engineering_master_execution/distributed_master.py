#!/usr/bin/env python3
"""
distributed_master.py

This master script dispatches prompt engineering experiments based on a group configuration.
You define groups (e.g., one group for mistral+llama+phi and one for deepseek) and for each group
you specify the allowed remote machines per format type (e.g., one machine for markdown and one for naturalized).

A sample group configuration (JSON) might be:

{
  "groups": [
    {
      "name": "mistral_llama_phi",
      "models": ["mistral", "llama", "phi"],
      "format_machines": {
        "markdown": ["serverA"],
        "naturalized": ["serverB"]
      }
    },
    {
      "name": "deepseek",
      "models": ["deepseek"],
      "format_machines": {
        "markdown": ["serverC"],
        "naturalized": ["serverD"]
      }
    }
  ]
}

Usage example:
--------------
python distributed_master.py \
  --remote_dir /users/eleves-a/2024/tim-luka.horstmann.m2/table-fact-checking-project/PromptEngineering \
  --repo_folder ../original_repo \
  --csv_folder data/all_csv/ \
  --dataset tokenized_data/test_examples.json \
  --learning_type zero_shot,one_shot \
  --group_config group_config.json \
  --max_workers 4 \
  --N 5 \
  --poll_interval 30

Make sure that:
  - SSH access (with key-based authentication) is set up for the remote servers.
  - The remote machines have the project (including prompt_engineering.py) in the directory specified by --remote_dir.
"""


BIRD_SERVERS = [
    "albatros", "autruche", "bengali", "coucou", "dindon", "epervier",
    "faisan", "gelinotte", "hibou", "harpie", "jabiru", "kamiche",
    "linotte", "loriol", "mouette", "nandou", "ombrette", "perdrix",
    "quetzal", "quiscale", "rouloul", "sitelle", "traquet", "urabu",
    "verdier"
]

import subprocess
import concurrent.futures
import time
import logging
import argparse
import os
import json

# Configure logging.
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def check_gpu(server):
    """
    Checks the GPU usage on a remote server via SSH and nvidia-smi.
    Returns a tuple (server, mem_used, gpu_util) if the server is considered "free"
    (criteria: GPU utilization < 10% and memory used < 2000 MiB), or None otherwise.
    """
    try:
        result = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", server,
             "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            output = result.stdout.strip()
            if output:
                free = False
                min_mem_used = None
                min_gpu_util = None
                for line in output.splitlines():
                    mem_used, mem_total, gpu_util = map(int, line.split(", "))
                    logging.info(f"{server}: GPU {gpu_util}% | Memory {mem_used}/{mem_total} MiB")
                    if gpu_util < 10 and mem_used < 2000:
                        free = True
                        if min_mem_used is None or mem_used < min_mem_used:
                            min_mem_used = mem_used
                            min_gpu_util = gpu_util
                if free:
                    return (server, min_mem_used, min_gpu_util)
        return None
    except Exception as e:
        logging.error(f"Error checking GPU on {server}: {e}")
        return None

def get_free_servers(allowed_servers):
    """
    Checks the list of allowed_servers concurrently and returns a list of tuples
    (server, mem_used, gpu_util) for those that are free.
    """
    free_servers = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(allowed_servers)) as executor:
        results = executor.map(check_gpu, allowed_servers)
        for res in results:
            if res:
                free_servers.append(res)
    return free_servers

def build_jobs_from_config(args, group_config):
    """
    Build a list of jobs based on the group configuration and command-line arguments.
    For each group, for each format type (using its allowed machines), and for each combination
    of dataset and learning_type, create a job with:
      - group name,
      - models (as a comma-separated string),
      - dataset,
      - learning_type,
      - format_type,
      - allowed_servers (list for that format),
      - max_workers and N (from args).
    """
    jobs = []
    datasets = [d.strip() for d in args.dataset.split(",")]
    learning_types = [l.strip() for l in args.learning_type.split(",")]
    
    for group in group_config.get("groups", []):
        group_name = group.get("name")
        models = group.get("models", [])
        models_str = ",".join(models)
        format_machines = group.get("format_machines", {})
        for fmt, allowed_servers in format_machines.items():
            for dataset in datasets:
                for lt in learning_types:
                    job = {
                        "group_name": group_name,
                        "models": models_str,
                        "dataset": dataset,
                        "learning_type": lt,
                        "format_type": fmt,
                        "allowed_servers": allowed_servers,
                        "max_workers": args.max_workers,
                        "N": args.N
                    }
                    jobs.append(job)
    return jobs

def dispatch_job(job, server, args):
    """
    Build and execute an SSH command to dispatch a single job to a remote server.
    The command changes to args.remote_dir on the remote machine and runs prompt_engineering.py
    with the job parameters.
    """
    cmd = (
        f"conda activate .conda && "
        f"cd {args.remote_dir} && "
        f"python prompt_engineering.py "
        f"--repo_folder {args.repo_folder} "
        f"--csv_folder {args.csv_folder} "
        f"--dataset {job['dataset']} "
        f"--learning_type {job['learning_type']} "
        f"--format_type {job['format_type']} "
        f"--models {job['models']} "
        f"--batch_prompts "
        f"--max_workers {job['max_workers']} "
        f"--N {job['N']}"
    )
    ssh_cmd = f"ssh {server} '{cmd}'"
    logging.info(f"Dispatching job for group '{job['group_name']}', format '{job['format_type']}' to {server}:\n{ssh_cmd}")
    try:
        subprocess.Popen(ssh_cmd, shell=True)
        return True
    except Exception as e:
        logging.error(f"Failed to dispatch job to {server}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Master script to dispatch distributed experiments via SSH based on group configuration"
    )
    parser.add_argument("--remote_dir", type=str, required=True,
                        help="Directory on remote machines where the project is located")
    parser.add_argument("--repo_folder", type=str, default="../original_repo", help="Path to repository folder")
    parser.add_argument("--csv_folder", type=str, default="data/all_csv/",
                        help="Folder containing CSV tables (relative to remote_dir)")
    parser.add_argument("--dataset", type=str, default="tokenized_data/test_examples.json",
                        help="Comma-separated list of dataset JSON files")
    parser.add_argument("--learning_type", type=str, default="zero_shot",
                        help="Comma-separated list of learning types")
    parser.add_argument("--group_config", type=str, required=True,
                        help="Path to JSON file with group configuration")
    parser.add_argument("--max_workers", type=int, default=4,
                        help="Number of worker processes for parallel claims (passed to remote script)")
    parser.add_argument("--N", type=int, default=5,
                        help="Number of tables to test")
    parser.add_argument("--poll_interval", type=int, default=30,
                        help="Polling interval (in seconds) to check for free servers")
    args = parser.parse_args()
    
    # Load the group configuration.
    if not os.path.exists(args.group_config):
        logging.error(f"Group configuration file not found: {args.group_config}")
        return
    with open(args.group_config, "r") as f:
        group_config = json.load(f)
    
    # Build the job list from the configuration.
    jobs = build_jobs_from_config(args, group_config)
    total_jobs = len(jobs)
    logging.info(f"Built {total_jobs} job(s) from group configuration.")
    
    # Main loop: while there are jobs, check for free servers (only among the allowed ones for each job)
    # and dispatch the job when a free server is available.
    while jobs:
        remaining_jobs = []
        for job in jobs:
            allowed = job["allowed_servers"]
            free_servers = get_free_servers(allowed)
            if free_servers:
                # Dispatch to the first available free server.
                server, mem_used, gpu_util = free_servers[0]
                success = dispatch_job(job, server, args)
                if success:
                    logging.info(f"Job for group '{job['group_name']}' with format '{job['format_type']}' dispatched to {server}.")
                else:
                    logging.error(f"Job {job} failed to dispatch to {server}. Requeuing job.")
                    remaining_jobs.append(job)
            else:
                # No free server available from the allowed list for this job; requeue it.
                remaining_jobs.append(job)
        jobs = remaining_jobs
        if jobs:
            logging.info(f"{len(jobs)} job(s) remaining; waiting {args.poll_interval} seconds before polling again...")
            time.sleep(args.poll_interval)
    
    logging.info("All jobs have been dispatched.")

if __name__ == "__main__":
    main()
