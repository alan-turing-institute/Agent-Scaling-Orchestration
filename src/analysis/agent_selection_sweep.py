import argparse
import subprocess
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SELECTIONS_DIR = os.path.join(REPO_ROOT, "results", "agent-selection")

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, choices=["gsm8k", "pro_medicine", "formal_logic", "humaneval", "truthfulqa", "arc", "winogrande"])

    # use llm chosen agents or default agents
    parser.add_argument("--agent_type", type=str, default="default", choices=["chosen", "default"])

    parser.add_argument("--choose_type", type=str, choices=["name", "full", "single"], default="name")

    return parser.parse_args()

def run_cmd(cmd):
    """Run command and stream output in real-time"""
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Read stdout line by line as it comes in
    for line in process.stdout:
        print(f"[OUTPUT] {line.rstrip()}")
        sys.stdout.flush()  # Force immediate print
    
    # Wait for process to finish and get stderr
    _, stderr = process.communicate()
    
    if stderr:
        print(f"[ERROR] {stderr}")
    
    return process.returncode

args = get_args()

# for choose_type in ["name", "full", "single"]:
for choose_type in ["name"]:
    with open(f"{SELECTIONS_DIR}/gpt-4.1-agents=4-{choose_type}-chosen-agents.json", "r") as f:
        data = json.load(f)

    jobs = []
    for dataset, agent_lists in data.items():
        # if dataset == args.dataset:
        for agents in agent_lists:
            cmd = ["python", "src/main.py", 
                   "--data", dataset,
                   "--max_new_tokens", "1024", 
                   "--num_agents", "4", 
                   "--data_size", "100",
                   "--agent_models", "ministral-3b",
                   # "--chosen_agents",
                   # "--agents", ",".join(agents),
                   "--solver", "vote",
                   "--debate_rounds", "0",
                   "--multi_persona"
                  ]
            if args.agent_type == "chosen":
                cmd.append("--comment")
                cmd.append(f"{choose_type}")
                cmd.append("--chosen_agents")
                cmd.append("--chosen_personas")
                cmd.append(",".join(agents))

                print(f"Agents: {agents}")

            jobs.append(cmd)
    
    print(f"Queued {len(jobs)} jobs.")
    # print(jobs)
    with ThreadPoolExecutor(max_workers=10) as ex:
        results = list(ex.map(run_cmd, jobs))

    print("Done")