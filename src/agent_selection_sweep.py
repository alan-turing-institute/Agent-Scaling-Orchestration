import argparse
import subprocess
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from defaults import MAX_NEW_TOKENS

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SELECTIONS_DIR = os.path.join(REPO_ROOT, "results", "agent-selection")

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, choices=["gsm8k", "pro_medicine", "formal_logic", "humaneval", "truthfulqa", "arc", "winogrande"])

    # use llm chosen agents or default agents
    parser.add_argument("--agent_type", type=str, default="default", choices=["chosen", "default"])

    parser.add_argument("--choose_type", type=str, choices=["name", "full", "single"], default="name")

    # Passed through to each src/main.py run
    parser.add_argument("--selection_model", type=str, default="gpt-4.1",
                        help="Model whose chosen-agents file supplies the teams")
    parser.add_argument("--agent_models", type=str, default="ministral-3b")
    parser.add_argument("--num_agents", type=int, default=4)
    parser.add_argument("--data_size", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--solver", type=str, default="vote", choices=["vote", "debate"])
    parser.add_argument("--debate_rounds", type=int, default=0)
    parser.add_argument("--max_workers", type=int, default=10,
                        help="Concurrent src/main.py runs")

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

if __name__ == "__main__":
    args = get_args()

    # for choose_type in ["name", "full", "single"]:
    for choose_type in ["name"]:
        selections = (f"{SELECTIONS_DIR}/{args.selection_model}"
                      f"-agents={args.num_agents}-{choose_type}-chosen-agents.json")
        with open(selections, "r") as f:
            data = json.load(f)

        jobs = []
        for dataset, agent_lists in data.items():
            # if dataset == args.dataset:
            for agents in agent_lists:
                cmd = [sys.executable, "src/main.py",
                       "--data", dataset,
                       "--max_new_tokens", str(args.max_new_tokens),
                       "--num_agents", str(args.num_agents),
                       "--data_size", str(args.data_size),
                       "--agent_models", args.agent_models,
                       "--solver", args.solver,
                       "--debate_rounds", str(args.debate_rounds),
                       "--multi_persona",
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
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            results = list(ex.map(run_cmd, jobs))

        print("Done")