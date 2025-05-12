import sys
from evaluation.base.main import main
from typing import List
import random

def run_humaneval_example(model="gpt4o", model_version="gpt-3.5-turbo"):
    base_system_prompt_path = "evaluation/examples/humaneval_base_system_prompt.txt"
    base_user_prompt_path = "evaluation/examples/humaneval_base_user_prompt.txt"
    zera_system_prompt_path = "evaluation/examples/humaneval_zera_system_prompt.txt"
    zera_user_prompt_path = "evaluation/examples/humaneval_zera_user_prompt.txt"

    with open(base_system_prompt_path, "r", encoding="utf-8") as f:
        base_system_prompt = f.read()
    with open(base_user_prompt_path, "r", encoding="utf-8") as f:
        base_user_prompt = f.read()
    with open(zera_system_prompt_path, "r", encoding="utf-8") as f:
        zera_system_prompt = f.read()
    with open(zera_user_prompt_path, "r", encoding="utf-8") as f:
        zera_user_prompt = f.read()

    sys.argv = [
        "humaneval_example.py",
        "--dataset", "humaneval",
        "--model", model,
        "--model_version", model_version,
        "--base_system_prompt", base_system_prompt,
        "--base_user_prompt", base_user_prompt,
        "--zera_system_prompt", zera_system_prompt,
        "--zera_user_prompt", zera_user_prompt,
        "--num_samples", "500",
    ]
    main()

if __name__ == "__main__":
    run_humaneval_example() 