"""
MBPP Dataset Evaluation Example

This example evaluates the model's programming problem-solving ability using the MBPP dataset.
It compares the existing prompt and Zera prompt on the same samples.
"""

import sys
from evaluation.base.main import main
from typing import List
import random

def run_mbpp_example(model="gpt4o", model_version="gpt-3.5-turbo"):
    base_system_prompt_path = "evaluation/examples/mbpp_base_system_prompt.txt"
    base_user_prompt_path = "evaluation/examples/mbpp_base_user_prompt.txt"
    zera_system_prompt_path = "evaluation/examples/mbpp_zera_system_prompt.txt"
    zera_user_prompt_path = "evaluation/examples/mbpp_zera_user_prompt.txt"

    with open(base_system_prompt_path, "r", encoding="utf-8") as f:
        base_system_prompt = f.read()
    with open(base_user_prompt_path, "r", encoding="utf-8") as f:
        base_user_prompt = f.read()
    with open(zera_system_prompt_path, "r", encoding="utf-8") as f:
        zera_system_prompt = f.read()
    with open(zera_user_prompt_path, "r", encoding="utf-8") as f:
        zera_user_prompt = f.read()

    sys.argv = [
        "mbpp_example.py",
        "--dataset", "mbpp",
        "--model", model,
        "--model_version", model_version,
        "--base_system_prompt", base_system_prompt,
        "--base_user_prompt", base_user_prompt,
        "--zera_system_prompt", zera_system_prompt,
        "--zera_user_prompt", zera_user_prompt,
        "--num_samples", "500",
    ]
    
    # Execute evaluation
    main()

def get_sample_indices(self, num_samples: int) -> List[int]:
    dataset = self.load_dataset("agent/dataset/mbpp_data/test.json")
    total_samples = len(dataset)
    if num_samples > total_samples:
        num_samples = total_samples
    return random.sample(range(total_samples), num_samples)

if __name__ == "__main__":
    run_mbpp_example() 