"""
MMLU Dataset Evaluation Example

This example evaluates the model's multiple-choice problem-solving ability using the MMLU dataset.
It compares the existing prompt and Zera prompt on the same samples.
"""

import sys
from evaluation.base.main import main

def run_mmlu_example(model="gpt4o", model_version="gpt-3.5-turbo"):
    base_system_prompt_path = "evaluation/examples/mmlu_base_system_prompt.txt"
    base_user_prompt_path = "evaluation/examples/mmlu_base_user_prompt.txt"
    zera_system_prompt_path = "evaluation/examples/mmlu_zera_system_prompt.txt"
    zera_user_prompt_path = "evaluation/examples/mmlu_zera_user_prompt.txt"

    with open(base_system_prompt_path, "r", encoding="utf-8") as f:
        base_system_prompt = f.read()
    with open(base_user_prompt_path, "r", encoding="utf-8") as f:
        base_user_prompt = f.read()
    with open(zera_system_prompt_path, "r", encoding="utf-8") as f:
        zera_system_prompt = f.read()
    with open(zera_user_prompt_path, "r", encoding="utf-8") as f:
        zera_user_prompt = f.read()

    sys.argv = [
        "mmlu_example.py",
        "--dataset", "mmlu",
        "--model", model,
        "--model_version", model_version,
        "--base_system_prompt", base_system_prompt,
        "--base_user_prompt", base_user_prompt,
        "--zera_system_prompt", zera_system_prompt,
        "--zera_user_prompt", zera_user_prompt,
        "--num_samples", "500",
        # Model parameters
        # "--temperature", "0.2",  # Use low temperature for more deterministic responses
        # "--top_p", "0.9"
    ]
    # Execute evaluation
    main()

if __name__ == "__main__":
    run_mmlu_example() 