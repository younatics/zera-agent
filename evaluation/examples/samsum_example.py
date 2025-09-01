"""
SamSum Dataset Evaluation Example

This example evaluates the model's dialogue summarization ability using the SamSum dataset.
It compares the existing prompt and Zera prompt on the same samples.
"""

from evaluation.base.main import main
import sys

def run_samsum_example(model="claude", model_version="claude-3-sonnet-20240229"):
    base_system_prompt_path = "evaluation/examples/samsum_base_system_prompt.txt"
    base_user_prompt_path = "evaluation/examples/samsum_base_user_prompt.txt"
    zera_system_prompt_path = "evaluation/examples/samsum_zera_system_prompt.txt"
    zera_user_prompt_path = "evaluation/examples/samsum_zera_user_prompt.txt"

    with open(base_system_prompt_path, "r", encoding="utf-8") as f:
        base_system_prompt = f.read()
    with open(base_user_prompt_path, "r", encoding="utf-8") as f:
        base_user_prompt = f.read()
    with open(zera_system_prompt_path, "r", encoding="utf-8") as f:
        zera_system_prompt = f.read()
    with open(zera_user_prompt_path, "r", encoding="utf-8") as f:
        zera_user_prompt = f.read()

    sys.argv = [
        "example.py",
        "--dataset", "samsum",
        "--model", model,
        "--model_version", model_version,
        "--base_system_prompt", base_system_prompt,
        "--base_user_prompt", base_user_prompt,
        # "--zera_system_prompt", zera_system_prompt,
        # "--zera_user_prompt", zera_user_prompt,
        "--num_samples", "500",
        "--base_num_shots", "5"

        # # Model parameters
        # "--temperature", "0.2",  # Use low temperature for more deterministic responses
        # "--top_p", "0.9"
    ]
    
    # Execute evaluation
    main()

if __name__ == "__main__":
    run_samsum_example() 