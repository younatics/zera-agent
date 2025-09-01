"""
CNN/DailyMail Dataset Evaluation Example

This example evaluates the model's news summarization ability using the CNN/DailyMail dataset.
It compares the existing prompt and Zera prompt on the same samples.
"""

from evaluation.base.main import main
import sys

def run_cnn_dailymail_example(model="gpt4o", model_version="gpt-3.5-turbo"):
    # Specify prompt file paths
    base_system_prompt_path = "evaluation/examples/cnn_base_system_prompt.txt"
    base_user_prompt_path = "evaluation/examples/cnn_base_user_prompt.txt"
    zera_system_prompt_path = "evaluation/examples/cnn_zera_system_prompt.txt"
    zera_user_prompt_path = "evaluation/examples/cnn_zera_user_prompt.txt"

    # Read prompts from files
    with open(base_system_prompt_path, "r", encoding="utf-8") as f:
        base_system_prompt = f.read()
    with open(base_user_prompt_path, "r", encoding="utf-8") as f:
        base_user_prompt = f.read()
    with open(zera_system_prompt_path, "r", encoding="utf-8") as f:
        zera_system_prompt = f.read()
    with open(zera_user_prompt_path, "r", encoding="utf-8") as f:
        zera_user_prompt = f.read()

    sys.argv = [
        "cnn_dailymail_example.py",
        "--dataset", "cnn_dailymail",
        "--model", model,
        "--model_version", model_version,
        "--base_system_prompt", base_system_prompt,
        "--base_user_prompt", base_user_prompt,
        # "--zera_system_prompt", zera_system_prompt,
        # "--zera_user_prompt", zera_user_prompt,
        "--num_samples", "500",
        "--base_num_shots", "5",
        # "--zera_num_shots", "5"
    ]
    # Execute evaluation
    main()

if __name__ == "__main__":
    run_cnn_dailymail_example() 