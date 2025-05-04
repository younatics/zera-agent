import sys
from evaluation.base.main import main

def run_gsm8k_fewshot_ablation(model="gpt4o", model_version="gpt-3.5-turbo"):
    sys.argv = [
        "gsm8k_ablation.py",
        "--dataset", "gsm8k",
        "--model", model,
        "--model_version", model_version,
        "--base_system_prompt", "Provide the final answer prefixed with '####'.",
        "--base_user_prompt", "Question:\n",
        "--base_num_shots", "1",
        "--zera_system_prompt", "Provide the final answer prefixed with '####'.",
        "--zera_user_prompt", "Question:\n",
        "--zera_num_shots", "5",
        "--num_samples", "1319",
        
    ]
    main()

def run_gsm8k_prompt_ablation(model="gpt4o", model_version="gpt-3.5-turbo"):
    sys.argv = [
        "gsm8k_ablation.py",
        "--dataset", "gsm8k",
        "--model", model,
        "--model_version", model_version,
        "--base_system_prompt", "You are a logical reasoning assistant. First reason through the problem naturally and clearly—ignoring formatting. Only at the final stage, concisely summarize critical numeric calculations using the designated \"<<calculation=result>>\" notation, and clearly report your final numeric answer.",
        "--base_user_prompt", "Solve the following problem step-by-step with clear, logical reasoning. Afterward, briefly present each critical calculation step explicitly using the \"<<calculation=result>>\" notation, concluding with your final numeric answer clearly marked after \"####\".\n\nExample:\nQuestion: Sara buys 4 bouquets of roses, each bouquet has 12 roses. She gives away 9 roses. How many roses does Sara have left?\n\nReasoning:\nSara first buys a total of 4 bouquets * 12 roses each = 48 roses. Then she gives away 9 roses, leaving her with 48 roses - 9 roses = 39 roses.\n\nCalculations:\nTotal roses bought: 4 * 12 = <<4*12=48>>  \nRoses remaining: 48 - 9 = <<48-9=39>>\n\n#### 39\n\nNow solve this problem:\n",
        "--zera_system_prompt", "Provide the final answer prefixed with '####'.",
        "--zera_user_prompt", "Question:\n",
        "--zera_num_shots", "5",
        
    ]
    main()


def run_bbh_fewshot_ablation(model="gpt4o", model_version="gpt-3.5-turbo"):
    sys.argv = [
        "bbh_ablation.py",
        "--dataset", "bbh",
        "--model", model,
        "--model_version", model_version,
        "--base_system_prompt", "Answer the following question.",
        "--base_user_prompt", "Question:",
        "--base_num_shots", "1",
        "--zera_system_prompt", "Answer the following question.",
        "--zera_user_prompt", "Question:",
        "--zera_num_shots", "5",
        "--num_samples", "1000"
    ]
    main()

def run_cnn_fewshot_dailymail_ablation(model="gpt4o", model_version="gpt-3.5-turbo"):
    sys.argv = [
        "cnn_dailymail_ablation.py",
        "--dataset", "cnn_dailymail",
        "--model", model,
        "--model_version", model_version,
        "--base_system_prompt", "You are a summarization assistant. Summarize the following article in 2–3 sentences, focusing on the main idea.",
        "--base_user_prompt", "Article:",
        "--base_num_shots", "1",
        "--zera_system_prompt", "You are a summarization assistant. Summarize the following article in 2–3 sentences, focusing on the main idea.",
        "--zera_user_prompt", "Article:",
        "--zera_num_shots", "5",
        "--num_samples", "1000",
    ]
    main()

def run_mbpp_fewshot_ablation(model="gpt4o", model_version="gpt-3.5-turbo"):
    sys.argv = [
        "mbpp_ablation.py",
        "--dataset", "mbpp",
        "--model", model,
        "--model_version", model_version,
        "--base_system_prompt", "Write a Python function that satisfies the following specification.",
        "--base_user_prompt", "Problem:",
        "--base_num_shots", "1",
        "--zera_system_prompt", "Write a Python function that satisfies the following specification.",
        "--zera_user_prompt", "Problem:",
        "--zera_num_shots", "5",
        "--num_samples", "1000",
    ]
    main()

def run_fewshot_ablation(model="gpt4o", model_version="gpt-3.5-turbo"):
    print("\n===== gsm8k ablation 평가 실행 =====")
    run_gsm8k_fewshot_ablation(model, model_version)
    print("\n===== bbh ablation 평가 실행 =====")
    run_bbh_fewshot_ablation(model, model_version)
    print("\n===== cnn_dailymail ablation 평가 실행 =====")
    run_cnn_fewshot_dailymail_ablation(model, model_version)
    print("\n===== mbpp ablation 평가 실행 =====")
    run_mbpp_fewshot_ablation(model, model_version)


def main():
    model = "local"
    model_version = "/data/project/private/kyle/hf_models/Meta-Llama-3-70B-Instruct"

    run_fewshot_ablation(model, model_version)



if __name__ == "__main__":
    main() 