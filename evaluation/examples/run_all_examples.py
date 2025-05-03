from evaluation.examples.bbh_example import run_bbh_example
from evaluation.examples.mmlu_pro_example import run_mmlu_pro_example
from evaluation.examples.humaneval_example import run_humaneval_example
from evaluation.examples.mmlu_example import run_mmlu_example
from evaluation.examples.gsm8k_example import run_gsm8k_example
from evaluation.examples.cnn_dailymail_example import run_cnn_dailymail_example
from evaluation.examples.mbpp_example import run_mbpp_example
from evaluation.examples.truthfulqa_example import run_truthfulqa_example
from evaluation.examples.xsum_example import run_xsum_example
from evaluation.examples.hellaswag_example import run_hellaswag_example

def main():
    # model = "gpt4o"
    # model_version = "gpt-3.5-turbo"

    model = "local"
    model_version = "/data/project/private/kyle/hf_models/Meta-Llama-3-70B-Instruct"
    
    print("\n===== mmlu_example.py 실행 =====")
    # run_mmlu_example(model, model_version)
    print("\n===== mmlu_pro_example.py 실행 =====")
    # run_mmlu_pro_example(model, model_version)
    print("\n===== gsm8k_example.py 실행 =====")
    run_gsm8k_example(model, model_version)
    print("\n===== cnn_dailymail_example.py 실행 =====")
    run_cnn_dailymail_example(model, model_version)
    print("\n===== mbpp_example.py 실행 =====")
    run_mbpp_example(model, model_version)
    print("\n===== humaneval_example.py 실행 =====")
    run_humaneval_example(model, model_version)
    print("\n===== truthfulqa_example.py 실행 =====")
    # run_truthfulqa_example(model, model_version)
    print("\n===== bbh_example.py 실행 =====")
    run_bbh_example(model, model_version)
    print("\n===== xsum_example.py 실행 =====")
    run_xsum_example(model, model_version)
    print("\n===== hellaswag_example.py 실행 =====")
    run_hellaswag_example(model, model_version)

if __name__ == "__main__":
    main() 