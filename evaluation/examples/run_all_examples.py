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
from evaluation.examples.samsum_example import run_samsum_example
from evaluation.examples.meetingbank_example import run_meetingbank_example

def main():
    # model = "gpt4o"
    # model_version = "gpt-3.5-turbo"

    model = "local1"
    model_version = "/data/project/private/kyle/hf_models/Meta-Llama-3-70B-Instruct"
    
    # print("\n===== Executing mmlu_example.py =====")
    # run_mmlu_example(model, model_version)

    # print("\n===== Executing mmlu_pro_example.py =====")
    # run_mmlu_pro_example(model, model_version)

    # print("\n===== Executing gsm8k_example.py =====")
    # run_gsm8k_example(model, model_version)
    # run_gsm8k_example(model, model_version)

    print("\n===== Executing cnn_dailymail_example.py =====")
    run_cnn_dailymail_example(model, model_version)

    print("\n===== Executing samsum_example.py =====")
    run_samsum_example(model, model_version)

    # print("\n===== Executing meetingbank_example.py =====")
    # run_meetingbank_example(model, model_version)

    # print("\n===== Executing mbpp_example.py =====")
    # run_mbpp_example(model, model_version)

    # print("\n===== Executing humaneval_example.py =====")
    # run_humaneval_example(model, model_version)

    # print("\n===== Executing bbh_example.py =====")
    # run_bbh_example(model, model_version)
    
    # print("\n===== Executing hellaswag_example.py =====")
    # run_hellaswag_example(model, model_version)

    # Send Slack notification after all evaluations complete
    # notify_slack(f"[Model version: {model_version}] All evaluations completed.", webhook_url)

if __name__ == "__main__":
    main() 