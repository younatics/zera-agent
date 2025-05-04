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
from agent.common.slack_notify import notify_slack

def main():
    # model = "gpt4o"
    # model_version = "gpt-3.5-turbo"

    model = "local"
    model_version = "/data/project/private/kyle/hf_models/Meta-Llama-3-70B-Instruct"
    
    webhook_url = "https://hooks.slack.com/services/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"
    print("\n===== mmlu_example.py 실행 =====")
    run_mmlu_example(model, model_version)
    notify_slack(f"[모델 버전: {model_version}] mmlu_example.py 평가가 완료되었습니다.", webhook_url)

    print("\n===== mmlu_pro_example.py 실행 =====")
    run_mmlu_pro_example(model, model_version)
    notify_slack(f"[모델 버전: {model_version}] mmlu_pro_example.py 평가가 완료되었습니다.", webhook_url)

    print("\n===== gsm8k_example.py 실행 =====")
    run_gsm8k_example(model, model_version)
    notify_slack(f"[모델 버전: {model_version}] gsm8k_example.py 평가가 완료되었습니다.", webhook_url)

    print("\n===== cnn_dailymail_example.py 실행 =====")
    run_cnn_dailymail_example(model, model_version)
    notify_slack(f"[모델 버전: {model_version}] cnn_dailymail_example.py 평가가 완료되었습니다.", webhook_url)

    print("\n===== mbpp_example.py 실행 =====")
    run_mbpp_example(model, model_version)
    notify_slack(f"[모델 버전: {model_version}] mbpp_example.py 평가가 완료되었습니다.", webhook_url)

    print("\n===== humaneval_example.py 실행 =====")
    run_humaneval_example(model, model_version)
    notify_slack(f"[모델 버전: {model_version}] humaneval_example.py 평가가 완료되었습니다.", webhook_url)

    print("\n===== truthfulqa_example.py 실행 =====")
    # run_truthfulqa_example(model, model_version)
    # notify_slack(f"[모델 버전: {model_version}] truthfulqa_example.py 평가가 완료되었습니다.", webhook_url)
    print("\n===== bbh_example.py 실행 =====")
    run_bbh_example(model, model_version)
    notify_slack(f"[모델 버전: {model_version}] bbh_example.py 평가가 완료되었습니다.", webhook_url)

    print("\n===== xsum_example.py 실행 =====")
    run_xsum_example(model, model_version)
    notify_slack(f"[모델 버전: {model_version}] xsum_example.py 평가가 완료되었습니다.", webhook_url)
    
    print("\n===== hellaswag_example.py 실행 =====")
    run_hellaswag_example(model, model_version)
    notify_slack(f"[모델 버전: {model_version}] hellaswag_example.py 평가가 완료되었습니다.", webhook_url)

    # 모든 평가 완료 후 슬랙 알림 전송
    notify_slack(f"[모델 버전: {model_version}] 모든 평가가 완료되었습니다.", webhook_url)

if __name__ == "__main__":
    main() 