"""
CNN/DailyMail 데이터셋 평가 예제

이 예제는 CNN/DailyMail 데이터셋을 사용하여 모델의 뉴스 요약 능력을 평가합니다.
기존 프롬프트와 제라 프롬프트를 동일한 샘플에 대해 비교 평가합니다.
"""

from evaluation.base.main import main
import sys

def run_cnn_dailymail_example(model="gpt4o", model_version="gpt-3.5-turbo"):
    # 프롬프트 파일 경로 지정
    base_system_prompt_path = "evaluation/examples/cnn_base_system_prompt.txt"
    base_user_prompt_path = "evaluation/examples/cnn_base_user_prompt.txt"
    zera_system_prompt_path = "evaluation/examples/cnn_zera_system_prompt.txt"
    zera_user_prompt_path = "evaluation/examples/cnn_zera_user_prompt.txt"

    # 파일에서 프롬프트 읽기
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
        "--num_samples", "500"
        "--num_shots", "5"
    ]
    # 평가 실행
    main()

if __name__ == "__main__":
    run_cnn_dailymail_example() 