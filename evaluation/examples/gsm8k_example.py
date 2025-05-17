"""
GSM8K 데이터셋 평가 예제

이 예제는 GSM8K 데이터셋을 사용하여 모델의 수학 문제 풀이 능력을 평가합니다.
기존 프롬프트와 제라 프롬프트를 동일한 샘플에 대해 비교 평가합니다.
"""

import sys
from evaluation.base.main import main

def run_gsm8k_example(model="gpt4o", model_version="gpt-3.5-turbo"):
    # 프롬프트 파일 경로 지정
    base_system_prompt_path = "evaluation/examples/gsm8k_base_system_prompt.txt"
    base_user_prompt_path = "evaluation/examples/gsm8k_base_user_prompt.txt"
    zera_system_prompt_path = "evaluation/examples/gsm8k_zera_system_prompt.txt"
    zera_user_prompt_path = "evaluation/examples/gsm8k_zera_user_prompt.txt"

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
        "gsm8k_example.py",
        "--dataset", "gsm8k",
        "--model", model,
        "--model_version", model_version,
        "--base_system_prompt", base_system_prompt,
        "--base_user_prompt", base_user_prompt,
        "--zera_system_prompt", zera_system_prompt,
        "--zera_user_prompt", zera_user_prompt,
        "--num_samples", "500",
        # 모델 파라미터
        # "--temperature", "0.2",  # 더 결정적인 응답을 위해 낮은 temperature 사용
        # "--top_p", "0.9"
    ]
    # 평가 실행
    main()

if __name__ == "__main__":
    run_gsm8k_example() 