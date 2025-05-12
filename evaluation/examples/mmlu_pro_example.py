"""
MMLU 데이터셋 평가 예제

이 예제는 MMLU 데이터셋을 사용하여 모델의 다중 선택형 문제 풀이 능력을 평가합니다.
기존 프롬프트와 제라 프롬프트를 동일한 샘플에 대해 비교 평가합니다.
"""

import sys
from evaluation.base.main import main

def run_mmlu_pro_example(model="gpt4o", model_version="gpt-3.5-turbo"):
    base_system_prompt_path = "evaluation/examples/mmlu_pro_base_system_prompt.txt"
    base_user_prompt_path = "evaluation/examples/mmlu_pro_base_user_prompt.txt"
    zera_system_prompt_path = "evaluation/examples/mmlu_pro_zera_system_prompt.txt"
    zera_user_prompt_path = "evaluation/examples/mmlu_pro_zera_user_prompt.txt"

    with open(base_system_prompt_path, "r", encoding="utf-8") as f:
        base_system_prompt = f.read()
    with open(base_user_prompt_path, "r", encoding="utf-8") as f:
        base_user_prompt = f.read()
    with open(zera_system_prompt_path, "r", encoding="utf-8") as f:
        zera_system_prompt = f.read()
    with open(zera_user_prompt_path, "r", encoding="utf-8") as f:
        zera_user_prompt = f.read()

    sys.argv = [
        "mmlu_pro_example.py",
        "--dataset", "mmlu_pro",
        "--model", model,
        "--model_version", model_version,
        "--base_system_prompt", base_system_prompt,
        "--base_user_prompt", base_user_prompt,
        "--zera_system_prompt", zera_system_prompt,
        "--zera_user_prompt", zera_user_prompt,
        "--num_samples", "500"
    ]
    # 평가 실행
    main()

if __name__ == "__main__":
    run_mmlu_pro_example() 