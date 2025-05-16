"""
SamSum 데이터셋 평가 예제

이 예제는 SamSum 데이터셋을 사용하여 모델의 대화 요약 능력을 평가합니다.
기존 프롬프트와 제라 프롬프트를 동일한 샘플에 대해 비교 평가합니다.
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
        "--zera_system_prompt", zera_system_prompt,
        "--zera_user_prompt", zera_user_prompt,
        "--num_samples", "500",
        # # 모델 파라미터
        # "--temperature", "0.2",  # 더 결정적인 응답을 위해 낮은 temperature 사용
        # "--top_p", "0.9"
    ]
    
    # 평가 실행
    main()

if __name__ == "__main__":
    run_samsum_example() 