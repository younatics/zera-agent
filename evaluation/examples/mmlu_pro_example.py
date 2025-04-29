"""
MMLU 데이터셋 평가 예제

이 예제는 MMLU 데이터셋을 사용하여 모델의 다중 선택형 문제 풀이 능력을 평가합니다.
"""

import sys
from evaluation.base.main import main

def run_mmlu_pro_example():
    # 명령행 인자 설정
    sys.argv = [
        "mmlu_pro_example.py",
        "--dataset", "mmlu_pro",
        "--model", "gpt4o",
        "--model_version", "gpt-3.5-turbo",
        # "--system_prompt", "Answer with only the letter of the correct choice.",
        # "--user_prompt", "Question:",
        "--system_prompt", "You are a helpful assistant skilled in step-by-step reasoning.",
        "--user_prompt", "Answer the following multiple-choice question. Provide a clear, logical explanation first, optionally referencing external resources, then conclude by clearly stating your selected choice in the format: \"The answer is (X).\"\nQuestion:",
        "--num_samples", "100"
    ]
    
    # 평가 실행
    main()

if __name__ == "__main__":
    run_mmlu_pro_example() 