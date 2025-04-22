"""
MMLU 데이터셋 평가 예제

이 예제는 MMLU 데이터셋을 사용하여 모델의 다중 선택형 문제 풀이 능력을 평가합니다.
"""

import sys
from evaluation.main import main

def run_mmlu_example():
    # 명령행 인자 설정
    sys.argv = [
        "mmlu_example.py",
        "--dataset", "mmlu",
        "--model", "gpt4o",
        "--model_version", "gpt-3.5-turbo",
        "--system_prompt", "You are a multiple-choice question solver. Choose the best answer among A, B, C, or D. Give only the letter.",
        "--user_prompt", "Question:",
        "--num_samples", "1000"
    ]
    
    # 평가 실행
    main()

if __name__ == "__main__":
    run_mmlu_example() 