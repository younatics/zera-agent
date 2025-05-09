"""
MMLU 데이터셋 평가 예제

이 예제는 MMLU 데이터셋을 사용하여 모델의 다중 선택형 문제 풀이 능력을 평가합니다.
기존 프롬프트와 제라 프롬프트를 동일한 샘플에 대해 비교 평가합니다.
"""

import sys
from evaluation.base.main import main

def run_mmlu_pro_example(model="gpt4o", model_version="gpt-3.5-turbo"):
    # 명령행 인자 설정
    sys.argv = [
        "mmlu_pro_example.py",
        "--dataset", "mmlu_pro",
        "--model", model,
        "--model_version", model_version,
        # 기존 프롬프트
        "--base_system_prompt", "Answer with only the letter of the correct choice.",
        "--base_user_prompt", "Question:",
        # 제라 프롬프트
        "--zera_system_prompt", "You are an AI assistant skilled at clear and concise step-by-step reasoning. Provide logical explanations freely, and precisely identify the correct final answer by stating its associated letter choice.",
        "--zera_user_prompt", """Solve the following questions by logically reasoning step-by-step. Clearly state your final answer at the end as "The answer is (Letter)."

Example:
Question: Predict the number of lines in the EPR spectrum of 13CH3• radical, assuming lines do not overlap.

Choices:
A. 10
B. 8
C. 4
D. 20

Let's think step by step:
1. The electron interacts with the single 13C nucleus (spin I = 1/2), splitting the line into (2 × 1/2 + 1) = 2 lines.
2. Each of these lines splits again due to three equivalent protons (each I = 1/2) into (2 × (3 × 1/2) + 1) = 4 lines.
3. Therefore, total number of lines = 2 × 4 = 8.

The answer is (B).
""",
        "--num_samples", "1000"
    ]
    
    # 평가 실행
    main()

if __name__ == "__main__":
    run_mmlu_pro_example() 