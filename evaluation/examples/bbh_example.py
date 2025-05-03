"""
BBH (Big-Bench Hard) 데이터셋 평가 예제

이 예제는 BBH 데이터셋을 사용하여 모델의 다양한 태스크 수행 능력을 평가합니다.
기존 프롬프트와 제라 프롬프트를 동일한 샘플에 대해 비교 평가합니다.
"""

import sys
from evaluation.base.main import main

def run_bbh_example(model="gpt4o", model_version="gpt-3.5-turbo"):
    # 명령행 인자 설정
    sys.argv = [
        "bbh_example.py",
        "--dataset", "bbh",
        "--model", model,
        "--model_version", model_version,
        # 기존 프롬프트
        "--base_system_prompt", "Answer the following question.",
        "--base_user_prompt", "Question:",
        # 제라 프롬프트
        "--zera_system_prompt", "You are a logical reasoning expert. Clearly reason each question step-by-step in natural, explicit language. Upon completing your analysis, distinctly separate it from your final concise answer, which must strictly follow the provided formatting instructions.",
        "--zera_user_prompt", """Solve these logical reasoning problems by explicitly thinking through them step-by-step before providing your final answer.

Examples:

Question: Sort alphabetically: horse dolphin cat bird
bird cat dolphin horse

Question: Jim scored higher than Sam. Sam scored higher than Eve. Who scored lowest?
Options:
(A) Jim
(B) Sam
(C) Eve
(C)

Question: Check validity:
"No cars can fly. All Toyotas are cars. Therefore, no Toyotas can fly."
Options:
(A) valid
(B) invalid
(A)

Now, begin solving.
""",
        "--num_samples", "1000"
    ]
    # 평가 실행
    main()

if __name__ == "__main__":
    run_bbh_example() 