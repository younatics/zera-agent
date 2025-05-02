"""
MMLU 데이터셋 평가 예제

이 예제는 MMLU 데이터셋을 사용하여 모델의 다중 선택형 문제 풀이 능력을 평가합니다.
기존 프롬프트와 제라 프롬프트를 동일한 샘플에 대해 비교 평가합니다.
"""

import sys
from evaluation.base.main import main

def run_mmlu_example():
    # 명령행 인자 설정
    sys.argv = [
        "mmlu_example.py",
        "--dataset", "mmlu",
        "--model", "gpt4o",
        # "--model_version", "gpt-3.5-turbo",
        "--model_version", "gpt-4o",
        # 기존 프롬프트
        "--base_system_prompt", "Answer with only the letter of the correct choice.",
        "--base_user_prompt", "Question:",
        # "--base_user_prompt", "Question:",
        # 제라 프롬프트
        "--zera_system_prompt", "You are an AI proficient in logical reasoning. Carefully analyze problems step-by-step, openly assessing each provided option. Keep your evaluation concise, logically clear, and distinctly separate from your final formatted answer.",
        "--zera_user_prompt", "Answer the following multiple-choice question by providing only the single letter (A, B, C, or D) of the correct choice.\n\nExample:\nQuestion: Why did Congress oppose Wilson's proposal for the League of Nations?\nA. It feared the League would encourage Soviet influence in the US  \nB. It feared the League would be anti-democratic  \nC. It feared the League would commit the US to an international alliance  \nD. Both A and B  \nAnswer: C\n\nQuestion: {Insert new question and choices here}  \nAnswer:",
        "--num_samples", "1000",
        # 모델 파라미터
        # "--temperature", "0.2",  # 더 결정적인 응답을 위해 낮은 temperature 사용
        # "--top_p", "0.9"
    ]
    
    # 평가 실행
    main()

if __name__ == "__main__":
    run_mmlu_example() 