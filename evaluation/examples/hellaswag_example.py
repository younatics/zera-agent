"""
HellaSwag 데이터셋 평가 예제

이 예제는 HellaSwag 데이터셋을 사용하여 모델의 상식적 문장 완성 능력을 평가합니다.
기존 프롬프트와 제라 프롬프트를 동일한 샘플에 대해 비교 평가합니다.
"""

import sys
from evaluation.base.main import main

def run_hellaswag_example():
    # 명령행 인자 설정
    sys.argv = [
        "hellaswag_example.py",
        "--dataset", "hellaswag",
        "--model", "gpt4o",
        "--model_version", "gpt-3.5-turbo",
        # 기존 프롬프트
        "--base_system_prompt", "What happens next?",
        "--base_user_prompt", "Question:",
        # 제라 프롬프트
        "--zera_system_prompt", "You are a logical reasoning assistant. Carefully read the context and all possible endings. Reason step-by-step to eliminate implausible options, then select and state only the single best ending as a letter (A, B, C, or D) or number (1, 2, 3, or 4).",
        "--zera_user_prompt", "Read the following context and possible endings. After reasoning, answer with only the letter (A, B, C, or D) or number (1, 2, 3, or 4) of the best ending.\n\nContext and choices:",
        "--num_samples", "1000",
        # 모델 파라미터
        # "--temperature", "0.2",  # 더 결정적인 응답을 위해 낮은 temperature 사용
        # "--top_p", "0.9"
    ]
    
    # 평가 실행
    main()

if __name__ == "__main__":
    run_hellaswag_example() 