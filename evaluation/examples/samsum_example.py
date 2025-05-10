"""
SamSum 데이터셋 평가 예제

이 예제는 SamSum 데이터셋을 사용하여 모델의 대화 요약 능력을 평가합니다.
기존 프롬프트와 제라 프롬프트를 동일한 샘플에 대해 비교 평가합니다.
"""

from evaluation.base.main import main
import sys

def run_samsum_example(model="gpt4o", model_version="gpt-3.5-turbo"):
    # 명령줄 인자 설정
    sys.argv = [
        "example.py",
        "--dataset", "samsum",
        "--model", model,
        "--model_version", model_version,
        # 기존 프롬프트
        # "--base_system_prompt", "You are a dialogue summarization assistant. Summarize the following conversation in 2-3 sentences, focusing on the main points and key decisions.",
        # "--base_user_prompt", "Conversation:",
        # 제라 프롬프트
        "--zera_system_prompt", "You are a summarizer.",
        "--zera_user_prompt", """Summarize the given conversation in a declarative sentence in the passive voice. If examples can help clarify output structure or improve model performance, include them directly in the USER_PROMPT—not in the instructions.""",
        "--num_samples", "1000",
        # # 모델 파라미터
        # "--temperature", "0.2",  # 더 결정적인 응답을 위해 낮은 temperature 사용
        # "--top_p", "0.9"
    ]
    
    # 평가 실행
    main()

if __name__ == "__main__":
    run_samsum_example() 