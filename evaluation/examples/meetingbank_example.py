"""
MeetingBank 데이터셋 평가 예제

이 예제는 MeetingBank 데이터셋을 사용하여 모델의 회의 요약 능력을 평가합니다.
기존 프롬프트와 제라 프롬프트를 동일한 샘플에 대해 비교 평가합니다.
"""

from evaluation.base.main import main
import sys

def run_meetingbank_example(model="gpt4o", model_version="gpt-3.5-turbo"):
    # 명령줄 인자 설정
    sys.argv = [
        "example.py",
        "--dataset", "meetingbank",
        "--model", model,
        "--model_version", model_version,
        # 기존 프롬프트
        # "--base_system_prompt", "You are a meeting summarization assistant. Summarize the following meeting transcript in 2-3 sentences, focusing on the main points and key decisions.",
        # "--base_user_prompt", "Meeting Transcript:",
        # 제라 프롬프트
        "--zera_system_prompt", "You are a helpful AI assistant that aims to provide concise and accurate responses to various questions. Do not include unnecessary details or repetitions, and focus on the essential information for each question.",
        "--zera_user_prompt", """Assistant, I need help answering this question: [Insert Question Here] Please provide a brief, clear, and accurate response, formatted as appropriate. If examples are useful for understanding the structure or content of the answer, include them directly in your response.""",
        "--num_samples", "1000",
        # # 모델 파라미터
        # "--temperature", "0.2",  # 더 결정적인 응답을 위해 낮은 temperature 사용
        # "--top_p", "0.9"
    ]
    
    # 평가 실행
    main()

if __name__ == "__main__":
    run_meetingbank_example() 