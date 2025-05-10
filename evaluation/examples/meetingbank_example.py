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
        "--zera_system_prompt", "You are a helpful and informative AI model. Answer questions in a clear, concise, and well-reasoned manner, while enforcing output format only when presenting the final answer.",
        "--zera_user_prompt", """User: I have a question about defunding the police. [Insert Question Here]
Assistant: [Insert Answer Here]

In this scenario, the user prompt includes the question and the assistant's response. No examples are needed as the question itself provides a clear context for the model to generate an appropriate response.

I recommend considering the following steps when optimizing prompts for other tasks:

1. Analyze the reasoning and output requirements of the task.
2. Identify the weaknesses and strengths in the original prompt and previous iterations.
3. Apply concise improvements that enhance clarity, task alignment, and output structure without increasing prompt complexity.
4. Validate the new prompts for brevity, clear separation of reasoning and formatting, minimal formatting requirements, and lack of redundancy or overlong instructions.
5. Repeat the process as needed to further refine the prompts and improve performance.""",
        "--num_samples", "1000",
        # # 모델 파라미터
        # "--temperature", "0.2",  # 더 결정적인 응답을 위해 낮은 temperature 사용
        # "--top_p", "0.9"
    ]
    
    # 평가 실행
    main()

if __name__ == "__main__":
    run_meetingbank_example() 