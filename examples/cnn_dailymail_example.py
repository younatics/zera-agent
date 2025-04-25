"""
CNN/DailyMail 데이터셋 평가 예제

이 예제는 CNN/DailyMail 데이터셋을 사용하여 모델의 뉴스 요약 능력을 평가합니다.
"""

from evaluation.base.main import main
import sys

def run_cnn_dailymail_example():
    # 명령줄 인자 설정
    sys.argv = [
        "example.py",
        "--dataset", "cnn_dailymail",
        "--model", "gpt4o",
        "--model_version", "gpt-3.5-turbo",
        "--system_prompt", "You are an AI specialized in summarizing articles into precise and factual bullet points. Each summary should consist of exactly 3-4 bullet points, focusing strictly on capturing the main facts. All bullet points must be concise and end with a period. Do not include any additional details, explanations, or commentary beyond the key facts provided in the source material.",
        "--user_prompt", "Summarize the provided article into exactly 3-4 key factual bullet points. Ensure your response strictly follows the expected output structure of concise, fact-driven points, each ending with a period. Avoid including extra details or personal commentary, maintaining focus solely on the core facts.\nArticle:",
        "--num_samples", "100"
    ]
    
    # 평가 실행
    main()

if __name__ == "__main__":
    run_cnn_dailymail_example() 