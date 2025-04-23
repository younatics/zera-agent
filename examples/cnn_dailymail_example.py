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
        "--system_prompt", "You are an advanced text analysis and summarization assistant. Your role is to extract key information from complex texts and articulate them concisely. Focus on identifying the main arguments, supporting details, and context. Omit unnecessary personal opinions or irrelevant data. Present the findings in a structured, coherent, and precise manner aligned with professional summarization standards.",
        "--user_prompt", "Please provide a concise summary of the core themes and details presented in this text. Highlight the main arguments, significant facts, and any relevant contextual information or examples. Ensure your summary is precise, objective, and free of unnecessary commentary, reflecting the structured pathway of the article.\nArticle:",
        "--num_samples", "100"
    ]
    
    # 평가 실행
    main()

if __name__ == "__main__":
    run_cnn_dailymail_example() 