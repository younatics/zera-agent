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
        "--system_prompt", "You are an expert AI specializing in detailed news article summarization. Your role is to transform news articles into detailed and succinct summaries while extracting critical facts, events, and figures. Ensure that summaries are comprehensive, capturing the core message and context accurately while maintaining a logical structure throughout. Prioritize essential information and adhere to the evaluation criteria focusing on accuracy, completeness, relevance, conciseness, and clarity.",
        "--user_prompt", "Please summarize the following news article into a detailed yet concise summary, highlighting and presenting the critical facts and key events. Focus on providing a comprehensive overview that accurately and clearly reflects the article's core message and context. If additional context or clarification is needed, feel free to request more information.\nArticle:",
        "--num_samples", "1000"
    ]
    
    # 평가 실행
    main()

if __name__ == "__main__":
    run_cnn_dailymail_example() 