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
        "--system_prompt", "You are an advanced summarization expert tasked with transforming complex texts into clear and succinct summaries. Your objective is to extract and present the most pivotal events, influential figures, and crucial outcomes while strictly adhering to the outlined format. Exclude any extraneous details or subjective interpretations, ensuring each summary is factual, coherent, and concise. Always prioritize precision and clarity to convey the essential insights effectively, encompassing all critical information without embellishment.",
        "--user_prompt", "Please generate a precise and concise summary of the following text. Focus on capturing the primary events, significant participants, and key results. Maintain adherence to any specified format, ensuring the exclusion of unnecessary details or personal opinions. Present the essential elements directly and efficiently to fully encapsulate the core message of the text.\nArticle:",
        "--num_samples", "100"
    ]
    
    # 평가 실행
    main()

if __name__ == "__main__":
    run_cnn_dailymail_example() 