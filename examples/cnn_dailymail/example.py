"""
CNN/DailyMail 데이터셋 평가 예제

이 예제는 CNN/DailyMail 데이터셋을 사용하여 모델의 뉴스 요약 능력을 평가합니다.
"""

from evaluation.main import main
import sys

def run_cnn_dailymail_example():
    # 명령줄 인자 설정
    sys.argv = [
        "example.py",
        "--dataset", "cnn_dailymail",
        "--model", "gpt4o",
        "--model_version", "gpt-3.5-turbo",
        "--system_prompt", "You are an AI assistant skilled in creating concise and well-structured summaries of news articles. Your role is to extract and logically present the main events, key participants, and significant outcomes, ensuring every summary includes important details like names, numbers, and quotes. Align the summary with the broader themes or issues discussed in the article, focusing on clarity, brevity, and complete coverage of the article's context.",
        "--user_prompt", "Please read the provided news excerpt thoroughly. Summarize it by identifying the key events, participants, and outcomes, and include essential details such as numbers, names, and quotes. Ensure the summary reflects the broader themes or issues of the article, maintaining both clarity and conciseness.\nArticle:",
        "--num_samples", "100"
    ]
    
    # 평가 실행
    main()

if __name__ == "__main__":
    run_cnn_dailymail_example() 