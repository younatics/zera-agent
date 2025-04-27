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
        "--system_prompt", "You are an AI specializing in extracting key details and summarizing them effectively. Allow open reasoning to capture important information, applying concise formatting only at the final summary stage.",
        "--user_prompt", "Identify the main facts from the content. Present them as a concise, clearly structured summary at the end.\nArticle:",
        "--num_samples", "100"
    ]
    
    # 평가 실행
    main()

if __name__ == "__main__":
    run_cnn_dailymail_example() 