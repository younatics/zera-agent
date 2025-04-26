"""
SAMSum 데이터셋 평가 예제

이 예제는 SAMSum 데이터셋을 사용하여 모델의 대화 요약 능력을 평가합니다.
"""

from evaluation.main import main
import argparse

def run_samsum_example():
    # 기본 인자 설정
    args = argparse.Namespace(
        dataset="samsum",
        model="gpt4o",
        model_version="gpt-3.5-turbo",
        system_prompt="You are a helpful assistant that summarizes conversations effectively.",
        user_prompt="Please summarize the following conversation:",
        num_samples=5
    )
    
    # 평가 실행
    main(args)

if __name__ == "__main__":
    run_samsum_example() 