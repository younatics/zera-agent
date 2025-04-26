"""
BBH (Big-Bench Hard) 데이터셋 평가 예제

이 예제는 BBH 데이터셋을 사용하여 모델의 다양한 태스크 수행 능력을 평가합니다.
"""

from evaluation.main import main
import argparse

def run_bbh_example():
    # 기본 인자 설정
    args = argparse.Namespace(
        dataset="bbh",
        model="gpt4o",
        model_version="gpt-3.5-turbo",
        system_prompt="You are a helpful assistant that can perform various reasoning tasks.",
        user_prompt="Please solve the following task:",
        num_samples=5
    )
    
    # 평가 실행
    main(args)

if __name__ == "__main__":
    run_bbh_example() 