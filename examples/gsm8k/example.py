"""
GSM8K 데이터셋 평가 예제

이 예제는 GSM8K 데이터셋을 사용하여 모델의 수학 문제 풀이 능력을 평가합니다.
"""

from evaluation.main import main
import argparse

def run_gsm8k_example():
    # 기본 인자 설정
    args = argparse.Namespace(
        dataset="gsm8k",
        model="gpt4o",
        model_version="gpt-3.5-turbo",
        system_prompt="You are a helpful assistant that solves math problems step by step.",
        user_prompt="Solve the following math problem step by step:",
        num_samples=5
    )
    
    # 평가 실행
    main(args)

if __name__ == "__main__":
    run_gsm8k_example() 