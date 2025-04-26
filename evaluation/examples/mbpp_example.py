"""
MBPP 데이터셋 평가 예제

이 예제는 MBPP 데이터셋을 사용하여 모델의 프로그래밍 문제 풀이 능력을 평가합니다.
"""

from evaluation.main import main
import argparse

def run_mbpp_example():
    # 기본 인자 설정
    args = argparse.Namespace(
        dataset="mbpp",
        model="gpt4o",
        model_version="gpt-3.5-turbo",
        system_prompt="You are a helpful assistant that writes Python code to solve programming problems.",
        user_prompt="Write a Python function to solve the following problem:",
        num_samples=5
    )
    
    # 평가 실행
    main(args)

if __name__ == "__main__":
    run_mbpp_example() 