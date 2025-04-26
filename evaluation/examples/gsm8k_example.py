"""
GSM8K 데이터셋 평가 예제

이 예제는 GSM8K 데이터셋을 사용하여 모델의 수학 문제 풀이 능력을 평가합니다.
"""

from evaluation.base.main import main
import argparse

def run_gsm8k_example():
    # 기본 인자 설정
    args = argparse.Namespace(
        dataset="gsm8k",
        model="gpt4o",
        model_version="gpt-3.5-turbo",
        # system_prompt="Let's think step-by-step.",
        # user_prompt="Question:",
        system_prompt=" ",
        user_prompt=" ",
        # system_prompt="You are a math word problem solver dedicated to accuracy and format precision. Your task is to resolve math problems with a clear, structured approach, showcasing each calculation step distinctly with inline calculation brackets (<<>>). Conclude every solution with the final answer prefixed by '####', ensuring the format is strictly adhered to at all times.",
        # user_prompt="Solve the given math problem by articulating each calculation as a standalone step. Utilize inline calculation brackets (<<>>) for each arithmetic expression, and present the final answer clearly prefixed by '####'. Follow the specified format rigorously, without extra explanations.\nQuestion:",
        num_samples=100
    )
    
    # 평가 실행
    main(args)

if __name__ == "__main__":
    run_gsm8k_example() 