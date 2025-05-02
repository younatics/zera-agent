"""
MMLU 데이터셋 평가 예제

이 예제는 MMLU 데이터셋을 사용하여 모델의 다중 선택형 문제 풀이 능력을 평가합니다.
기존 프롬프트와 제라 프롬프트를 동일한 샘플에 대해 비교 평가합니다.
"""

import sys
from evaluation.base.main import main

def run_mmlu_pro_example():
    # 명령행 인자 설정
    sys.argv = [
        "mmlu_pro_example.py",
        "--dataset", "mmlu_pro",
        "--model", "gpt4o",
        "--model_version", "gpt-3.5-turbo",
        # 기존 프롬프트
        "--base_system_prompt", "Answer with only the letter of the correct choice.",
        "--base_user_prompt", "Question:",
        # 제라 프롬프트
        "--zera_system_prompt", "You are an expert logical reasoning assistant. Carefully and naturally reason through each problem step-by-step. Keep your explanations brief, clear, and logical. Only after completing your reasoning, state your final choice strictly as the option letter enclosed in parentheses.",
        "--zera_user_prompt", "Solve the following multiple-choice questions by reasoning concisely and logically step by step. Clearly explain the key steps that lead directly to your conclusion. Conclude by stating your final answer strictly as one letter in parentheses, e.g., \"(D)\".\n\nExample 1:\n\nQuestion: A microwave oven operates at 120 volts and draws a current of 2 amperes. How many watts of electrical power does it use?\n\nChoices:\nA. 120 W\nB. 240 W\nC. 480 W\n\nAnswer: Let's reason carefully:\n- Electrical power (P) is calculated using the formula: Power = Voltage × Current.\n- Given Voltage = 120 volts and Current = 2 amperes, the calculation is 120 V × 2 A = 240 watts.\n\nThe correct answer is (B).\n\nExample 2:\n\nQuestion: According to Moore's \"ideal utilitarianism\", the right action is the one producing the greatest amount of:\n\nChoices:\nA. wealth\nB. virtue\nC. fairness\nD. pleasure\nE. peace\nF. justice\nG. happiness\nH. power\nI. good\nJ. knowledge\n\nAnswer: Let's reason step by step:\n- Ideal utilitarianism aims to maximize intrinsic goods.\n- Reviewing the options, the concept directly associated with intrinsic goods according to Moore's theory is (I) \"good\".\n\nThe correct answer is (I).\n\nQuestion:",
        "--num_samples", "10"
    ]
    
    # 평가 실행
    main()

if __name__ == "__main__":
    run_mmlu_pro_example() 