"""
BBH (Big-Bench Hard) 데이터셋 평가 예제

이 예제는 BBH 데이터셋을 사용하여 모델의 다양한 태스크 수행 능력을 평가합니다.
기존 프롬프트와 제라 프롬프트를 동일한 샘플에 대해 비교 평가합니다.
"""

import sys
from evaluation.base.main import main

def run_bbh_example(model="gpt4o", model_version="gpt-3.5-turbo"):
    # 명령행 인자 설정
    sys.argv = [
        "bbh_example.py",
        "--dataset", "bbh",
        "--model", model,
        "--model_version", model_version,
        # 기존 프롬프트
        # "--base_system_prompt", "Answer the following question.",
        # "--base_user_prompt", "Question:",
        # 제라 프롬프트
        "--zera_system_prompt", "You are an expert AI specialized in precise, structured logical reasoning. Freely think step-by-step through each reasoning question, carefully and explicitly stating each logical inference in clear, natural language. Only after fully completing your structured reasoning, separately provide your concise final answer strictly in the minimal format specified by the question, without further commentary.",
        "--zera_user_prompt", """Solve the following structured reasoning question. First, explicitly present each step of your reasoning clearly.

After finishing your reasoning, separately provide your final concise answer in exactly the required minimal format indicated by the question.

Example:
Question:
At a fruit-eating competition, Alice eats more fruits than Ben, but fewer than Cara. Dave eats the most fruits. Who eats the second-most fruits?
Options:
(A) Alice 
(B) Ben  
(C) Cara  
(D) Dave  

Expected Output:  
(C)

Your Response:
Reasoning:
1. Alice eats more fruits than Ben but fewer than Cara, so we have Cara > Alice > Ben.
2. Dave eats the most fruits, meaning he is ahead of everyone else. Dave > Cara > Alice > Ben.
3. Therefore, the second-most fruits eater is Cara.

Final Answer:  
(C)

Now solve the following:

\{Question\}""",
        "--num_samples", "1000"
    ]
    # 평가 실행
    main()

if __name__ == "__main__":
    run_bbh_example() 