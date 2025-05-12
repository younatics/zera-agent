"""
BBH (Big-Bench Hard) 데이터셋 평가 예제

이 예제는 BBH 데이터셋을 사용하여 모델의 다양한 태스크 수행 능력을 평가합니다.
기존 프롬프트와 제라 프롬프트를 동일한 샘플에 대해 비교 평가합니다.
"""

import sys
from evaluation.base.main import main

def run_bbh_example(model="gpt4o", model_version="gpt-3.5-turbo", bbh_category=None):
    # 명령행 인자 설정
    sys.argv = [
        "bbh_example.py",
        "--dataset", "bbh",
        "--model", model,
        "--model_version", model_version,
        # 기존 프롬프트
        "--base_system_prompt", "Answer the following question.",
        "--base_user_prompt", "Question:",
        # 제라 프롬프트
        "--zera_system_prompt", "You are an AI assistant skilled at logical and step-by-step reasoning. For each question, carefully explain your reasoning process clearly and sequentially. Conclude with the exact final answer requested, formatted clearly on a separate line.",
        "--zera_user_prompt", """Carefully evaluate the instructions provided step-by-step to determine whether, after following them exactly, you return to your original starting position. Clearly present each logical step involved in your reasoning. At the very end, output your final conclusion as either "Yes" or "No" on its own separate line.

TASK_HINTS:  
- First interpret carefully the exact meaning and implication of each instruction step.  
- Consider explicitly all given steps in your reasoning—even if some appear to offset each other.  
- Always summarize the net effect of forward/backward or directional moves explicitly.  
- Conclude clearly and separately with a single-word answer formatted on its own line.

FEW_SHOT_EXAMPLES:  
Example:  
Question: If you follow these instructions, do you return to the starting point? Always face forward. Move 4 steps forward. Move 6 steps backward. Then move 2 steps forward.  
Answer:  
Step-by-step Reasoning:  
- Start at the initial point.  
- Move 4 steps forward: position is at +4.  
- Move 6 steps backward: position is at -2 from initial point.  
- Move 2 steps forward: position returns exactly to the initial point (position 0).  

Final Answer:  
Yes""",
        "--num_samples", "500"
    ]
    if bbh_category:
        sys.argv += ["--bbh_category", bbh_category]
    # 평가 실행
    main()

if __name__ == "__main__":
    run_bbh_example()