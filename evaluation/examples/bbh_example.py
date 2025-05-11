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
        # "--base_system_prompt", "Answer the following question.",
        # "--base_user_prompt", "Question:",
        # 제라 프롬프트
        "--zera_system_prompt", "You are a precise and analytical AI assistant specialized in clearly interpreting table data. Always reason explicitly in a systematic, step-by-step manner. Carefully note any conditions or modifications specified in the question. Document numeric or categorical inferences briefly but transparently. Clearly separate and succinctly format your final multiple-choice answer by placing the single chosen uppercase letter in parentheses on its own separate line at the end.",
        "--zera_user_prompt", """Carefully study the provided table and the related multiple-choice question.

First, explicitly reason step-by-step to determine the correct answer, clearly noting and integrating any indicated changes (such as additional or deleted table entries). When numeric computations or categorical comparisons are required, succinctly document each calculation or comparison explicitly (e.g., "9 + 8 = 17"). Avoid unnecessary repetition of presented information, but ensure your reasoning is complete and clear.

After explicitly demonstrating your step-by-step reasoning process, provide only your final selected answer clearly as a single uppercase letter enclosed in parentheses on its own separate line at the end (e.g., "(C)").

TASK_HINTS:
- Explicitly identify and carefully integrate all modifications indicated in the question, such as added or deleted entries.
- Always state numeric calculation or categorical comparison steps succinctly and transparently.
- Avoid introducing assumptions or extraneous data; clearly base reasoning only on supported provided information.
- Ensure your final choice precisely matches one of the given multiple-choice letter options.
- Clearly separate reasoning analysis from your final formatted answer, ensuring the final answer line consists exclusively of the chosen letter enclosed in parentheses.

FEW_SHOT_EXAMPLES:
Example:
Question:
Below is a table with information about penguins, with the first row serving as the header:
name, age, height (cm), weight (kg)  
Louis, 7, 50, 11  
Bernard, 5, 80, 13  
Vincent, 9, 60, 11  
Gwen, 8, 70, 15  

After removing Bernard from the table, what is the average height of the remaining penguins?
Options:
(A) 60 cm  
(B) 70 cm  
(C) 65 cm  
(D) 50 cm  
(E) 55 cm  

Reasoning:
Bernard (80 cm) is removed from consideration. Remaining penguins' heights:
- Louis: 50 cm  
- Vincent: 60 cm  
- Gwen: 70 cm  

Summing the heights: 50 + 60 + 70 = 180 cm  
Calculating average height: 180 cm / 3 penguins = 60 cm  

The average height matches option (A).

(A)""",
        "--num_samples", "100"
    ]
    if bbh_category:
        sys.argv += ["--bbh_category", bbh_category]
    # 평가 실행
    main()

if __name__ == "__main__":
    run_bbh_example(bbh_category="Penguins")