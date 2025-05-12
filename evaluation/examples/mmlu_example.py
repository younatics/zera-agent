"""
MMLU 데이터셋 평가 예제

이 예제는 MMLU 데이터셋을 사용하여 모델의 다중 선택형 문제 풀이 능력을 평가합니다.
기존 프롬프트와 제라 프롬프트를 동일한 샘플에 대해 비교 평가합니다.
"""

import sys
from evaluation.base.main import main

def run_mmlu_example(model="gpt4o", model_version="gpt-3.5-turbo"):
    # 명령행 인자 설정
    sys.argv = [
        "mmlu_example.py",
        "--dataset", "mmlu",
        "--model", model,
        "--model_version", model_version,
        # 기존 프롬프트
        "--base_system_prompt", "Answer with only the letter of the correct choice.",
        "--base_user_prompt", "Question:",
        # 제라 프롬프트
        "--zera_system_prompt", "You are an AI assistant specialized in clear and structured logical reasoning. Carefully examine each provided statement step-by-step, briefly defining any relevant key terms. Your explanations should transparently show your thought process leading explicitly to your final answer, keeping the final answer format strictly as requested.",
        "--zera_user_prompt", """Carefully evaluate each statement provided in the question. Determine clearly and step-by-step if the statements are True or False, briefly defining any key concepts necessary for clarity. Avoid repeating the given answer choices in your reasoning. After finishing your detailed reasoning, explicitly state only the correct choice letter (A, B, C, or D) separately on the final line.

Example:

Question:
According to philosopher Cohen, if I promise to give you a dollar, then:

Choices:  
A. you have a right to my dollar.  
B. I am obligated to give you my dollar.  
C. both A and B  
D. neither A nor B  

Answer:  
Firstly, a "promise" is a voluntary commitment to perform an action. A "right" means a justified entitlement to receive something from another party, typically enforceable morally or legally, whereas an "obligation" is a moral or legal duty to act. According to Cohen, when someone promises to give another something, that promise creates both a moral entitlement in the promisee—meaning you have a right to expect the promised item—and a corresponding moral obligation binding the promisor to fulfill that promise. Therefore, both statements A and B hold true under Cohen's viewpoint.

C

TASK_HINTS:
- Separate your step-by-step reasoning distinctly from your final answer.
- Clearly and briefly define relevant concepts crucial to your reasoning.
- Do not restate or repeat the provided multiple-choice answers within your reasoning.
- Ensure your final choice is explicitly provided as a single letter on a separate final line.

FEW_SHOT_EXAMPLES:
Included above in the example.""",
        "--num_samples", "500",
        # 모델 파라미터
        # "--temperature", "0.2",  # 더 결정적인 응답을 위해 낮은 temperature 사용
        # "--top_p", "0.9"
    ]
    
    # 평가 실행
    main()

if __name__ == "__main__":
    run_mmlu_example() 