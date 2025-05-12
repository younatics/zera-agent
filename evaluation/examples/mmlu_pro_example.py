"""
MMLU 데이터셋 평가 예제

이 예제는 MMLU 데이터셋을 사용하여 모델의 다중 선택형 문제 풀이 능력을 평가합니다.
기존 프롬프트와 제라 프롬프트를 동일한 샘플에 대해 비교 평가합니다.
"""

import sys
from evaluation.base.main import main

def run_mmlu_pro_example(model="gpt4o", model_version="gpt-3.5-turbo"):
    # 명령행 인자 설정
    sys.argv = [
        "mmlu_pro_example.py",
        "--dataset", "mmlu_pro",
        "--model", model,
        "--model_version", model_version,
        # 기존 프롬프트
        "--base_system_prompt", "Answer with only the letter of the correct choice.",
        "--base_user_prompt", "Question:",
        # 제라 프롬프트
        "--zera_system_prompt", "You are an expert AI assistant specializing in economics and analytical reasoning. Think naturally and carefully before finalizing your answers. First briefly define explicitly relevant economic concepts. Conduct explicit, step-by-step evaluation of each option, grouping and concisely stating common errors among incorrect choices. Clearly and succinctly justify your final correct answer choice, explicitly demonstrating how it logically stands apart. Provide your chosen final answer explicitly as a single letter enclosed in parentheses, distinctly placed on a separate line at the end of your analysis.",
        "--zera_user_prompt", """Answer the following economics-related multiple-choice question explicitly and step-by-step:

1. Begin by explicitly defining only those economic concepts directly relevant to understanding this particular question.
2. Explicitly and individually evaluate each of the provided answer choices.
3. Explicitly group incorrect answers whenever relevant, concisely summarizing their common logical flaw.
4. Justify clearly why your chosen answer choice is correct by explicitly distinguishing it from incorrect alternatives.
5. Explicitly state your final answer as a single choice letter enclosed in parentheses, distinctly on a separate line at the end.

Example Question:  
Which policy action clearly represents expansionary fiscal policy?

Choices:  
A. Lowering interest rates by the central bank  
B. Reducing income taxes  
C. Increasing reserve requirements for banks  
D. Selling government bonds  
E. Raising the discount rate  

Example Answer:

Explicit definition of directly relevant economic concepts:  
- "Expansionary Fiscal Policy": Government actions (explicitly fiscal and not monetary) aimed at stimulating economic activity by increasing government spending or decreasing taxes, thus increasing overall demand.

Step-by-step explicit reasoning:

- Incorrect choices (A, C, E): Lowering interest rates (A), increasing reserve requirements (C), and raising the discount rate (E) explicitly represent monetary policy actions decided by the central bank—not fiscal actions by government.
  - Common logical flaw: All three explicitly represent monetary policy, not fiscal policy.

- Incorrect choice (D): Selling government bonds explicitly reduces money available in the economy, effectively representing contractionary monetary policy.
  - Logical flaw: Explicitly contractionary monetary, wrongly identified as expansionary fiscal.

- Correct choice (B): Reducing income taxes explicitly increases disposable income, consumer spending, and overall economic demand, clearly exemplifying expansionary fiscal policy.

Clearly justified selection:  
Option (B) explicitly matched the definition, visibly stimulating economic activity by decreasing taxes, in sharp contrast to explicitly monetary (A, C, E) and explicitly contractionary (D) actions listed.

Final Answer:  
(B)

TASK_HINTS:  
- Explicitly define essential economic concepts strictly related to the posed question before reasoning.  
- Explicitly group incorrect choices with a brief statement of their clearly shared logical flaw.  
- Clearly provide reasoning to explicitly differentiate the correct answer choice from incorrect alternatives.  
- Format your final choice clearly as a single letter enclosed in parentheses, distinctly set apart on its own line.""",
        "--num_samples", "500"
    ]
    
    # 평가 실행
    main()

if __name__ == "__main__":
    run_mmlu_pro_example() 