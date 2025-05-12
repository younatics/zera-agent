"""
GSM8K 데이터셋 평가 예제

이 예제는 GSM8K 데이터셋을 사용하여 모델의 수학 문제 풀이 능력을 평가합니다.
기존 프롬프트와 제라 프롬프트를 동일한 샘플에 대해 비교 평가합니다.
"""

import sys
from evaluation.base.main import main

def run_gsm8k_example(model="gpt4o", model_version="gpt-3.5-turbo"):
    # 명령행 인자 설정
    sys.argv = [
        "gsm8k_example.py",
        "--dataset", "gsm8k",
        "--model", model,
        "--model_version", model_version,
        # 기존 프롬프트
        "--base_system_prompt", "Provide the final answer prefixed with '####'.",
        "--base_user_prompt", "Question:\n",
        # 제라 프롬프트
        "--zera_system_prompt", "You are an assistant skilled in step-by-step reasoning for arithmetic-based word problems. Carefully restate critical numerical values and explicitly include units. Transparently show your arithmetic calculations inline using \"<<...>>\" notation. Concisely clarify the reasoning connecting intermediate steps to achieve a logical flow. Clearly present your final numeric answer isolated on its own line, prefixed by \"####\".",
        "--zera_user_prompt", """Solve the following arithmetic word problem step-by-step.

TASK_HINTS:  
- Begin each step by briefly summarizing critical numeric details from the problem explicitly with appropriate units.  
- Transparently show each arithmetic calculation explicitly using inline "<<...>>" notation.  
- Clearly and concisely explain the reasoning linking each step, focusing particularly on intermediate logical connections.  
- Avoid redundant phrasing; use concise language and logically combine simple calculations when appropriate.  
- Explicitly isolate the final numeric answer clearly on its own line, prefixed by "####".

FEW_SHOT_EXAMPLES:  
Example:  
Question:  
There are 4 baskets of apples, each basket containing 30 apples. If John takes half of the apples from each basket and then donates 25 apples to the local shelter, how many apples does John have left?

Answer:  
There are 4 baskets with 30 apples each, giving a total of 4 × 30 = <<4 × 30 = 120>> 120 apples.  
John takes half of each basket's apples, which totals 120 ÷ 2 = <<120 ÷ 2 = 60>> 60 apples.  
John then donates 25 apples, leaving him with 60 − 25 = <<60 − 25 = 35>> 35 apples.  

#### 35""",
        "--num_samples", "500",
        # 모델 파라미터
        # "--temperature", "0.2",  # 더 결정적인 응답을 위해 낮은 temperature 사용
        # "--top_p", "0.9"
    ]
    
    # 평가 실행
    main()

if __name__ == "__main__":
    run_gsm8k_example() 