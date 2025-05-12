"""
HellaSwag 데이터셋 평가 예제

이 예제는 HellaSwag 데이터셋을 사용하여 모델의 상식적 문장 완성 능력을 평가합니다.
기존 프롬프트와 제라 프롬프트를 동일한 샘플에 대해 비교 평가합니다.
"""

import sys
from evaluation.base.main import main

def run_hellaswag_example(model="gpt4o", model_version="gpt-3.5-turbo"):
    # 명령행 인자 설정
    sys.argv = [
        "hellaswag_example.py",
        "--dataset", "hellaswag",
        "--model", model,
        "--model_version", model_version,
        # 기존 프롬프트
        "--base_system_prompt", "Answer with only the letter of the correct choice.",
        "--base_user_prompt", "Question:",
        # 제라 프롬프트
        "--zera_system_prompt", "You are an attentive and precise AI assistant. Consider the scenario carefully, reason logically through what makes sense given the described activity and context, then provide a clear and concise final answer in the format requested.",
        "--zera_user_prompt", """Select the most appropriate ending to the given context for the described activity. First, consider each option logically and briefly reason about why it might or might not be appropriate. After reasoning, clearly state the letter of the correct option in parentheses as your final answer.

Example:
Question: Activity: Shaving  
Context: He gives a list of things you will need to take care of a bald head. He places the items on the sink. He  
A. puts garnishes on the hair.  
B. grabs a razor and shaves his eyebrow.  
C. then lathers and shaves the skin on his head.  
D. looks in the mirror to make sure his hair is down.

Reasoning:  
- Option A makes no logical sense; garnishes aren't relevant to hair care or shaving.  
- Option B could be a shaving activity but shaving eyebrows is uncommon and not indicated in context.  
- Option C is logical and explicitly matches the described activity (taking care of a bald head through shaving).  
- Option D references "hair," which contradicts the baldness indicated in the scenario.

Final Answer: (C)

TASK_HINTS:
  - Provide only the letter of the correct option enclosed in parentheses as the final answer.
  - Clearly separate your reasoning from your final answer.
  - Avoid repeating the full text of the selected option in the final answer.

FEW_SHOT_EXAMPLES:
Example:
Question: Activity: Cooking soup  
Context: She chopped carrots and onions and put them into the pot. She stirred the ingredients gently and then  
A. set the pot on fire to burn off excess vegetables.  
B. sprinkled sugar on the broth for sweetness.  
C. covered the pot to simmer gently.  
D. placed the pot in the refrigerator immediately.

Reasoning:  
- Option A is unrealistic; burning off vegetables is not a common cooking practice.  
- Option B involves sugar, which doesn't typically go in savory soup with vegetables.  
- Option C is a sensible next step for soup preparation, allowing it to cook slowly.  
- Option D disrupts cooking and wouldn't allow flavors to blend or cook properly.

Final Answer: (C)""",
        "--num_samples", "500",
        # 모델 파라미터
        # "--temperature", "0.2",  # 더 결정적인 응답을 위해 낮은 temperature 사용
        # "--top_p", "0.9"
    ]
    
    # 평가 실행
    main()

if __name__ == "__main__":
    run_hellaswag_example() 