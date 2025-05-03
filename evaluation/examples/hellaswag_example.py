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
        "--base_system_prompt", "What happens next?",
        "--base_user_prompt", "Question:",
        # 제라 프롬프트
        "--zera_system_prompt", "You are an AI assistant skilled at logical reasoning through contextual scenarios. For each scenario, reason carefully and naturally to assess the logical flow. Clearly analyze the suitability of each continuation. Only after completing your reasoning, provide your final answer strictly as a single letter (A, B, C, or D), without further explanation.",
        "--zera_user_prompt", """
Below is a short incomplete scenario with four possible continuations labeled A, B, C, and D.

Reason briefly and clearly about which option most naturally continues the scenario, mentioning explicitly why it fits the context best, and concisely explaining why each of the other options is comparatively less suitable.

Conclude your response strictly as a single letter (A, B, C, or D).

Example:

Context:
[header] Safe Campfire Procedures
[step] Choose a clear, flat site and remove flammable debris thoroughly.

Continuations:
A. Immediately cook with large flames.
B. Spread your tools messily around the site.
C. Build a small, controlled fire structure with kindling.
D. Go swimming in a nearby lake.

Reasoning:
Option C directly and safely follows the context, as it clearly continues a logical sequence of building a campfire.
Option A introduces risk prematurely, option B creates hazards, and option D is irrelevant to the task.

Answer:
C

Example:

Context:
[header] Baking Bread Dough
[step] Once the dough is well kneaded, shape it into a smooth ball on a lightly floured surface.

Continuations:
A. Let the dough rise in a lightly greased bowl.
B. Immediately put the dough into cold storage.
C. Decorate the dough ball with frosting.
D. Slice the dough thinly right away.

Reasoning:
Option A naturally follows bread-making steps, because dough typically needs to rise after kneading.
Option B interrupts yeast activity, option C suggests decorating too early, and option D bypasses the important rising step.

Answer:
A

Now, complete the following scenario:

Context:
[Insert new context here]

Continuations:
A. [Option A text]
B. [Option B text]
C. [Option C text]
D. [Option D text]

Reasoning:""",
        "--num_samples", "1000",
        # 모델 파라미터
        # "--temperature", "0.2",  # 더 결정적인 응답을 위해 낮은 temperature 사용
        # "--top_p", "0.9"
    ]
    
    # 평가 실행
    main()

if __name__ == "__main__":
    run_hellaswag_example() 