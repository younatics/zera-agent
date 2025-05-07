"""
MBPP 데이터셋 평가 예제

이 예제는 MBPP 데이터셋을 사용하여 모델의 프로그래밍 문제 풀이 능력을 평가합니다.
기존 프롬프트와 제라 프롬프트를 동일한 샘플에 대해 비교 평가합니다.
"""

import sys
from evaluation.base.main import main
from typing import List
import random

def run_mbpp_example(model="gpt4o", model_version="gpt-3.5-turbo"):
    # 명령행 인자 설정
    sys.argv = [
        "mbpp_example.py",
        "--dataset", "mbpp",
        "--model", model,
        "--model_version", model_version,
        # 기존 프롬프트
        # "--base_system_prompt", "Write a Python function that satisfies the following specification.",
        # "--base_user_prompt", "Problem:",
        # 제라 프롬프트
        "--zera_system_prompt", "You are an expert Python coding assistant. First reason freely and clearly about the logical solution for the provided programming problem. Then, produce only a concise and accurate Python function implementation, strictly observing Python syntax, naming conventions, and clarity requirements. Do not include any explanations or commentary beyond the final code.",
        "--zera_user_prompt", """Write the Python function that directly solves the following programming problem.

Question: {Insert user question here}

{If appropriate, use one or two concise examples to implicitly clarify required structure, syntax, or logic patterns. For typical cases straightforwardly expressed in Python, omit examples.}

Your solution should be provided strictly in the following format:

```python
# your Python function implementation
```""",
        "--num_samples", "1000",
    ]
    
    # 평가 실행
    main()

def get_sample_indices(self, num_samples: int) -> List[int]:
    dataset = self.load_dataset("agent/dataset/mbpp_data/test.json")
    total_samples = len(dataset)
    if num_samples > total_samples:
        num_samples = total_samples
    return random.sample(range(total_samples), num_samples)

if __name__ == "__main__":
    run_mbpp_example() 