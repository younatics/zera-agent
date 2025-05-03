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
        # # # 기존 프롬프트
        "--base_system_prompt", "Write a Python function that satisfies the following specification.",
        "--base_user_prompt", "Problem:",
        # 제라 프롬프트
        "--zera_system_prompt", "You are a highly skilled coding assistant. For each problem, reason step-by-step and then output the final code solution in a single code block, with no extra explanation.",
        "--zera_user_prompt", """Solve the following Python programming problem. 

Provide your final answer as a concise, executable Python function definition only. You may add short inline comments or minimal essential test cases afterward, if needed.

Example:

Question: Write a Python function to determine if a number is prime.

```python
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5)+1):
        if n % i == 0:
            return False
    return True

# is_prime(7) ➞ True
# is_prime(10) ➞ False
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