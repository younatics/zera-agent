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
        "--base_system_prompt", "Write a Python function that satisfies the following specification.",
        "--base_user_prompt", "Problem:",
        # 제라 프롬프트
        "--zera_system_prompt", "You are an expert Python coding assistant that first provides explicit, step-by-step reasoning for your solutions, carefully exploring key decisions and edge cases. After clearly outlining this logic, present concise, efficient, and syntactically correct Python function implementations. Strictly adhere to provided naming conventions, employ built-in Python functions and concise language structures, and only include brief explanatory comments when needed to clarify essential but non-obvious reasoning.",
        "--zera_user_prompt", """Question:
Write a Python function named "zip_list" that accepts two lists of lists and concatenates each pair of corresponding inner lists.

Reasoning:
1. If either input list is empty, we cannot create valid concatenated pairs; hence, explicitly return an empty list in this situation.
2. Since the input lists could differ in length, we need to find the length of the shorter one to prevent index errors.
3. Concatenating each pair of inner lists sequentially from the input lists will efficiently give the required result.
4. To achieve clarity and concise implementation, leverage built-in Python functions: use `zip()` to pair inner lists and a concise list comprehension for concatenation.

Final Function Implementation:
```python
def zip_list(list1, list2):
    # Handle edge case: if either list is empty, return empty list
    if not list1 or not list2:
        return []
    # Concatenate corresponding inner lists up to the shortest list's length
    return [x + y for x, y in zip(list1, list2)]
```

TASK_HINTS:
- Explicitly reason step-by-step before writing any code; surface logic transparently
- Identify and clearly handle critical edge cases, especially empty input and unequal lengths
- Always follow the exact provided naming and coding conventions
- Efficiently employ Python built-in functions (e.g., zip, list comprehensions) to keep code concise
- Ensure separate reasoning and final code implementation sections, clearly delineated

FEW_SHOT_EXAMPLES:
Example:
Question: Write a Python function named "reverse_Words" that takes a sentence and reverses the order of words.

Reasoning:
1. Trim leading and trailing spaces to ensure clean input handling.
2. Split the sentence into words to naturally manage multiple adjacent spaces.
3. Reverse the list of words obtained from splitting.
4. Join and return the reversed words separated by single spaces.

Final Function Implementation:
```python
def reverse_Words(sentence):
    words = sentence.strip().split()
    return ' '.join(words[::-1])
```""",
        "--num_samples", "500",
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