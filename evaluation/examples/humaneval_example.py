import sys
from evaluation.base.main import main
from typing import List
import random

def run_humaneval_example(model="gpt4o", model_version="gpt-3.5-turbo"):
    sys.argv = [
        "humaneval_example.py",
        "--dataset", "humaneval",
        "--model", model,
        "--model_version", model_version,
        # 기존 프롬프트
        "--base_system_prompt", "Write a Python function that satisfies the following specification.",
        "--base_user_prompt", "Problem:",
        # 제라 프롬프트
        "--zera_system_prompt", "You are an expert Python assistant. Clearly reason step-by-step through your approach to solving the given task without initially worrying about format or style. After thoroughly completing your logical reasoning, provide a concise, strictly accurate Python implementation within a cleanly formatted code block. Include only essential inline comments to explain genuinely complex logic or unique design choices. Adhere rigorously to the provided instructions and explicitly manage any specified conditions or edge cases.",
        "--zera_user_prompt", """
Implement the Python function according to the provided instructions or docstring. Begin with clear, thorough logical reasoning about how you will approach and solve the task, ensuring you explicitly discuss special cases or important conditions mentioned.

Example:

"""
Write a function 'digits(n)' that takes a positive integer n and returns the product of its odd digits. If all digits are even, return 0.

Examples:  
digits(1)  => 1  
digits(4)  => 0  
digits(235) => 15  
"""

Logical Reasoning:
Let's approach this step by step:
- We'll initialize a product variable (`product`) with 1 and a boolean flag (`has_odd_digit`) set to False.
- Convert the integer `n` to a string to iterate through each digit.
- For each digit, check if it is odd (digit%2 == 1):
  - If odd, multiply it into `product` and set the flag (`has_odd_digit`) to True.
- Finally, return the `product` if at least one odd digit is found, otherwise return 0.

Implementation:
```python
def digits(n):
    product = 1
    has_odd_digit = False

    for digit in str(n):
        digit = int(digit)
        if digit % 2 == 1:
            product *= digit  # multiply odd digit into product
            has_odd_digit = True

    return product if has_odd_digit else 0
```""",
        "--num_samples", "1000",
    ]
    main()

if __name__ == "__main__":
    run_humaneval_example() 