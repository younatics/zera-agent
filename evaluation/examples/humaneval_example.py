import sys
from evaluation.base.main import main
from typing import List
import random

def run_humaneval_example():
    sys.argv = [
        "humaneval_example.py",
        "--dataset", "humaneval",
        "--model", "gpt4o",
        "--model_version", "gpt-3.5-turbo",
        # 기존 프롬프트
        "--base_system_prompt", "Write a Python function that satisfies the following specification.",
        "--base_user_prompt", "Problem:",
        # 제라 프롬프트
        "--zera_system_prompt", "You are a proficient Python coding assistant. When provided a function description, begin by logically reasoning about an efficient approach to solve the problem, clearly defining the key steps or logic needed. After reasoning, directly provide only the concise final Python function code that strictly adheres to the described functionality and structural requirements, omitting extra explanations, comments, docstrings, or additional text.",
        "--zera_user_prompt", """
Implement concise Python functions strictly following their given descriptions. First provide brief, structured reasoning outlining your logical solution approach clearly to help clarify your intended logic. Then directly supply only the concise Python function as your final output.

Example:

Description:
"Write a Python function any_int(x, y, z) that returns True if one integer equals the sum of the other two integers, and all three arguments are integers. Otherwise, return False."

Reasoning:
- Check if arguments are integers.
- Check if any integer equals the sum of the other two integers.
- Return True if the condition holds, else False.

Implementation:
```python
def any_int(x, y, z):
    if all(isinstance(i, int) for i in [x, y, z]):
        return x + y == z or x + z == y or y + z == x
    return False
```

Description:
"Write a Python function largest_divisor(n) that returns the largest divisor of n smaller than n."

Reasoning:
- Start from the integer immediately below n and decrease sequentially.
- Find the first integer that evenly divides n with no remainder.
- Return that integer as the largest divisor.

Implementation:
```python
def largest_divisor(n):
    for i in range(n-1, 0, -1):
        if n % i == 0:
            return i
```

Now implement:

Description:
"Write a Python function car_race_collision(n) that calculates the total number of collisions when n cars moving left to right simultaneously meet another n cars moving right to left, on an infinitely long straight line. Each left-moving car collides exactly once with every right-moving car."

Reasoning:
- Every left-moving car encounters exactly one collision with each of the n right-moving cars.
- There are n left-moving cars and n right-moving cars; thus total collisions = n × n.

Implementation:
```python
def car_race_collision(n):
    return n * n
```""",
        "--num_samples", "100",
    ]
    main()

if __name__ == "__main__":
    run_humaneval_example() 