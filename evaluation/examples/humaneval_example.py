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
        "--zera_system_prompt", "You are an expert Python programmer skilled in explicitly evaluating multiple competing implementation strategies before coding. Carefully reason and explicitly compare at least two valid implementation options step-by-step, examining readability, conciseness, efficiency, and Pythonic best practices. Clearly delineate your comparative reasoning before separately and explicitly presenting the rigorously structured final Python implementation that strictly meets the specified function signature, constraints, and edge cases. Your final implementation should integrate your findings from the explicit comparative analysis.",
        "--zera_user_prompt", """Below is a Python function signature with its explicit task description provided in the docstring:

```python
def largest_smallest_integers(lst):
    '''
    Create a function that returns a tuple (a, b), where 'a' is
    the largest of negative integers, and 'b' is the smallest
    of positive integers in a list.
    If there are no negative or positive integers, return them as None.

    Explicit Examples:
    >>> largest_smallest_integers([2, 4, 1, 3, 5, 7])
    (None, 1)
    >>> largest_smallest_integers([])
    (None, None)
    >>> largest_smallest_integers([0])
    (None, None)
    >>> largest_smallest_integers([-4, -2, -9, 0, 3, 7])
    (-2, 3)
    '''
```

Explicit Comparative Reasoning:

Two viable implementation strategies are:

1. **Iterative Approach (single traversal)**:
- Iterate through the list, tracking two variables: one for the largest negative integer (initialized to negative infinity) and one for the smallest positive integer (initialized to positive infinity). Update them whenever a better candidate is found.
- Readability: Excellent; logic is explicit and straightforward.
- Conciseness: Slightly verbose due to explicit value updates, but clear.
- Efficiency: Optimal time complexity of O(n), traverses the list exactly once.
- Pythonic Best Practices: Explicit, direct approach common in Python for clarity and readability.

Example snippet:
```python
max_neg = float('-inf')
min_pos = float('inf')
for num in lst:
    if num < 0 and num > max_neg:
        max_neg = num
    elif num > 0 and num < min_pos:
        min_pos = num
```

2. **Functional Approach (filter with built-ins)**:
- Use Python's built-in `filter()` function twice separately to obtain negatives and positives. Then apply `max()` and `min()` built-ins.
- Readability: Good, but slightly less intuitive due to split logic.
- Conciseness: Potentially more concise with fewer explicit steps.
- Efficiency: Traverses the list effectively twice (filtering negatives and then positives), slightly less efficient but remains O(n).
- Pythonic Best Practices: Demonstrates Python's built-in functions, which is idiomatic but marginally less intuitive here due to two-step filtering.

Example snippet:
```python
negatives = list(filter(lambda x: x < 0, lst))
positives = list(filter(lambda x: x > 0, lst))
max_neg = max(negatives) if negatives else None
min_pos = min(positives) if positives else None
```

Selected Optimal Implementation:

The iterative approach provides clearer readability, explicitly unified logic, optimal efficiency through single traversal, and more straightforward handling of edge cases (empty lists, no negatives/positives). Thus, it is the preferred solution here.

Final Implementation:
```python
def largest_smallest_integers(lst):
    max_neg = float('-inf')
    min_pos = float('inf')

    for num in lst:
        if num < 0 and num > max_neg:
            max_neg = num
        elif num > 0 and num < min_pos:
            min_pos = num

    return (max_neg if max_neg != float('-inf') else None,
            min_pos if min_pos != float('inf') else None)
```

TASK_HINTS:  
- Explicitly reason step-by-step through at least two valid implementation strategies and compare them systematically.
- Clearly separate comparative reasoning analysis explicitly from the structured, final Python implementation.
- Prioritize readability and Pythonic style unless explicit trade-offs justify otherwise.
- Ensure rigorous compliance with all explicitly stated edge cases and constraints (empty lists, absence of positive/negative integers).

FEW_SHOT_EXAMPLES:  
Example:

Function and explicit constraint:
```python
def squared_numbers(n: int) -> list:
    '''
    Return a list containing squares of numbers from 0 up to n inclusive.
    >>> squared_numbers(3)
    [0, 1, 4, 9]
    >>> squared_numbers(0)
    [0]
    '''
```

Explicit Comparative Reasoning:

Two approaches exist explicitly:

- **List comprehension**:
```python
return [x ** 2 for x in range(n + 1)]
```
(Readability: Explicit, highly intuitive; Conciseness: Compact; Efficiency: Optimal single-pass O(n); Pythonic: Strongly emphasized idiom.)

- **Map plus lambda function**:
```python
return list(map(lambda x: x ** 2, range(n + 1)))
```
(Readability: Concise, but slightly less immediately intuitive; Conciseness: Equally concise; Efficiency: Still O(n), small overhead; Pythonic: Slightly less common because of readability considerations.)

Choosing the list comprehension method due to superior readability, intuitive clarity, and Pythonic emphasis:

Final Implementation:
```python
def squared_numbers(n: int) -> list:
    return [x ** 2 for x in range(n + 1)]
```""",
        "--num_samples", "500",
    ]
    main()

if __name__ == "__main__":
    run_humaneval_example() 