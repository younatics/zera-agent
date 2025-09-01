from typing import List, Dict, Any, Optional
from evaluation.base.evaluator import BaseEvaluator
from agent.dataset.humaneval_dataset import HumanEvalDataset
import ast
import re
import logging
import subprocess
import tempfile
import json
import random

class HumanEvalEvaluator(BaseEvaluator):
    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        # Ignore dataset_path and always use HumanEvalDataset
        dataset = HumanEvalDataset()
        return dataset.get_split_data("test")

    def format_question(self, item: Dict[str, Any]) -> str:
        # Prompt + function name guidance
        return f"{item['prompt']}\n\nFunction name: {item['entry_point']}"

    def evaluate_response(self, response: str, ground_truth: Dict[str, Any]) -> bool:
        logger = logging.getLogger(__name__)
        try:
            # 1. Extract import statements from entire model response
            import_lines = "\n".join([line for line in response.splitlines() if line.strip().startswith("import") or line.strip().startswith("from ")])
            # 2. Extract code block (only the last code block)
            code_block = response
            code_blocks = re.findall(r"```python(.*?)```", response, re.DOTALL)
            if code_blocks:
                code_block = code_blocks[-1]
            # Extract function name
            func_def_match = re.search(r'def ([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code_block)
            func_name = func_def_match.group(1) if func_def_match else "unknown_function"
            if func_name != ground_truth['entry_point']:
                logger.error(f"[Function name mismatch] Model: {func_name}, Correct: {ground_truth['entry_point']}")
                return False
            # Extract check function code
            test_cases_code = ground_truth['test_cases']
            if not test_cases_code or not isinstance(test_cases_code, str) or not test_cases_code.strip():
                logger.error(f"[No test cases] {ground_truth['task_id']}")
                return False
            # Add candidate alias code
            candidate_alias = f"\ncandidate = {func_name}\n"
            # Combine all code (add check(candidate) call at the end)
            full_code = f"{import_lines}\n{code_block}\n{candidate_alias}\n{test_cases_code}\n\ncheck(candidate)"
            print("[FULL_CODE]====================\n" + full_code + "\n============================\n")
            # Execute with subprocess
            stdout, stderr, success = run_safely(full_code, timeout=180)
            print(f"[EVAL_RESULT] success={success}\nstdout=\n{stdout}\nstderr=\n{stderr}")
            if not success:
                logger.error(f"[Subprocess failed] stdout: {stdout} | stderr: {stderr}")
                return False
            return True
        except Exception as e:
            logger.error(f"[Exception] {e}\nresponse: {response}\nground_truth: {ground_truth}")
            return False

    def get_sample_indices(self, num_samples: int) -> List[int]:
        dataset = self.load_dataset(None)
        total_samples = len(dataset)
        if num_samples > total_samples:
            num_samples = total_samples
        return random.sample(range(total_samples), num_samples)

def run_safely(code: str, timeout=2):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_path = f.name
    try:
        result = subprocess.run(
            ["python3", temp_path],
            timeout=timeout,
            capture_output=True,
        )
        return result.stdout.decode(), result.stderr.decode(), result.returncode == 0
    except subprocess.TimeoutExpired:
        return "Timeout", "", False 