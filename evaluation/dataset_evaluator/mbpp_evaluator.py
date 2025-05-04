from typing import List, Dict, Any, Optional
from evaluation.base.evaluator import BaseEvaluator
import pandas as pd
import ast
import os
import random
import re
import logging
import signal
import subprocess
import tempfile
import math
import difflib

class MBPPEvaluator(BaseEvaluator):
    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """MBPP 데이터셋을 로드합니다."""
        if dataset_path == "mbpp":
            dataset_path = "agent/dataset/mbpp_data/test.csv"
        df = pd.read_csv(dataset_path)
        data = df.to_dict('records')
        return data
    
    def format_question(self, item: Dict[str, Any]) -> str:
        """MBPP 질문을 포맷팅합니다."""
        # code에서 함수명 추출
        match = re.search(r'def ([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', item['code'])
        func_name = match.group(1) if match else "unknown_function"
        return f"{item['text']}\n\nFunction name: {func_name}"
    
    def extract_code(self, response: str) -> str:
        # 코드블록이 있으면 내부만 추출
        code_match = re.search(r"```(?:python)?(.*?)```", response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        # 코드블록이 없으면 함수 정의부터 끝까지 추출 (fallback)
        func_def_match = re.search(r'(def [a-zA-Z_][a-zA-Z0-9_]*\s*\(.*)', response, re.DOTALL)
        if func_def_match:
            return func_def_match.group(1).strip()
        return response.strip()

    def _find_best_func_name(self, code_block: str, target_func: str) -> str:
        func_names = re.findall(r'def ([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code_block)
        if not func_names:
            return None
        best = difflib.get_close_matches(target_func, func_names, n=1, cutoff=0.6)
        return best[0] if best else func_names[0]

    def _auto_imports(self, code_block: str) -> str:
        imports = []
        if re.search(r'\bmath\.', code_block) and 'import math' not in code_block:
            imports.append('import math')
        if re.search(r'\bre\.', code_block) and 'import re' not in code_block:
            imports.append('import re')
        if re.search(r'\bcollections\.', code_block) and 'import collections' not in code_block:
            imports.append('import collections')
        if re.search(r'\bitertools\.', code_block) and 'import itertools' not in code_block:
            imports.append('import itertools')
        if re.search(r'\bfunctools\.', code_block) and 'import functools' not in code_block:
            imports.append('import functools')
        if re.search(r'\bheapq\.', code_block) and 'import heapq' not in code_block:
            imports.append('import heapq')
        if re.search(r'\bdatetime\.', code_block) and 'import datetime' not in code_block:
            imports.append('import datetime')
        return '\n'.join(imports) + ('\n' if imports else '') + code_block

    def _wrap_lambda(self, code_block: str, func_name: str, gt_func: str) -> str:
        lambda_match = re.search(r'(\w+)\s*=\s*lambda', code_block)
        if lambda_match:
            name = lambda_match.group(1)
            sig_match = re.search(r'def '+re.escape(gt_func)+r'\((.*?)\)', code_block)
            sig = sig_match.group(1) if sig_match else 'x'
            return code_block + f"\ndef {gt_func}({sig}):\n    return {name}({sig})"
        return code_block

    def _normalize_output(self, output):
        try:
            val = eval(output, {"__builtins__": {}})
            if isinstance(val, (list, tuple)):
                return list(val)
            if isinstance(val, float):
                return round(val, 6)
            if isinstance(val, str):
                return val.strip().lower()
            return val
        except Exception:
            return str(output).strip().lower()

    def evaluate_response(self, response: str, ground_truth: Dict[str, Any]) -> bool:
        logger = logging.getLogger(__name__)
        try:
            import_lines = "\n".join([line for line in response.splitlines() if line.strip().startswith("import") or line.strip().startswith("from ")])
            code_block = self.extract_code(response)
            gt_func_match = re.search(r'def ([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', ground_truth['code'])
            gt_func_name = gt_func_match.group(1) if gt_func_match else "unknown_function"
            best_func = self._find_best_func_name(code_block, gt_func_name)
            if not best_func:
                code_block = self._wrap_lambda(code_block, best_func, gt_func_name)
                best_func = gt_func_name
            code_block = re.sub(r'def ([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', f'def {gt_func_name}(', code_block, count=1)
            code_block = self._auto_imports(code_block)
            test_cases = ast.literal_eval(ground_truth['test_list'])
            full_code = import_lines + "\n" + code_block
            for test_case in test_cases:
                try:
                    test_case = test_case.replace("assert ", "")
                    call, expected = test_case.split("==")
                    call = call.strip()
                    expected = expected.strip()
                    test_code = f"{full_code}\nprint({call})"
                    stdout, stderr, success = run_safely(test_code, timeout=10)
                    if not success:
                        logger.error(f"[서브프로세스실패] {call} | stdout: {stdout} | stderr: {stderr}")
                        return False
                    actual_output = stdout.strip()
                    norm_actual = self._normalize_output(actual_output)
                    norm_expected = self._normalize_output(expected)
                    if isinstance(norm_actual, float) and isinstance(norm_expected, float):
                        if not math.isclose(norm_actual, norm_expected, rel_tol=1e-4, abs_tol=1e-4):
                            logger.error(f"[테스트실패] {call} -> {norm_actual} (예상: {norm_expected})")
                            return False
                    elif norm_actual != norm_expected:
                        logger.error(f"[테스트실패] {call} -> {norm_actual} (예상: {norm_expected})")
                        return False
                except Exception as e:
                    logger.error(f"[테스트케이스실행실패] {test_case} | {e}")
                    return False
            return True
        except Exception as e:
            logger.error(f"[예외] {e}\nresponse: {response}\nground_truth: {ground_truth}")
            return False

    def get_sample_indices(self, num_samples: int) -> List[int]:
        dataset = self.load_dataset("agent/dataset/mbpp_data/test.csv")
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