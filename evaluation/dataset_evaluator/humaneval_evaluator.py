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
        # dataset_path는 무시하고 항상 HumanEvalDataset을 사용
        dataset = HumanEvalDataset()
        return dataset.get_split_data("test")

    def format_question(self, item: Dict[str, Any]) -> str:
        # 프롬프트 + 함수명 안내
        return f"{item['prompt']}\n\nFunction name: {item['entry_point']}"

    def evaluate_response(self, response: str, ground_truth: Dict[str, Any]) -> bool:
        logger = logging.getLogger(__name__)
        try:
            # 1. 모델 답변 전체에서 import문 추출
            import_lines = "\n".join([line for line in response.splitlines() if line.strip().startswith("import") or line.strip().startswith("from ")])
            # 2. 코드블록 추출 (맨 마지막 코드블록만)
            code_block = response
            code_blocks = re.findall(r"```python(.*?)```", response, re.DOTALL)
            if code_blocks:
                code_block = code_blocks[-1]
            # 함수명 추출
            func_def_match = re.search(r'def ([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code_block)
            func_name = func_def_match.group(1) if func_def_match else "unknown_function"
            if func_name != ground_truth['entry_point']:
                logger.error(f"[함수명불일치] 모델: {func_name}, 정답: {ground_truth['entry_point']}")
                return False
            # check 함수 코드 추출
            test_cases_code = ground_truth['test_cases']
            if not test_cases_code or not isinstance(test_cases_code, str) or not test_cases_code.strip():
                logger.error(f"[테스트케이스없음] {ground_truth['task_id']}")
                return False
            # candidate alias 코드 추가
            candidate_alias = f"\ncandidate = {func_name}\n"
            # 전체 코드 합치기 (마지막에 check(candidate) 호출 추가)
            full_code = f"{import_lines}\n{code_block}\n{candidate_alias}\n{test_cases_code}\n\ncheck(candidate)"
            print("[FULL_CODE]====================\n" + full_code + "\n============================\n")
            # subprocess로 실행
            stdout, stderr, success = run_safely(full_code, timeout=180)
            print(f"[EVAL_RESULT] success={success}\nstdout=\n{stdout}\nstderr=\n{stderr}")
            if not success:
                logger.error(f"[서브프로세스실패] stdout: {stdout} | stderr: {stderr}")
                return False
            return True
        except Exception as e:
            logger.error(f"[예외] {e}\nresponse: {response}\nground_truth: {ground_truth}")
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