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
    
    def evaluate_response(self, response: str, ground_truth: Dict[str, Any]) -> bool:
        """MBPP 응답을 평가합니다."""
        logger = logging.getLogger(__name__)
        try:
            # 1. 모델 답변 전체에서 import문 추출
            import_lines = "\n".join([line for line in response.splitlines() if line.strip().startswith("import") or line.strip().startswith("from ")])
            # 2. 코드블록 추출
            code_block = response
            code_match = re.search(r"```python(.*?)```", response, re.DOTALL)
            if code_match:
                code_block = code_match.group(1)
            # 응답에서 함수 정의 추출
            func_def_match = re.search(r'def ([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code_block)
            func_name = func_def_match.group(1) if func_def_match else "unknown_function"
            if func_name == "unknown_function":
                logger.error(f"[파싱실패] 함수명 추출 실패. response: {response}")
                return False
            # 정답 함수명 추출
            gt_func_match = re.search(r'def ([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', ground_truth['code'])
            gt_func_name = gt_func_match.group(1) if gt_func_match else "unknown_function"
            if func_name != gt_func_name:
                logger.error(f"[함수명불일치] 모델: {func_name}, 정답: {gt_func_name}")
                return False
            # 테스트 케이스 실행
            test_cases = ast.literal_eval(ground_truth['test_list'])
            # 3. 합쳐서 실행
            full_code = import_lines + "\n" + code_block
            for test_case in test_cases:
                try:
                    test_case = test_case.replace("assert ", "")
                    call, expected = test_case.split("==")
                    call = call.strip()
                    expected = expected.strip()
                    # 테스트 실행 코드 생성
                    test_code = f"{full_code}\nprint({call})"
                    stdout, stderr, success = run_safely(test_code, timeout=180)
                    if not success:
                        logger.error(f"[서브프로세스실패] {call} | stdout: {stdout} | stderr: {stderr}")
                        return False
                    actual_output = stdout.strip()
                    # 타입 유연 비교
                    try:
                        expected_eval = str(eval(expected))
                        if actual_output != expected_eval:
                            logger.error(f"[테스트실패] {call} -> {actual_output} (예상: {expected_eval})")
                            return False
                    except Exception:
                        if actual_output != expected:
                            logger.error(f"[테스트실패] {call} -> {actual_output} (예상: {expected})")
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