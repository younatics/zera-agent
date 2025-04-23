from typing import List, Dict, Any, Optional
from evaluation.base.evaluator import BaseEvaluator
import json
import ast
import os

class MBPPEvaluator(BaseEvaluator):
    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """MBPP 데이터셋을 로드합니다."""
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    
    def format_question(self, item: Dict[str, Any]) -> str:
        """MBPP 질문을 포맷팅합니다."""
        return f"{item['text']}\n\nFunction name: {item['entry_point']}"
    
    def evaluate_response(self, response: str, ground_truth: Dict[str, Any]) -> bool:
        """MBPP 응답을 평가합니다."""
        try:
            # 응답에서 함수 정의 추출
            func_def = response.split("def")[1].split("\n")[0]
            func_name = func_def.split("(")[0].strip()
            
            # 함수 이름이 일치하는지 확인
            if func_name != ground_truth['entry_point']:
                return False
                
            # 테스트 케이스 실행
            test_cases = ground_truth['tests']
            local_vars = {}
            exec(response, {}, local_vars)
            func = local_vars[func_name]
            
            for test_case in test_cases:
                # 테스트 케이스에서 입력값과 예상 출력값 추출
                test_input = test_case.split("assert")[1].split("==")[0].strip()
                expected_output = test_case.split("==")[1].strip()
                
                # 테스트 실행
                actual_output = str(eval(f"func({test_input})"))
                if actual_output != expected_output:
                    return False
                    
            return True
        except:
            return False 