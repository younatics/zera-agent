from typing import List, Dict, Any
import json
import re
from .evaluator import BaseEvaluator

class GSM8KEvaluator(BaseEvaluator):
    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """GSM8K 데이터셋을 로드합니다."""
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    
    def format_question(self, item: Dict[str, Any]) -> str:
        """GSM8K 질문을 포맷팅합니다."""
        return item['question']
    
    def evaluate_response(self, response: str, ground_truth: Dict[str, Any]) -> bool:
        """GSM8K 응답을 평가합니다."""
        # 정답 추출
        answer_pattern = r"\\boxed{([^}]+)}"
        matches = re.findall(answer_pattern, response)
        if not matches:
            return False
            
        # 마지막 boxed 값이 최종 답안
        model_answer = matches[-1].strip()
        try:
            # 숫자만 추출하여 비교
            model_num = float(re.sub(r'[^\d.]', '', model_answer))
            correct_num = float(ground_truth['answer'].split('####')[1].strip())
            return abs(model_num - correct_num) < 1e-6
        except:
            return False 