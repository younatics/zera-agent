from typing import List, Dict, Any
import json
from .evaluator import BaseEvaluator

class BBHEvaluator(BaseEvaluator):
    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """BBH 데이터셋을 로드합니다."""
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    
    def format_question(self, item: Dict[str, Any]) -> str:
        """BBH 질문을 포맷팅합니다."""
        return f"{item['input']}\n\n{item['question']}"
    
    def evaluate_response(self, response: str, ground_truth: Dict[str, Any]) -> bool:
        """BBH 응답을 평가합니다."""
        # 정답과 모델 응답을 소문자로 변환하고 공백 제거
        model_answer = response.strip().lower()
        correct_answer = ground_truth['answer'].strip().lower()
        
        # 정확한 문자열 매칭
        return model_answer == correct_answer 