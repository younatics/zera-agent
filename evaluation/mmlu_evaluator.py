from typing import List, Dict, Any
import json
from .evaluator import BaseEvaluator

class MMLUEvaluator(BaseEvaluator):
    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """MMLU 데이터셋을 로드합니다."""
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    
    def format_question(self, item: Dict[str, Any]) -> str:
        """MMLU 질문을 포맷팅합니다."""
        question = item['question']
        choices = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(item['choices'])])
        return f"{question}\n\n{choices}"
    
    def evaluate_response(self, response: str, ground_truth: Dict[str, Any]) -> bool:
        """MMLU 응답을 평가합니다."""
        # 응답에서 첫 번째 알파벳 추출
        response = response.strip().upper()
        if not response:
            return False
            
        model_answer = response[0]
        correct_answer = chr(65 + ground_truth['answer'])  # 0->A, 1->B, ...
        return model_answer == correct_answer 