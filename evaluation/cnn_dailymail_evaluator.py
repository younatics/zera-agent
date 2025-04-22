from typing import List, Dict, Any
from evaluation.evaluator import BaseEvaluator
from rouge import Rouge
import json

class CNNDailyMailEvaluator(BaseEvaluator):
    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """CNN/DailyMail 데이터셋을 로드합니다."""
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    
    def format_question(self, item: Dict[str, Any]) -> str:
        """CNN/DailyMail 기사를 포맷팅합니다."""
        return item['article']
    
    def evaluate_response(self, response: str, ground_truth: Dict[str, Any]) -> bool:
        """CNN/DailyMail 요약을 평가합니다."""
        # ROUGE 점수를 사용하여 평가
        rouge = Rouge()
        try:
            scores = rouge.get_scores(response, ground_truth['highlights'])
            # ROUGE-L F1 점수가 0.5 이상이면 통과
            return scores[0]['rouge-l']['f'] >= 0.5
        except Exception as e:
            print(f"ROUGE 평가 중 오류 발생: {str(e)}")
            return False 