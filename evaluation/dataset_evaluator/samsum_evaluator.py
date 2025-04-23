from typing import List, Dict, Any, Optional
from evaluation.base.evaluator import BaseEvaluator
from rouge import Rouge

class SAMSumEvaluator(BaseEvaluator):
    def __init__(self, model_name: str = "gpt4o", model_version: str = "gpt-3.5-turbo"):
        super().__init__(model_name, model_version)
        self.rouge = Rouge()
        
    def format_question(self, item: Dict[str, Any]) -> str:
        """SAMSum 질문을 포맷팅합니다."""
        return item['dialogue']
    
    def evaluate_response(self, response: str, ground_truth: Dict[str, Any]) -> bool:
        """SAMSum 응답을 평가합니다."""
        try:
            # ROUGE 점수 계산
            scores = self.rouge.get_scores(response, ground_truth['summary'])[0]
            # ROUGE-L F1 점수가 0.3 이상이면 합격
            return scores['rouge-l']['f'] >= 0.3
        except:
            return False 