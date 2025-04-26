from typing import List, Dict, Any, Optional
import json
import re
import pandas as pd
from evaluation.base.evaluator import BaseEvaluator

class GSM8KEvaluator(BaseEvaluator):
    def load_dataset(self, dataset_name: str) -> List[Dict[str, Any]]:
        """GSM8K 데이터셋을 로드합니다."""
        dataset_path = "agent/dataset/gsm8k_data/test.csv"
        df = pd.read_csv(dataset_path)
        data = df.to_dict('records')
        return data
    
    def format_question(self, item: Dict[str, Any]) -> str:
        """GSM8K 질문을 포맷팅합니다."""
        return item['question']
    
    def evaluate_response(self, response: str, ground_truth: Dict[str, Any]) -> bool:
        """GSM8K 응답을 평가합니다."""
        try:
            # 모델 답변에서 마지막 숫자를 찾습니다
            numbers = re.findall(r'\d+(?:\.\d+)?', response)
            if not numbers:
                return False
            model_num = float(numbers[-1])
            
            # 정답에서 #### 뒤의 숫자를 찾습니다
            answer_match = re.search(r'####\s*(\d+(?:\.\d+)?)', str(ground_truth['answer']))
            if not answer_match:
                return False
            correct_num = float(answer_match.group(1))
            
            return abs(model_num - correct_num) < 1e-6
        except:
            return False 