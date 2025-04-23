from typing import List, Dict, Any, Optional
from evaluation.base.evaluator import BaseEvaluator
from datasets import load_dataset
import json
import re

class MMLUEvaluator(BaseEvaluator):
    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """MMLU 데이터셋을 로드합니다."""
        # Hugging Face에서 MMLU 데이터셋 로드
        dataset = load_dataset("cais/mmlu", "all")
        test_data = dataset["test"]
        
        # 필요한 형식으로 변환
        formatted_data = []
        for item in test_data:
            formatted_item = {
                "question": item["question"],
                "choices": item["choices"],
                "answer": item["answer"]
            }
            formatted_data.append(formatted_item)
            
        return formatted_data
    
    def format_question(self, item: Dict[str, Any]) -> str:
        """MMLU 질문을 포맷팅합니다."""
        question = item['question']
        choices = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(item['choices'])])
        return f"{question}\n\n{choices}\n\nAnswer:"
    
    def evaluate_response(self, response: str, ground_truth: Dict[str, Any]) -> bool:
        """MMLU 응답을 평가합니다."""
        # 응답에서 첫 번째 알파벳 추출 (정규식 사용)
        match = re.match(r'^([A-D])', response.strip().upper())
        if not match:
            return False
            
        model_answer = match.group(1)  # 'A', 'B', 'C', 'D'
        correct_answer = chr(65 + ground_truth['answer'])  # 0->A, 1->B, 2->C, 3->D
        
        print(f"모델 답변: {model_answer}, 정답: {correct_answer}")  # 디버깅용
        return model_answer == correct_answer 