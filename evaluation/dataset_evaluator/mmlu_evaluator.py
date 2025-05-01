from typing import List, Dict, Any, Optional
from evaluation.base.evaluator import BaseEvaluator
from datasets import load_dataset
import json
import re
import random

class MMLUEvaluator(BaseEvaluator):
    def __init__(self, model_name: str, model_version: str, temperature: float = 0.7, top_p: float = 0.9):
        """MMLU 평가기를 초기화합니다."""
        super().__init__(model_name, model_version, temperature, top_p)
        self.dataset_cache = None

    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """MMLU 데이터셋을 로드합니다."""
        if self.dataset_cache is None:
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
            
            self.dataset_cache = formatted_data
            
        return self.dataset_cache
    
    def get_sample_indices(self, num_samples: int) -> List[int]:
        """평가에 사용할 샘플의 인덱스를 반환합니다."""
        if self.dataset_cache is None:
            self.load_dataset("")
        
        total_samples = len(self.dataset_cache)
        # 중복 없이 랜덤하게 인덱스 선택
        indices = random.sample(range(total_samples), min(num_samples, total_samples))
        return indices

    def format_question(self, item: Dict[str, Any]) -> str:
        """MMLU 질문을 포맷팅합니다."""
        question = item['question']
        choices = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(item['choices'])])
        return f"{question}\n\n{choices}\n\nAnswer:"
    
    def evaluate_response(self, response: str, ground_truth: Dict[str, Any]) -> bool:
        """MMLU 응답을 평가합니다."""
        response_clean = response.strip().upper()
        
        # 괄호 안에 있는 알파벳(A~D) 우선 추출
        match = re.search(r'\(([A-D])\)', response_clean)
        if match:
            model_answer = match.group(1)
        else:
            # "final answer:" 또는 "the answer is" 패턴 검색
            match = re.search(r'(?:FINAL|THE)\s+ANSWER(?:\s+IS)?[:\s]*([A-D])', response_clean)
            if not match:
                # 첫 번째 알파벳 추출 (기존 방식)
                match = re.match(r'^([A-D])', response_clean)
            if not match:
                # 마지막에 등장하는 한 글자 알파벳 추출 (A~D)
                matches = re.findall(r'([A-D])', response_clean)
                if matches:
                    model_answer = matches[-1]
                else:
                    return False
            else:
                model_answer = match.group(1)
            
        correct_answer = chr(65 + ground_truth['answer'])  # 0->A, 1->B, 2->C, 3->D
        
        print(f"모델 답변: {model_answer}, 정답: {correct_answer}")  # 디버깅용
        return model_answer == correct_answer 