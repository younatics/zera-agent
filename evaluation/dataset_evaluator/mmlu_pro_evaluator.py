from typing import List, Dict, Any, Optional
from evaluation.base.evaluator import BaseEvaluator
from datasets import load_dataset
import json
import re
import random

class MMLUProEvaluator(BaseEvaluator):
    def __init__(self, model_name: str, model_version: str, temperature: float = 0.7, top_p: float = 0.9):
        """MMLU Pro 평가기를 초기화합니다."""
        super().__init__(model_name, model_version, temperature, top_p)
        self.dataset_cache = None

    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """MMLU Pro 데이터셋을 로드합니다."""
        if self.dataset_cache is None:
            # Hugging Face에서 MMLU 데이터셋 로드
            dataset = load_dataset("TIGER-Lab/MMLU-Pro", "default")
            test_data = dataset["test"]
            
            # 필요한 형식으로 변환
            formatted_data = []
            for item in test_data:
                formatted_item = {
                    "question": item["question"],
                    "choices": item["options"],
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
        """MMLU Pro 질문을 포맷팅합니다."""
        question = item['question']
        choices = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(item['choices'])])
        return f"{question}\n\n{choices}\n\nAnswer:"
    
    def evaluate_response(self, response: str, ground_truth: Dict[str, Any]) -> bool:
        """MMLU Pro 응답을 평가합니다."""
        response_clean = response.strip().upper()
        # 0. 괄호 안에 있는 알파벳(A~J) 우선 추출
        match = re.search(r'\(([A-J])\)', response_clean)
        if match:
            model_answer = match.group(1)
        else:
            # 1. '**J. ...**' 또는 '**J**' 또는 'J. ...' 또는 'J ...' 등 다양한 패턴
            match = re.search(r'\*\*?([A-J])\*\*?[\s\.]', response_clean)  # '**J. ...' '**J**' 등
            if not match:
                match = re.search(r'([A-J])\.[\s]', response_clean)  # 'J. ...'
            if not match:
                match = re.search(r'([A-J])[\s]', response_clean)  # 'J ...'
            if not match:
                # 2. "the answer is X" 또는 "the answer is (X)" 패턴
                match = re.search(r'answer is[\s:]*\(?([A-J])\)?', response_clean)
            if not match:
                # 3. "정답은 X" 또는 "정답은 (X)" 등 한글 패턴도 추가
                match = re.search(r'정답[은는]?[\s:]*\(?([A-J])\)?', response_clean)
            if not match:
                # 4. 마지막에 등장하는 한 글자 알파벳 추출 (A~J)
                matches = re.findall(r'([A-J])', response_clean)
                if matches:
                    model_answer = matches[-1]
                else:
                    return False
            else:
                model_answer = match.group(1)
        # 정답 인덱스와 비교 (타입에 따라 분기)
        correct_answer = ground_truth['answer']
        if isinstance(correct_answer, int):
            model_answer_idx = ord(model_answer) - 65
            print(f"모델 답변: {model_answer_idx}, 정답: {correct_answer}")  # 디버깅용
            return model_answer_idx == correct_answer
        elif isinstance(correct_answer, str) and len(correct_answer) == 1 and correct_answer.isalpha():
            print(f"모델 답변: {model_answer}, 정답: {correct_answer}")  # 디버깅용
            return model_answer == correct_answer.upper()
        else:
            return False 