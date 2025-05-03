from typing import List, Dict, Any, Optional
from evaluation.base.evaluator import BaseEvaluator
from agent.dataset.hellaswag_dataset import HellaSwagDataset
import re
import random

class HellaSwagEvaluator(BaseEvaluator):
    def __init__(self, model_name: str, model_version: str, temperature: float = 0.7, top_p: float = 0.9):
        super().__init__(model_name, model_version, temperature, top_p)
        self.dataset = HellaSwagDataset()
        self.data_cache = None

    def load_dataset(self, split: str = "validation") -> List[Dict[str, Any]]:
        if self.data_cache is None:
            self.data_cache = self.dataset.get_split_data(split)
        return self.data_cache

    def get_sample_indices(self, num_samples: int) -> List[int]:
        data = self.load_dataset()
        total_samples = len(data)
        return random.sample(range(total_samples), min(num_samples, total_samples))

    def format_question(self, item: Dict[str, Any]) -> str:
        context = item['context']
        choices = item['choices']
        choices_str = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        return f"{context}\n\n{choices_str}\n\nAnswer:"

    def evaluate_response(self, response: str, ground_truth: Dict[str, Any]) -> bool:
        response_clean = response.strip().upper()
        # 'Answer:' 이후에 나오는 첫 번째 A/B/C/D 또는 1/2/3/4 추출
        match = re.search(r'ANSWER[:\s]*([A-D1-4])', response_clean)
        if match:
            model_answer = match.group(1)
        else:
            # 괄호 안의 숫자/알파벳 우선 추출
            match = re.search(r'\(([A-D1-4])\)', response_clean)
            if match:
                model_answer = match.group(1)
            else:
                # "final answer:" 등 패턴
                match = re.search(r'(?:FINAL|THE)\s+ANSWER(?:\s+IS)?[:\s]*([A-D1-4])', response_clean)
                if not match:
                    # 첫 알파벳/숫자
                    match = re.match(r'^([A-D1-4])', response_clean)
                if not match:
                    # 마지막 등장하는 알파벳/숫자
                    matches = re.findall(r'([A-D1-4])', response_clean)
                    if matches:
                        model_answer = matches[-1]
                    else:
                        return False
                else:
                    model_answer = match.group(1)

        # 정답 변환
        correct_idx = int(ground_truth['answer'])
        correct_letter = chr(65 + correct_idx)  # 0->A, 1->B, ...
        correct_number = str(correct_idx + 1)   # 0->1, 1->2, ...

        # 디버깅용 상세 로그 추가
        print(f"모델 답변: {model_answer}, correct_letter: {correct_letter}, correct_number: {correct_number}, 실제 답변: {ground_truth['answer']}")
        print(f"비교 결과: {model_answer in [correct_letter, correct_number]}")
        return model_answer in [correct_letter, correct_number] 