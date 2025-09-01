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

    def load_dataset(self, *args, **kwargs) -> List[Dict[str, Any]]:
        if self.data_cache is None:
            self.data_cache = self.dataset.get_split_data('validation')
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
        # Extract first A/B/C/D or 1/2/3/4 after 'Answer:'
        match = re.search(r'ANSWER[:\s]*([A-D1-4])', response_clean)
        if match:
            model_answer = match.group(1)
        else:
            # Prioritize numbers/alphabet in parentheses
            match = re.search(r'\(([A-D1-4])\)', response_clean)
            if match:
                model_answer = match.group(1)
            else:
                # "final answer:" etc. patterns
                match = re.search(r'(?:FINAL|THE)\s+ANSWER(?:\s+IS)?[:\s]*([A-D1-4])', response_clean)
                if not match:
                    # First alphabet/number
                    match = re.match(r'^([A-D1-4])', response_clean)
                if not match:
                    # Last occurring alphabet/number
                    matches = re.findall(r'([A-D1-4])', response_clean)
                    if matches:
                        model_answer = matches[-1]
                    else:
                        return False
                else:
                    model_answer = match.group(1)

        # Convert correct answer
        correct_idx = int(ground_truth['answer'])
        correct_letter = chr(65 + correct_idx)  # 0->A, 1->B, ...
        correct_number = str(correct_idx + 1)   # 0->1, 1->2, ...

        # Add detailed debug logs
        print(f"Model answer: {model_answer}, correct_letter: {correct_letter}, correct_number: {correct_number}, actual answer: {ground_truth['answer']}")
        print(f"Comparison result: {model_answer in [correct_letter, correct_number]}")
        return model_answer in [correct_letter, correct_number] 