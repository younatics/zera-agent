from typing import List, Dict, Any, Optional
from evaluation.base.evaluator import BaseEvaluator
from datasets import load_dataset
import json
import re
import random

class MMLUProEvaluator(BaseEvaluator):
    def __init__(self, model_name: str, model_version: str, temperature: float = 0.7, top_p: float = 0.9):
        """Initialize MMLU Pro evaluator."""
        super().__init__(model_name, model_version, temperature, top_p)
        self.dataset_cache = None

    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load MMLU Pro dataset."""
        if self.dataset_cache is None:
            # Load MMLU dataset from Hugging Face
            dataset = load_dataset("TIGER-Lab/MMLU-Pro", "default")
            test_data = dataset["test"]
            
            # Convert to required format
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
        """Return indices of samples to use for evaluation."""
        if self.dataset_cache is None:
            self.load_dataset("")
        
        total_samples = len(self.dataset_cache)
        # Randomly select indices without duplication
        indices = random.sample(range(total_samples), min(num_samples, total_samples))
        return indices

    def format_question(self, item: Dict[str, Any]) -> str:
        """Format MMLU Pro question."""
        question = item['question']
        choices = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(item['choices'])])
        return f"{question}\n\n{choices}\n\nAnswer:"
    
    def evaluate_response(self, response: str, ground_truth: Dict[str, Any]) -> bool:
        """Evaluate MMLU Pro response."""
        response_clean = response.strip().upper()
        # 0. Extract alphabet (A~J) in parentheses (prioritize the last one)
        matches = re.findall(r'\(([A-J])\)', response_clean)
        if matches:
            model_answer = matches[-1]  # Use the last alphabet in parentheses
        else:
            # 1. Various patterns like '**J. ...**' or '**J**' or 'J. ...' or 'J ...'
            match = re.search(r'\*\*?([A-J])\*\*?[\s\.]', response_clean)  # '**J. ...' '**J**' etc.
            if not match:
                match = re.search(r'([A-J])\.[\s]', response_clean)  # 'J. ...'
            if not match:
                match = re.search(r'([A-J])[\s]', response_clean)  # 'J ...'
            if not match:
                # 2. "the answer is X" or "the answer is (X)" pattern
                match = re.search(r'answer is[\s:]*\(?([A-J])\)?', response_clean)
            if not match:
                # 3. Korean patterns like "정답은 X" or "정답은 (X)" also added
                match = re.search(r'정답[은는]?[\s:]*\(?([A-J])\)?', response_clean)
            if not match:
                # 4. Extract the last single alphabet (A~J)
                matches = re.findall(r'([A-J])', response_clean)
                if matches:
                    model_answer = matches[-1]
                else:
                    return False
            else:
                model_answer = match.group(1)
        # Compare with correct answer index (branch by type)
        correct_answer = ground_truth['answer']
        if isinstance(correct_answer, int):
            model_answer_idx = ord(model_answer) - 65
            print(f"Model answer: {model_answer_idx}, Correct answer: {correct_answer}")  # For debugging
            return model_answer_idx == correct_answer
        elif isinstance(correct_answer, str) and len(correct_answer) == 1 and correct_answer.isalpha():
            print(f"Model answer: {model_answer}, Correct answer: {correct_answer}")  # For debugging
            return model_answer == correct_answer.upper()
        else:
            return False 