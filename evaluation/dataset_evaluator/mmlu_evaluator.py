from typing import List, Dict, Any, Optional
from evaluation.base.evaluator import BaseEvaluator
from datasets import load_dataset
import json
import re
import random

class MMLUEvaluator(BaseEvaluator):
    def __init__(self, model_name: str, model_version: str, temperature: float = 0.7, top_p: float = 0.9):
        """Initialize MMLU evaluator."""
        super().__init__(model_name, model_version, temperature, top_p)
        self.dataset_cache = None

    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load MMLU dataset."""
        if self.dataset_cache is None:
            # Load MMLU dataset from Hugging Face
            dataset = load_dataset("cais/mmlu", "all")
            test_data = dataset["test"]
            
            # Convert to required format
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
        """Return indices of samples to use for evaluation."""
        if self.dataset_cache is None:
            self.load_dataset("")
        
        total_samples = len(self.dataset_cache)
        # Randomly select indices without duplicates
        indices = random.sample(range(total_samples), min(num_samples, total_samples))
        return indices

    def format_question(self, item: Dict[str, Any]) -> str:
        """Format MMLU question."""
        question = item['question']
        choices = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(item['choices'])])
        return f"{question}\n\n{choices}\n\nAnswer:"
    
    def evaluate_response(self, response: str, ground_truth: Dict[str, Any]) -> bool:
        """Evaluate MMLU response."""
        response_clean = response.strip().upper()
        
        # First extract alphabet (A~D) in parentheses
        match = re.search(r'\(([A-D])\)', response_clean)
        if match:
            model_answer = match.group(1)
        else:
            # Search for "final answer:" or "the answer is" pattern
            match = re.search(r'(?:FINAL|THE)\s+ANSWER(?:\s+IS)?[:\s]*([A-D])', response_clean)
            if not match:
                # Extract first alphabet (existing method)
                match = re.match(r'^([A-D])', response_clean)
            if not match:
                # Extract last single character alphabet (A~D)
                matches = re.findall(r'([A-D])', response_clean)
                if matches:
                    model_answer = matches[-1]
                else:
                    return False
            else:
                model_answer = match.group(1)
            
        correct_answer = chr(65 + ground_truth['answer'])  # 0->A, 1->B, 2->C, 3->D
        
        print(f"Model answer: {model_answer}, Correct answer: {correct_answer}")  # For debugging
        return model_answer == correct_answer 