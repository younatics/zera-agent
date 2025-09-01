from typing import List, Dict, Any, Optional
import json
import re
import pandas as pd
import random
from evaluation.base.evaluator import BaseEvaluator

class GSM8KEvaluator(BaseEvaluator):
    def load_dataset(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Load GSM8K dataset."""
        dataset_path = "agent/dataset/gsm8k_data/test.csv"
        df = pd.read_csv(dataset_path)
        data = df.to_dict('records')
        print(f"\nTotal dataset size: {len(df)} samples")
        print(f"Actual dataset size used: {len(data)} samples")
        print("-" * 50)
        return data
    
    def format_question(self, item: Dict[str, Any]) -> str:
        """Format GSM8K question."""
        return item['question']
    
    def get_sample_indices(self, num_samples: int) -> List[int]:
        """Return indices of samples to evaluate."""
        dataset = self.load_dataset("gsm8k")
        total_samples = len(dataset)
        if num_samples > total_samples:
            num_samples = total_samples
        return random.sample(range(total_samples), num_samples)
    
    def evaluate_response(self, response: str, ground_truth: str) -> bool:
        """Evaluate GSM8K response."""
        try:
            def extract_last_number(text: str) -> float:
                """Extract the last number from text."""
                numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', str(text))
                if not numbers:
                    return None
                return float(numbers[-1].replace(',', ''))

            def extract_answer_after_hash(text: str) -> float:
                """Extract the number that appears after ####."""
                match = re.search(r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', str(text))
                if not match:
                    return None
                return float(match.group(1).replace(',', ''))

            # Try to extract number after ####
            model_number = extract_answer_after_hash(response)
            
            # Use last number if #### format is not found
            if model_number is None:
                model_number = extract_last_number(response)
                print("\n[WARNING] #### format not found, using last number.")

            ground_truth_number = extract_last_number(ground_truth)

            if not model_number:
                print("\n[PARSING FAILED] Could not find number in model response.")
                return False

            if not ground_truth_number:
                print("\n[PARSING FAILED] Could not find number in ground truth.")
                return False

            print(f"\n[PARSED ANSWER] Model: {model_number}")
            print(f"[PARSED ANSWER] Ground Truth: {ground_truth_number}")

            # Allow small error for floating point comparison
            return abs(model_number - ground_truth_number) < 0.01

        except Exception as e:
            print(f"\n[PARSING ERROR] {str(e)}")
            return False 