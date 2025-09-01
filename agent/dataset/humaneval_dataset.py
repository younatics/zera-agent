import os
from typing import Dict, List
from datasets import load_dataset
import pandas as pd
from pathlib import Path
import json

class HumanEvalDataset:
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.join(current_dir, 'humaneval_data')
        self.base_dir = base_dir
        
        # Create base directory
        Path(self.base_dir).mkdir(parents=True, exist_ok=True)
        
        # Check if dataset is already downloaded
        if not self._check_dataset_exists():
            print("Dataset not found. Downloading and processing dataset...")
            try:
                self._download_and_process_dataset()
            except Exception as e:
                print(f"Error downloading and processing dataset: {e}")
                raise
    
    def _check_dataset_exists(self) -> bool:
        """Check if data exists"""
        return os.path.exists(os.path.join(self.base_dir, "test.csv"))
    
    def _download_and_process_dataset(self) -> None:
        """Download and process HumanEval dataset"""
        try:
            # Load dataset from Hugging Face
            dataset = load_dataset("openai_humaneval")
            
            data = []
            for item in dataset['test']:
                # Combine prompt and entry point
                full_prompt = f"{item['prompt']}\n\n{item['entry_point']}"
                
                data.append({
                    'task_id': item['task_id'],
                    'prompt': full_prompt,
                    'canonical_solution': item['canonical_solution'],
                    'test_cases': json.dumps(item['test']),  # Save as JSON string
                    'entry_point': item['entry_point']
                })
            
            # Save data as CSV
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(self.base_dir, "test.csv"), index=False)
            print("Saved test data")
                
        except Exception as e:
            print(f"Error processing dataset: {str(e)}")
    
    def get_split_data(self, split: str = "test") -> List[Dict]:
        """Get data (HumanEval only has test split)"""
        if split != "test":
            raise ValueError("HumanEval dataset only has test split")
        
        csv_path = os.path.join(self.base_dir, "test.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found: {csv_path}")
        
        # Load data from CSV file
        df = pd.read_csv(csv_path)
        data = []
        
        for _, row in df.iterrows():
            # test_cases is multi-line Python code, so use as string
            test_cases = row['test_cases'] if isinstance(row['test_cases'], str) else ""
            data.append({
                'task_id': row['task_id'],
                'prompt': row['prompt'],
                'canonical_solution': row['canonical_solution'],
                'test_cases': test_cases,
                'entry_point': row['entry_point']
            })
        
        return data

    def get_all_data(self) -> Dict[str, List[Dict]]:
        """Get all data"""
        try:
            test_data = self.get_split_data("test")
            return {"test": test_data}
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return {"test": []}

if __name__ == "__main__":
    # Usage example
    dataset = HumanEvalDataset()
    
    try:
        test_data = dataset.get_split_data("test")
        print(f"Number of test examples: {len(test_data)}")
        
        # Output first example
        if test_data:
            first_example = test_data[0]
            print("\nFirst test example:")
            print(f"Task ID: {first_example['task_id']}")
            print(f"Prompt:\n{first_example['prompt']}")
            print(f"\nAnswer:\n{first_example['canonical_solution']}")
            print(f"\nTest cases:\n{first_example['test_cases']}")
        # Diagnose empty test_cases samples
        empty_cases = [d['task_id'] for d in test_data if not d['test_cases'] or not str(d['test_cases']).strip()]
        print(f"Number of samples with empty test_cases: {len(empty_cases)}")
        print(f"Example task_ids: {empty_cases[:5]}")
    except ValueError as e:
        print(e) 