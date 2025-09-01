import os
from typing import Dict, List, Optional
from datasets import load_dataset
import pandas as pd
from pathlib import Path

class TruthfulQADataset:
    def __init__(self, data_dir: str = "agent/dataset/truthfulqa_data"):
        self.data_dir = os.path.abspath(data_dir)
        print(f"Dataset directory: {self.data_dir}")
        self._ensure_data_dir()
        if not self._check_dataset_exists():
            print("Dataset not found, downloading...")
            self._download_and_process_dataset()
        else:
            print("Dataset already exists, skipping download")

    def _ensure_data_dir(self):
        """Check if data directory exists and create if not"""
        os.makedirs(self.data_dir, exist_ok=True)
        print(f"Ensured data directory exists at: {self.data_dir}")

    def _check_dataset_exists(self) -> bool:
        """Check if dataset file exists"""
        splits = ['test']  # TruthfulQA only has test data
        for split in splits:
            file_path = os.path.join(self.data_dir, f"{split}.csv")
            print(f"Checking file: {file_path}")
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return False
            if os.path.getsize(file_path) == 0:
                print(f"File is empty: {file_path}")
                return False
        print("All dataset files exist and are not empty")
        return True

    def _download_and_process_dataset(self) -> None:
        """Download and process TruthfulQA dataset"""
        try:
            # Load dataset from Hugging Face
            dataset = load_dataset("truthful_qa", "generation", trust_remote_code=True)
            
            # Save data as CSV
            data = []
            for item in dataset['validation']:  # TruthfulQA uses validation as test
                data.append({
                    'question': item['question'],  # Question
                    'best_answer': item['best_answer'],  # Best answer
                    'correct_answers': item['correct_answers'],  # All correct answers list
                    'incorrect_answers': item['incorrect_answers']  # Incorrect answers list
                })
            
            df = pd.DataFrame(data)
            # Convert list data to strings for storage
            df['correct_answers'] = df['correct_answers'].apply(str)
            df['incorrect_answers'] = df['incorrect_answers'].apply(str)
            df.to_csv(os.path.join(self.data_dir, "test.csv"), index=False)
            print(f"Saved test data with {len(data)} examples")
                
        except Exception as e:
            print(f"Error processing TruthfulQA dataset: {str(e)}")

    def get_split_data(self, split: str) -> List[Dict]:
        """Get data for specific split"""
        if split not in ['test']:
            raise ValueError(f"Invalid split name: {split}")
        
        csv_path = os.path.join(self.data_dir, f"{split}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found: {csv_path}")
        
        # Load data from CSV file (with data type specification)
        df = pd.read_csv(csv_path, dtype={
            'question': str,
            'best_answer': str,
            'correct_answers': str,
            'incorrect_answers': str
        })
        data = []
        
        for _, row in df.iterrows():
            # Convert string-stored lists back to lists
            correct_answers = eval(row['correct_answers'])
            incorrect_answers = eval(row['incorrect_answers'])
            
            data.append({
                'question': str(row['question']),
                'best_answer': str(row['best_answer']),
                'correct_answers': correct_answers,
                'incorrect_answers': incorrect_answers
            })
        
        return data

    def get_all_data(self) -> Dict[str, List[Dict]]:
        """Get data for all splits"""
        all_data = {}
        for split in ['test']:
            try:
                split_data = self.get_split_data(split)
                all_data[split] = split_data
            except Exception as e:
                print(f"Error loading data for {split} split: {str(e)}")
                continue
        return all_data

if __name__ == "__main__":
    # Example usage
    dataset = TruthfulQADataset()
    
    # Example access to specific split data
    try:
        test_data = dataset.get_split_data("test")
        print(f"Number of test examples: {len(test_data)}")
        
        # Print the first example
        if test_data:
            first_example = test_data[0]
            print("\nFirst test example:")
            print(f"Question: {first_example['question']}")
            print(f"Best answer: {first_example['best_answer']}")
            print("Correct answers:")
            for i, answer in enumerate(first_example['correct_answers'], 1):
                print(f"{i}. {answer}")
            print("\nIncorrect answers:")
            for i, answer in enumerate(first_example['incorrect_answers'], 1):
                print(f"{i}. {answer}")
    except ValueError as e:
        print(e) 