import os
from typing import Dict, List
from datasets import load_dataset
import pandas as pd
from pathlib import Path

class MBPPDataset:
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.join(current_dir, 'mbpp_data')
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
        """Check if dataset files exist"""
        for split in ['train', 'test', 'validation']:
            if not os.path.exists(os.path.join(self.base_dir, f"{split}.csv")):
                return False
        return True

    def _download_and_process_dataset(self) -> None:
        """Download and process MBPP dataset"""
        try:
            # Load dataset from Hugging Face
            dataset = load_dataset("mbpp")
            
            # Save each split data
            for split in ['train', 'test', 'validation']:
                data = []
                split_name = 'prompt' if split == 'validation' else split
                
                for item in dataset[split_name]:
                    data.append({
                        'task_id': item['task_id'],
                        'text': item['text'],  # Problem description
                        'code': item['code'],  # Correct code
                        'test_list': item['test_list'],  # Test case list
                        'test_setup_code': item['test_setup_code'] if 'test_setup_code' in item else '',
                        'challenge_test_list': item['challenge_test_list'] if 'challenge_test_list' in item else []
                    })
                
                # Save data as CSV
                df = pd.DataFrame(data)
                df.to_csv(os.path.join(self.base_dir, f"{split}.csv"), index=False)
                print(f"Saved {split} data")
                
        except Exception as e:
            print(f"Error processing MBPP dataset: {str(e)}")

    def get_split_data(self, split: str) -> List[Dict]:
        """Get data for specific split"""
        if split not in ['train', 'test', 'validation']:
            raise ValueError(f"Invalid split name: {split}")
        
        csv_path = os.path.join(self.base_dir, f"{split}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found: {csv_path}")
        
        # Load data from CSV file
        df = pd.read_csv(csv_path)
        data = []
        
        for _, row in df.iterrows():
            # Convert string representations of lists to actual lists
            test_list = eval(row['test_list']) if isinstance(row['test_list'], str) else row['test_list']
            challenge_test_list = eval(row['challenge_test_list']) if isinstance(row['challenge_test_list'], str) else row['challenge_test_list']
            
            data.append({
                'task_id': row['task_id'],
                'text': row['text'],
                'code': row['code'],
                'test_list': test_list,
                'test_setup_code': row['test_setup_code'],
                'challenge_test_list': challenge_test_list
            })
        
        return data

    def get_all_data(self) -> Dict[str, List[Dict]]:
        """Get data for all splits"""
        all_data = {}
        for split in ['train', 'test', 'validation']:
            try:
                split_data = self.get_split_data(split)
                all_data[split] = split_data
            except Exception as e:
                print(f"Error loading data for {split} split: {str(e)}")
                continue
        
        return all_data

    def get_validation_data(self) -> List[Dict]:
        """Get validation data (alias for get_split_data)"""
        return self.get_split_data('validation')

    def get_test_data(self) -> List[Dict]:
        """Get test data (alias for get_split_data)"""
        return self.get_split_data('test')

    def get_train_data(self) -> List[Dict]:
        """Get train data (alias for get_split_data)"""
        return self.get_split_data('train')

if __name__ == "__main__":
    # Example usage
    dataset = MBPPDataset()
    
    # Example access to specific split data
    try:
        test_data = dataset.get_split_data("test")
        print(f"Number of test examples: {len(test_data)}")
        
        # Print the first example
        if test_data:
            first_example = test_data[0]
            print("\nFirst test example:")
            print(f"Task ID: {first_example['task_id']}")
            print(f"Problem: {first_example['text']}")
            print(f"Correct code:\n{first_example['code']}")
            print("\nTest cases:")
            for test in first_example['test_list']:
                print(f"- {test}")
    except ValueError as e:
        print(e) 