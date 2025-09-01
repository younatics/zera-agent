import os
from typing import Dict, List
from datasets import load_dataset
import pandas as pd
from pathlib import Path

class GSM8KDataset:
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.join(current_dir, 'gsm8k_data')
        self.base_dir = base_dir
        
        # Create base directory
        Path(self.base_dir).mkdir(parents=True, exist_ok=True)
        
        # Check if dataset is already downloaded
        if not self._check_dataset_exists():
            print("Dataset not found. Downloading and processing dataset...")
            try:
                self._download_and_process_dataset()
            except Exception as e:
                print(f"Error occurred during dataset download and processing: {e}")
                raise
    
    def _check_dataset_exists(self) -> bool:
        """Check if dataset exists"""
        for split in ['train', 'test']:
            if not os.path.exists(os.path.join(self.base_dir, f"{split}.csv")):
                return False
        return True
    
    def _download_and_process_dataset(self) -> None:
        """Download and process GSM8K dataset"""
        try:
            # Load dataset from Hugging Face
            dataset = load_dataset("gsm8k", "main")
            
            # Save each split data
            for split in ['train', 'test']:
                data = []
                for item in dataset[split]:
                    data.append({
                        'question': item['question'],
                        'answer': item['answer']
                    })
                
                # Save data as CSV
                df = pd.DataFrame(data)
                df.to_csv(os.path.join(self.base_dir, f"{split}.csv"), index=False)
                print(f"{split} data saved successfully")
                
        except Exception as e:
            print(f"Error occurred during dataset processing: {str(e)}")
    
    def get_data(self, split: str = 'train') -> List[Dict]:
        """Get data for a specific split"""
        if split not in ['train', 'test']:
            raise ValueError("Split must be 'train' or 'test'")
        
        csv_path = os.path.join(self.base_dir, f"{split}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found: {csv_path}")
        
        # Load data from CSV file
        df = pd.read_csv(csv_path)
        data = []
        
        for _, row in df.iterrows():
            data.append({
                'question': row['question'],
                'answer': row['answer']
            })
        
        return data
    
    def load_data(self, split: str = 'train') -> List[Dict]:
        """Wrapper method for get_data for compatibility with Streamlit app"""
        if split == 'validation':
            split = 'test'  # GSM8K has no validation set, so use test set
        return self.get_data(split)

if __name__ == "__main__":
    # Usage example
    dataset = GSM8KDataset()
    
    # Example of data access
    try:
        train_data = dataset.get_data('train')
        test_data = dataset.get_data('test')
        print(f"Number of training examples: {len(train_data)}")
        print(f"Number of test examples: {len(test_data)}")
        
        # Output first example
        if train_data:
            first_example = train_data[0]
            print("\nFirst training example:")
            print(f"Question: {first_example['question']}")
            print(f"Answer: {first_example['answer']}")
    except Exception as e:
        print(e) 