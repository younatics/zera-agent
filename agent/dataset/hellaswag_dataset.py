import os
from typing import Dict, List
from datasets import load_dataset
import pandas as pd
from pathlib import Path

class HellaSwagDataset:
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.join(current_dir, 'hellaswag_data')
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
        for split in ['validation', 'train']:
            if not os.path.exists(os.path.join(self.base_dir, f"{split}.csv")):
                return False
        return True
    
    def _download_and_process_dataset(self) -> None:
        """Download and process HellaSwag dataset"""
        try:
            # Load dataset from Hugging Face
            dataset = load_dataset("Rowan/hellaswag")
            
            # Save each split data
            for split in ['validation', 'train']:
                data = []
                for item in dataset[split]:
                    data.append({
                        'activity_label': item['activity_label'],
                        'context': f"{item['ctx_a']} {item['ctx_b']}",
                        'choices': item['endings'],
                        'answer': item['label']
                    })
                
                # Save data as CSV
                df = pd.DataFrame(data)
                df.to_csv(os.path.join(self.base_dir, f"{split}.csv"), index=False)
                print(f"Saved {split} data")
                    
        except Exception as e:
            print(f"Error processing dataset: {str(e)}")
    
    def get_split_data(self, split: str) -> List[Dict]:
        """Get data for specific split"""
        if split not in ['validation', 'train']:
            raise ValueError(f"Invalid split: {split}")
        
        csv_path = os.path.join(self.base_dir, f"{split}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found: {csv_path}")
        
        # Load data from CSV file
        df = pd.read_csv(csv_path)
        data = []
        
        for _, row in df.iterrows():
            # Convert choices string to list
            choices = eval(row['choices']) if isinstance(row['choices'], str) else row['choices']
            data.append({
                'activity_label': row['activity_label'],
                'context': row['context'],
                'choices': choices,
                'answer': row['answer']
            })
        
        return data

    def get_all_data(self) -> Dict[str, List[Dict]]:
        """Get data for all splits"""
        all_data = {}
        for split in ['validation', 'train']:
            try:
                split_data = self.get_split_data(split)
                all_data[split] = split_data
            except Exception as e:
                print(f"Error loading data for {split}: {str(e)}")
                continue
        return all_data

if __name__ == "__main__":
    # Usage example
    dataset = HellaSwagDataset()
    
    # Example of accessing specific split data
    try:
        validation_data = dataset.get_split_data("validation")
        print(f"Number of validation examples: {len(validation_data)}")
        
        # Output first example
        if validation_data:
            first_example = validation_data[0]
            print("\nFirst validation example:")
            print(f"Activity: {first_example['activity_label']}")
            print(f"Context: {first_example['context']}")
            print("Choices:")
            for i, choice in enumerate(first_example['choices'], 1):
                print(f"{i}. {choice}")
            print(f"Answer: {int(first_example['answer']) + 1}")  # Convert 0-based to 1-based
    except ValueError as e:
        print(e) 