import os
from typing import Dict, List, Optional
from datasets import load_dataset
import pandas as pd
from pathlib import Path

class SamsumDataset:
    def __init__(self, data_dir: str = "agent/dataset/samsum_data"):
        self.data_dir = os.path.abspath(data_dir)
        print(f"Dataset directory: {self.data_dir}")
        self._ensure_data_dir()
        if not self._check_dataset_exists():
            print("Dataset not found, downloading...")
            self._download_and_process_dataset()
        else:
            print("Dataset already exists, skipping download")

    def _ensure_data_dir(self):
        os.makedirs(self.data_dir, exist_ok=True)
        print(f"Ensured data directory exists at: {self.data_dir}")

    def _check_dataset_exists(self) -> bool:
        splits = ['train', 'validation', 'test']
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
        try:
            dataset = load_dataset("samsum")
            split_map = {"train": "train", "validation": "validation", "test": "test"}
            for split, split_name in split_map.items():
                data = []
                for item in dataset[split_name]:
                    data.append({
                        'dialogue': item['dialogue'],
                        'summary': item['summary'],
                        'id': item['id']
                    })
                df = pd.DataFrame(data)
                df.to_csv(os.path.join(self.data_dir, f"{split}.csv"), index=False)
                print(f"Saved {split} data")
        except Exception as e:
            print(f"Error processing Samsum dataset: {str(e)}")

    def get_split_data(self, split: str) -> List[Dict]:
        if split not in ['train', 'test', 'validation']:
            raise ValueError(f"Invalid split name: {split}")
        csv_path = os.path.join(self.data_dir, f"{split}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found: {csv_path}")
        df = pd.read_csv(csv_path, dtype={
            'dialogue': str,
            'summary': str,
            'id': str
        })
        data = []
        for _, row in df.iterrows():
            data.append({
                'dialogue': str(row['dialogue']),
                'summary': str(row['summary']),
                'id': str(row['id'])
            })
        return data

    def get_all_data(self) -> Dict[str, List[Dict]]:
        all_data = {}
        for split in ['train', 'test', 'validation']:
            try:
                split_data = self.get_split_data(split)
                all_data[split] = split_data
            except Exception as e:
                print(f"Error loading data for {split} split: {str(e)}")
                continue
        return all_data

if __name__ == "__main__":
    dataset = SamsumDataset()
    try:
        test_data = dataset.get_split_data("test")
        print(f"테스트 예제 수: {len(test_data)}")
        if test_data:
            first_example = test_data[0]
            print("\n첫 번째 테스트 예제:")
            print(f"ID: {first_example['id']}")
            print(f"대화:\n{first_example['dialogue'][:200]}...")
            print(f"요약문:\n{first_example['summary']}")
    except ValueError as e:
        print(e) 