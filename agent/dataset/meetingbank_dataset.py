import os
from typing import Dict, List
from datasets import load_dataset
import pandas as pd
from pathlib import Path

class MeetingBankDataset:
    def __init__(self, data_dir: str = "agent/dataset/meetingbank_data"):
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
        splits = ['validation', 'test']
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
            dataset = load_dataset("huuuyeah/meetingbank")
            split_map = {"validation": "validation", "test": "test"}
            for split, split_name in split_map.items():
                data = []
                for item in dataset[split_name]:
                    data.append({
                        'id': item['id'],
                        'uid': item['uid'],
                        'summary': item['summary'],
                        'transcript': item['transcript']
                    })
                df = pd.DataFrame(data)
                df.to_csv(os.path.join(self.data_dir, f"{split}.csv"), index=False)
                print(f"Saved {split} data")
        except Exception as e:
            print(f"Error processing MeetingBank dataset: {str(e)}")

    def get_split_data(self, split: str) -> List[Dict]:
        if split not in ['validation', 'test']:
            raise ValueError(f"Invalid split name: {split}")
        csv_path = os.path.join(self.data_dir, f"{split}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found: {csv_path}")
        df = pd.read_csv(csv_path, dtype={
            'id': int,
            'uid': str,
            'summary': str,
            'transcript': str
        })
        data = []
        for _, row in df.iterrows():
            data.append({
                'id': int(row['id']),
                'uid': str(row['uid']),
                'summary': str(row['summary']),
                'transcript': str(row['transcript'])
            })
        return data

    def get_all_data(self) -> Dict[str, List[Dict]]:
        all_data = {}
        for split in ['validation', 'test']:
            try:
                split_data = self.get_split_data(split)
                all_data[split] = split_data
            except Exception as e:
                print(f"Error loading data for {split} split: {str(e)}")
                continue
        return all_data

if __name__ == "__main__":
    dataset = MeetingBankDataset()
    try:
        test_data = dataset.get_split_data("test")
        print(f"테스트 예제 수: {len(test_data)}")
        if test_data:
            first_example = test_data[0]
            print("\n첫 번째 테스트 예제:")
            print(f"ID: {first_example['id']}")
            print(f"UID: {first_example['uid']}")
            print(f"Transcript (앞 200자):\n{first_example['transcript'][:200]}...")
            print(f"Summary:\n{first_example['summary']}")
    except ValueError as e:
        print(e) 