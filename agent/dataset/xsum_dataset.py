import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
from datasets import load_dataset


class XSumDataset:
    """Loader for the XSum extreme summarization dataset."""

    def __init__(self, base_dir: str = None):
        if base_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.join(current_dir, "xsum_data")
        self.base_dir = base_dir
        Path(self.base_dir).mkdir(parents=True, exist_ok=True)

        if not self._check_dataset_exists():
            print("Dataset not found. Downloading and processing dataset...")
            self._download_and_process_dataset()

    def _check_dataset_exists(self) -> bool:
        for split in ["train", "validation", "test"]:
            if not os.path.exists(os.path.join(self.base_dir, f"{split}.csv")):
                return False
        return True

    def _download_and_process_dataset(self) -> None:
        dataset = load_dataset("xsum")

        for split in ["train", "validation", "test"]:
            data = [
                {
                    "document": item["document"],
                    "summary": item["summary"],
                }
                for item in dataset[split]
            ]
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(self.base_dir, f"{split}.csv"), index=False)
            print(f"Saved {split} data")

    def get_split_data(self, split: str) -> List[Dict]:
        if split not in ["train", "validation", "test"]:
            raise ValueError("Split must be one of 'train', 'validation', or 'test'")

        csv_path = os.path.join(self.base_dir, f"{split}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        return [
            {
                "document": row["document"],
                "summary": row["summary"],
            }
            for _, row in df.iterrows()
        ]

    def get_validation_data(self) -> List[Dict]:
        return self.get_split_data("validation")

    def get_test_data(self) -> List[Dict]:
        return self.get_split_data("test")

    def get_train_data(self) -> List[Dict]:
        return self.get_split_data("train")
