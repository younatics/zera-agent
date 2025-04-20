import os
import pandas as pd
from typing import Dict, Optional
from datasets import load_dataset
from pathlib import Path

class CNNDataset:
    """
    A class for handling CNN news articles dataset.
    Supports loading from Hugging Face datasets and saving as CSV files.
    """
    
    def __init__(self, data_dir: str = None, hf_dataset_name: str = "cnn_dailymail"):
        """
        Initialize the CNN dataset.
        
        Args:
            data_dir (str): Directory containing the CNN dataset files
            hf_dataset_name (str): Name of the Hugging Face dataset to load
        """
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), 'cnn_data')
        self.data_dir = data_dir
        
        # Create data directory if it doesn't exist
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        
        # Try to load from Hugging Face first
        try:
            self._load_from_huggingface(hf_dataset_name)
        except Exception as e:
            print(f"Failed to load from Hugging Face: {e}")
            # Fall back to local files
            self.load_data()
    
    def _load_from_huggingface(self, dataset_name: str):
        """
        Load dataset from Hugging Face and save as CSV files.
        
        Args:
            dataset_name (str): Name of the Hugging Face dataset
        """
        # Load the dataset with version 3.0.0
        dataset = load_dataset(dataset_name, '3.0.0')
        
        # Convert and save each split
        for split in ['train', 'validation', 'test']:
            if split in dataset:
                print(f"Processing {split} split...")
                data = []
                for article in dataset[split]:
                    data.append({
                        'input': article['article'],
                        'expected_answer': article['highlights']
                    })
                
                # Convert to DataFrame and save as CSV
                df = pd.DataFrame(data)
                csv_path = os.path.join(self.data_dir, f"{split}.csv")
                df.to_csv(csv_path, index=False)
                print(f"Saved {split} split to {csv_path}")
    
    def load_data(self, split: str = None) -> pd.DataFrame:
        """
        Load CNN articles from CSV files.
        
        Args:
            split (str, optional): Which split to load ('train', 'validation', 'test').
                                If None, loads all splits.
        
        Returns:
            pd.DataFrame: DataFrame containing the articles
        """
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        if split is not None:
            # Load specific split
            csv_path = os.path.join(self.data_dir, f"{split}.csv")
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Split file not found: {csv_path}")
            return pd.read_csv(csv_path)
        else:
            # Load all splits
            data = {}
            for split_name in ['train', 'validation', 'test']:
                csv_path = os.path.join(self.data_dir, f"{split_name}.csv")
                if os.path.exists(csv_path):
                    data[split_name] = pd.read_csv(csv_path)
            return data
    
    def get_split_sizes(self) -> Dict[str, int]:
        """
        Get the number of examples in each split.
        
        Returns:
            Dict[str, int]: Dictionary containing the size of each split
        """
        sizes = {}
        for split in ['train', 'validation', 'test']:
            csv_path = os.path.join(self.data_dir, f"{split}.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                sizes[split] = len(df)
        return sizes

    def download_dataset(self):
        """Download the dataset from Hugging Face and save it locally."""
        print("Downloading CNN dataset from Hugging Face...")
        try:
            self._load_from_huggingface("cnn_dailymail")
            sizes = self.get_split_sizes()
            print("\nDataset downloaded and processed successfully!")
            print("Split sizes:")
            for split, size in sizes.items():
                print(f"- {split}: {size:,} examples")
            print(f"Data saved in: {self.data_dir}")
        except Exception as e:
            print(f"Error downloading dataset: {e}")

if __name__ == "__main__":
    # 데이터 디렉토리 설정
    data_dir = os.path.join(os.path.dirname(__file__), 'cnn_data')
    
    # CNNDataset 인스턴스 생성 및 다운로드
    dataset = CNNDataset(data_dir=data_dir)
    dataset.download_dataset()
    
    # 데이터 로드 예시
    try:
        # 특정 split 로드
        train_data = dataset.load_data('train')
        print("\n첫 번째 학습 예제:")
        print(f"입력: {train_data['input'].iloc[0][:200]}...")
        print(f"기대 출력: {train_data['expected_answer'].iloc[0]}")
        
        # 모든 split 크기 확인
        sizes = dataset.get_split_sizes()
        print("\n각 split 크기:")
        for split, size in sizes.items():
            print(f"{split}: {size:,} examples")
    except Exception as e:
        print(f"Error loading data: {e}") 