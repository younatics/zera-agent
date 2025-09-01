import os
import pandas as pd
from typing import Dict, Optional, List
from datasets import load_dataset
from pathlib import Path
import shutil
import glob

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
        
        # Check if chunk directories exist
        all_chunks_exist = True
        for split in ['train', 'validation', 'test']:
            chunk_dir = os.path.join(self.data_dir, f"{split}_chunks")
            if not os.path.exists(chunk_dir):
                all_chunks_exist = False
                break
        
        if not all_chunks_exist:
            print("Chunk directories not found. Downloading and processing dataset...")
            try:
                self._load_from_huggingface(hf_dataset_name)
                self.split_and_save_chunks(chunk_size=200)
            except Exception as e:
                print(f"Error downloading and processing dataset: {e}")
                raise
        else:
            print("Using existing chunk files...")
    
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
    
    def split_and_save_chunks(self, chunk_size: int = 200):
        """
        Split and save the dataset into chunks of specified size (chunk_size).
        
        Args:
            chunk_size (int): Size of each chunk (default: 200)
        """
        # Process each split
        for split in ['train', 'validation', 'test']:
            # Original CSV file path
            csv_path = os.path.join(self.data_dir, f"{split}.csv")
            
            if not os.path.exists(csv_path):
                print(f"{split} split file does not exist: {csv_path}")
                continue
                
            # Directory to save chunks
            chunk_dir = os.path.join(self.data_dir, f"{split}_chunks")
            if os.path.exists(chunk_dir):
                shutil.rmtree(chunk_dir)  # Remove existing directory
            os.makedirs(chunk_dir)
            
            # Load data
            df = pd.read_csv(csv_path)
            total_examples = len(df)
            
            # Split into chunks
            for i in range(0, total_examples, chunk_size):
                chunk = df.iloc[i:i+chunk_size]
                chunk_path = os.path.join(chunk_dir, f"{split}_chunk_{i//chunk_size}.csv")
                chunk.to_csv(chunk_path, index=False)
                print(f"{split} split {i//chunk_size}th chunk saved: {len(chunk)} examples")
            
            print(f"\n{split} split processing completed:")
            print(f"- Total examples: {total_examples}")
            print(f"- Chunk size: {chunk_size}")
            print(f"- Total chunks: {(total_examples + chunk_size - 1) // chunk_size}")
            print(f"- Save location: {chunk_dir}")
            
            # Delete original CSV file
            os.remove(csv_path)
            print(f"Original {split}.csv file deletion completed")

    def load_data(self, split: str, chunk_index: int = None) -> List[Dict]:
        """
        Load data from chunk files.
        
        Args:
            split (str): One of 'train', 'validation', or 'test'
            chunk_index (int, optional): Index of the chunk to load. If None, loads all chunks.
        
        Returns:
            List[Dict]: Loaded data as a list of dictionaries
        """
        if split not in ['train', 'validation', 'test']:
            raise ValueError("split must be one of 'train', 'validation', or 'test'")
            
        chunk_dir = os.path.join(self.data_dir, f"{split}_chunks")
        if not os.path.exists(chunk_dir):
            raise FileNotFoundError(f"Chunk directory not found: {chunk_dir}")
            
        if chunk_index is not None:
            # Load specific chunk
            chunk_file = os.path.join(chunk_dir, f"{split}_chunk_{chunk_index}.csv")
            if not os.path.exists(chunk_file):
                raise FileNotFoundError(f"Chunk file not found: {chunk_file}")
            df = pd.read_csv(chunk_file)
            return df.to_dict('records')
        else:
            # Load all chunks
            all_data = []
            chunk_files = sorted(glob.glob(os.path.join(chunk_dir, f"{split}_chunk_*.csv")))
            for chunk_file in chunk_files:
                df = pd.read_csv(chunk_file)
                all_data.extend(df.to_dict('records'))
            return all_data

    def get_num_chunks(self, split: str) -> int:
        """
        Get the number of chunks for a given split.
        
        Args:
            split (str): One of 'train', 'validation', or 'test'
            
        Returns:
            int: Number of chunks
        """
        chunk_dir = os.path.join(self.data_dir, f"{split}_chunks")
        return len(glob.glob(os.path.join(chunk_dir, f"{split}_chunk_*.csv")))
    
    def load_all_data(self, split: str) -> List[Dict]:
        """
        Load all data from a given split.
        
        Args:
            split (str): One of 'train', 'validation', or 'test'
            
        Returns:
            List[Dict]: All data from the specified split as a list of dictionaries
        """
        return self.load_data(split, chunk_index=None)

if __name__ == "__main__":
    # Example usage
    dataset = CNNDataset()
    
    # Get number of chunks for each split
    for split in ['train', 'validation', 'test']:
        num_chunks = dataset.get_num_chunks(split)
        print(f"Number of chunks in {split} split: {num_chunks}")
        
        # Load first chunk as example
        if num_chunks > 0:
            chunk_data = dataset.load_data(split, chunk_index=0)
            print(f"First chunk of {split} split has {len(chunk_data)} examples") 