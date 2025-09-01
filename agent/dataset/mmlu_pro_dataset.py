import os
from typing import Dict, List
from datasets import load_dataset
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import glob
import json
import time
import csv

class MMLUProDataset:
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.join(current_dir, 'mmlu_pro_data')
        self.base_dir = base_dir
        self.subjects = [
            "computer science", "economics", "engineering", "history", "law",
            "other", "philosophy", "psychology", "biology", "business",
            "chemistry", "health", "math", "physics"
        ]
        
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
        """Check if data exists for all subjects"""
        for subject in self.subjects:
            subject_dir = os.path.join(self.base_dir, subject)
            if not os.path.exists(subject_dir):
                return False
            for split in ['test', 'validation']:
                if not os.path.exists(os.path.join(subject_dir, f"{split}.csv")):
                    return False
        return True
    
    def _download_and_process_dataset(self) -> None:
        """Download and process MMLU Pro dataset"""
        print("Downloading MMLU-Pro dataset...")
        
        try:
            # Load dataset from Hugging Face
            dataset = load_dataset("TIGER-Lab/MMLU-Pro")
            
            # Process each split data
            for split in ['test', 'validation']:
                if split not in dataset:
                    print(f"Warning: {split} split not found in dataset")
                    continue
                
                split_data = dataset[split]
                
                # Group data by category
                category_data = {}
                for item in split_data:
                    category = item['category'].lower()
                    if category not in category_data:
                        category_data[category] = []
                    
                    # Determine value to store in answer based on split
                    if split == 'test':
                        answer_value = item['answer']
                    else:  # validation
                        answer_value = item['cot_content']
                    question_data = {
                        'question': item['question'],
                        'choices': item['options'],
                        'answer': answer_value
                    }
                    category_data[category].append(question_data)
                
                # Save each category's data as CSV
                for category, questions in category_data.items():
                    category_dir = os.path.join(self.base_dir, category)
                    os.makedirs(category_dir, exist_ok=True)
                    
                    # Create DataFrame and save as CSV
                    df = pd.DataFrame(questions)
                    print(df.head())  # Step 2
                    csv_path = os.path.join(category_dir, f"{split}.csv")
                    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_ALL)
                    print(f"Saved {category}'s {split} data with {len(questions)} questions")
                
        except Exception as e:
            print(f"Error downloading and processing dataset: {str(e)}")
            raise
    
    def get_subject_data(self, subject: str) -> Dict[str, List[Dict]]:
        """Get data for a specific subject"""
        if subject not in self.subjects:
            raise ValueError(f"Subject {subject} not found in MMLU Pro dataset")
        
        subject_dir = os.path.join(self.base_dir, subject)
        data = {}
        
        for split in ['test', 'validation']:
            csv_path = os.path.join(subject_dir, f"{split}.csv")
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Data file not found: {csv_path}")
            
            # Load data from CSV file
            df = pd.read_csv(csv_path, dtype=str)
            data[split] = []
            
            for _, row in df.iterrows():
                # Convert choices string to list
                choices = eval(row['choices']) if isinstance(row['choices'], str) else row['choices']
                data[split].append({
                    'question': row['question'],
                    'choices': choices,
                    'answer': row['answer']  # cot_content goes into answer
                })
        
        return data

    def get_all_subjects_data(self) -> Dict[str, Dict[str, List[Dict]]]:
        """Get data for all subjects"""
        all_data = {}
        for subject in self.subjects:
            try:
                subject_data = self.get_subject_data(subject)
                all_data[subject] = subject_data
            except Exception as e:
                print(f"Error loading data for {subject}: {str(e)}")
                continue
        return all_data

if __name__ == "__main__":
    # Usage example
    dataset = MMLUProDataset()
    
    # Example of accessing specific subject data
    try:
        subject_data = dataset.get_subject_data("computer science")
        for split in ['test', 'validation']:
            if split in subject_data:
                print(f"Number of {split} examples: {len(subject_data[split])}")
        
        # Print first example
        if subject_data['test']:
            first_example = subject_data['test'][0]
            print("\nFirst test example:")
            print(f"Question: {first_example['question']}")
            print("Choices:")
            for i, choice in enumerate(first_example['choices'], 1):
                print(f"{i}. {choice}")
            print(f"Answer: {first_example['answer']}")  # cot_content goes into answer
    except ValueError as e:
        print(e) 