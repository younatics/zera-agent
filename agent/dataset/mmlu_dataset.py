import os
from typing import Dict, List
from datasets import load_dataset
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import glob

class MMLUDataset:
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.join(current_dir, 'mmlu_data')
        self.base_dir = base_dir
        self.subjects = [
            "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
            "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
            "college_medicine", "college_physics", "computer_security", "conceptual_physics",
            "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
            "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
            "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
            "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
            "high_school_physics", "high_school_psychology", "high_school_statistics", "high_school_us_history",
            "high_school_world_history", "human_aging", "human_sexuality", "international_law",
            "jurisprudence", "logical_fallacies", "machine_learning", "management", "marketing",
            "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
            "philosophy", "prehistory", "professional_accounting", "professional_law", "professional_medicine",
            "professional_psychology", "public_relations", "security_studies", "sociology", "us_foreign_policy",
            "virology", "world_religions"
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
        """Download and process MMLU dataset for all subjects"""
        for subject in self.subjects:
            print(f"Processing {subject} subject...")
            subject_dir = os.path.join(self.base_dir, subject)
            os.makedirs(subject_dir, exist_ok=True)
            
            try:
                # Load dataset from Hugging Face
                dataset = load_dataset("cais/mmlu", subject)
                
                # Save each split data
                for split in ['test', 'validation']:
                    data = []
                    for item in dataset[split]:
                        data.append({
                            'question': item['question'],
                            'choices': item['choices'],
                            'answer': item['answer']
                        })
                    
                    # Save data as CSV
                    df = pd.DataFrame(data)
                    df.to_csv(os.path.join(subject_dir, f"{split}.csv"), index=False)
                    print(f"Saved {subject}'s {split} data")
                    
            except Exception as e:
                print(f"Error processing {subject} subject: {str(e)}")
    
    def get_subject_data(self, subject: str) -> Dict[str, List[Dict]]:
        """Get data for a specific subject"""
        if subject not in self.subjects:
            raise ValueError(f"Subject {subject} not found in MMLU dataset")
        
        subject_dir = os.path.join(self.base_dir, subject)
        data = {}
        
        for split in ['test', 'validation']:
            csv_path = os.path.join(subject_dir, f"{split}.csv")
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Data file not found: {csv_path}")
            
            # Load data from CSV file
            df = pd.read_csv(csv_path)
            data[split] = []
            
            for _, row in df.iterrows():
                # Convert choices string to list
                choices = eval(row['choices']) if isinstance(row['choices'], str) else row['choices']
                data[split].append({
                    'question': row['question'],
                    'choices': choices,
                    'answer': row['answer']
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
    dataset = MMLUDataset()
    
    # Example of accessing specific subject data
    try:
        subject_data = dataset.get_subject_data("high_school_physics")
        print(f"Number of test examples: {len(subject_data['test'])}")
        print(f"Number of validation examples: {len(subject_data['validation'])}")
        
        # Output first example
        if subject_data['test']:
            first_example = subject_data['test'][0]
            print("\nFirst test example:")
            print(f"Question: {first_example['question']}")
            print("Choices:")
            for i, choice in enumerate(first_example['choices'], 1):
                print(f"{i}. {choice}")
            print(f"Answer: {first_example['answer'] + 1}")  # Convert 0-based to 1-based
    except ValueError as e:
        print(e) 