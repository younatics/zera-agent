import os
from typing import Dict, List, Optional
from datasets import load_dataset
import pandas as pd
from pathlib import Path

class BBHDataset:
    CATEGORY_TO_SUBTASKS = {
        'Penguins': ['penguins_in_a_table'],
        'Geometry': ['geometric_shapes'],
        'Epistemic': ['web_of_lies'],
        'ObjectCount': ['object_counting'],
        'Temporal': ['temporal_sequences'],
        'CausalJudge': ['causal_judgement'],
    }

    def __init__(self, data_dir: str = "agent/dataset/bbh_data"):
        self.data_dir = os.path.abspath(data_dir)
        print(f"Dataset directory: {self.data_dir}")
        self._ensure_data_dir()
        if not self._check_dataset_exists():
            print("Dataset not found, downloading...")
            self._download_and_process_dataset()
        else:
            print("Dataset already exists, skipping download")

    def _ensure_data_dir(self):
        """Check if data directory exists and create if not"""
        os.makedirs(self.data_dir, exist_ok=True)
        print(f"Ensured data directory exists at: {self.data_dir}")

    def _check_dataset_exists(self) -> bool:
        """Check if dataset files exist (test.csv or subtask-specific csv)"""
        test_csv = os.path.join(self.data_dir, "test.csv")
        if os.path.exists(test_csv) and os.path.getsize(test_csv) > 0:
            print("Found merged test.csv file.")
            return True
        # Or True if at least one subtask-specific file exists
        for subtask in self.get_all_subtasks():
            file_path = os.path.join(self.data_dir, f"{subtask}.csv")
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                print(f"Found subtask file: {file_path}")
                return True
        print("No dataset files found.")
        return False

    def _download_and_process_dataset(self) -> None:
        """Download and process BBH dataset (save by subtask and category)"""
        try:
            subtasks = [
                'boolean_expressions', 'causal_judgement', 'date_understanding',
                'disambiguation_qa', 'dyck_languages', 'formal_fallacies',
                'geometric_shapes', 'hyperbaton', 'logical_deduction_five_objects',
                'logical_deduction_seven_objects', 'logical_deduction_three_objects',
                'movie_recommendation', 'multistep_arithmetic_two', 'navigate',
                'object_counting', 'penguins_in_a_table', 'reasoning_about_colored_objects',
                'ruin_names', 'salient_translation_error_detection', 'snarks',
                'sports_understanding', 'temporal_sequences', 'tracking_shuffled_objects_five_objects',
                'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_three_objects',
                'web_of_lies', 'word_sorting'
            ]
            all_data = []
            subtask_data_map = {}
            for subtask in subtasks:
                print(f"Loading {subtask}...")
                dataset = load_dataset("lukaemon/bbh", subtask, trust_remote_code=True)
                subtask_data = []
                for item in dataset['test']:
                    row = {
                        'task': subtask,
                        'input': item['input'],
                        'target': item['target'],
                    }
                    all_data.append(row)
                    subtask_data.append(row)
                # Save by subtask
                df_sub = pd.DataFrame(subtask_data)
                df_sub.to_csv(os.path.join(self.data_dir, f"{subtask}.csv"), index=False)
                subtask_data_map[subtask] = subtask_data
                print(f"Saved {subtask} with {len(subtask_data)} examples")
            # Save by category
            for category, subtask_list in self.CATEGORY_TO_SUBTASKS.items():
                cat_data = []
                for subtask in subtask_list:
                    cat_data.extend(subtask_data_map.get(subtask, []))
                df_cat = pd.DataFrame(cat_data)
                df_cat.to_csv(os.path.join(self.data_dir, f"{category}.csv"), index=False)
                print(f"Saved category {category} with {len(cat_data)} examples")
            # Also save overall test.csv (for compatibility)
            df = pd.DataFrame(all_data)
            df.to_csv(os.path.join(self.data_dir, "test.csv"), index=False)
            print(f"Saved merged test data with {len(all_data)} examples")
        except Exception as e:
            print(f"Error processing BBH dataset: {str(e)}")

    def get_split_data(self, split: str) -> List[Dict]:
        """Get data for a specific split"""
        if split not in ['test']:
            raise ValueError(f"Invalid split name: {split}")
        
        csv_path = os.path.join(self.data_dir, f"{split}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found: {csv_path}")
        
        # Load data from CSV file (specify data types)
        df = pd.read_csv(csv_path, dtype={
            'task': str,
            'input': str,
            'target': str
        })
        data = []
        
        for _, row in df.iterrows():
            data.append({
                'task': str(row['task']),
                'input': str(row['input']),
                'target': str(row['target'])
            })
        
        return data

    def get_all_data(self) -> Dict[str, List[Dict]]:
        """Get data for all splits"""
        all_data = {}
        for split in ['test']:
            try:
                split_data = self.get_split_data(split)
                all_data[split] = split_data
            except Exception as e:
                print(f"Error loading data for {split} split: {str(e)}")
                continue
        return all_data

    def get_subtask_data(self, subtask: str) -> List[Dict]:
        """Get data for a specific subtask"""
        csv_path = os.path.join(self.data_dir, f"{subtask}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Subtask data file not found: {csv_path}")
        df = pd.read_csv(csv_path, dtype={
            'task': str,
            'input': str,
            'target': str
        })
        data = []
        for _, row in df.iterrows():
            data.append({
                'task': str(row['task']),
                'input': str(row['input']),
                'target': str(row['target'])
            })
        return data

    def get_all_subtasks(self) -> List[str]:
        """Return all subtask names stored in the data directory (based on csv files)"""
        files = os.listdir(self.data_dir)
        subtasks = [f[:-4] for f in files if f.endswith('.csv') and f != 'test.csv']
        return subtasks

    def get_category_data(self, category: str) -> List[Dict]:
        """Get data for a specific category"""
        if category not in self.CATEGORY_TO_SUBTASKS:
            raise ValueError(f"Invalid category name: {category}")
        csv_path = os.path.join(self.data_dir, f"{category}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Category data file not found: {csv_path}")
        df = pd.read_csv(csv_path, dtype={
            'task': str,
            'input': str,
            'target': str
        })
        data = []
        for _, row in df.iterrows():
            data.append({
                'task': str(row['task']),
                'input': str(row['input']),
                'target': str(row['target'])
            })
        return data

    def get_all_categories(self) -> List[str]:
        """Return list of category names"""
        return list(self.CATEGORY_TO_SUBTASKS.keys())

if __name__ == "__main__":
    # Usage example
    dataset = BBHDataset()
    
    # Example of accessing specific split data
    try:
        test_data = dataset.get_split_data("test")
        print(f"Number of test examples: {len(test_data)}")
        
        # Print first example
        if test_data:
            first_example = test_data[0]
            print("\nFirst test example:")
            print(f"Task: {first_example['task']}")
            print(f"Input:\n{first_example['input']}")
            print(f"Target: {first_example['target']}")
    except ValueError as e:
        print(e)

    # Example: Access specific subtask data
    try:
        penguins_data = dataset.get_subtask_data("penguins_in_a_table")
        print(f"Number of Penguins examples: {len(penguins_data)}")
        if penguins_data:
            first = penguins_data[0]
            print("\nFirst Penguins example:")
            print(f"Task: {first['task']}")
            print(f"Input:\n{first['input']}")
            print(f"Target: {first['target']}")
    except Exception as e:
        print(e)

    # Example: Access data by category
    try:
        cat_data = dataset.get_category_data("Penguins")
        print(f"Number of Penguins category examples: {len(cat_data)}")
        if cat_data:
            first = cat_data[0]
            print("\nFirst Penguins category example:")
            print(f"Task: {first['task']}")
            print(f"Input:\n{first['input']}")
            print(f"Target: {first['target']}")
    except Exception as e:
        print(e) 