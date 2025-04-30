import os
from typing import Dict, List, Optional
from datasets import load_dataset
import pandas as pd
from pathlib import Path

class BBHDataset:
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
        """데이터 디렉토리가 존재하는지 확인하고 없으면 생성"""
        os.makedirs(self.data_dir, exist_ok=True)
        print(f"Ensured data directory exists at: {self.data_dir}")

    def _check_dataset_exists(self) -> bool:
        """데이터셋 파일이 존재하는지 확인"""
        splits = ['test']  # BBH는 test 데이터만 있음
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
        """BBH 데이터셋을 다운로드하고 처리"""
        try:
            # 모든 서브태스크 목록
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
            
            # 각 서브태스크의 데이터를 로드
            for subtask in subtasks:
                print(f"Loading {subtask}...")
                dataset = load_dataset("lukaemon/bbh", subtask, trust_remote_code=True)
                
                for item in dataset['test']:
                    all_data.append({
                        'task': subtask,  # 태스크 이름
                        'input': item['input'],  # 입력 텍스트
                        'target': item['target'],  # 정답
                    })
            
            # 모든 데이터를 CSV로 저장
            df = pd.DataFrame(all_data)
            df.to_csv(os.path.join(self.data_dir, "test.csv"), index=False)
            print(f"Saved test data with {len(all_data)} examples")
                
        except Exception as e:
            print(f"Error processing BBH dataset: {str(e)}")

    def get_split_data(self, split: str) -> List[Dict]:
        """특정 분할의 데이터를 가져옴"""
        if split not in ['test']:
            raise ValueError(f"Invalid split name: {split}")
        
        csv_path = os.path.join(self.data_dir, f"{split}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found: {csv_path}")
        
        # CSV 파일에서 데이터 로드 (데이터 타입 명시)
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
        """모든 분할의 데이터를 가져옴"""
        all_data = {}
        for split in ['test']:
            try:
                split_data = self.get_split_data(split)
                all_data[split] = split_data
            except Exception as e:
                print(f"Error loading data for {split} split: {str(e)}")
                continue
        return all_data

if __name__ == "__main__":
    # 사용 예시
    dataset = BBHDataset()
    
    # 특정 분할 데이터 접근 예시
    try:
        test_data = dataset.get_split_data("test")
        print(f"테스트 예제 수: {len(test_data)}")
        
        # 첫 번째 예제 출력
        if test_data:
            first_example = test_data[0]
            print("\n첫 번째 테스트 예제:")
            print(f"Task: {first_example['task']}")
            print(f"Input:\n{first_example['input']}")
            print(f"Target: {first_example['target']}")
    except ValueError as e:
        print(e) 