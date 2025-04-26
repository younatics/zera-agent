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
        
        # 기본 디렉토리 생성
        Path(self.base_dir).mkdir(parents=True, exist_ok=True)
        
        # 데이터셋이 이미 다운로드되어 있는지 확인
        if not self._check_dataset_exists():
            print("Dataset not found. Downloading and processing dataset...")
            try:
                self._download_and_process_dataset()
            except Exception as e:
                print(f"Error downloading and processing dataset: {e}")
                raise
    
    def _check_dataset_exists(self) -> bool:
        """모든 과목의 데이터가 존재하는지 확인"""
        for subject in self.subjects:
            subject_dir = os.path.join(self.base_dir, subject)
            if not os.path.exists(subject_dir):
                return False
            for split in ['test', 'validation']:
                if not os.path.exists(os.path.join(subject_dir, f"{split}.csv")):
                    return False
        return True
    
    def _download_and_process_dataset(self) -> None:
        """모든 과목의 MMLU 데이터셋을 다운로드하고 처리"""
        for subject in self.subjects:
            print(f"Processing {subject} subject...")
            subject_dir = os.path.join(self.base_dir, subject)
            os.makedirs(subject_dir, exist_ok=True)
            
            try:
                # Hugging Face에서 데이터셋 로드
                dataset = load_dataset("cais/mmlu", subject)
                
                # 각 분할 데이터를 저장
                for split in ['test', 'validation']:
                    data = []
                    for item in dataset[split]:
                        data.append({
                            'question': item['question'],
                            'choices': item['choices'],
                            'answer': item['answer']
                        })
                    
                    # 데이터를 CSV로 저장
                    df = pd.DataFrame(data)
                    df.to_csv(os.path.join(subject_dir, f"{split}.csv"), index=False)
                    print(f"Saved {subject}'s {split} data")
                    
            except Exception as e:
                print(f"Error processing {subject} subject: {str(e)}")
    
    def get_subject_data(self, subject: str) -> Dict[str, List[Dict]]:
        """특정 과목의 데이터를 가져옴"""
        if subject not in self.subjects:
            raise ValueError(f"MMLU 데이터셋에서 {subject} 과목을 찾을 수 없습니다")
        
        subject_dir = os.path.join(self.base_dir, subject)
        data = {}
        
        for split in ['test', 'validation']:
            csv_path = os.path.join(subject_dir, f"{split}.csv")
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Data file not found: {csv_path}")
            
            # CSV 파일에서 데이터 로드
            df = pd.read_csv(csv_path)
            data[split] = []
            
            for _, row in df.iterrows():
                # choices 문자열을 리스트로 변환
                choices = eval(row['choices']) if isinstance(row['choices'], str) else row['choices']
                data[split].append({
                    'question': row['question'],
                    'choices': choices,
                    'answer': row['answer']
                })
        
        return data

    def get_all_subjects_data(self) -> Dict[str, Dict[str, List[Dict]]]:
        """모든 과목의 데이터를 가져옴"""
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
    # 사용 예시
    dataset = MMLUDataset()
    
    # 특정 과목 데이터 접근 예시
    try:
        subject_data = dataset.get_subject_data("high_school_physics")
        print(f"테스트 예제 수: {len(subject_data['test'])}")
        print(f"검증 예제 수: {len(subject_data['validation'])}")
        
        # 첫 번째 예제 출력
        if subject_data['test']:
            first_example = subject_data['test'][0]
            print("\n첫 번째 테스트 예제:")
            print(f"질문: {first_example['question']}")
            print("선택지:")
            for i, choice in enumerate(first_example['choices'], 1):
                print(f"{i}. {choice}")
            print(f"정답: {first_example['answer'] + 1}")  # 0-based를 1-based로 변환
    except ValueError as e:
        print(e) 