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
        """MMLU Pro 데이터셋을 다운로드하고 처리"""
        print("Downloading MMLU-Pro dataset...")
        
        try:
            # Hugging Face에서 데이터셋 로드
            dataset = load_dataset("TIGER-Lab/MMLU-Pro")
            
            # 각 분할 데이터를 처리
            for split in ['test', 'validation']:
                if split not in dataset:
                    print(f"Warning: {split} split not found in dataset")
                    continue
                
                split_data = dataset[split]
                
                # 카테고리별로 데이터 그룹화
                category_data = {}
                for item in split_data:
                    category = item['category'].lower()
                    if category not in category_data:
                        category_data[category] = []
                    
                    # split에 따라 answer에 저장할 값 결정
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
                
                # 각 카테고리의 데이터를 CSV로 저장
                for category, questions in category_data.items():
                    category_dir = os.path.join(self.base_dir, category)
                    os.makedirs(category_dir, exist_ok=True)
                    
                    # DataFrame 생성 및 CSV 저장
                    df = pd.DataFrame(questions)
                    print(df.head())  # 2단계
                    csv_path = os.path.join(category_dir, f"{split}.csv")
                    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_ALL)
                    print(f"Saved {category}'s {split} data with {len(questions)} questions")
                
        except Exception as e:
            print(f"Error downloading and processing dataset: {str(e)}")
            raise
    
    def get_subject_data(self, subject: str) -> Dict[str, List[Dict]]:
        """특정 과목의 데이터를 가져옴"""
        if subject not in self.subjects:
            raise ValueError(f"MMLU Pro 데이터셋에서 {subject} 과목을 찾을 수 없습니다")
        
        subject_dir = os.path.join(self.base_dir, subject)
        data = {}
        
        for split in ['test', 'validation']:
            csv_path = os.path.join(subject_dir, f"{split}.csv")
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Data file not found: {csv_path}")
            
            # CSV 파일에서 데이터 로드
            df = pd.read_csv(csv_path, dtype=str)
            data[split] = []
            
            for _, row in df.iterrows():
                # choices 문자열을 리스트로 변환
                choices = eval(row['choices']) if isinstance(row['choices'], str) else row['choices']
                data[split].append({
                    'question': row['question'],
                    'choices': choices,
                    'answer': row['answer']  # cot_content가 answer에 들어감
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
    dataset = MMLUProDataset()
    
    # 특정 과목 데이터 접근 예시
    try:
        subject_data = dataset.get_subject_data("computer science")
        for split in ['test', 'validation']:
            if split in subject_data:
                print(f"{split} 예제 수: {len(subject_data[split])}")
        
        # 첫 번째 예제 출력
        if subject_data['test']:
            first_example = subject_data['test'][0]
            print("\n첫 번째 테스트 예제:")
            print(f"질문: {first_example['question']}")
            print("선택지:")
            for i, choice in enumerate(first_example['choices'], 1):
                print(f"{i}. {choice}")
            print(f"정답: {first_example['answer']}")  # cot_content가 answer에 들어감
    except ValueError as e:
        print(e) 