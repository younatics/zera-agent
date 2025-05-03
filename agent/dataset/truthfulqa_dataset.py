import os
from typing import Dict, List, Optional
from datasets import load_dataset
import pandas as pd
from pathlib import Path

class TruthfulQADataset:
    def __init__(self, data_dir: str = "agent/dataset/truthfulqa_data"):
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
        splits = ['test']  # TruthfulQA는 test 데이터만 있음
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
        """TruthfulQA 데이터셋을 다운로드하고 처리"""
        try:
            # Hugging Face에서 데이터셋 로드
            dataset = load_dataset("truthful_qa", "generation", trust_remote_code=True)
            
            # 데이터를 CSV로 저장
            data = []
            for item in dataset['validation']:  # TruthfulQA는 validation을 test로 사용
                data.append({
                    'question': item['question'],  # 질문
                    'best_answer': item['best_answer'],  # 가장 좋은 정답
                    'correct_answers': item['correct_answers'],  # 모든 정답 목록
                    'incorrect_answers': item['incorrect_answers']  # 오답 목록
                })
            
            df = pd.DataFrame(data)
            # 리스트 형태의 데이터를 문자열로 변환하여 저장
            df['correct_answers'] = df['correct_answers'].apply(str)
            df['incorrect_answers'] = df['incorrect_answers'].apply(str)
            df.to_csv(os.path.join(self.data_dir, "test.csv"), index=False)
            print(f"Saved test data with {len(data)} examples")
                
        except Exception as e:
            print(f"Error processing TruthfulQA dataset: {str(e)}")

    def get_split_data(self, split: str) -> List[Dict]:
        """특정 분할의 데이터를 가져옴"""
        if split not in ['test']:
            raise ValueError(f"Invalid split name: {split}")
        
        csv_path = os.path.join(self.data_dir, f"{split}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found: {csv_path}")
        
        # CSV 파일에서 데이터 로드 (데이터 타입 명시)
        df = pd.read_csv(csv_path, dtype={
            'question': str,
            'best_answer': str,
            'correct_answers': str,
            'incorrect_answers': str
        })
        data = []
        
        for _, row in df.iterrows():
            # 문자열로 저장된 리스트를 다시 리스트로 변환
            correct_answers = eval(row['correct_answers'])
            incorrect_answers = eval(row['incorrect_answers'])
            
            data.append({
                'question': str(row['question']),
                'best_answer': str(row['best_answer']),
                'correct_answers': correct_answers,
                'incorrect_answers': incorrect_answers
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
    dataset = TruthfulQADataset()
    
    # 특정 분할 데이터 접근 예시
    try:
        test_data = dataset.get_split_data("test")
        print(f"테스트 예제 수: {len(test_data)}")
        
        # 첫 번째 예제 출력
        if test_data:
            first_example = test_data[0]
            print("\n첫 번째 테스트 예제:")
            print(f"질문: {first_example['question']}")
            print(f"가장 좋은 정답: {first_example['best_answer']}")
            print("정답 목록:")
            for i, answer in enumerate(first_example['correct_answers'], 1):
                print(f"{i}. {answer}")
            print("\n오답 목록:")
            for i, answer in enumerate(first_example['incorrect_answers'], 1):
                print(f"{i}. {answer}")
    except ValueError as e:
        print(e) 