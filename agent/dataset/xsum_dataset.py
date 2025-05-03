import os
from typing import Dict, List, Optional
from datasets import load_dataset
import pandas as pd
from pathlib import Path

class XSumDataset:
    def __init__(self, data_dir: str = "agent/dataset/xsum_data"):
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
        splits = ['test', 'validation']  # train 제외
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
        """XSum 데이터셋을 다운로드하고 처리"""
        try:
            # Hugging Face에서 데이터셋 로드
            dataset = load_dataset("xsum", trust_remote_code=True)
            
            # 각 분할 데이터를 저장 (train 제외)
            for split in ['test', 'validation']:
                data = []
                split_name = 'validation' if split == 'validation' else split
                
                for item in dataset[split_name]:
                    data.append({
                        'document': item['document'],  # 원본 뉴스 기사
                        'summary': item['summary'],    # 요약문
                        'id': item['id']              # 기사 ID
                    })
                
                # 데이터를 CSV로 저장
                df = pd.DataFrame(data)
                df.to_csv(os.path.join(self.data_dir, f"{split}.csv"), index=False)
                print(f"Saved {split} data")
                
        except Exception as e:
            print(f"Error processing XSum dataset: {str(e)}")

    def get_split_data(self, split: str) -> List[Dict]:
        """특정 분할의 데이터를 가져옴"""
        if split not in ['test', 'validation']:  # train 제외
            raise ValueError(f"Invalid split name: {split}")
        
        csv_path = os.path.join(self.data_dir, f"{split}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found: {csv_path}")
        
        # CSV 파일에서 데이터 로드 (데이터 타입 명시)
        df = pd.read_csv(csv_path, dtype={
            'document': str,
            'summary': str,
            'id': str
        })
        data = []
        
        for _, row in df.iterrows():
            data.append({
                'document': str(row['document']),
                'summary': str(row['summary']),
                'id': str(row['id'])
            })
        
        return data

    def get_all_data(self) -> Dict[str, List[Dict]]:
        """모든 분할의 데이터를 가져옴"""
        all_data = {}
        for split in ['train', 'test', 'validation']:
            try:
                split_data = self.get_split_data(split)
                all_data[split] = split_data
            except Exception as e:
                print(f"Error loading data for {split} split: {str(e)}")
                continue
        return all_data

if __name__ == "__main__":
    # 사용 예시
    dataset = XSumDataset()
    
    # 특정 분할 데이터 접근 예시
    try:
        test_data = dataset.get_split_data("test")
        print(f"테스트 예제 수: {len(test_data)}")
        
        # 첫 번째 예제 출력
        if test_data:
            first_example = test_data[0]
            print("\n첫 번째 테스트 예제:")
            print(f"ID: {first_example['id']}")
            print(f"원본 기사:\n{first_example['document'][:200]}...")  # 처음 200자만 출력
            print(f"요약문:\n{first_example['summary']}")
    except ValueError as e:
        print(e) 