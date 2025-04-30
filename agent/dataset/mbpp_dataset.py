import os
from typing import Dict, List
from datasets import load_dataset
import pandas as pd
from pathlib import Path

class MBPPDataset:
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.join(current_dir, 'mbpp_data')
        self.base_dir = base_dir
        
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
        """데이터셋 파일들이 존재하는지 확인"""
        for split in ['train', 'test', 'validation']:
            if not os.path.exists(os.path.join(self.base_dir, f"{split}.csv")):
                return False
        return True

    def _download_and_process_dataset(self) -> None:
        """MBPP 데이터셋을 다운로드하고 처리"""
        try:
            # Hugging Face에서 데이터셋 로드
            dataset = load_dataset("mbpp")
            
            # 각 분할 데이터를 저장
            for split in ['train', 'test', 'validation']:
                data = []
                split_name = 'prompt' if split == 'validation' else split
                
                for item in dataset[split_name]:
                    data.append({
                        'task_id': item['task_id'],
                        'text': item['text'],  # 문제 설명
                        'code': item['code'],  # 정답 코드
                        'test_list': item['test_list'],  # 테스트 케이스 목록
                        'test_setup_code': item['test_setup_code'] if 'test_setup_code' in item else '',
                        'challenge_test_list': item['challenge_test_list'] if 'challenge_test_list' in item else []
                    })
                
                # 데이터를 CSV로 저장
                df = pd.DataFrame(data)
                df.to_csv(os.path.join(self.base_dir, f"{split}.csv"), index=False)
                print(f"Saved {split} data")
                
        except Exception as e:
            print(f"Error processing MBPP dataset: {str(e)}")

    def get_split_data(self, split: str) -> List[Dict]:
        """특정 분할의 데이터를 가져옴"""
        if split not in ['train', 'test', 'validation']:
            raise ValueError(f"Invalid split name: {split}")
        
        csv_path = os.path.join(self.base_dir, f"{split}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found: {csv_path}")
        
        # CSV 파일에서 데이터 로드
        df = pd.read_csv(csv_path)
        data = []
        
        for _, row in df.iterrows():
            # 리스트 형태의 문자열을 실제 리스트로 변환
            test_list = eval(row['test_list']) if isinstance(row['test_list'], str) else row['test_list']
            challenge_test_list = eval(row['challenge_test_list']) if isinstance(row['challenge_test_list'], str) else row['challenge_test_list']
            
            data.append({
                'task_id': row['task_id'],
                'text': row['text'],
                'code': row['code'],
                'test_list': test_list,
                'test_setup_code': row['test_setup_code'],
                'challenge_test_list': challenge_test_list
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
    dataset = MBPPDataset()
    
    # 특정 분할 데이터 접근 예시
    try:
        test_data = dataset.get_split_data("test")
        print(f"테스트 예제 수: {len(test_data)}")
        
        # 첫 번째 예제 출력
        if test_data:
            first_example = test_data[0]
            print("\n첫 번째 테스트 예제:")
            print(f"Task ID: {first_example['task_id']}")
            print(f"문제: {first_example['text']}")
            print(f"정답 코드:\n{first_example['code']}")
            print("\n테스트 케이스:")
            for test in first_example['test_list']:
                print(f"- {test}")
    except ValueError as e:
        print(e) 