import os
from typing import Dict, List
from datasets import load_dataset
import pandas as pd
from pathlib import Path

class HellaSwagDataset:
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.join(current_dir, 'hellaswag_data')
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
        """데이터가 존재하는지 확인"""
        for split in ['validation', 'train']:
            if not os.path.exists(os.path.join(self.base_dir, f"{split}.csv")):
                return False
        return True
    
    def _download_and_process_dataset(self) -> None:
        """HellaSwag 데이터셋을 다운로드하고 처리"""
        try:
            # Hugging Face에서 데이터셋 로드
            dataset = load_dataset("Rowan/hellaswag")
            
            # 각 분할 데이터를 저장
            for split in ['validation', 'train']:
                data = []
                for item in dataset[split]:
                    data.append({
                        'activity_label': item['activity_label'],
                        'context': f"{item['ctx_a']} {item['ctx_b']}",
                        'choices': item['endings'],
                        'answer': item['label']
                    })
                
                # 데이터를 CSV로 저장
                df = pd.DataFrame(data)
                df.to_csv(os.path.join(self.base_dir, f"{split}.csv"), index=False)
                print(f"Saved {split} data")
                    
        except Exception as e:
            print(f"Error processing dataset: {str(e)}")
    
    def get_split_data(self, split: str) -> List[Dict]:
        """특정 분할의 데이터를 가져옴"""
        if split not in ['validation', 'train']:
            raise ValueError(f"Invalid split: {split}")
        
        csv_path = os.path.join(self.base_dir, f"{split}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found: {csv_path}")
        
        # CSV 파일에서 데이터 로드
        df = pd.read_csv(csv_path)
        data = []
        
        for _, row in df.iterrows():
            # choices 문자열을 리스트로 변환
            choices = eval(row['choices']) if isinstance(row['choices'], str) else row['choices']
            data.append({
                'activity_label': row['activity_label'],
                'context': row['context'],
                'choices': choices,
                'answer': row['answer']
            })
        
        return data

    def get_all_data(self) -> Dict[str, List[Dict]]:
        """모든 분할의 데이터를 가져옴"""
        all_data = {}
        for split in ['validation', 'train']:
            try:
                split_data = self.get_split_data(split)
                all_data[split] = split_data
            except Exception as e:
                print(f"Error loading data for {split}: {str(e)}")
                continue
        return all_data

if __name__ == "__main__":
    # 사용 예시
    dataset = HellaSwagDataset()
    
    # 특정 분할 데이터 접근 예시
    try:
        validation_data = dataset.get_split_data("validation")
        print(f"검증 예제 수: {len(validation_data)}")
        
        # 첫 번째 예제 출력
        if validation_data:
            first_example = validation_data[0]
            print("\n첫 번째 검증 예제:")
            print(f"활동: {first_example['activity_label']}")
            print(f"문맥: {first_example['context']}")
            print("선택지:")
            for i, choice in enumerate(first_example['choices'], 1):
                print(f"{i}. {choice}")
            print(f"정답: {int(first_example['answer']) + 1}")  # 0-based를 1-based로 변환
    except ValueError as e:
        print(e) 