import os
from typing import Dict, List
from datasets import load_dataset
import pandas as pd
from pathlib import Path

class GSM8KDataset:
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.join(current_dir, 'gsm8k_data')
        self.base_dir = base_dir
        
        # 기본 디렉토리 생성
        Path(self.base_dir).mkdir(parents=True, exist_ok=True)
        
        # 데이터셋이 이미 다운로드되어 있는지 확인
        if not self._check_dataset_exists():
            print("데이터셋을 찾을 수 없습니다. 데이터셋을 다운로드하고 처리합니다...")
            try:
                self._download_and_process_dataset()
            except Exception as e:
                print(f"데이터셋 다운로드 및 처리 중 오류 발생: {e}")
                raise
    
    def _check_dataset_exists(self) -> bool:
        """데이터셋이 존재하는지 확인"""
        for split in ['train', 'test']:
            if not os.path.exists(os.path.join(self.base_dir, f"{split}.csv")):
                return False
        return True
    
    def _download_and_process_dataset(self) -> None:
        """GSM8K 데이터셋을 다운로드하고 처리"""
        try:
            # Hugging Face에서 데이터셋 로드
            dataset = load_dataset("gsm8k", "main")
            
            # 각 분할 데이터를 저장
            for split in ['train', 'test']:
                data = []
                for item in dataset[split]:
                    data.append({
                        'question': item['question'],
                        'answer': item['answer']
                    })
                
                # 데이터를 CSV로 저장
                df = pd.DataFrame(data)
                df.to_csv(os.path.join(self.base_dir, f"{split}.csv"), index=False)
                print(f"{split} 데이터 저장 완료")
                
        except Exception as e:
            print(f"데이터셋 처리 중 오류 발생: {str(e)}")
    
    def get_data(self, split: str = 'train') -> List[Dict]:
        """특정 분할의 데이터를 가져옴"""
        if split not in ['train', 'test']:
            raise ValueError("분할은 'train' 또는 'test'여야 합니다")
        
        csv_path = os.path.join(self.base_dir, f"{split}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {csv_path}")
        
        # CSV 파일에서 데이터 로드
        df = pd.read_csv(csv_path)
        data = []
        
        for _, row in df.iterrows():
            data.append({
                'question': row['question'],
                'answer': row['answer']
            })
        
        return data
    
    def load_data(self, split: str = 'train') -> List[Dict]:
        """Streamlit 앱과의 호환성을 위한 get_data의 래퍼 메서드"""
        if split == 'validation':
            split = 'test'  # GSM8K는 validation set이 없으므로 test set을 사용
        return self.get_data(split)

if __name__ == "__main__":
    # 사용 예시
    dataset = GSM8KDataset()
    
    # 데이터 접근 예시
    try:
        train_data = dataset.get_data('train')
        test_data = dataset.get_data('test')
        print(f"훈련 예제 수: {len(train_data)}")
        print(f"테스트 예제 수: {len(test_data)}")
        
        # 첫 번째 예제 출력
        if train_data:
            first_example = train_data[0]
            print("\n첫 번째 훈련 예제:")
            print(f"질문: {first_example['question']}")
            print(f"답변: {first_example['answer']}")
    except Exception as e:
        print(e) 