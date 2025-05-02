import os
from typing import Dict, List
from datasets import load_dataset
import pandas as pd
from pathlib import Path
import json

class HumanEvalDataset:
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.join(current_dir, 'humaneval_data')
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
        return os.path.exists(os.path.join(self.base_dir, "test.csv"))
    
    def _download_and_process_dataset(self) -> None:
        """HumanEval 데이터셋을 다운로드하고 처리"""
        try:
            # Hugging Face에서 데이터셋 로드
            dataset = load_dataset("openai_humaneval")
            
            data = []
            for item in dataset['test']:
                # 프롬프트와 엔트리포인트 결합
                full_prompt = f"{item['prompt']}\n\n{item['entry_point']}"
                
                data.append({
                    'task_id': item['task_id'],
                    'prompt': full_prompt,
                    'canonical_solution': item['canonical_solution'],
                    'test_cases': json.dumps(item['test']),  # JSON 문자열로 저장
                    'entry_point': item['entry_point']
                })
            
            # 데이터를 CSV로 저장
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(self.base_dir, "test.csv"), index=False)
            print("Saved test data")
                
        except Exception as e:
            print(f"Error processing dataset: {str(e)}")
    
    def get_split_data(self, split: str = "test") -> List[Dict]:
        """데이터를 가져옴 (HumanEval은 test split만 있음)"""
        if split != "test":
            raise ValueError("HumanEval dataset only has test split")
        
        csv_path = os.path.join(self.base_dir, "test.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found: {csv_path}")
        
        # CSV 파일에서 데이터 로드
        df = pd.read_csv(csv_path)
        data = []
        
        for _, row in df.iterrows():
            # test_cases는 멀티라인 파이썬 코드이므로 문자열 그대로 사용
            test_cases = row['test_cases'] if isinstance(row['test_cases'], str) else ""
            data.append({
                'task_id': row['task_id'],
                'prompt': row['prompt'],
                'canonical_solution': row['canonical_solution'],
                'test_cases': test_cases,
                'entry_point': row['entry_point']
            })
        
        return data

    def get_all_data(self) -> Dict[str, List[Dict]]:
        """모든 데이터를 가져옴"""
        try:
            test_data = self.get_split_data("test")
            return {"test": test_data}
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return {"test": []}

if __name__ == "__main__":
    # 사용 예시
    dataset = HumanEvalDataset()
    
    try:
        test_data = dataset.get_split_data("test")
        print(f"테스트 예제 수: {len(test_data)}")
        
        # 첫 번째 예제 출력
        if test_data:
            first_example = test_data[0]
            print("\n첫 번째 테스트 예제:")
            print(f"Task ID: {first_example['task_id']}")
            print(f"프롬프트:\n{first_example['prompt']}")
            print(f"\n정답:\n{first_example['canonical_solution']}")
            print(f"\n테스트 케이스:\n{first_example['test_cases']}")
        # test_cases가 빈 샘플 진단
        empty_cases = [d['task_id'] for d in test_data if not d['test_cases'] or not str(d['test_cases']).strip()]
        print(f"test_cases가 빈 샘플 수: {len(empty_cases)}")
        print(f"예시 task_id: {empty_cases[:5]}")
    except ValueError as e:
        print(e) 