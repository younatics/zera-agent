from typing import List, Dict, Any, Optional
import json
import re
import pandas as pd
from evaluation.base.evaluator import BaseEvaluator

class GSM8KEvaluator(BaseEvaluator):
    def load_dataset(self, dataset_name: str) -> List[Dict[str, Any]]:
        """GSM8K 데이터셋을 로드합니다."""
        dataset_path = "agent/dataset/gsm8k_data/test.csv"
        df = pd.read_csv(dataset_path)
        data = df.to_dict('records')
        print(f"\n전체 데이터셋 크기: {len(df)}개")
        print(f"실제 사용되는 데이터셋 크기: {len(data)}개")
        print("-" * 50)
        return data
    
    def format_question(self, item: Dict[str, Any]) -> str:
        """GSM8K 질문을 포맷팅합니다."""
        return item['question']
    
    def evaluate_response(self, response: str, ground_truth: str) -> bool:
        """GSM8K 응답을 평가합니다."""
        try:
            def extract_last_number(text: str) -> float:
                """텍스트에서 마지막 숫자를 추출합니다."""
                numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', str(text))
                if not numbers:
                    return None
                return float(numbers[-1].replace(',', ''))

            # 답변에서 마지막 숫자 추출
            model_number = extract_last_number(response)
            ground_truth_number = extract_last_number(ground_truth)

            if not model_number:
                print("\n[파싱 실패] 모델 답변에서 숫자를 찾을 수 없습니다.")
                return False

            if not ground_truth_number:
                print("\n[파싱 실패] 정답에서 숫자를 찾을 수 없습니다.")
                return False

            print(f"\n[파싱된 답] 모델: {model_number}")
            print(f"[파싱된 답] 정답: {ground_truth_number}")

            # 부동소수점 비교시 작은 오차 허용
            return abs(model_number - ground_truth_number) < 0.01

        except Exception as e:
            print(f"\n[파싱 에러] {str(e)}")
            return False 