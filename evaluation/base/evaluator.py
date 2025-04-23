from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from common.api_client import Model
import json
import time
from pathlib import Path
import logging
from .dataset_loader import DatasetLoader
import os
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseEvaluator(ABC):
    def __init__(self, model_name: str = "gpt4o", model_version: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self.model_version = model_version
        self.model = Model(model_name).set_version(model_version)
        self.model.set_temperature(0.7)
        self.model.set_top_p(0.9)

        self.results_dir = Path("evaluation/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def load_dataset(self, dataset_name: str, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """데이터셋을 로드하는 메서드"""
        dataset_path = f"data/{dataset_name}/test.json"
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if num_samples is not None:
            data = random.sample(data, min(num_samples, len(data)))
            
        return data
        
    @abstractmethod
    def format_question(self, item: Dict[str, Any]) -> str:
        """각 데이터셋의 질문을 모델 입력 형식으로 변환"""
        pass
        
    @abstractmethod
    def evaluate_response(self, response: str, ground_truth: Dict[str, Any]) -> bool:
        """모델의 응답을 평가하는 메서드"""
        pass
        
    def run_evaluation(self, 
                      dataset_name: str, 
                      system_prompt: Optional[str] = None,
                      user_prompt: Optional[str] = None,
                      num_samples: Optional[int] = None) -> Dict[str, Any]:
        """전체 평가를 실행하는 메서드"""
        dataset = self.load_dataset(dataset_name)
        
        # 랜덤 샘플링
        if num_samples:
            dataset = random.sample(dataset, min(num_samples, len(dataset)))
            
        results = {
            "total": len(dataset),
            "correct": 0,
            "samples": [],  # 각 샘플의 상세 정보를 저장
            "system_prompt": system_prompt,
            "user_prompt": user_prompt
        }
        
        for idx, item in enumerate(dataset):
            try:
                question = self.format_question(item)
                
                response = self.model.ask(question, system_prompt, user_prompt)
                
                is_correct = self.evaluate_response(response, item)
                
                results["correct"] += 1 if is_correct else 0
                
                # 각 샘플의 상세 정보 저장
                sample_info = {
                    "question": question,
                    "model_response": response,
                    "actual_answer": item.get("answer", item),  # answer 필드가 있으면 사용, 없으면 전체 item
                    "is_correct": is_correct
                }
                results["samples"].append(sample_info)
                
                # 상세 정보 출력
                print(f"\n샘플 {idx+1}/{len(dataset)}:")
                print(f"문제: {question}")
                print(f"모델 답변: {response}")
                print(f"실제 답변: {sample_info['actual_answer']}")
                print(f"정답 여부: {'정답' if is_correct else '오답'}")
                print("-" * 50)
                
                logger.info(f"Processed {idx+1}/{len(dataset)} samples")
                time.sleep(1)  # API rate limit 방지
                
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {str(e)}")
                continue
                
        accuracy = results["correct"] / results["total"] if results["total"] > 0 else 0
        results["accuracy"] = accuracy
        
        # 결과 저장
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_file = self.results_dir / f"{self.__class__.__name__}_{timestamp}.json"
        self.save_results(results, str(result_file))
            
        return results 

    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """결과를 저장합니다."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2) 