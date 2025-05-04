from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from agent.common.api_client import Model
import json
import time
from pathlib import Path
import logging
from .dataset_loader import DatasetLoader
import os
import random
from agent.common.slack_notify import send_file_to_slack, notify_slack
import subprocess
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseEvaluator(ABC):
    def __init__(self, model_name: str, model_version: str, temperature: float = None, top_p: float = None):
        """
        평가기를 초기화합니다.
        
        Args:
            model_name: 사용할 모델의 이름
            model_version: 사용할 모델의 버전
            temperature: 모델의 temperature 값 (선택사항)
            top_p: 모델의 top_p 값 (선택사항)
        """
        self.model_name = model_name
        self.model_version = model_version
        self.model = Model(model_name).set_version(model_version)
        
        # temperature와 top_p가 제공된 경우에만 설정
        if temperature is not None:
            self.model.set_temperature(temperature)
            self.temperature = temperature
        if top_p is not None:
            self.model.set_top_p(top_p)
            self.top_p = top_p

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
                      num_samples: Optional[int] = None,
                      sample_indices: Optional[List[int]] = None) -> Dict[str, Any]:
        """전체 평가를 실행하는 메서드"""
        dataset = self.load_dataset(dataset_name)
        
        # 샘플 인덱스가 제공된 경우 해당 샘플만 사용
        if sample_indices is not None:
            dataset = [dataset[i] for i in sample_indices]
        # 샘플 수가 지정된 경우 랜덤 샘플링
        elif num_samples:
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
                print(f"시스템 프롬프트: {system_prompt}")
                print(f"사용자 프롬프트: {user_prompt}")
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
        # 모델 버전만 파일명에 포함
        model_version_safe = self.model_version.replace('/', '_')
        result_file = self.results_dir / f"{self.__class__.__name__}_{model_version_safe}_{timestamp}.json"
        self.save_results(results, str(result_file))
            
        return results

    def save_results(self, results: List[Dict[str, Any]], output_path: str, slack_file_upload: bool = True):
        """결과를 저장합니다. slack_file_upload가 True면 슬랙 웹훅으로 간단한 메시지를 전송합니다."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        # MBPP 평가 결과라면 추가 분석 파일 생성
        if 'MBPPEvaluator' in os.path.basename(output_path):
            try:
                # 분석 파일명 결정
                analysis_path = output_path.replace('.json', '_analysis.json')
                # analyze_mbpp_json_eval.py 실행
                subprocess.run([
                    sys.executable, 'evaluation/code_analysis/analyze_mbpp_json_eval.py', output_path
                ], check=True)
                print(f"MBPP 추가 분석 파일 생성: {analysis_path}")
            except Exception as e:
                print(f"MBPP 분석 파일 생성 실패: {e}")
        if slack_file_upload:
            webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
            model_version = getattr(self, 'model_version', 'unknown')
            accuracy = results.get("accuracy", 'N/A')
            if webhook_url:
                if isinstance(accuracy, float):
                    accuracy_str = f"{accuracy:.2%}"
                else:
                    accuracy_str = str(accuracy)
                msg = f"[평가 결과] 모델 버전: {model_version}\n정확도: {accuracy_str}\n결과 파일: {os.path.basename(output_path)}"
                notify_slack(msg, webhook_url)
            else:
                print("슬랙 웹훅 알림을 위해 SLACK_WEBHOOK_URL 환경변수가 필요합니다.") 