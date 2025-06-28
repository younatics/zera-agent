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
        self.model_version = model_version or "unknown"
        
        # model_version이 None이면 Model 클래스에서 기본값 사용
        if model_version:
            self.model = Model(model_name).set_version(model_version)
        else:
            self.model = Model(model_name)  # 기본 버전 사용
        
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
        
    def send_slack_notification(self, message: str):
        webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
        if webhook_url:
            notify_slack(message, webhook_url)
        else:
            print("슬랙 웹훅 알림을 위해 SLACK_WEBHOOK_URL 환경변수가 필요합니다.")

    def run_evaluation(self, 
                      dataset_name, 
                      system_prompt: Optional[str] = None,
                      user_prompt: Optional[str] = None,
                      num_samples: Optional[int] = None,
                      sample_indices: Optional[List[int]] = None,
                      is_zera: Optional[bool] = None,
                      num_shots: Optional[int] = None,
                      dataset_display_name: Optional[str] = None) -> Dict[str, Any]:
        """전체 평가를 실행하는 메서드"""
        # --- 평가 시작 슬랙 알림 ---
        model_version = getattr(self, 'model_version', 'unknown')
        if is_zera is True:
            prompt_type = "🧬 제라 프롬프트"
        elif is_zera is False:
            prompt_type = "📝 베이스 프롬프트"
        else:
            prompt_type = "🤖 프롬프트"
        if dataset_display_name:
            dataset_desc = dataset_display_name
        elif isinstance(dataset_name, list):
            dataset_desc = f"Loaded dataset (size={len(dataset_name)})"
        else:
            dataset_desc = str(dataset_name)
        start_msg = f"{prompt_type} 평가 시작!\n모델 버전: {model_version}\n데이터셋: {dataset_desc}"
        self.send_slack_notification(start_msg)
        # dataset_name이 이미 list면 그대로 사용, 아니면 load_dataset 호출
        if isinstance(dataset_name, list):
            dataset = dataset_name
        else:
            dataset = self.load_dataset(dataset_name)
        if sample_indices is not None:
            dataset = [dataset[i] for i in sample_indices]
        elif num_samples:
            dataset = random.sample(dataset, min(num_samples, len(dataset)))
        results = {
            "total": len(dataset),
            "correct": 0,
            "samples": [],
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "num_shots": num_shots
        }
        # few-shot 예시 생성
        few_shot_examples = []
        if num_shots is not None and num_shots > 0:
            # 평가에 사용되는 샘플과 겹치지 않게 예시를 뽑음
            available_indices = set(range(len(dataset)))
            if len(dataset) > num_shots:
                few_shot_indices = random.sample(list(available_indices), num_shots)
            else:
                few_shot_indices = list(available_indices)
            for idx in few_shot_indices:
                ex_item = dataset[idx]
                ex_question = self.format_question(ex_item)
                ex_answer = ex_item.get("answer", ex_item)
                few_shot_examples.append(f"[Example {len(few_shot_examples)+1}]\nQuestion: {ex_question}\nAnswer: {ex_answer}\n")
            few_shot_prompt = "\n".join(few_shot_examples)
        else:
            few_shot_prompt = ""
        for idx, item in enumerate(dataset):
            try:
                question = self.format_question(item)
                # user_prompt는 그대로, question 위에만 few-shot 예시 추가
                full_question = ""
                if few_shot_prompt:
                    full_question += few_shot_prompt + "---\n"  # 예시와 질문 사이에 구분자 추가
                full_question += question
                # 모델 응답에서 텍스트 부분만 추출 (메타데이터 제외)
                response_data = self.model.ask(full_question, system_prompt, user_prompt)
                if isinstance(response_data, tuple):
                    response = response_data[0]  # 텍스트 부분만 사용
                else:
                    response = response_data  # 이미 텍스트인 경우
                
                is_correct = self.evaluate_response(response, item)
                results["correct"] += 1 if is_correct else 0
                sample_info = {
                    "question": full_question,
                    "model_response": response,
                    "actual_answer": item.get("answer", item),
                    "is_correct": is_correct
                }
                results["samples"].append(sample_info)
                print(f"\n샘플 {idx+1}/{len(dataset)}:")
                print(f"시스템 프롬프트: {system_prompt}")
                print(f"사용자 프롬프트: {user_prompt}")
                print(f"문제: {full_question}")
                print(f"모델 답변: {response}")
                print(f"실제 답변: {sample_info['actual_answer']}")
                print(f"정답 여부: {'정답' if is_correct else '오답'}")
                print("-" * 50)
                logger.info(f"Processed {idx+1}/{len(dataset)} samples")
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {str(e)}")
                continue
        accuracy = results["correct"] / results["total"] if results["total"] > 0 else 0
        results["accuracy"] = accuracy
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_version_safe = (self.model_version or "unknown").replace('/', '_')
        result_file = self.results_dir / f"{self.__class__.__name__}_{model_version_safe}_{timestamp}.json"
        self.save_results(results, str(result_file), is_zera=is_zera)
        return results

    def save_results(self, results: List[Dict[str, Any]], output_path: str, slack_file_upload: bool = True, is_zera: bool = None):
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
            model_version = getattr(self, 'model_version', 'unknown')
            accuracy = results.get("accuracy", 'N/A')
            if isinstance(accuracy, float):
                accuracy_str = f"{accuracy:.2%}"
            else:
                accuracy_str = str(accuracy)
            if is_zera is True:
                prompt_type = "🧬 제라 프롬프트"
            elif is_zera is False:
                prompt_type = "📝 베이스 프롬프트"
            else:
                prompt_type = "🤖 프롬프트"
            msg = f"{prompt_type} 평가 결과\n모델 버전: {model_version}\n정확도: {accuracy_str}\n결과 파일: {os.path.basename(output_path)}\n🎉 수고하셨습니다!"
            self.send_slack_notification(msg) 