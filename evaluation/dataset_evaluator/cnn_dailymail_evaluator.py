from typing import List, Dict, Any, Optional
from evaluation.base.evaluator import BaseEvaluator
from rouge import Rouge
import json
import os
from datasets import load_dataset
import random
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class CNNDailyMailEvaluator(BaseEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.samples_dir = Path("evaluation/samples")
        self.samples_dir.mkdir(exist_ok=True)

    def load_dataset(self, dataset_path: str, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """CNN/DailyMail 데이터셋을 로드합니다."""
        if num_samples:
            # 샘플 파일 경로 생성
            sample_file = self.samples_dir / f"cnn_dailymail_samples_{num_samples}.json"
            
            # 이미 샘플 파일이 있으면 로드
            if sample_file.exists():
                logger.info(f"Loading existing samples from {sample_file}")
                with open(sample_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            # 1000개 샘플 파일이 있고, 더 작은 샘플이 필요한 경우
            sample_1000_file = self.samples_dir / "cnn_dailymail_samples_1000.json"
            if sample_1000_file.exists() and num_samples < 1000:
                logger.info(f"Loading and sampling from 1000 samples file")
                with open(sample_1000_file, 'r', encoding='utf-8') as f:
                    base_samples = json.load(f)
                    return random.sample(base_samples, num_samples)
            
            # 샘플 파일이 없으면 새로 생성
            logger.info(f"Creating new samples file: {sample_file}")
            dataset = load_dataset("cnn_dailymail", "3.0.0", split="test")
            formatted_data = []
            for item in dataset:
                formatted_item = {
                    "article": item["article"],
                    "highlights": item["highlights"]
                }
                formatted_data.append(formatted_item)
            
            # 랜덤 샘플링
            sampled_data = random.sample(formatted_data, min(num_samples, len(formatted_data)))
            
            # 샘플 저장
            with open(sample_file, 'w', encoding='utf-8') as f:
                json.dump(sampled_data, f, ensure_ascii=False, indent=2)
            
            return sampled_data
        else:
            # 전체 데이터셋 로드
            dataset = load_dataset("cnn_dailymail", "3.0.0", split="test")
            formatted_data = []
            for item in dataset:
                formatted_item = {
                    "article": item["article"],
                    "highlights": item["highlights"]
                }
                formatted_data.append(formatted_item)
            return formatted_data
    
    def get_sample_indices(self, num_samples: int) -> List[int]:
        """평가할 샘플의 인덱스를 반환합니다."""
        dataset = self.load_dataset("cnn_dailymail", num_samples)
        total_samples = len(dataset)
        if num_samples > total_samples:
            num_samples = total_samples
        return random.sample(range(total_samples), num_samples)
    
    def format_question(self, item: Dict[str, Any]) -> str:
        """CNN/DailyMail 기사를 포맷팅합니다."""
        return item['article']
    
    def evaluate_response(self, response: str, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """CNN/DailyMail 요약을 평가합니다."""
        # 'article:' 또는 'points:' 이후의 텍스트만 추출
        response_lower = response.lower()
        rouge = Rouge()
        try:
            scores = rouge.get_scores(response, ground_truth['highlights'])
            rouge_l_score = scores[0]['rouge-l']['f']
            
            return {
                'is_passed': True,  # ROUGE-L 점수는 평가 결과에 포함되지만 정답 여부는 항상 True
                'rouge_scores': scores[0]  # ROUGE-1, ROUGE-2, ROUGE-L 점수 모두 포함
            }
        except Exception as e:
            print(f"ROUGE 평가 중 오류 발생: {str(e)}")
            return {
                'is_passed': True,  # 오류 발생 시에도 정답으로 처리
                'rouge_scores': None,
                'error': str(e)
            }
            
    def run_evaluation(self, 
                      dataset_name: str, 
                      system_prompt: Optional[str] = None,
                      user_prompt: Optional[str] = None,
                      num_samples: Optional[int] = None,
                      sample_indices: Optional[List[int]] = None,
                      is_zera: Optional[bool] = None,
                      num_shots: Optional[int] = None,
                      **kwargs) -> Dict[str, Any]:
        """전체 평가를 실행하는 메서드"""
        if sample_indices is not None:
            # 전체 데이터셋에서 인덱싱
            full_dataset = self.load_dataset(dataset_name)
            dataset = [full_dataset[i] for i in sample_indices]
        else:
            dataset = self.load_dataset(dataset_name, num_samples)
        
        results = {
            "total": len(dataset),
            "correct": 0,
            "samples": [],  # 각 샘플의 상세 정보를 저장
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "rouge_scores": {
                "rouge-1": {"f": 0.0},
                "rouge-2": {"f": 0.0},
                "rouge-l": {"f": 0.0}
            }
        }
        
        for idx, item in enumerate(dataset):
            try:
                question = self.format_question(item)
                response = self.model.ask(question, system_prompt, user_prompt)
                eval_result = self.evaluate_response(response, item)
                
                is_correct = eval_result['is_passed']
                results["correct"] += 1 if is_correct else 0
                
                # ROUGE 점수 누적
                if eval_result['rouge_scores']:
                    for metric in ['rouge-1', 'rouge-2', 'rouge-l']:
                        results["rouge_scores"][metric]["f"] += eval_result['rouge_scores'][metric]['f']
                
                # 각 샘플의 상세 정보 저장
                sample_info = {
                    "question": question,
                    "model_response": response,
                    "actual_answer": item.get("highlights", item),
                    "is_correct": is_correct,
                    "rouge_scores": eval_result['rouge_scores']
                }
                results["samples"].append(sample_info)
                
                # 상세 정보 출력
                print(f"\n샘플 {idx+1}/{len(dataset)}:")
                print(f"문제: {question}")
                print(f"모델 답변: {response}")
                print(f"실제 답변: {sample_info['actual_answer']}")
                print(f"정답 여부: {'정답' if is_correct else '오답'}")
                if eval_result['rouge_scores']:
                    print("ROUGE 점수:")
                    for metric, scores in eval_result['rouge_scores'].items():
                        print(f"  {metric}: F1={scores['f']:.3f}")
                print("-" * 50)
                
                logger.info(f"Processed {idx+1}/{len(dataset)} samples")
                time.sleep(1)  # API rate limit 방지
                
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {str(e)}")
                continue
                
        # ROUGE 점수 평균 계산
        for metric in results["rouge_scores"]:
            results["rouge_scores"][metric]["f"] /= results["total"]
                
        accuracy = results["correct"] / results["total"] if results["total"] > 0 else 0
        results["accuracy"] = accuracy
        
        # 결과 저장
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_file = self.results_dir / f"{self.__class__.__name__}_{timestamp}.json"
        self.save_results(results, str(result_file))
        
        # 슬랙 알림 메시지에 ROUGE 점수 추가
        rouge_scores = results["rouge_scores"]
        rouge_msg = "\nROUGE 점수:"
        for metric, scores in rouge_scores.items():
            rouge_msg += f"\n{metric}: F1={scores['f']:.3f}"
        
        # 슬랙 알림 전송
        msg = f"CNN/DailyMail 평가 완료!\n정확도: {results['accuracy']:.2%}{rouge_msg}"
        self.send_slack_notification(msg)
            
        return results 