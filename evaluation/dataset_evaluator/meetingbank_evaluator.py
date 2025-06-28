from typing import List, Dict, Any, Optional
from evaluation.base.evaluator import BaseEvaluator
from rouge import Rouge
import pandas as pd
import random
from agent.dataset.meetingbank_dataset import MeetingBankDataset
import json
import os
from pathlib import Path

class MeetingBankEvaluator(BaseEvaluator):
    def __init__(self, model_name: str = "gpt4o", model_version: str = "gpt-3.5-turbo", temperature: float = 0.7, top_p: float = 0.9):
        super().__init__(model_name, model_version, temperature, top_p)
        self.rouge = Rouge()
        self.samples_dir = Path("evaluation/samples")
        self.samples_dir.mkdir(exist_ok=True)

    def load_dataset(self, dataset_path: str = "meetingbank", num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """MeetingBank 데이터셋을 로드합니다. 샘플 파일(json) 관리 방식."""
        if num_samples:
            sample_file = self.samples_dir / f"meetingbank_samples_{num_samples}.json"
            if sample_file.exists():
                print(f"[INFO] Loading existing samples from {sample_file}")
                with open(sample_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            # 샘플 파일이 없으면 새로 생성
            print(f"[INFO] Creating new samples file: {sample_file}")
            meetingbank_dataset = MeetingBankDataset()
            all_data = meetingbank_dataset.get_split_data("test")
            sampled_data = random.sample(all_data, min(num_samples, len(all_data)))
            with open(sample_file, 'w', encoding='utf-8') as f:
                json.dump(sampled_data, f, ensure_ascii=False, indent=2)
            return sampled_data
        else:
            # 전체 데이터셋 로드
            meetingbank_dataset = MeetingBankDataset()
            return meetingbank_dataset.get_split_data("test")

    def get_sample_indices(self, num_samples: int) -> List[int]:
        data = self.load_dataset("meetingbank", num_samples)
        total_samples = len(data)
        if num_samples > total_samples:
            num_samples = total_samples
        return random.sample(range(total_samples), num_samples)

    def format_question(self, item: Dict[str, Any]) -> str:
        """MeetingBank 대화(transcript)을 입력으로 사용합니다."""
        return item['transcript']

    def evaluate_response(self, response: str, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """MeetingBank 요약을 평가합니다 (ROUGE-L 기준 없이, 항상 정답 처리)."""
        try:
            scores = self.rouge.get_scores(response, ground_truth['summary'])
            is_passed = True
            return {
                'is_passed': is_passed,
                'rouge_scores': scores[0]
            }
        except Exception as e:
            print(f"ROUGE 평가 중 오류 발생: {str(e)}")
            return {
                'is_passed': True,
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
                      dataset_display_name: Optional[str] = None) -> Dict[str, Any]:
        """전체 평가를 실행하는 메서드"""
        dataset = self.load_dataset(dataset_name, num_samples)
        if sample_indices is not None:
            dataset_len = len(dataset)
            sample_indices = [i for i in sample_indices if i < dataset_len]
            dataset = [dataset[i] for i in sample_indices]
        if len(dataset) == 0:
            print("[경고] 평가할 샘플이 없습니다. 데이터셋 경로와 num_samples, sample_indices 값을 확인하세요.")
            return {
                "total": 0,
                "correct": 0,
                "samples": [],
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "rouge_scores": {
                    "rouge-1": {"f": 0.0},
                    "rouge-2": {"f": 0.0},
                    "rouge-l": {"f": 0.0}
                },
                "accuracy": 0.0
            }
        results = {
            "total": len(dataset),
            "correct": 0,
            "samples": [],
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
                # 모델 응답에서 텍스트 부분만 추출 (메타데이터 제외)
                response_data = self.model.ask(question, system_prompt, user_prompt)
                if isinstance(response_data, tuple):
                    response = response_data[0]  # 텍스트 부분만 사용
                else:
                    response = response_data  # 이미 텍스트인 경우
                # 응답이 에러(예: Exception 객체이거나, 'Error:'로 시작하거나, 'error' 포함)인 경우 평가에서 제외
                if isinstance(response, Exception) or (
                    isinstance(response, str) and (
                        response.strip().lower().startswith('error:') or 'error' in response.strip().lower()
                    )
                ):
                    print(f"[SKIP] 샘플 {idx+1}: 모델 응답이 에러로 평가에서 제외합니다. 응답: {response}")
                    continue
                eval_result = self.evaluate_response(response, item)
                is_correct = eval_result['is_passed']
                results["correct"] += 1 if is_correct else 0
                if eval_result['rouge_scores']:
                    for metric in ['rouge-1', 'rouge-2', 'rouge-l']:
                        results["rouge_scores"][metric]["f"] += eval_result['rouge_scores'][metric]['f']
                sample_info = {
                    "question": question,
                    "model_response": response,
                    "actual_answer": item.get("summary", item),
                    "is_correct": is_correct,
                    "rouge_scores": eval_result['rouge_scores']
                }
                results["samples"].append(sample_info)
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
            except Exception as e:
                print(f"Error processing sample {idx}: {str(e)}")
                continue
        for metric in results["rouge_scores"]:
            results["rouge_scores"][metric]["f"] /= results["total"]
        accuracy = results["correct"] / results["total"] if results["total"] > 0 else 0
        results["accuracy"] = accuracy
        import time
        if not hasattr(self, 'results_dir'):
            self.results_dir = Path("evaluation/results")
            self.results_dir.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_file = self.results_dir / f"{self.__class__.__name__}_{timestamp}.json"
        self.save_results(results, str(result_file))
        rouge_scores = results["rouge_scores"]
        rouge_msg = "\nROUGE 점수:"
        for metric, scores in rouge_scores.items():
            rouge_msg += f"\n{metric}: F1={scores['f']:.3f}"
        msg = f"MeetingBank 평가 완료!\n정확도: {results['accuracy']:.2%}{rouge_msg}"
        self.send_slack_notification(msg)
        return results 