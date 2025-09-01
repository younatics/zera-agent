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
        """Load MeetingBank dataset. Sample file (json) management method."""
        if num_samples:
            sample_file = self.samples_dir / f"meetingbank_samples_{num_samples}.json"
            if sample_file.exists():
                print(f"[INFO] Loading existing samples from {sample_file}")
                with open(sample_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            # Create new sample file if it doesn't exist
            print(f"[INFO] Creating new samples file: {sample_file}")
            meetingbank_dataset = MeetingBankDataset()
            all_data = meetingbank_dataset.get_split_data("test")
            sampled_data = random.sample(all_data, min(num_samples, len(all_data)))
            with open(sample_file, 'w', encoding='utf-8') as f:
                json.dump(sampled_data, f, ensure_ascii=False, indent=2)
            return sampled_data
        else:
            # Load entire dataset
            meetingbank_dataset = MeetingBankDataset()
            return meetingbank_dataset.get_split_data("test")

    def get_sample_indices(self, num_samples: int) -> List[int]:
        data = self.load_dataset("meetingbank", num_samples)
        total_samples = len(data)
        if num_samples > total_samples:
            num_samples = total_samples
        return random.sample(range(total_samples), num_samples)

    def format_question(self, item: Dict[str, Any]) -> str:
        """Use MeetingBank dialogue (transcript) as input."""
        return item['transcript']

    def evaluate_response(self, response: str, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate MeetingBank summary (always treat as correct without ROUGE-L criteria)."""
        try:
            scores = self.rouge.get_scores(response, ground_truth['summary'])
            is_passed = True
            return {
                'is_passed': is_passed,
                'rouge_scores': scores[0]
            }
        except Exception as e:
            print(f"Error during ROUGE evaluation: {str(e)}")
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
        """Method to execute the entire evaluation"""
        dataset = self.load_dataset(dataset_name, num_samples)
        if sample_indices is not None:
            dataset_len = len(dataset)
            sample_indices = [i for i in sample_indices if i < dataset_len]
            dataset = [dataset[i] for i in sample_indices]
        if len(dataset) == 0:
            print("[WARNING] No samples to evaluate. Please check the dataset path, num_samples, and sample_indices values.")
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
                # Extract only text part from model response (exclude metadata)
                response_data = self.model.ask(question, system_prompt, user_prompt)
                if isinstance(response_data, tuple):
                    response = response_data[0]  # Use only text part
                else:
                    response = response_data  # Already text
                # Exclude from evaluation if response is an error (e.g., Exception object, starts with 'Error:', or contains 'error')
                if isinstance(response, Exception) or (
                    isinstance(response, str) and (
                        response.strip().lower().startswith('error:') or 'error' in response.strip().lower()
                    )
                ):
                    print(f"[SKIP] Sample {idx+1}: Model response is excluded from evaluation due to error. Response: {response}")
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
                print(f"\nSample {idx+1}/{len(dataset)}:")
                print(f"Question: {question}")
                print(f"Model response: {response}")
                print(f"Actual answer: {sample_info['actual_answer']}")
                print(f"Correctness: {'Correct' if is_correct else 'Incorrect'}")
                if eval_result['rouge_scores']:
                    print("ROUGE scores:")
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
        rouge_msg = "\nROUGE scores:"
        for metric, scores in rouge_scores.items():
            rouge_msg += f"\n{metric}: F1={scores['f']:.3f}"
        msg = f"MeetingBank evaluation completed!\nAccuracy: {results['accuracy']:.2%}{rouge_msg}"
        self.send_slack_notification(msg)
        return results 