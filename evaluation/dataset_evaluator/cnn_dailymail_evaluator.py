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
        """Load CNN/DailyMail dataset."""
        if num_samples:
            # Create sample file path
            sample_file = self.samples_dir / f"cnn_dailymail_samples_{num_samples}.json"
            
            # Load if sample file already exists
            if sample_file.exists():
                logger.info(f"Loading existing samples from {sample_file}")
                with open(sample_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            # If 1000 sample file exists and smaller sample is needed
            sample_1000_file = self.samples_dir / "cnn_dailymail_samples_1000.json"
            if sample_1000_file.exists() and num_samples < 1000:
                logger.info(f"Loading and sampling from 1000 samples file")
                with open(sample_1000_file, 'r', encoding='utf-8') as f:
                    base_samples = json.load(f)
                    return random.sample(base_samples, num_samples)
            
            # Create new sample file if none exists
            logger.info(f"Creating new samples file: {sample_file}")
            dataset = load_dataset("cnn_dailymail", "3.0.0", split="test")
            formatted_data = []
            for item in dataset:
                formatted_item = {
                    "article": item["article"],
                    "highlights": item["highlights"]
                }
                formatted_data.append(formatted_item)
            
            # Random sampling
            sampled_data = random.sample(formatted_data, min(num_samples, len(formatted_data)))
            
            # Save samples
            with open(sample_file, 'w', encoding='utf-8') as f:
                json.dump(sampled_data, f, ensure_ascii=False, indent=2)
            
            return sampled_data
        else:
            # Load full dataset
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
        """Return indices of samples to evaluate."""
        dataset = self.load_dataset("cnn_dailymail", num_samples)
        total_samples = len(dataset)
        if num_samples > total_samples:
            num_samples = total_samples
        return random.sample(range(total_samples), num_samples)
    
    def format_question(self, item: Dict[str, Any]) -> str:
        """Format CNN/DailyMail article."""
        return item['article']
    
    def evaluate_response(self, response: str, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate CNN/DailyMail summary."""
        # Extract only text after 'article:' or 'points:'
        response_lower = response.lower()
        if 'article:' in response_lower:
            response = response_lower.split('article:')[1].strip()
            # Additional processing if points: exists after article:
        rouge = Rouge()
        try:
            scores = rouge.get_scores(response, ground_truth['highlights'])
            rouge_l_score = scores[0]['rouge-l']['f']
            
            return {
                'is_passed': True,  # ROUGE-L score is included in evaluation results but correctness is always True
                'rouge_scores': scores[0]  # Include ROUGE-1, ROUGE-2, ROUGE-L scores
            }
        except Exception as e:
            print(f"Error during ROUGE evaluation: {str(e)}")
            return {
                'is_passed': True,  # Treat as correct even if error occurs
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
        """Method to execute full evaluation"""
        if sample_indices is not None:
            # Index from full dataset
            full_dataset = self.load_dataset(dataset_name)
            dataset = [full_dataset[i] for i in sample_indices]
        else:
            dataset = self.load_dataset(dataset_name, num_samples)
        
        results = {
            "total": len(dataset),
            "correct": 0,
            "samples": [],  # Save detailed information for each sample
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
                eval_result = self.evaluate_response(response, item)
                
                is_correct = eval_result['is_passed']
                results["correct"] += 1 if is_correct else 0
                
                # Accumulate ROUGE scores
                if eval_result['rouge_scores']:
                    for metric in ['rouge-1', 'rouge-2', 'rouge-l']:
                        results["rouge_scores"][metric]["f"] += eval_result['rouge_scores'][metric]['f']
                
                # Save detailed information for each sample
                sample_info = {
                    "question": question,
                    "model_response": response,
                    "actual_answer": item.get("highlights", item),
                    "is_correct": is_correct,
                    "rouge_scores": eval_result['rouge_scores']
                }
                results["samples"].append(sample_info)
                
                # Output detailed information
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
                
                logger.info(f"Processed {idx+1}/{len(dataset)} samples")
                time.sleep(1)  # Prevent API rate limit
                
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {str(e)}")
                continue
                
        # Calculate average ROUGE scores
        for metric in results["rouge_scores"]:
            results["rouge_scores"][metric]["f"] /= results["total"]
                
        accuracy = results["correct"] / results["total"] if results["total"] > 0 else 0
        results["accuracy"] = accuracy
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_file = self.results_dir / f"{self.__class__.__name__}_{timestamp}.json"
        self.save_results(results, str(result_file))
        
        # Add ROUGE scores to Slack notification message
        rouge_scores = results["rouge_scores"]
        rouge_msg = "\nROUGE scores:"
        for metric, scores in rouge_scores.items():
            rouge_msg += f"\n{metric}: F1={scores['f']:.3f}"
        
        # Send Slack notification
        msg = f"CNN/DailyMail evaluation completed!\nAccuracy: {results['accuracy']:.2%}{rouge_msg}"
        self.send_slack_notification(msg)
            
        return results 