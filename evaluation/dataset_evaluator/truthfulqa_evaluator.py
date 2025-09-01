from typing import List, Dict, Any, Optional
from evaluation.base.evaluator import BaseEvaluator
from agent.dataset.truthfulqa_dataset import TruthfulQADataset
from evaluation.dataset_evaluator.llm_judge.judge import LLMJudge
import json
import random
import time
import logging
from pathlib import Path
from rouge import Rouge

logger = logging.getLogger(__name__)

class TruthfulQAEvaluator(BaseEvaluator):
    def __init__(self, *args, **kwargs):
        """Initialize TruthfulQA evaluator."""
        super().__init__(*args, **kwargs)
        self.dataset_cache = None
        self.samples_dir = Path("evaluation/samples")
        self.samples_dir.mkdir(exist_ok=True)
        self.rouge = Rouge()
        self.judge = LLMJudge()

    def load_dataset(self, dataset_path: str, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load TruthfulQA dataset."""
        if self.dataset_cache is None:
            # Load TruthfulQA dataset
            dataset = TruthfulQADataset()
            test_data = dataset.get_split_data("test")
            self.dataset_cache = test_data
            
        if num_samples:
            return random.sample(self.dataset_cache, min(num_samples, len(self.dataset_cache)))
        return self.dataset_cache
    
    def get_sample_indices(self, num_samples: int) -> List[int]:
        """Return indices of samples to use for evaluation."""
        # Load dataset if not already loaded
        if self.dataset_cache is None:
            self.dataset_cache = self.load_dataset("")
        
        total_samples = len(self.dataset_cache)
        print(f"Total available samples: {total_samples}")
        
        # Randomly select indices without duplicates
        indices = random.sample(range(total_samples), min(num_samples, total_samples))
        print(f"Selected {len(indices)} samples: {indices}")
        return indices

    def format_question(self, item: Dict[str, Any]) -> str:
        """Format TruthfulQA question."""
        return item['question']
    
    def evaluate_response(self, response: str, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate TruthfulQA response."""
        try:
            # Request evaluation from LLM judge
            judge_result = self.judge.evaluate(
                ground_truth['question'],
                response,
                ground_truth
            )
            
            # Calculate ROUGE scores for reference only
            rouge_scores = self.rouge.get_scores(response, ground_truth['best_answer'])[0]
            
            return {
                'is_passed': judge_result['is_passed'],
                'judge_score': judge_result['judge_score'],
                'judge_response': judge_result['judge_response'],
                'rouge_scores': rouge_scores
            }
            
        except Exception as e:
            logger.error(f"Error occurred during evaluation: {str(e)}")
            return {
                'is_passed': False,
                'rouge_scores': None,
                'error': str(e)
            }
            
    def run_evaluation(self, 
                      dataset_name: str, 
                      system_prompt: Optional[str] = None,
                      user_prompt: Optional[str] = None,
                      num_samples: Optional[int] = None,
                      sample_indices: Optional[List[int]] = None) -> Dict[str, Any]:
        """Method to execute the entire evaluation"""
        # Load dataset
        if self.dataset_cache is None:
            self.dataset_cache = self.load_dataset(dataset_name)
        
        # Select samples
        if sample_indices is not None:
            dataset = [self.dataset_cache[i] for i in sample_indices]
        else:
            dataset = random.sample(self.dataset_cache, min(num_samples or len(self.dataset_cache), len(self.dataset_cache)))
        
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
                    "best_answer": item["best_answer"],
                    "correct_answers": item["correct_answers"],
                    "incorrect_answers": item["incorrect_answers"],
                    "is_correct": is_correct,
                    "rouge_scores": eval_result['rouge_scores'],
                    "judge_score": eval_result.get('judge_score'),
                    "judge_response": eval_result.get('judge_response')
                }
                results["samples"].append(sample_info)
                
                # Output detailed information
                print(f"\nSample {idx+1}/{len(dataset)}:")
                print(f"Question: {question}")
                print(f"Model response: {response}")
                print(f"Best answer: {item['best_answer']}")
                print(f"Correctness: {'Correct' if is_correct else 'Incorrect'}")
                if eval_result['rouge_scores']:
                    print("ROUGE scores:")
                    for metric, scores in eval_result['rouge_scores'].items():
                        print(f"  {metric}: F1={scores['f']:.3f}")
                print(f"Judge evaluation score: {eval_result.get('judge_score', 0):.3f}")
                print(f"Judge evaluation explanation: {eval_result.get('judge_response')}")
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
            
        return results 