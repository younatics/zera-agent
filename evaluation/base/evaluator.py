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
        Initialize the evaluator.
        
        Args:
            model_name: Name of the model to use
            model_version: Version of the model to use
            temperature: Temperature value for the model (optional)
            top_p: Top-p value for the model (optional)
        """
        self.model_name = model_name
        self.model_version = model_version or "unknown"
        
        # If model_version is None, use default value from Model class
        if model_version:
            self.model = Model(model_name).set_version(model_version)
        else:
            self.model = Model(model_name)  # Use default version
        
        # Set temperature and top_p only if provided
        if temperature is not None:
            self.model.set_temperature(temperature)
            self.temperature = temperature
        if top_p is not None:
            self.model.set_top_p(top_p)
            self.top_p = top_p

        self.results_dir = Path("evaluation/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def load_dataset(self, dataset_name: str, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Method to load the dataset"""
        dataset_path = f"data/{dataset_name}/test.json"
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if num_samples is not None:
            data = random.sample(data, min(num_samples, len(data)))
            
        return data
        
    @abstractmethod
    def format_question(self, item: Dict[str, Any]) -> str:
        """Convert each dataset question to model input format"""
        pass
        
    @abstractmethod
    def evaluate_response(self, response: str, ground_truth: Dict[str, Any]) -> bool:
        """Method to evaluate model response"""
        pass
        
    def send_slack_notification(self, message: str):
        webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
        if webhook_url:
            notify_slack(message, webhook_url)
        else:
            print("SLACK_WEBHOOK_URL environment variable is required for Slack webhook notifications.")

    def run_evaluation(self, 
                      dataset_name, 
                      system_prompt: Optional[str] = None,
                      user_prompt: Optional[str] = None,
                      num_samples: Optional[int] = None,
                      sample_indices: Optional[List[int]] = None,
                      is_zera: Optional[bool] = None,
                      num_shots: Optional[int] = None,
                      dataset_display_name: Optional[str] = None) -> Dict[str, Any]:
        """Method to execute the complete evaluation"""
        # --- Slack notification for evaluation start ---
        model_version = getattr(self, 'model_version', 'unknown')
        if is_zera is True:
            prompt_type = "🧬 Zera Prompt"
        elif is_zera is False:
            prompt_type = "📝 Base Prompt"
        else:
            prompt_type = "🤖 Prompt"
        if dataset_display_name:
            dataset_desc = dataset_display_name
        elif isinstance(dataset_name, list):
            dataset_desc = f"Loaded dataset (size={len(dataset_name)})"
        else:
            dataset_desc = str(dataset_name)
        start_msg = f"{prompt_type} evaluation started!\nModel version: {model_version}\nDataset: {dataset_desc}"
        self.send_slack_notification(start_msg)
        # If dataset_name is already a list, use it as is, otherwise call load_dataset
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
        # Generate few-shot examples
        few_shot_examples = []
        if num_shots is not None and num_shots > 0:
            # Select examples that don't overlap with samples used for evaluation
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
                # user_prompt remains the same, only add few-shot examples above the question
                full_question = ""
                if few_shot_prompt:
                    full_question += few_shot_prompt + "---\n"  # Add separator between examples and question
                full_question += question
                # Extract only text part from model response (exclude metadata)
                response_data = self.model.ask(full_question, system_prompt, user_prompt)
                if isinstance(response_data, tuple):
                    response = response_data[0]  # Use only text part
                else:
                    response = response_data  # Already text
                
                is_correct = self.evaluate_response(response, item)
                results["correct"] += 1 if is_correct else 0
                sample_info = {
                    "question": full_question,
                    "model_response": response,
                    "actual_answer": item.get("answer", item),
                    "is_correct": is_correct
                }
                results["samples"].append(sample_info)
                print(f"\nSample {idx+1}/{len(dataset)}:")
                print(f"System prompt: {system_prompt}")
                print(f"User prompt: {user_prompt}")
                print(f"Question: {full_question}")
                print(f"Model answer: {response}")
                print(f"Actual answer: {sample_info['actual_answer']}")
                print(f"Correct: {'Yes' if is_correct else 'No'}")
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
        """Save the results. If slack_file_upload is True, send a simple message via Slack webhook."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        # If it's MBPP evaluation result, create additional analysis file
        if 'MBPPEvaluator' in os.path.basename(output_path):
            try:
                # Determine analysis filename
                analysis_path = output_path.replace('.json', '_analysis.json')
                # Execute analyze_mbpp_json_eval.py
                subprocess.run([
                    sys.executable, 'evaluation/code_analysis/analyze_mbpp_json_eval.py', output_path
                ], check=True)
                print(f"MBPP additional analysis file created: {analysis_path}")
            except Exception as e:
                print(f"Failed to create MBPP analysis file: {e}")
        if slack_file_upload:
            model_version = getattr(self, 'model_version', 'unknown')
            accuracy = results.get("accuracy", 'N/A')
            if isinstance(accuracy, float):
                accuracy_str = f"{accuracy:.2%}"
            else:
                accuracy_str = str(accuracy)
            if is_zera is True:
                prompt_type = "🧬 Zera Prompt"
            elif is_zera is False:
                prompt_type = "📝 Base Prompt"
            else:
                prompt_type = "🤖 Prompt"
            msg = f"{prompt_type} evaluation result\nModel version: {model_version}\nAccuracy: {accuracy_str}\nResult file: {os.path.basename(output_path)}\n🎉 Great job!"
            self.send_slack_notification(msg) 