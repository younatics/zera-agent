import argparse
from dotenv import load_dotenv
import os
import sys
from pathlib import Path
from evaluation.dataset_evaluator.gsm8k_evaluator import GSM8KEvaluator
from evaluation.dataset_evaluator.mmlu_evaluator import MMLUEvaluator
from evaluation.dataset_evaluator.bbh_evaluator import BBHEvaluator
from evaluation.dataset_evaluator.cnn_dailymail_evaluator import CNNDailyMailEvaluator
from evaluation.dataset_evaluator.mbpp_evaluator import MBPPEvaluator
from evaluation.dataset_evaluator.mmlu_pro_evaluator import MMLUProEvaluator
from evaluation.dataset_evaluator.truthfulqa_evaluator import TruthfulQAEvaluator
from evaluation.dataset_evaluator.humaneval_evaluator import HumanEvalEvaluator
from evaluation.dataset_evaluator.xsum_evaluator import XSUMEvaluator
from evaluation.dataset_evaluator.hellaswag_evaluator import HellaSwagEvaluator
from evaluation.dataset_evaluator.samsum_evaluator import SamSumEvaluator
from evaluation.dataset_evaluator.meetingbank_evaluator import MeetingBankEvaluator
import random

def setup_environment():
    # Try to load from different possible locations
    env_paths = [
        '.env',  # Current directory
        '../.env',  # Parent directory
        '../../.env',  # Parent's parent directory
        str(Path.home() / '.env'),  # User's home directory
    ]
    
    env_loaded = False
    for env_path in env_paths:
        if os.path.exists(env_path):
            load_dotenv(env_path)
            env_loaded = True
            break
    
    # Verify required environment variables
    required_vars = ['OPENAI_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please ensure these variables are set in your .env file or environment")
        sys.exit(1)

# Call setup at import time
setup_environment()

def run_single_evaluation(evaluator, dataset, system_prompt, user_prompt, num_samples, sample_indices=None, is_zera=None, num_shots=None, dataset_display_name=None):
    results = evaluator.run_evaluation(
        dataset,
        system_prompt,
        user_prompt,
        num_samples,
        sample_indices=sample_indices,
        is_zera=is_zera,
        num_shots=num_shots,
        dataset_display_name=dataset_display_name
    )
    return results

def print_evaluation_results(results, prompt_type=""):
    print(f"\n{prompt_type} evaluation results:")
    print(f"Total samples: {results['total']}")
    print(f"Correct answers: {results['correct']}")
    print(f"Accuracy: {results['accuracy']:.2%}")
    
    if "rouge_scores" in results:
        print(f"\n{prompt_type} ROUGE scores (average):")
        for metric, scores in results["rouge_scores"].items():
            print(f"  {metric}:")
            print(f"    F1: {scores['f']:.3f}")
    
    return results['accuracy']

def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description="LLM evaluation script")
        parser.add_argument("--dataset", type=str, required=True, 
                          choices=["gsm8k", "mmlu", "mmlu_pro", "bbh", "cnn_dailymail", "mbpp", "truthfulqa", "humaneval", "xsum", "hellaswag", "samsum", "meetingbank"],
                          help="Dataset to evaluate")
        parser.add_argument("--model", type=str, default="gpt4o",
                          help="Model to use (default: gpt4o)")
        parser.add_argument("--model_version", type=str, default="gpt-3.5-turbo",
                          help="Model version (default: gpt-3.5-turbo)")
        parser.add_argument("--base_system_prompt", type=str,
                          help="Existing system prompt")
        parser.add_argument("--base_user_prompt", type=str,
                          help="Existing user prompt")
        parser.add_argument("--zera_system_prompt", type=str,
                          help="Zera system prompt")
        parser.add_argument("--zera_user_prompt", type=str,
                          help="Zera user prompt")
        parser.add_argument("--num_samples", type=int, default=10,
                          help="Number of samples to evaluate (default: 10)")
        parser.add_argument("--temperature", type=float, default=0.7,
                          help="Model temperature value (default: 0.7)")
        parser.add_argument("--top_p", type=float, default=0.9,
                          help="Model top_p value (default: 0.9)")
        parser.add_argument("--base_num_shots", type=int, default=0,
                          help="Number of few-shot examples for existing prompt (default: 0)")
        parser.add_argument("--zera_num_shots", type=int, default=0,
                          help="Number of few-shot examples for Zera prompt (default: 0)")
        parser.add_argument("--bbh_category", type=str, default=None,
                          help="Category name to use for BBH evaluation (e.g., Penguins, Geometry, etc.)")
        
        args = parser.parse_args()
    
    # Select evaluator
    evaluators = {
        "gsm8k": GSM8KEvaluator,
        "mmlu": MMLUEvaluator,
        "bbh": BBHEvaluator,
        "cnn_dailymail": CNNDailyMailEvaluator,
        "mbpp": MBPPEvaluator,
        "mmlu_pro": MMLUProEvaluator,
        "truthfulqa": TruthfulQAEvaluator,
        "humaneval": HumanEvalEvaluator,
        "xsum": XSUMEvaluator,
        "hellaswag": HellaSwagEvaluator,
        "samsum": SamSumEvaluator,
        "meetingbank": MeetingBankEvaluator
    }
    
    evaluator_class = evaluators[args.dataset]
    evaluator = evaluator_class(args.model, args.model_version, args.temperature, args.top_p)
    
    # Change dataset path when BBH category is specified
    dataset_arg = args.dataset
    if args.dataset == "bbh" and getattr(args, "bbh_category", None):
        dataset_arg = f"agent/dataset/bbh_data/{args.bbh_category}.csv"

    # Load actual dataset to use
    dataset_loaded = evaluator.load_dataset(dataset_arg)
    # Generate sample indices (based on actual dataset size)
    sample_indices = random.sample(range(len(dataset_loaded)), min(args.num_samples, len(dataset_loaded)))

    # Run evaluation only if existing prompts are available
    base_accuracy = None
    dataset_display_name = args.bbh_category if getattr(args, "bbh_category", None) else args.dataset
    if args.base_system_prompt is not None and args.base_user_prompt is not None:
        base_results = run_single_evaluation(
            evaluator,
            dataset_loaded,
            args.base_system_prompt,
            args.base_user_prompt,
            args.num_samples,
            sample_indices,
            is_zera=False,
            num_shots=args.base_num_shots,
            dataset_display_name=dataset_display_name
        )
        base_accuracy = print_evaluation_results(base_results, "Existing prompt")
    
    # Run evaluation with Zera prompt
    zera_results = run_single_evaluation(
        evaluator,
        dataset_loaded,
        args.zera_system_prompt,
        args.zera_user_prompt,
        args.num_samples,
        sample_indices,
        is_zera=True,
        num_shots=args.zera_num_shots,
        dataset_display_name=dataset_display_name
    )
    zera_accuracy = print_evaluation_results(zera_results, "Zera prompt")
    
    # Output comparison results (only if existing prompt is available)
    if base_accuracy is not None:
        print("\n=== Final Comparison Results ===")
        print(f"Existing prompt accuracy: {base_accuracy:.2%}")
        print(f"Zera prompt accuracy: {zera_accuracy:.2%}")
        print(f"Accuracy difference (Zera - Existing): {(zera_accuracy - base_accuracy):.2%}")
        
        if "rouge_scores" in base_results and "rouge_scores" in zera_results:
            print("\nROUGE score difference (Zera - Existing):")
            for metric in base_results["rouge_scores"].keys():
                base_rouge = base_results["rouge_scores"][metric]["f"]
                zera_rouge = zera_results["rouge_scores"][metric]["f"]
                diff = zera_rouge - base_rouge
                print(f"  {metric} F1 (Existing): {base_rouge:.3f}  |  (Zera): {zera_rouge:.3f}  |  Difference: {diff:.3f}")
            # Highlight ROUGE-L separately
            if "rouge-l" in base_results["rouge_scores"]:
                base_rouge_l = base_results["rouge_scores"]["rouge-l"]["f"]
                zera_rouge_l = zera_results["rouge_scores"]["rouge-l"]["f"]
                print(f"\nROUGE-L F1 (Existing): {base_rouge_l:.3f}  |  (Zera): {zera_rouge_l:.3f}  |  Difference: {zera_rouge_l - base_rouge_l:.3f}")

if __name__ == "__main__":
    main() 