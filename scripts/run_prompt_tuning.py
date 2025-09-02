#!/usr/bin/env python3
"""
Prompt Auto-tuning Execution Script

Usage:
    python run_prompt_tuning.py --dataset bbh --total_samples 20 --iteration_samples 5 --iterations 10 --model solar --evaluator solar --meta_model solar --output_dir ./results
"""

import argparse
import os
import sys
import logging
from datetime import datetime
from pathlib import Path
import json
import random

# Add project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from agent.core.prompt_tuner import PromptTuner
from agent.common.api_client import Model
from agent.dataset.mmlu_dataset import MMLUDataset
from agent.dataset.mmlu_pro_dataset import MMLUProDataset
from agent.dataset.cnn_dataset import CNNDataset
from agent.dataset.gsm8k_dataset import GSM8KDataset
from agent.dataset.mbpp_dataset import MBPPDataset
from agent.dataset.xsum_dataset import XSumDataset
from agent.dataset.bbh_dataset import BBHDataset
from agent.dataset.truthfulqa_dataset import TruthfulQADataset
from agent.dataset.hellaswag_dataset import HellaSwagDataset
from agent.dataset.humaneval_dataset import HumanEvalDataset
from agent.dataset.samsum_dataset import SamsumDataset
from agent.dataset.meetingbank_dataset import MeetingBankDataset

def setup_logging(output_dir):
    """Setup logging configuration"""
    log_file = os.path.join(output_dir, f"prompt_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Log to both console and file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Log file: {log_file}")
    return logger

def load_dataset(dataset_name, total_samples, logger):
    """Load dataset"""
    logger.info(f"Loading dataset: {dataset_name}")
    
    test_cases = []
    
    if dataset_name.lower() == "mmlu":
        dataset = MMLUDataset()
        all_subjects_data = dataset.get_all_subjects_data()
        data = []
        for subject_data in all_subjects_data.values():
            data.extend(subject_data["validation"])
        
        for item in data:
            choices_str = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(item['choices'])])
            question = f"{item['question']}\n\nChoices:\n{choices_str}"
            expected = chr(65 + item['answer']) if isinstance(item['answer'], int) else item['answer']
            test_cases.append({
                'question': question,
                'expected': expected
            })
    
    elif dataset_name.lower() == "mmlu_pro":
        dataset = MMLUProDataset()
        all_subjects_data = dataset.get_all_subjects_data()
        data = []
        for subject_data in all_subjects_data.values():
            data.extend(subject_data["validation"])
        
        for item in data:
            choices_str = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(item['choices'])])
            question = f"{item['question']}\n\nChoices:\n{choices_str}"
            expected = chr(65 + item['answer']) if isinstance(item['answer'], int) else item['answer']
            test_cases.append({
                'question': question,
                'expected': expected
            })
    
    elif dataset_name.lower() == "bbh":
        dataset = BBHDataset()
        all_data_dict = dataset.get_all_data()
        data = []
        for split_data in all_data_dict.values():
            data.extend(split_data)
        
        for item in data:
            question = item['question']
            expected = item['answer']
            test_cases.append({
                'question': question,
                'expected': expected
            })
    
    elif dataset_name.lower() == "cnn":
        dataset = CNNDataset()
        data = dataset.get_validation_data()
        
        for item in data:
            question = f"Summarize the following article:\n\n{item['article']}"
            expected = item['summary']
            test_cases.append({
                'question': question,
                'expected': expected
            })
    
    elif dataset_name.lower() == "gsm8k":
        dataset = GSM8KDataset()
        data = dataset.get_validation_data()
        
        for item in data:
            question = f"Solve the following math problem step by step:\n\n{item['question']}"
            expected = item['answer']
            test_cases.append({
                'question': question,
                'expected': expected
            })
    
    elif dataset_name.lower() == "mbpp":
        dataset = MBPPDataset()
        data = dataset.get_validation_data()
        
        for item in data:
            question = f"Write a Python function to solve the following problem:\n\n{item['prompt']}"
            expected = item['code']
            test_cases.append({
                'question': question,
                'expected': expected
            })
    
    elif dataset_name.lower() == "xsum":
        dataset = XSumDataset()
        data = dataset.get_validation_data()
        
        for item in data:
            question = f"Summarize the following article in one sentence:\n\n{item['document']}"
            expected = item['summary']
            test_cases.append({
                'question': question,
                'expected': expected
            })
    
    elif dataset_name.lower() == "truthfulqa":
        dataset = TruthfulQADataset()
        data = dataset.get_validation_data()
        
        for item in data:
            question = f"Answer the following question truthfully:\n\n{item['question']}"
            expected = item['answer']
            test_cases.append({
                'question': question,
                'expected': expected
            })
    
    elif dataset_name.lower() == "hellaswag":
        dataset = HellaSwagDataset()
        data = dataset.get_validation_data()
        
        for item in data:
            choices_str = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(item['choices'])])
            question = f"Complete the following sentence:\n\n{item['context']}\n\nChoices:\n{choices_str}"
            expected = chr(65 + item['answer']) if isinstance(item['answer'], int) else item['answer']
            test_cases.append({
                'question': question,
                'expected': expected
            })
    
    elif dataset_name.lower() == "humaneval":
        dataset = HumanEvalDataset()
        data = dataset.get_validation_data()
        
        for item in data:
            question = f"Complete the following Python function:\n\n{item['prompt']}"
            expected = item['canonical_solution']
            test_cases.append({
                'question': question,
                'expected': expected
            })
    
    elif dataset_name.lower() == "samsum":
        dataset = SamsumDataset()
        data = dataset.get_validation_data()
        
        for item in data:
            question = f"Summarize the following conversation:\n\n{item['dialogue']}"
            expected = item['summary']
            test_cases.append({
                'question': question,
                'expected': expected
            })
    
    elif dataset_name.lower() == "meetingbank":
        dataset = MeetingBankDataset()
        data = dataset.get_validation_data()
        
        for item in data:
            question = f"Summarize the following meeting transcript:\n\n{item['transcript']}"
            expected = item['summary']
            test_cases.append({
                'question': question,
                'expected': expected
            })
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Sample from entire dataset
    if total_samples > 0 and total_samples < len(test_cases):
        test_cases = random.sample(test_cases, total_samples)
    
    logger.info(f"Dataset loading completed: {len(test_cases)} samples")
    return test_cases

def save_results(tuner, output_dir, dataset_name, config, logger):
    """Save results"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save configuration information
    config_file = os.path.join(output_dir, f"config_{dataset_name}_{timestamp}.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    logger.info(f"Configuration saved: {config_file}")
    
    # Save complete results CSV
    csv_data = tuner.save_results_to_csv()
    csv_file = os.path.join(output_dir, f"results_{dataset_name}_{timestamp}.csv")
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write(csv_data)
    logger.info(f"Complete results saved: {csv_file}")
    
    # Save cost summary CSV
    cost_csv_data = tuner.export_cost_summary_to_csv()
    cost_file = os.path.join(output_dir, f"cost_summary_{dataset_name}_{timestamp}.csv")
    with open(cost_file, 'w', encoding='utf-8') as f:
        f.write(cost_csv_data)
    logger.info(f"Cost summary saved: {cost_file}")
    
    # Save best performance prompt
    if tuner.iteration_results:
        best_result = max(tuner.iteration_results, key=lambda x: x.avg_score)
        best_prompt_file = os.path.join(output_dir, f"best_prompt_{dataset_name}_{timestamp}.json")
        best_prompt_data = {
            "iteration": best_result.iteration,
            "avg_score": best_result.avg_score,
            "std_dev": best_result.std_dev,
            "top3_avg_score": best_result.top3_avg_score,
            "best_avg_score": best_result.best_avg_score,
            "best_sample_score": best_result.best_sample_score,
            "task_type": best_result.task_type,
            "task_description": best_result.task_description,
            "system_prompt": best_result.system_prompt,
            "user_prompt": best_result.user_prompt,
            "created_at": best_result.created_at.isoformat()
        }
        
        with open(best_prompt_file, 'w', encoding='utf-8') as f:
            json.dump(best_prompt_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Best performance prompt saved: {best_prompt_file}")
        logger.info(f"Best performance: Average score {best_result.avg_score:.3f}")

def main():
    parser = argparse.ArgumentParser(description="Prompt Auto-tuning Execution")
    
    # Dataset configuration
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["mmlu", "mmlu_pro", "bbh", "cnn", "gsm8k", "mbpp", "xsum", 
                               "truthfulqa", "hellaswag", "humaneval", "samsum", "meetingbank"],
                       help="Dataset to use")
    
    # Sampling configuration
    parser.add_argument("--total_samples", type=int, 
                       choices=[5, 20, 50, 100, 200], default=20,
                       help="Number of samples to sample from entire data (5, 20, 50, 100, 200)")
    
    parser.add_argument("--iteration_samples", type=int, default=5,
                       help="Number of samples to use per iteration")
    
    parser.add_argument("--iterations", type=int, default=10,
                       help="Number of iterations")
    
    # Model configuration
    parser.add_argument("--model", type=str, default="solar",
                       choices=["solar", "gpt4o", "claude", "local1", "local2", "solar_strawberry"],
                       help="Main model")
    
    parser.add_argument("--evaluator", type=str, default="solar",
                       choices=["solar", "gpt4o", "claude", "local1", "local2", "solar_strawberry"],
                       help="Evaluation model")
    
    parser.add_argument("--meta_model", type=str, default="solar",
                       choices=["solar", "gpt4o", "claude", "local1", "local2", "solar_strawberry"],
                       help="Meta prompt generation model")
    
    # Tuning configuration
    parser.add_argument("--use_meta_prompt", action="store_true", default=True,
                       help="Whether to use meta prompt")
    
    parser.add_argument("--evaluation_threshold", type=float, default=0.8,
                       help="Evaluation prompt score threshold")
    
    parser.add_argument("--score_threshold", type=float, default=None,
                       help="Average score threshold (None if not used)")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="./results",
                       help="Result storage directory")
    
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    
    # Set random seed
    random.seed(args.seed)
    logger.info(f"Random seed set: {args.seed}")
    
    # Utilize model information from api_client
    from agent.common.api_client import Model
    
    # Configuration information (add model version information)
    config = vars(args).copy()
    config["model_version"] = Model.get_model_info(args.model)["default_version"]
    config["evaluator_version"] = Model.get_model_info(args.evaluator)["default_version"]
    config["meta_model_version"] = Model.get_model_info(args.meta_model)["default_version"]
    
    logger.info("=== Prompt Tuning Started ===")
    logger.info(f"Configuration: {json.dumps(config, ensure_ascii=False, indent=2)}")
    
    try:
        # Load dataset
        test_cases = load_dataset(args.dataset, args.total_samples, logger)
        
        # Initialize PromptTuner
        logger.info("Initializing PromptTuner...")
        tuner = PromptTuner(
            model_name=args.model,
            model_version=config["model_version"],
            evaluator_model_name=args.evaluator,
            evaluator_model_version=config["evaluator_version"],
            meta_prompt_model_name=args.meta_model,
            meta_prompt_model_version=config["meta_model_version"]
        )
        
        # Load prompt files
        prompts_dir = os.path.join(os.path.dirname(__file__), 'agent', 'prompts')
        
        with open(os.path.join(prompts_dir, 'initial_system_prompt.txt'), 'r', encoding='utf-8') as f:
            initial_system_prompt = f.read()
        with open(os.path.join(prompts_dir, 'initial_user_prompt.txt'), 'r', encoding='utf-8') as f:
            initial_user_prompt = f.read()
        
        # Load and configure metaprompt files (same as Streamlit app)
        with open(os.path.join(prompts_dir, 'meta_system_prompt.txt'), 'r', encoding='utf-8') as f:
            meta_system_prompt = f.read()
        with open(os.path.join(prompts_dir, 'meta_user_prompt.txt'), 'r', encoding='utf-8') as f:
            meta_user_prompt = f.read()
        
        # Metaprompt configuration (same logic as Streamlit app)
        if meta_system_prompt.strip() and meta_user_prompt.strip():
            tuner.set_meta_prompt(meta_system_prompt, meta_user_prompt)
            logger.info("✅ Metaprompt configuration completed")
        else:
            logger.warning("⚠️ Metaprompt is empty.")
        
        # Progress callback configuration
        def progress_callback(iteration, test_case_index):
            progress = ((iteration - 1) * args.iteration_samples + test_case_index) / (args.iterations * args.iteration_samples)
            logger.info(f"Progress: {progress*100:.1f}% - Iteration {iteration}/{args.iterations}, Test Case {test_case_index}/{args.iteration_samples}")
        
        def iteration_callback(result):
            logger.info(f"Iteration {result.iteration} completed - Average score: {result.avg_score:.3f}, Standard deviation: {result.std_dev:.3f}")
        
        def best_prompt_callback(iteration, avg_score, system_prompt, user_prompt):
            """Save in real-time whenever a new best prompt is discovered"""
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            best_prompt_file = os.path.join(args.output_dir, f"best_prompt_{args.dataset}_{timestamp}.json")
            
            best_prompt_data = {
                "iteration": iteration,
                "avg_score": avg_score,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "updated_at": datetime.now().isoformat(),
                "note": "Real-time updated best prompt"
            }
            
            with open(best_prompt_file, 'w', encoding='utf-8') as f:
                json.dump(best_prompt_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"🏆 New best prompt saved: {best_prompt_file} (Iteration {iteration}, Score: {avg_score:.3f})")
        
        # Prompt change process tracking callbacks
        def prompt_improvement_start_callback(iteration, avg_score, current_system_prompt, current_user_prompt):
            """Callback when prompt improvement starts"""
            logger.info(f"\n🔄 [Iteration {iteration}] Prompt improvement started (Current score: {avg_score:.3f})")
            logger.info(f"   📋 Current system prompt: {current_system_prompt[:100]}{'...' if len(current_system_prompt) > 100 else ''}")
            logger.info(f"   📝 Current user prompt: {current_user_prompt[:100]}{'...' if len(current_user_prompt) > 100 else ''}")
        
        def meta_prompt_generated_callback(iteration, meta_prompt):
            """Callback when metaprompt generation is completed"""
            logger.info(f"\n📊 [Iteration {iteration}] Metaprompt generation completed")
            logger.info(f"   🧠 Metaprompt length: {len(meta_prompt)} characters")
            # Show only part of metaprompt (may be too long)
            meta_preview = meta_prompt[:200] + "..." if len(meta_prompt) > 200 else meta_prompt
            logger.info(f"   📜 Metaprompt preview: {meta_preview}")
        
        def prompt_updated_callback(iteration, previous_system_prompt, previous_user_prompt, 
                                  previous_task_type, previous_task_description,
                                  new_system_prompt, new_user_prompt, 
                                  new_task_type, new_task_description, raw_improved_prompts):
            """Callback when prompt update is completed"""
            logger.info(f"\n✨ [Iteration {iteration}] Prompt update completed!")
            
            # Task information changes
            if previous_task_type != new_task_type:
                logger.info(f"   🎯 Task type changed: '{previous_task_type}' → '{new_task_type}'")
            if previous_task_description != new_task_description:
                logger.info(f"   📖 Task description changed: '{previous_task_description[:50]}...' → '{new_task_description[:50]}...'")
            
            # System prompt changes
            if previous_system_prompt != new_system_prompt:
                logger.info(f"   🔧 System prompt changed:")
                logger.info(f"      Previous: {previous_system_prompt[:100]}{'...' if len(previous_system_prompt) > 100 else ''}")
                logger.info(f"      New: {new_system_prompt[:100]}{'...' if len(new_system_prompt) > 100 else ''}")
            else:
                logger.info(f"   🔧 System prompt: No changes")
            
            # User prompt changes  
            if previous_user_prompt != new_user_prompt:
                logger.info(f"   📝 User prompt changed:")
                logger.info(f"      Previous: {previous_user_prompt[:100]}{'...' if len(previous_user_prompt) > 100 else ''}")
                logger.info(f"      New: {new_user_prompt[:100]}{'...' if len(new_user_prompt) > 100 else ''}")
            else:
                logger.info(f"   📝 User prompt: No changes")
        
        tuner.progress_callback = progress_callback
        tuner.iteration_callback = iteration_callback
        tuner.best_prompt_callback = best_prompt_callback
        tuner.prompt_improvement_start_callback = prompt_improvement_start_callback
        tuner.meta_prompt_generated_callback = meta_prompt_generated_callback
        tuner.prompt_updated_callback = prompt_updated_callback
        
        # Execute prompt tuning
        logger.info("Executing prompt tuning...")
        results = tuner.tune_prompt(
            initial_system_prompt=initial_system_prompt,
            initial_user_prompt=initial_user_prompt,
            initial_test_cases=test_cases,
            num_iterations=args.iterations,
            score_threshold=args.score_threshold,
            evaluation_score_threshold=args.evaluation_threshold,
            use_meta_prompt=args.use_meta_prompt,
            num_samples=args.iteration_samples
        )
        
        # Save results
        logger.info("Saving results...")
        save_results(tuner, args.output_dir, args.dataset, config, logger)
        
        # Output cost summary
        cost_summary = tuner.get_cost_summary()
        logger.info("=== Cost Summary ===")
        logger.info(f"Total cost: ${cost_summary['total_cost']:.4f}")
        logger.info(f"Total tokens: {cost_summary['total_tokens']:,}")
        logger.info(f"Total time: {cost_summary['total_duration']:.1f} seconds")
        logger.info(f"Total calls: {cost_summary['total_calls']}")
        
        logger.info("=== Prompt Tuning Completed ===")
        
    except Exception as e:
        logger.error(f"Error occurred during prompt tuning: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 