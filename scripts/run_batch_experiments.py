#!/usr/bin/env python3
"""
Script to run multiple prompt tuning experiments in batch

Usage:
    python run_batch_experiments.py --config experiments_config.json
"""

import json
import subprocess
import time
import argparse
import os
from datetime import datetime
import logging
import sys
from pathlib import Path


# Import evaluation system
sys.path.append(str(Path(__file__).parent))
from evaluation.base.main import main as evaluation_main

def setup_logging():
    """Setup logging configuration"""
    log_file = f"batch_experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Custom formatter class
    class ColoredFormatter(logging.Formatter):
        """Log formatter with colors and emojis"""
        
        # ANSI color codes
        COLORS = {
            'DEBUG': '\033[36m',      # Cyan
            'INFO': '\033[32m',       # Green  
            'WARNING': '\033[33m',    # Yellow
            'ERROR': '\033[31m',      # Red
            'CRITICAL': '\033[35m',   # Magenta
            'RESET': '\033[0m'        # Reset
        }
        
        # Emoji mapping
        EMOJIS = {
            'DEBUG': '🔍',
            'INFO': '📝',
            'WARNING': '⚠️',
            'ERROR': '❌',
            'CRITICAL': '🚨'
        }
        
        def format(self, record):
            # Time format
            time_str = self.formatTime(record, '%H:%M:%S')
            
            # Level-specific colors and emojis
            level_color = self.COLORS.get(record.levelname, '')
            reset_color = self.COLORS['RESET']
            emoji = self.EMOJIS.get(record.levelname, '📝')
            
            # Add additional emojis for special keywords in messages
            message = record.getMessage()
            if 'Experiment started:' in message:
                emoji = '🚀'
            elif 'Experiment completed:' in message:
                emoji = '✅'
            elif 'Experiment failed:' in message:
                emoji = '❌'
            elif 'Evaluation started' in message:
                emoji = '🔍'
            elif 'Evaluation completed' in message:
                emoji = '🎯'
            elif 'New best' in message:
                emoji = '🏆'
            elif 'Progress:' in message:
                emoji = '⏳'
            elif 'Waiting' in message:
                emoji = '⏸️'
            elif 'Batch experiment completed' in message:
                emoji = '🎉'
            
            # Formatted log message
            formatted = f"{level_color}{emoji} [{time_str}] {message}{reset_color}"
            return formatted
    
    # Console handler (with colors)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter())
    
    # File handler (without colors)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Root logger setup
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def create_default_config():
    """Create default experiment configuration - GSM8K sample size variation experiment"""
    # Generate timestamp (based on execution time)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    config = {
        "experiments": [
            {
                "name": "GSM8K_Sample_5",
                "dataset": "gsm8k",
                "total_samples": 5,
                "iteration_samples": 5,
                "iterations": 10,
                "model": "local1",
                "evaluator": "local1",
                "meta_model": "local1",
                "output_dir": f"./results/gsm8k_sample_5_{timestamp}",
                "enabled": True
            },
            {
                "name": "GSM8K_Sample_20",
                "dataset": "gsm8k",
                "total_samples": 20,
                "iteration_samples": 5,
                "iterations": 10,
                "model": "local1",
                "evaluator": "local1",
                "meta_model": "local1",
                "output_dir": f"./results/gsm8k_sample_20_{timestamp}",
                "enabled": True
            },
            {
                "name": "GSM8K_Sample_50",
                "dataset": "gsm8k",
                "total_samples": 50,
                "iteration_samples": 5,
                "iterations": 10,
                "model": "local1",
                "evaluator": "local1",
                "meta_model": "local1",
                "output_dir": f"./results/gsm8k_sample_50_{timestamp}",
                "enabled": True
            },
            {
                "name": "GSM8K_Sample_100",
                "dataset": "gsm8k",
                "total_samples": 100,
                "iteration_samples": 5,
                "iterations": 10,
                "model": "local1",
                "evaluator": "local1",
                "meta_model": "local1",
                "output_dir": f"./results/gsm8k_sample_100_{timestamp}",
                "enabled": True
            },
            {
                "name": "GSM8K_Sample_200",
                "dataset": "gsm8k",
                "total_samples": 200,
                "iteration_samples": 5,
                "iterations": 10,
                "model": "local1",
                "evaluator": "local1",
                "meta_model": "local1",
                "output_dir": f"./results/gsm8k_sample_200_{timestamp}",
                "enabled": True
            }
        ],
        "global_settings": {
            "use_meta_prompt": True,
            "evaluation_threshold": 0.95,  # Set to high value to almost always improve
            "score_threshold": None,
            "seed": 42,
            "delay_between_experiments": 5,  # Wait time between experiments (seconds)
            "run_evaluation": True,  # Whether to run evaluation after experiment completion
            "evaluation_samples": 500  # Number of samples for evaluation
        }
    }
    return config

def run_evaluation_after_experiment(experiment_config, output_dir, global_settings, logger):
    """Run GSM8K evaluation after experiment completion"""
    try:
        # Check if evaluation should be run
        if not global_settings.get("run_evaluation", True):
            logger.info("Evaluation execution is disabled.")
            return True
        
        # Find latest config_*.json file (for model information)
        config_files = list(Path(output_dir).glob("config_*.json"))
        if not config_files:
            logger.warning(f"Configuration file not found: {output_dir}")
            return False
        
        # Select latest config file
        latest_config = max(config_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Using configuration file: {latest_config}")
        
        # Load model information from config file
        with open(latest_config, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # Find latest best_prompt_*.json file
        best_prompt_files = list(Path(output_dir).glob("best_prompt_*.json"))
        if not best_prompt_files:
            logger.warning(f"Best performance prompt file not found: {output_dir}")
            return False
        
        # Select latest file
        latest_best_prompt = max(best_prompt_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Using best performance prompt file: {latest_best_prompt}")
        
        # Load best performance prompt
        with open(latest_best_prompt, 'r', encoding='utf-8') as f:
            best_prompt_data = json.load(f)
        
        system_prompt = best_prompt_data.get('system_prompt', '')
        user_prompt = best_prompt_data.get('user_prompt', '')
        avg_score = best_prompt_data.get('avg_score', 0.0)
        
        # Validate prompts
        if not system_prompt or not user_prompt:
            logger.error("Prompts are empty. Skipping evaluation.")
            return False
        
        print(f"🏆 Using best performance prompt (tuning score: {avg_score:.3f})")
        print(f"📝 System prompt: {len(system_prompt)} characters")
        print(f"📝 User prompt: {len(user_prompt)} characters")
        print(f"🤖 Model used: {config_data.get('model', 'solar')} (version: {config_data.get('model_version', 'N/A')})")
        print(f"📊 Evaluation samples: {global_settings.get('evaluation_samples', 500)}")
        
        # Import model information from api_client
        from agent.common.api_client import Model as ApiModel
        
        # Create argparse.Namespace object
        class EvaluationArgs:
            def __init__(self):
                self.dataset = "gsm8k"
                # Use model information from config file
                self.model = config_data.get("model", "local1")
                
                # Get model_version from config, or use default from api_client if not available
                model_name = config_data.get("model", "local1")
                self.model_version = config_data.get("model_version") or ApiModel.get_model_info(model_name)["default_version"]
                
                self.base_system_prompt = None
                self.base_user_prompt = None
                self.zera_system_prompt = system_prompt
                self.zera_user_prompt = user_prompt
                self.num_samples = global_settings.get("evaluation_samples", 500)
                self.temperature = 0.7
                self.top_p = 0.9
                self.base_num_shots = 0
                self.zera_num_shots = 0
                self.bbh_category = None
        
        eval_args = EvaluationArgs()
        
        # Execute evaluation and capture results
        print(f"🔍 Running GSM8K evaluation... (sample count: {eval_args.num_samples})")
        
        # Capture evaluation results to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        eval_output_file = os.path.join(output_dir, f"evaluation_output_{timestamp}.txt")
        
        # Backup original sys.argv and stdout
        original_argv = sys.argv.copy()
        original_stdout = sys.stdout
        
        try:
            # Redirect stdout to file
            with open(eval_output_file, 'w', encoding='utf-8') as f:
                sys.stdout = f
                # Call evaluation_main function
                evaluation_main(eval_args)
            
            # Extract accuracy from evaluation result file
            accuracy = extract_accuracy_from_output(eval_output_file)
            
            print(f"🎯 GSM8K evaluation completed!")
            print(f"📊 Accuracy: {accuracy:.2%}")
            print(f"💾 Result file: {eval_output_file}")
            
            # Save evaluation summary information
            eval_summary = {
                "experiment_name": experiment_config["name"],
                "tuning_avg_score": avg_score,
                "gsm8k_accuracy": accuracy,
                "evaluation_samples": eval_args.num_samples,
                "timestamp": timestamp,
                "best_prompt_file": str(latest_best_prompt),
                "output_file": eval_output_file
            }
            
            eval_summary_file = os.path.join(output_dir, f"evaluation_summary_{timestamp}.json")
            with open(eval_summary_file, 'w', encoding='utf-8') as f:
                json.dump(eval_summary, f, ensure_ascii=False, indent=2)
            
            print(f"📋 Summary saved: {eval_summary_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error during evaluation execution: {str(e)}")
            return False
        finally:
            # Restore stdout
            sys.stdout = original_stdout
            # Restore sys.argv
            sys.argv = original_argv
            
    except Exception as e:
        logger.error(f"Error during evaluation preparation: {str(e)}")
        return False

def extract_accuracy_from_output(output_file):
    """Extract accuracy from evaluation result file"""
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find "Accuracy: XX.XX%" pattern
        import re
        accuracy_match = re.search(r'Accuracy:\s*([\d.]+)%', content)
        if accuracy_match:
            return float(accuracy_match.group(1)) / 100.0
        
        # Find "Zera prompt accuracy: XX.XX%" pattern
        zera_accuracy_match = re.search(r'Zera prompt.*?accuracy:\s*([\d.]+)%', content)
        if zera_accuracy_match:
            return float(zera_accuracy_match.group(1)) / 100.0
            
        return 0.0
        
    except Exception as e:
        print(f"Error extracting accuracy: {str(e)}")
        return 0.0

def print_experiment_header(experiment_name, experiment_num, total_experiments, logger):
    """Print experiment start header"""
    print("\n" + "="*80)
    print(f"🚀 Experiment {experiment_num}/{total_experiments}: {experiment_name}")
    print("="*80)

def print_section_divider(title, logger):
    """Print section divider"""
    print(f"\n{'─'*60}")
    print(f"🔸 {title}")
    print("─"*60)

def run_experiment(experiment_config, global_settings, logger):
    """Execute single experiment"""
    experiment_name = experiment_config["name"]
    
    # Create output directory
    output_dir = experiment_config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    # Build command
    cmd = [
        "python3", "run_prompt_tuning.py",
        "--dataset", experiment_config["dataset"],
        "--total_samples", str(experiment_config["total_samples"]),
        "--iteration_samples", str(experiment_config["iteration_samples"]),
        "--iterations", str(experiment_config["iterations"]),
        "--model", experiment_config["model"],
        "--evaluator", experiment_config["evaluator"],
        "--meta_model", experiment_config["meta_model"],
        "--output_dir", output_dir,
        "--seed", str(global_settings["seed"])
    ]
    
    # Add global settings
    if global_settings["use_meta_prompt"]:
        cmd.append("--use_meta_prompt")
    
    cmd.extend(["--evaluation_threshold", str(global_settings["evaluation_threshold"])])
    
    if global_settings["score_threshold"] is not None:
        cmd.extend(["--score_threshold", str(global_settings["score_threshold"])])
    
    logger.info(f"Execution command: {' '.join(cmd)}")
    
    print_section_divider("Prompt Tuning Execution", logger)
    print(f"🔥 Command: {' '.join(cmd)}")
    
    # Execute experiment
    start_time = time.time()
    try:
        print("🚀 Starting prompt tuning...")
        print("─" * 60)
        print("📝 Real-time log:")
        print("─" * 60)
        
        # Change to capture_output=False for real-time output
        result = subprocess.run(cmd, text=True, encoding='utf-8')
        end_time = time.time()
        duration = end_time - start_time
        
        print("─" * 60)
        if result.returncode == 0:
            print(f"✅ Prompt tuning completed! (Time taken: {duration:.1f} seconds)")
            
            # Run evaluation after successful experiment
            print_section_divider("GSM8K Evaluation Execution", logger)
            eval_success = run_evaluation_after_experiment(experiment_config, output_dir, global_settings, logger)
            if eval_success:
                print("🎯 Evaluation completed!")
            else:
                print("⚠️ Evaluation failed")
                
        else:
            print(f"❌ Prompt tuning failed! (Return code: {result.returncode})")
            
        return result.returncode == 0, duration
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"❌ Exception occurred during experiment execution: {str(e)}")
        logger.error(f"Exception details: {str(e)}")
        return False, duration

def main():
    parser = argparse.ArgumentParser(description="Batch prompt tuning experiment execution")
    parser.add_argument("--config", type=str, default="experiments_config.json",
                       help="Experiment configuration JSON file")
    parser.add_argument("--create_config", action="store_true",
                       help="Create default configuration file")
    parser.add_argument("--dry_run", action="store_true",
                       help="Output commands only without actual execution")
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    # Create default configuration file
    if args.create_config:
        config = create_default_config()
        with open(args.config, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        logger.info(f"Default configuration file created: {args.config}")
        return
    
    # Load configuration file (use default if not exists)
    if os.path.exists(args.config):
        logger.info(f"Using configuration file: {args.config}")
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        logger.info("Configuration file not found, using default configuration.")
        config = create_default_config()
    
    experiments = config["experiments"]
    global_settings = config["global_settings"]
    
    # Filter only enabled experiments
    enabled_experiments = [exp for exp in experiments if exp.get("enabled", True)]
    
    logger.info(f"Executing {len(enabled_experiments)} experiments.")
    
    if args.dry_run:
        logger.info("=== DRY RUN MODE ===")
        for i, experiment in enumerate(enabled_experiments):
            logger.info(f"Experiment {i+1}: {experiment['name']}")
            logger.info(f"  Dataset: {experiment['dataset']}")
            logger.info(f"  Total samples: {experiment['total_samples']}")
            logger.info(f"  Iteration samples: {experiment['iteration_samples']}")
            logger.info(f"  Iterations: {experiment['iterations']}")
            logger.info(f"  Model: {experiment['model']}")
            logger.info(f"  Output directory: {experiment['output_dir']}")
        return
    
    # Batch experiment start header
    print("\n" + "🎯" + "="*78 + "🎯")
    print(f"🎯 Starting batch prompt tuning experiments! ({len(enabled_experiments)} experiments)")
    print("🎯" + "="*78 + "🎯")
    
    # Execute experiments
    total_start_time = time.time()
    successful_experiments = 0
    failed_experiments = 0
    
    for i, experiment in enumerate(enabled_experiments):
        # Print experiment header
        print_experiment_header(experiment['name'], i+1, len(enabled_experiments), logger)
        
        # Print experiment configuration summary
        print(f"📊 Dataset: {experiment['dataset']}")
        print(f"📈 Total samples: {experiment['total_samples']}")
        print(f"🔄 Iterations: {experiment['iterations']}")
        print(f"🤖 Model: {experiment['model']}")
        print(f"📁 Output: {experiment['output_dir']}")
        
        progress_percent = ((i) / len(enabled_experiments)) * 100
        progress_bar = "█" * int(progress_percent // 5) + "░" * (20 - int(progress_percent // 5))
        print(f"\nOverall progress: [{progress_bar}] {progress_percent:.1f}%")
        
        success, duration = run_experiment(experiment, global_settings, logger)
        
        if success:
            successful_experiments += 1
            print(f"\n✅ Experiment completed! (Time taken: {duration:.1f} seconds)")
        else:
            failed_experiments += 1
            print(f"\n❌ Experiment failed! (Time taken: {duration:.1f} seconds)")
        
        # Wait before next experiment (if not the last one)
        if i < len(enabled_experiments) - 1:
            delay = global_settings.get("delay_between_experiments", 60)
            print(f"\n⏸️ Waiting {delay} seconds until next experiment...")
            
            # Display countdown
            for remaining in range(delay, 0, -1):
                print(f"\r⏳ Waiting... {remaining} seconds remaining", end="", flush=True)
                time.sleep(1)
            print("\r" + " " * 30 + "\r", end="")  # Clear line
    
    total_duration = time.time() - total_start_time
    
    # Final summary header
    print("\n" + "🎉" + "="*78 + "🎉")
    print("🎉 Batch experiments completed!")
    print("🎉" + "="*78 + "🎉")
    
    # Format execution time
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = int(total_duration % 60)
    
    if hours > 0:
        time_str = f"{hours} hours {minutes} minutes {seconds} seconds"
    elif minutes > 0:
        time_str = f"{minutes} minutes {seconds} seconds"
    else:
        time_str = f"{seconds} seconds"
    
    # Final summary
    print(f"⏱️  Total execution time: {time_str}")
    print(f"✅ Successful experiments: {successful_experiments}")
    print(f"❌ Failed experiments: {failed_experiments}")
    print(f"📊 Total experiments: {len(enabled_experiments)}")
    
    # Calculate and display success rate
    if len(enabled_experiments) > 0:
        success_rate = (successful_experiments / len(enabled_experiments)) * 100
        success_bar = "█" * int(success_rate // 5) + "░" * (20 - int(success_rate // 5))
        print(f"📈 Success rate: [{success_bar}] {success_rate:.1f}%")
    
    # Generate evaluation result summary
    if global_settings.get("run_evaluation", True):
        print_section_divider("Evaluation Result Summary", logger)
        generate_evaluation_summary(enabled_experiments, logger)

def generate_evaluation_summary(experiments, logger):
    """Generate evaluation result summary for all experiments"""
    summary_data = []
    
    for experiment in experiments:
        output_dir = experiment["output_dir"]
        
        # Find evaluation summary files
        eval_summary_files = list(Path(output_dir).glob("evaluation_summary_*.json"))
        if eval_summary_files:
            # Select latest file
            latest_summary = max(eval_summary_files, key=lambda p: p.stat().st_mtime)
            
            try:
                with open(latest_summary, 'r', encoding='utf-8') as f:
                    eval_data = json.load(f)
                summary_data.append(eval_data)
            except Exception as e:
                logger.warning(f"Failed to read evaluation summary file: {latest_summary} - {str(e)}")
    
    if summary_data:
        # Sort results by GSM8K accuracy
        summary_data.sort(key=lambda x: x.get("gsm8k_accuracy", 0), reverse=True)
        
        print("\n🏆 Performance ranking by experiment:")
        print("─" * 80)
        print(f"{'Rank':<4} {'Experiment':<20} {'Tuning Score':<12} {'GSM8K Accuracy':<15} {'Samples':<8}")
        print("─" * 80)
        
        for i, data in enumerate(summary_data, 1):
            exp_name = data.get("experiment_name", "N/A")[:18]
            tuning_score = data.get("tuning_avg_score", 0)
            gsm8k_acc = data.get("gsm8k_accuracy", 0)
            samples = data.get("evaluation_samples", 0)
            
            # Medal emoji by rank
            if i == 1:
                rank_emoji = "🥇"
            elif i == 2:
                rank_emoji = "🥈"
            elif i == 3:
                rank_emoji = "🥉"
            else:
                rank_emoji = f"{i:2d}"
            
            print(f"{rank_emoji:<4} {exp_name:<20} {tuning_score:<12.3f} {gsm8k_acc:<15.2%} {samples:<8}")
        
        print("─" * 80)
        
        # Highlight best performance
        best_data = summary_data[0]
        print(f"\n🎖️  Best performance: {best_data.get('experiment_name', 'N/A')}")
        print(f"   📊 GSM8K accuracy: {best_data.get('gsm8k_accuracy', 0):.2%}")
        print(f"   🎯 Tuning score: {best_data.get('tuning_avg_score', 0):.3f}")
        
        # Create overall summary CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Check/create results folder
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        summary_csv_file = results_dir / f"batch_evaluation_summary_{timestamp}.csv"
        
        try:
            import pandas as pd
            df = pd.DataFrame(summary_data)
            df.to_csv(summary_csv_file, index=False, encoding='utf-8')
            print(f"\n💾 Overall evaluation summary CSV saved: {summary_csv_file}")
        except ImportError:
            logger.warning("⚠️ pandas not installed, skipping CSV file creation.")
        except Exception as e:
            logger.error(f"❌ Error creating CSV file: {str(e)}")
    else:
        print("⚠️ No evaluation result summary data found.")

if __name__ == "__main__":
    main() 