import argparse
from dotenv import load_dotenv
import os
import sys
from pathlib import Path
from evaluation.dataset_evaluator.gsm8k_evaluator import GSM8KEvaluator
from evaluation.dataset_evaluator.mmlu_evaluator import MMLUEvaluator
from evaluation.dataset_evaluator.bbh_evaluator import BBHEvaluator
from evaluation.dataset_evaluator.cnn_dailymail_evaluator import CNNDailyMailEvaluator
from evaluation.dataset_evaluator.samsum_evaluator import SAMSumEvaluator
from evaluation.dataset_evaluator.mbpp_evaluator import MBPPEvaluator

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

def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description="LLM 평가 스크립트")
        parser.add_argument("--dataset", type=str, required=True, 
                          choices=["gsm8k", "mmlu", "bbh", "cnn_dailymail", "samsum", "mbpp"],
                          help="평가할 데이터셋")
        parser.add_argument("--model", type=str, default="gpt4o",
                          help="사용할 모델 (기본값: gpt4o)")
        parser.add_argument("--model_version", type=str, default="gpt-3.5-turbo",
                          help="모델 버전 (기본값: gpt-3.5-turbo)")
        parser.add_argument("--system_prompt", type=str,
                          help="시스템 프롬프트")
        parser.add_argument("--user_prompt", type=str,
                          help="유저 프롬프트")
        parser.add_argument("--num_samples", type=int, default=10,
                          help="평가할 샘플 수 (기본값: 10)")
        
        args = parser.parse_args()
    
    # 평가기 선택
    evaluators = {
        "gsm8k": GSM8KEvaluator,
        "mmlu": MMLUEvaluator,
        "bbh": BBHEvaluator,
        "cnn_dailymail": CNNDailyMailEvaluator,
        "samsum": SAMSumEvaluator,
        "mbpp": MBPPEvaluator
    }
    
    evaluator_class = evaluators[args.dataset]
    evaluator = evaluator_class(args.model, args.model_version)
    
    # 평가 실행
    results = evaluator.run_evaluation(
        args.dataset,
        args.system_prompt,
        args.user_prompt,
        args.num_samples
    )
    
    print(f"\n평가 결과:")
    print(f"총 샘플 수: {results['total']}")
    print(f"정답 수: {results['correct']}")
    print(f"정확도: {results['accuracy']:.2%}")
    
    # ROUGE 점수 출력
    if "rouge_scores" in results:
        print("\nROUGE 점수 (평균):")
        for metric, scores in results["rouge_scores"].items():
            print(f"  {metric}:")
            print(f"    F1: {scores['f']:.3f}")

if __name__ == "__main__":
    main() 