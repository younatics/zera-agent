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
    print(f"\n{prompt_type} 평가 결과:")
    print(f"총 샘플 수: {results['total']}")
    print(f"정답 수: {results['correct']}")
    print(f"정확도: {results['accuracy']:.2%}")
    
    if "rouge_scores" in results:
        print(f"\n{prompt_type} ROUGE 점수 (평균):")
        for metric, scores in results["rouge_scores"].items():
            print(f"  {metric}:")
            print(f"    F1: {scores['f']:.3f}")
    
    return results['accuracy']

def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description="LLM 평가 스크립트")
        parser.add_argument("--dataset", type=str, required=True, 
                          choices=["gsm8k", "mmlu", "mmlu_pro", "bbh", "cnn_dailymail", "mbpp", "truthfulqa", "humaneval", "xsum", "hellaswag"],
                          help="평가할 데이터셋")
        parser.add_argument("--model", type=str, default="gpt4o",
                          help="사용할 모델 (기본값: gpt4o)")
        parser.add_argument("--model_version", type=str, default="gpt-3.5-turbo",
                          help="모델 버전 (기본값: gpt-3.5-turbo)")
        parser.add_argument("--base_system_prompt", type=str,
                          help="기존 시스템 프롬프트")
        parser.add_argument("--base_user_prompt", type=str,
                          help="기존 유저 프롬프트")
        parser.add_argument("--zera_system_prompt", type=str,
                          help="제라 시스템 프롬프트")
        parser.add_argument("--zera_user_prompt", type=str,
                          help="제라 유저 프롬프트")
        parser.add_argument("--num_samples", type=int, default=10,
                          help="평가할 샘플 수 (기본값: 10)")
        parser.add_argument("--temperature", type=float, default=0.7,
                          help="모델의 temperature 값 (기본값: 0.7)")
        parser.add_argument("--top_p", type=float, default=0.9,
                          help="모델의 top_p 값 (기본값: 0.9)")
        parser.add_argument("--base_num_shots", type=int, default=0,
                          help="기존 프롬프트에 사용할 few-shot 예시 개수 (기본값: 0)")
        parser.add_argument("--zera_num_shots", type=int, default=0,
                          help="제라 프롬프트에 사용할 few-shot 예시 개수 (기본값: 0)")
        parser.add_argument("--bbh_category", type=str, default=None,
                          help="BBH 평가 시 사용할 카테고리 이름 (예: Penguins, Geometry 등)")
        
        args = parser.parse_args()
    
    # 평가기 선택
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
        "hellaswag": HellaSwagEvaluator
    }
    
    evaluator_class = evaluators[args.dataset]
    evaluator = evaluator_class(args.model, args.model_version, args.temperature, args.top_p)
    
    # BBH 카테고리 지정 시 데이터셋 경로 변경
    dataset_arg = args.dataset
    if args.dataset == "bbh" and getattr(args, "bbh_category", None):
        dataset_arg = f"agent/dataset/bbh_data/{args.bbh_category}.csv"

    # 실제 사용할 데이터셋 로드
    dataset_loaded = evaluator.load_dataset(dataset_arg)
    # 샘플 인덱스 생성 (실제 데이터셋 크기 기준)
    sample_indices = random.sample(range(len(dataset_loaded)), min(args.num_samples, len(dataset_loaded)))

    # 기존 프롬프트가 있는 경우에만 평가 실행
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
        base_accuracy = print_evaluation_results(base_results, "기존 프롬프트")
    
    # 제라 프롬프트로 평가 실행
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
    zera_accuracy = print_evaluation_results(zera_results, "제라 프롬프트")
    
    # 비교 결과 출력 (기존 프롬프트가 있는 경우에만)
    if base_accuracy is not None:
        print("\n=== 최종 비교 결과 ===")
        print(f"기존 프롬프트 정확도: {base_accuracy:.2%}")
        print(f"제라 프롬프트 정확도: {zera_accuracy:.2%}")
        print(f"정확도 차이 (제라 - 기존): {(zera_accuracy - base_accuracy):.2%}")
        
        if "rouge_scores" in base_results and "rouge_scores" in zera_results:
            print("\nROUGE 점수 차이 (제라 - 기존):")
            for metric in base_results["rouge_scores"].keys():
                base_rouge = base_results["rouge_scores"][metric]["f"]
                zera_rouge = zera_results["rouge_scores"][metric]["f"]
                diff = zera_rouge - base_rouge
                print(f"  {metric} F1 (기존): {base_rouge:.3f}  |  (제라): {zera_rouge:.3f}  |  차이: {diff:.3f}")
            # ROUGE-L만 별도 강조
            if "rouge-l" in base_results["rouge_scores"]:
                base_rouge_l = base_results["rouge_scores"]["rouge-l"]["f"]
                zera_rouge_l = zera_results["rouge_scores"]["rouge-l"]["f"]
                print(f"\nROUGE-L F1 (기존): {base_rouge_l:.3f}  |  (제라): {zera_rouge_l:.3f}  |  차이: {zera_rouge_l - base_rouge_l:.3f}")

if __name__ == "__main__":
    main() 