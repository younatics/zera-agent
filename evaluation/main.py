import argparse
from dotenv import load_dotenv
import os
from evaluation.gsm8k_evaluator import GSM8KEvaluator
from evaluation.mmlu_evaluator import MMLUEvaluator
from evaluation.bbh_evaluator import BBHEvaluator
from evaluation.cnn_dailymail_evaluator import CNNDailyMailEvaluator
from evaluation.samsum_evaluator import SAMSumEvaluator
from evaluation.mbpp_evaluator import MBPPEvaluator

# .env 파일 로드
load_dotenv()

def main():
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

if __name__ == "__main__":
    main() 