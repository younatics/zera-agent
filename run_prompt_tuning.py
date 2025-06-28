#!/usr/bin/env python3
"""
프롬프트 자동 튜닝 실행 스크립트

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

# 프로젝트 루트 디렉토리를 Python 경로에 추가
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
    """로깅 설정"""
    log_file = os.path.join(output_dir, f"prompt_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # 콘솔과 파일 모두에 로깅
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"로그 파일: {log_file}")
    return logger

def load_dataset(dataset_name, total_samples, logger):
    """데이터셋 로드"""
    logger.info(f"데이터셋 로드 중: {dataset_name}")
    
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
            test_cases.append({
                'question': item['input'],
                'expected': item['target']
            })
    
    elif dataset_name.lower() == "cnn":
        dataset = CNNDataset()
        data = dataset.load_all_data("validation")
        
        for item in data:
            normalized_expected = ' '.join(
                line.strip()
                for line in item['expected_answer'].split('\n')
                if line.strip() and not line.strip().startswith(('-', '*'))
            )
            test_cases.append({
                'question': item['input'],
                'expected': normalized_expected
            })
    
    elif dataset_name.lower() == "gsm8k":
        dataset = GSM8KDataset()
        data = dataset.load_data("test")
        
        for item in data:
            test_cases.append({
                'question': item['question'],
                'expected': item['answer']
            })
    
    elif dataset_name.lower() == "mbpp":
        dataset = MBPPDataset()
        data = dataset.get_split_data("test")
        
        for item in data:
            test_cases.append({
                'question': item['text'],
                'expected': item['code']
            })
    
    elif dataset_name.lower() == "xsum":
        dataset = XSumDataset()
        data = dataset.get_split_data("validation")
        
        for item in data:
            test_cases.append({
                'question': item['document'],
                'expected': item['summary']
            })
    
    elif dataset_name.lower() == "truthfulqa":
        dataset = TruthfulQADataset()
        data = dataset.get_split_data("test")
        
        for item in data:
            test_cases.append({
                'question': item['input'],
                'expected': item['target']
            })
    
    elif dataset_name.lower() == "hellaswag":
        dataset = HellaSwagDataset()
        data = dataset.get_split_data("validation")
        
        for item in data:
            choices_str = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(item['choices'])])
            question = f"Activity: {item['activity_label']}\nContext: {item['context']}\n\nComplete the context with the most appropriate ending:\n{choices_str}"
            test_cases.append({
                'question': question,
                'expected': chr(65 + item['answer'])
            })
    
    elif dataset_name.lower() == "humaneval":
        dataset = HumanEvalDataset()
        data = dataset.get_split_data("test")
        
        for item in data:
            test_cases.append({
                'question': item['prompt'],
                'expected': item['canonical_solution']
            })
    
    elif dataset_name.lower() == "samsum":
        dataset = SamsumDataset()
        data = dataset.get_split_data("validation")
        
        for item in data:
            test_cases.append({
                'question': item['dialogue'],
                'expected': item['summary']
            })
    
    elif dataset_name.lower() == "meetingbank":
        dataset = MeetingBankDataset()
        data = dataset.get_split_data("validation")
        
        for item in data:
            test_cases.append({
                'question': item['transcript'],
                'expected': item['summary']
            })
    
    else:
        raise ValueError(f"지원되지 않는 데이터셋: {dataset_name}")
    
    # 전체 데이터에서 샘플링
    if total_samples > 0 and total_samples < len(test_cases):
        test_cases = random.sample(test_cases, total_samples)
    
    logger.info(f"데이터셋 로드 완료: {len(test_cases)}개 샘플")
    return test_cases

def save_results(tuner, output_dir, dataset_name, config, logger):
    """결과 저장"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 설정 정보 저장
    config_file = os.path.join(output_dir, f"config_{dataset_name}_{timestamp}.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    logger.info(f"설정 저장: {config_file}")
    
    # 전체 결과 CSV 저장
    csv_data = tuner.save_results_to_csv()
    csv_file = os.path.join(output_dir, f"results_{dataset_name}_{timestamp}.csv")
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write(csv_data)
    logger.info(f"전체 결과 저장: {csv_file}")
    
    # 비용 요약 CSV 저장
    cost_csv_data = tuner.export_cost_summary_to_csv()
    cost_file = os.path.join(output_dir, f"cost_summary_{dataset_name}_{timestamp}.csv")
    with open(cost_file, 'w', encoding='utf-8') as f:
        f.write(cost_csv_data)
    logger.info(f"비용 요약 저장: {cost_file}")
    
    # 최고 성능 프롬프트 저장
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
        logger.info(f"최고 성능 프롬프트 저장: {best_prompt_file}")
        logger.info(f"최고 성능: 평균 점수 {best_result.avg_score:.3f}")

def main():
    parser = argparse.ArgumentParser(description="프롬프트 자동 튜닝 실행")
    
    # 데이터셋 설정
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["mmlu", "mmlu_pro", "bbh", "cnn", "gsm8k", "mbpp", "xsum", 
                               "truthfulqa", "hellaswag", "humaneval", "samsum", "meetingbank"],
                       help="사용할 데이터셋")
    
    # 샘플링 설정
    parser.add_argument("--total_samples", type=int, 
                       choices=[5, 20, 50, 100, 200], default=20,
                       help="전체 데이터에서 샘플링할 개수 (5, 20, 50, 100, 200)")
    
    parser.add_argument("--iteration_samples", type=int, default=5,
                       help="매 이터레이션마다 사용할 샘플 수")
    
    parser.add_argument("--iterations", type=int, default=10,
                       help="이터레이션 수")
    
    # 모델 설정
    parser.add_argument("--model", type=str, default="solar",
                       choices=["solar", "gpt4o", "claude", "local1", "local2", "solar_strawberry"],
                       help="메인 모델")
    
    parser.add_argument("--evaluator", type=str, default="solar",
                       choices=["solar", "gpt4o", "claude", "local1", "local2", "solar_strawberry"],
                       help="평가 모델")
    
    parser.add_argument("--meta_model", type=str, default="solar",
                       choices=["solar", "gpt4o", "claude", "local1", "local2", "solar_strawberry"],
                       help="메타 프롬프트 생성 모델")
    
    # 튜닝 설정
    parser.add_argument("--use_meta_prompt", action="store_true", default=True,
                       help="메타 프롬프트 사용 여부")
    
    parser.add_argument("--evaluation_threshold", type=float, default=0.8,
                       help="평가 프롬프트 점수 임계값")
    
    parser.add_argument("--score_threshold", type=float, default=None,
                       help="평균 점수 임계값 (None이면 사용 안함)")
    
    # 출력 설정
    parser.add_argument("--output_dir", type=str, default="./results",
                       help="결과 저장 디렉토리")
    
    parser.add_argument("--seed", type=int, default=42,
                       help="랜덤 시드")
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 로깅 설정
    logger = setup_logging(args.output_dir)
    
    # 랜덤 시드 설정
    random.seed(args.seed)
    logger.info(f"랜덤 시드 설정: {args.seed}")
    
    # api_client의 모델 정보 활용
    from agent.common.api_client import Model
    
    # 설정 정보 (모델 버전 정보 추가)
    config = vars(args).copy()
    config["model_version"] = Model.get_model_info(args.model)["default_version"]
    config["evaluator_version"] = Model.get_model_info(args.evaluator)["default_version"]
    config["meta_model_version"] = Model.get_model_info(args.meta_model)["default_version"]
    
    logger.info("=== 프롬프트 튜닝 시작 ===")
    logger.info(f"설정: {json.dumps(config, ensure_ascii=False, indent=2)}")
    
    try:
        # 데이터셋 로드
        test_cases = load_dataset(args.dataset, args.total_samples, logger)
        
        # PromptTuner 초기화
        logger.info("PromptTuner 초기화 중...")
        tuner = PromptTuner(
            model_name=args.model,
            model_version=config["model_version"],
            evaluator_model_name=args.evaluator,
            evaluator_model_version=config["evaluator_version"],
            meta_prompt_model_name=args.meta_model,
            meta_prompt_model_version=config["meta_model_version"]
        )
        
        # 프롬프트 파일 로드
        prompts_dir = os.path.join(os.path.dirname(__file__), 'agent', 'prompts')
        
        with open(os.path.join(prompts_dir, 'initial_system_prompt.txt'), 'r', encoding='utf-8') as f:
            initial_system_prompt = f.read()
        with open(os.path.join(prompts_dir, 'initial_user_prompt.txt'), 'r', encoding='utf-8') as f:
            initial_user_prompt = f.read()
        
        # 프로그레스 콜백 설정
        def progress_callback(iteration, test_case_index):
            progress = ((iteration - 1) * args.iteration_samples + test_case_index) / (args.iterations * args.iteration_samples)
            logger.info(f"진행도: {progress*100:.1f}% - Iteration {iteration}/{args.iterations}, Test Case {test_case_index}/{args.iteration_samples}")
        
        def iteration_callback(result):
            logger.info(f"Iteration {result.iteration} 완료 - 평균 점수: {result.avg_score:.3f}, 표준편차: {result.std_dev:.3f}")
        
        tuner.progress_callback = progress_callback
        tuner.iteration_callback = iteration_callback
        
        # 프롬프트 튜닝 실행
        logger.info("프롬프트 튜닝 실행 중...")
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
        
        # 결과 저장
        logger.info("결과 저장 중...")
        save_results(tuner, args.output_dir, args.dataset, config, logger)
        
        # 비용 요약 출력
        cost_summary = tuner.get_cost_summary()
        logger.info("=== 비용 요약 ===")
        logger.info(f"총 비용: ${cost_summary['total_cost']:.4f}")
        logger.info(f"총 토큰: {cost_summary['total_tokens']:,}")
        logger.info(f"총 시간: {cost_summary['total_duration']:.1f}초")
        logger.info(f"총 호출: {cost_summary['total_calls']}")
        
        logger.info("=== 프롬프트 튜닝 완료 ===")
        
    except Exception as e:
        logger.error(f"프롬프트 튜닝 중 오류 발생: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 