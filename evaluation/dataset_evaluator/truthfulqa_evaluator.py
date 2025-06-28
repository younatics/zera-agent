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
        """TruthfulQA 평가기를 초기화합니다."""
        super().__init__(*args, **kwargs)
        self.dataset_cache = None
        self.samples_dir = Path("evaluation/samples")
        self.samples_dir.mkdir(exist_ok=True)
        self.rouge = Rouge()
        self.judge = LLMJudge()

    def load_dataset(self, dataset_path: str, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """TruthfulQA 데이터셋을 로드합니다."""
        if self.dataset_cache is None:
            # TruthfulQA 데이터셋 로드
            dataset = TruthfulQADataset()
            test_data = dataset.get_split_data("test")
            self.dataset_cache = test_data
            
        if num_samples:
            return random.sample(self.dataset_cache, min(num_samples, len(self.dataset_cache)))
        return self.dataset_cache
    
    def get_sample_indices(self, num_samples: int) -> List[int]:
        """평가에 사용할 샘플의 인덱스를 반환합니다."""
        # 데이터셋이 로드되어 있지 않으면 로드
        if self.dataset_cache is None:
            self.dataset_cache = self.load_dataset("")
        
        total_samples = len(self.dataset_cache)
        print(f"Total available samples: {total_samples}")
        
        # 중복 없이 랜덤하게 인덱스 선택
        indices = random.sample(range(total_samples), min(num_samples, total_samples))
        print(f"Selected {len(indices)} samples: {indices}")
        return indices

    def format_question(self, item: Dict[str, Any]) -> str:
        """TruthfulQA 질문을 포맷팅합니다."""
        return item['question']
    
    def evaluate_response(self, response: str, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """TruthfulQA 응답을 평가합니다."""
        try:
            # LLM 판사에게 평가를 요청
            judge_result = self.judge.evaluate(
                ground_truth['question'],
                response,
                ground_truth
            )
            
            # ROUGE 점수는 참고용으로만 계산
            rouge_scores = self.rouge.get_scores(response, ground_truth['best_answer'])[0]
            
            return {
                'is_passed': judge_result['is_passed'],
                'judge_score': judge_result['judge_score'],
                'judge_response': judge_result['judge_response'],
                'rouge_scores': rouge_scores
            }
            
        except Exception as e:
            logger.error(f"평가 중 오류 발생: {str(e)}")
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
        """전체 평가를 실행하는 메서드"""
        # 데이터셋 로드
        if self.dataset_cache is None:
            self.dataset_cache = self.load_dataset(dataset_name)
        
        # 샘플 선택
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
                # 모델 응답에서 텍스트 부분만 추출 (메타데이터 제외)
                response_data = self.model.ask(question, system_prompt, user_prompt)
                if isinstance(response_data, tuple):
                    response = response_data[0]  # 텍스트 부분만 사용
                else:
                    response = response_data  # 이미 텍스트인 경우
                eval_result = self.evaluate_response(response, item)
                
                is_correct = eval_result['is_passed']
                results["correct"] += 1 if is_correct else 0
                
                # ROUGE 점수 누적
                if eval_result['rouge_scores']:
                    for metric in ['rouge-1', 'rouge-2', 'rouge-l']:
                        results["rouge_scores"][metric]["f"] += eval_result['rouge_scores'][metric]['f']
                
                # 각 샘플의 상세 정보 저장
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
                
                # 상세 정보 출력
                print(f"\n샘플 {idx+1}/{len(dataset)}:")
                print(f"질문: {question}")
                print(f"모델 답변: {response}")
                print(f"가장 좋은 정답: {item['best_answer']}")
                print(f"정답 여부: {'정답' if is_correct else '오답'}")
                if eval_result['rouge_scores']:
                    print("ROUGE 점수:")
                    for metric, scores in eval_result['rouge_scores'].items():
                        print(f"  {metric}: F1={scores['f']:.3f}")
                print(f"판사의 평가 점수: {eval_result.get('judge_score', 0):.3f}")
                print(f"판사의 평가 설명: {eval_result.get('judge_response')}")
                print("-" * 50)
                
                logger.info(f"Processed {idx+1}/{len(dataset)} samples")
                time.sleep(1)  # API rate limit 방지
                
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {str(e)}")
                continue
                
        # ROUGE 점수 평균 계산
        for metric in results["rouge_scores"]:
            results["rouge_scores"][metric]["f"] /= results["total"]
                
        accuracy = results["correct"] / results["total"] if results["total"] > 0 else 0
        results["accuracy"] = accuracy
        
        # 결과 저장
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_file = self.results_dir / f"{self.__class__.__name__}_{timestamp}.json"
        self.save_results(results, str(result_file))
            
        return results 