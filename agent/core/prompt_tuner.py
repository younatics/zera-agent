from typing import List, Dict, Optional, Callable
import logging
from agent.common.api_client import Model
import os
import random
import statistics
import csv
from datetime import datetime
import io
import json
from .iteration_result import IterationResult, TestCaseResult
import pandas as pd

class PromptTuner:
    """
    A class for automatically fine-tuning system prompts for LLMs.
    """
    
    def __init__(self, model_name: str = "solar", evaluator_model_name: str = "solar", meta_prompt_model_name: str = "solar", model_version: str = None, evaluator_model_version: str = None, meta_prompt_model_version: str = None):
        """
        Initialize the PromptTuner with specific models.
        
        Args:
            model_name (str): The name of the model to use for tuning (default: "solar")
            evaluator_model_name (str): The name of the model to use for evaluation (default: "solar")
            meta_prompt_model_name (str): The name of the model to use for meta prompt generation (default: "solar")
            model_version (str): The version of the model to use for tuning (default: None)
            evaluator_model_version (str): The version of the model to use for evaluation (default: None)
            meta_prompt_model_version (str): The version of the model to use for meta prompt generation (default: None)
        """
        self.model = Model(model_name, version=model_version)
        self.evaluator = Model(evaluator_model_name, version=evaluator_model_version)
        self.meta_prompt_model = Model(meta_prompt_model_name, version=meta_prompt_model_version)
        self.iteration_results = []
        self.progress_callback = None
        self.iteration_callback = None
        # 새로운 콜백들 추가
        self.prompt_improvement_start_callback = None
        self.meta_prompt_generated_callback = None
        self.prompt_updated_callback = None
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 비용 트랙킹을 위한 통계 초기화
        self.model_stats = self._initialize_stats("모델 호출")
        self.evaluator_stats = self._initialize_stats("평가자 호출") 
        self.meta_prompt_stats = self._initialize_stats("메타 프롬프트 생성")
        
        # 프롬프트 파일 경로
        prompts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompts')
        
        # 기본 initial_system_prompt 로드
        with open(os.path.join(prompts_dir, 'initial_system_prompt.txt'), 'r', encoding='utf-8') as f:
            self.initial_system_prompt = f.read()
        # 기본 initial_user_prompt 로드
        with open(os.path.join(prompts_dir, 'initial_user_prompt.txt'), 'r', encoding='utf-8') as f:
            self.initial_user_prompt = f.read()
        
        # 기본 평가 프롬프트 로드
        with open(os.path.join(prompts_dir, 'evaluation_system_prompt.txt'), 'r', encoding='utf-8') as f:
            self.evaluation_system_prompt_template = f.read()
        with open(os.path.join(prompts_dir, 'evaluation_user_prompt.txt'), 'r', encoding='utf-8') as f:
            self.evaluation_user_prompt_template = f.read()
        
        # 기본 메타프롬프트 로드
        with open(os.path.join(prompts_dir, 'meta_system_prompt.txt'), 'r', encoding='utf-8') as f:
            self.meta_system_prompt_template = f.read()
        with open(os.path.join(prompts_dir, 'meta_user_prompt.txt'), 'r', encoding='utf-8') as f:
            self.meta_user_prompt_template = f.read()
    
    def _initialize_stats(self, stat_type: str) -> Dict:
        """통계 추적을 위한 딕셔너리를 초기화합니다."""
        return {
            "type": stat_type,
            "total_calls": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "total_duration": 0.0,
            "calls_by_iteration": {}
        }

    def _update_stats(self, stats: Dict, metadata: Dict, iteration: int = None):
        """통계를 업데이트합니다."""
        stats["total_calls"] += 1
        stats["total_input_tokens"] += metadata.get("input_tokens", 0)
        stats["total_output_tokens"] += metadata.get("output_tokens", 0)
        stats["total_tokens"] += metadata.get("total_tokens", 0)
        stats["total_cost"] += metadata.get("cost", 0.0)
        stats["total_duration"] += metadata.get("duration", 0.0)
        
        if iteration is not None:
            if iteration not in stats["calls_by_iteration"]:
                stats["calls_by_iteration"][iteration] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost": 0.0,
                    "duration": 0.0
                }
            
            iteration_stats = stats["calls_by_iteration"][iteration]
            iteration_stats["calls"] += 1
            iteration_stats["input_tokens"] += metadata.get("input_tokens", 0)
            iteration_stats["output_tokens"] += metadata.get("output_tokens", 0)
            iteration_stats["total_tokens"] += metadata.get("total_tokens", 0)
            iteration_stats["cost"] += metadata.get("cost", 0.0)
            iteration_stats["duration"] += metadata.get("duration", 0.0)

    def set_evaluation_prompt(self, system_prompt_template: str, user_prompt_template: str):
        """
        평가 프롬프트 템플릿을 설정합니다.
        
        Args:
            system_prompt_template (str): 평가 시스템 프롬프트 템플릿
            user_prompt_template (str): 평가 유저 프롬프트 템플릿
        """
        self.evaluation_system_prompt_template = system_prompt_template
        self.evaluation_user_prompt_template = user_prompt_template
    
    def set_meta_prompt(self, system_prompt_template: str, user_prompt_template: str):
        """
        메타프롬프트 템플릿을 설정합니다.
        
        Args:
            system_prompt_template (str): 메타 시스템 프롬프트 템플릿
            user_prompt_template (str): 메타 유저 프롬프트 템플릿
        """
        self.meta_system_prompt_template = system_prompt_template
        self.meta_user_prompt_template = user_prompt_template
    
    def set_initial_prompt(self, system_prompt: str, user_prompt: str):
        """
        초기 프롬프트를 설정합니다.
        
        Args:
            system_prompt (str): 초기 시스템 프롬프트
            user_prompt (str): 초기 유저 프롬프트
        """
        self.initial_system_prompt = system_prompt
        self.initial_user_prompt = user_prompt
    
    def _evaluate_output(self, output: str, expected: str, question: str, task_type: str, task_description: str, iteration: int = None) -> tuple[float, Dict]:
        """
        Evaluate an output using the evaluator model.
        
        Args:
            output (str): The actual output to evaluate
            expected (str): The expected output
            question (str): The original question
            task_type (str): The type of task being evaluated
            task_description (str): The description of the task being evaluated
            iteration (int): Current iteration number for cost tracking
            
        Returns:
            tuple[float, Dict]: A tuple containing the score and evaluation details
        """
        try:
            # 평가 유저 프롬프트 생성
            evaluation_prompt = self.evaluation_user_prompt_template.format(
                response=output,
                expected=expected,
                question=question,
                task_type=task_type,
                task_description=task_description
            )
            
            # 평가 시스템 프롬프트 생성
            evaluation_system_prompt = self.evaluation_system_prompt_template.format(
                task_type=task_type,
                task_description=task_description
            )
            
            # 평가 모델로 평가 수행 및 통계 업데이트
            evaluation, metadata = self.evaluator.ask(
                question=evaluation_prompt,
                system_prompt=evaluation_system_prompt
            )
            
            # 평가자 통계 업데이트
            self._update_stats(self.evaluator_stats, metadata, iteration)
            self.logger.info(f"Evaluating output:")
            self.logger.info(f"Question: {question}")
            self.logger.info(f"Actual output: {output}")
            self.logger.info(f"Expected output: {expected}")
            self.logger.info(f"Evaluation: {evaluation}")
            
            # JSON 문자열 추출 및 파싱
            try:
                # 응답에서 JSON 부분만 추출
                evaluation = evaluation.strip()
                json_start = evaluation.find('{')
                json_end = evaluation.rfind('}') + 1
                
                if json_start == -1 or json_end == 0:
                    raise ValueError("응답에서 JSON 객체를 찾을 수 없습니다")
                
                json_str = evaluation[json_start:json_end]
                
                # JSON 파싱
                evaluation_data = json.loads(json_str)
                
                # 필수 필드 검증
                if 'scores' not in evaluation_data:
                    raise ValueError("JSON 응답에 scores 필드가 없습니다")
                
                # 카테고리별 점수와 가중치 추출
                scores_data = evaluation_data.get('scores', {})
                evaluation_details = {'category_scores': {}}
                total_weighted_score = 0.0
                total_weight = 0.0
                
                # 각 카테고리별 점수와 피드백 정보 추출
                for category, details in scores_data.items():
                    if isinstance(details, str):
                        # 문자열인 경우 (예: 'PASS', 'FAIL') 직접 변환
                        score = self._convert_to_float(details)
                        weight = 0.5  # 기본 가중치를 0.5로 변경
                        evaluation_details['category_scores'][category] = {
                            'score': score,
                            'current_state': details,
                            'improvement_action': '',
                            'weight': weight
                        }
                    else:
                        # 딕셔너리인 경우 기존 로직 사용
                        score = self._convert_to_float(details.get('score', 0))
                        weight = self._convert_to_float(details.get('weight', 0.5))  # 기본 가중치를 0.5로 변경
                        evaluation_details['category_scores'][category] = {
                            'score': score,
                            'current_state': details.get('current_state', ''),
                            'improvement_action': details.get('improvement_action', ''),
                            'weight': weight
                        }
                    
                    # 가중치가 적용된 점수 누적
                    total_weighted_score += score * weight
                    total_weight += weight
                
                # 최종 점수 계산 (가중치 합으로 나누어 정규화)
                final_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
                
                self.logger.info(f"Evaluation score: {final_score}")
                self.logger.info(f"Evaluation details: {evaluation_details}")
                
                return final_score, evaluation_details
                
            except (ValueError, TypeError, json.JSONDecodeError) as e:
                self.logger.error(f"평가 중 오류 발생: {str(e)}")
                # 오류 발생 시 기본값 반환
                return 0.0, {'final_score': 0.0, 'category_scores': {}, 'error': str(e)}
            
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            self.logger.error(f"Error during evaluation: {str(e)}")
            return 0.0, {'final_score': 0.0, 'category_scores': {}}

    def _convert_to_float(self, value) -> float:
        """
        Convert a value to float, handling special cases like 'PASS'
        
        Args:
            value: The value to convert
            
        Returns:
            float: The converted value
        """
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            value = value.upper().strip()
            if value == 'PASS':
                return 0.5
            elif value == 'FAIL':
                return 0.0
            try:
                return float(value)
            except (ValueError, TypeError):
                return 0.0
        return 0.0

    def tune_prompt(self, initial_system_prompt: str, initial_user_prompt: str, initial_test_cases: List[Dict], num_iterations: int = 3, score_threshold: Optional[float] = None, evaluation_score_threshold: float = 0.8, use_meta_prompt: bool = True, num_samples: Optional[int] = None) -> List[IterationResult]:
        """
        Tune a system prompt using a set of test cases.
        
        Args:
            initial_system_prompt (str): The initial system prompt
            initial_user_prompt (str): The initial user prompt
            initial_test_cases (List[Dict]): List of test cases, each containing 'question' and 'expected'
            num_iterations (int): Number of iterations to perform
            score_threshold (Optional[float]): Threshold to stop tuning if average score exceeds this value
            evaluation_score_threshold (float): Threshold to trigger prompt improvement
            use_meta_prompt (bool): Whether to use meta prompt for improvement
            num_samples (Optional[int]): Number of samples to use for evaluate prompt
            
        Returns:
            List[IterationResult]: List of iteration results
        """
        current_system_prompt = initial_system_prompt
        current_user_prompt = initial_user_prompt
        best_system_prompt = initial_system_prompt
        best_user_prompt = initial_user_prompt
        best_avg_score = 0.0
        
        # 초기 프롬프트 로깅
        self.logger.info(f"📝 초기 프롬프트:")
        self.logger.info(f"   시스템 프롬프트: {initial_system_prompt[:200]}{'...' if len(initial_system_prompt) > 200 else ''}")
        self.logger.info(f"   유저 프롬프트: {initial_user_prompt[:200]}{'...' if len(initial_user_prompt) > 200 else ''}")
        
        # 초기 task_type과 task_description 설정
        current_task_type = "General Task"
        current_task_description = "General task requiring outputs to various questions"
        
        # 튜닝 설정 정보 로깅
        self.logger.info(f"🎯 프롬프트 튜닝 설정:")
        self.logger.info(f"   num_iterations: {num_iterations}")
        self.logger.info(f"   score_threshold: {score_threshold}")
        self.logger.info(f"   evaluation_score_threshold: {evaluation_score_threshold}")
        self.logger.info(f"   use_meta_prompt: {use_meta_prompt}")
        self.logger.info(f"   num_samples: {num_samples}")
        self.logger.info(f"   총 테스트 케이스 수: {len(initial_test_cases)}")
        
        for iteration in range(num_iterations):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"🔄 Iteration {iteration + 1}/{num_iterations} 시작")
            self.logger.info(f"{'='*60}")
            self.logger.info(f"📋 현재 이터레이션 프롬프트:")
            self.logger.info(f"   시스템: {current_system_prompt[:150]}{'...' if len(current_system_prompt) > 150 else ''}")
            self.logger.info(f"   유저: {current_user_prompt[:150]}{'...' if len(current_user_prompt) > 150 else ''}")
            self.logger.info(f"   태스크 타입: {current_task_type}")
            self.logger.info(f"   태스크 설명: {current_task_description[:100]}{'...' if len(current_task_description) > 100 else ''}")
            
            iteration_scores = []
            test_case_results = []
            iteration_best_sample_score = 0.0  # 이터레이션별 최고 점수 초기화
            
            # 각 이터레이션마다 랜덤 샘플링
            test_cases = random.sample(initial_test_cases, num_samples) if num_samples is not None and num_samples < len(initial_test_cases) else initial_test_cases
            
            # 테스트 케이스 실행 및 평가
            for i, test_case in enumerate(test_cases):
                self.logger.info(f"\nTest Case {i}/{len(test_cases)}")
                self.logger.info(f"Question: {test_case['question']}")
                
                # 현재 프롬프트로 출력 생성 및 통계 업데이트
                output, model_metadata = self.model.ask(test_case['question'], system_prompt=current_system_prompt, user_prompt=current_user_prompt)
                self._update_stats(self.model_stats, model_metadata, iteration + 1)
                self.logger.info(f"Output: {output}")
                
                # 출력 평가
                score, evaluation_details = self._evaluate_output(
                    output=output,
                    expected=test_case['expected'],
                    question=test_case['question'],
                    task_type=current_task_type,
                    task_description=current_task_description,
                    iteration=iteration + 1
                )
                self.logger.info(f"📊 Score: {score}")
                self.logger.info(f"📝 Evaluation details: {evaluation_details}")
                
                # 점수와 출력 저장
                iteration_scores.append(score)
                self.logger.info(f"🎯 테스트 케이스 {i+1}/{len(test_cases)} 완료 - 점수: {score}")
                
                # TestCaseResult 생성
                test_case_result = TestCaseResult(
                    test_case=i,
                    question=test_case['question'],
                    expected_output=test_case['expected'],
                    actual_output=output,
                    score=score,
                    evaluation_details=evaluation_details if isinstance(evaluation_details, dict) else {'error': str(evaluation_details), 'category_scores': {}}
                )
                test_case_results.append(test_case_result)
                
                # 최고 개별 점수 업데이트
                if score is not None and score > iteration_best_sample_score:
                    iteration_best_sample_score = score
                
                # 프로그레스 바 업데이트
                if self.progress_callback:
                    self.progress_callback(iteration + 1, i + 1)
            
            # iteration이 끝난 후 평균 점수 계산
            valid_scores = [score for score in iteration_scores if score is not None]
            self.logger.info(f"\n📊 Iteration {iteration + 1} 점수 요약:")
            self.logger.info(f"   전체 점수: {iteration_scores}")
            self.logger.info(f"   유효 점수: {valid_scores} (총 {len(valid_scores)}개)")
            
            if valid_scores:
                avg_score = sum(valid_scores) / len(valid_scores)
                # 표준편차 계산
                std_dev = statistics.stdev(valid_scores) if len(valid_scores) > 1 else 0.0
                # top3 평균 점수 계산
                top3_scores = sorted(valid_scores, reverse=True)[:3]
                top3_avg_score = sum(top3_scores) / len(top3_scores)
                
                self.logger.info(f"   평균 점수: {avg_score:.3f}")
                self.logger.info(f"   표준편차: {std_dev:.3f}")
                self.logger.info(f"   Top3 점수: {top3_scores}")
                self.logger.info(f"   Top3 평균: {top3_avg_score:.3f}")
            else:
                avg_score = 0.0
                std_dev = 0.0
                top3_avg_score = 0.0
                self.logger.warning(f"   ⚠️ 유효한 점수가 없습니다!")
            
            # 현재까지의 최고 평균 점수와 비교
            self.logger.info(f"🏆 베스트 점수 비교: 현재 {avg_score:.3f} vs 이전 최고 {best_avg_score:.3f}")
            if avg_score > best_avg_score:
                self.logger.info(f"🎉 새로운 베스트 점수 달성! {best_avg_score:.3f} → {avg_score:.3f}")
                best_avg_score = avg_score
                best_system_prompt = current_system_prompt
                best_user_prompt = current_user_prompt
                
                # 실시간 베스트 프롬프트 저장을 위한 콜백 호출
                if hasattr(self, 'best_prompt_callback') and self.best_prompt_callback:
                    self.best_prompt_callback(iteration + 1, avg_score, current_system_prompt, current_user_prompt)
            else:
                self.logger.info(f"📊 베스트 점수 유지: {best_avg_score:.3f} (현재: {avg_score:.3f})")
            
            # IterationResult 생성
            iteration_result = IterationResult(
                iteration=iteration + 1,
                system_prompt=current_system_prompt,
                user_prompt=current_user_prompt,
                avg_score=avg_score,
                std_dev=std_dev,
                top3_avg_score=top3_avg_score,
                best_avg_score=best_avg_score,
                best_sample_score=iteration_best_sample_score,
                test_case_results=test_case_results,
                meta_prompt=None,
                task_type=current_task_type,
                task_description=current_task_description
            )
            self.iteration_results.append(iteration_result)
            
            # 점수 임계값 체크
            if score_threshold is not None and avg_score >= score_threshold:
                self.logger.info(f"평균 점수가 임계값({score_threshold}) 이상입니다. 튜닝을 종료합니다.")
                if self.progress_callback:
                    self.progress_callback(num_iterations, len(test_cases))

                if self.iteration_callback:
                    self.iteration_callback(iteration_result)

                break
            
            # 프롬프트 개선 (평균 점수가 평가 임계값 미만인 경우)
            self.logger.info(f"🔍 메타프롬프트 트리거 조건 체크:")
            self.logger.info(f"   use_meta_prompt: {use_meta_prompt}")
            self.logger.info(f"   avg_score: {avg_score:.3f}")
            self.logger.info(f"   evaluation_score_threshold: {evaluation_score_threshold}")
            self.logger.info(f"   조건 만족: {use_meta_prompt and avg_score < evaluation_score_threshold}")
            
            if use_meta_prompt and avg_score < evaluation_score_threshold:
                self.logger.info("🔄 프롬프트 개선 조건 만족! 메타프롬프트 실행합니다...")
                
                # 프롬프트 개선 시작 콜백 호출
                if self.prompt_improvement_start_callback:
                    self.prompt_improvement_start_callback(
                        iteration=iteration + 1,
                        avg_score=avg_score,
                        current_system_prompt=current_system_prompt,
                        current_user_prompt=current_user_prompt
                    )
            else:
                self.logger.info(f"⏭️ 프롬프트 개선 생략 - 조건 불만족 또는 임계값 초과")
                
                # 메타프롬프트를 사용하여 현재 프롬프트를 개선
                improvement_prompt = self._generate_meta_prompt(
                    current_system_prompt, 
                    current_user_prompt, 
                    self._get_recent_prompts(), 
                    test_case_results,
                    current_task_type,
                    current_task_description
                )
                
                # 메타프롬프트 생성 완료 콜백 호출
                if self.meta_prompt_generated_callback:
                    self.meta_prompt_generated_callback(
                        iteration=iteration + 1,
                        meta_prompt=improvement_prompt
                    )
                
                # 결과에 메타프롬프트 추가
                iteration_result.meta_prompt = improvement_prompt
                
                # 메타프롬프트를 사용하여 프롬프트 개선 및 통계 업데이트
                self.logger.info(f"🤖 메타프롬프트 모델에 질의 중...")
                improved_prompts, meta_metadata = self.meta_prompt_model.ask(
                    question=improvement_prompt,
                    system_prompt=self.meta_system_prompt_template
                )
                self._update_stats(self.meta_prompt_stats, meta_metadata, iteration + 1)
                
                self.logger.info(f"🔍 메타프롬프트 응답 수신 완료 (길이: {len(improved_prompts) if improved_prompts else 0} 문자)")
                self.logger.info(f"📄 메타프롬프트 원본 응답:\n{'-'*50}\n{improved_prompts}\n{'-'*50}")
                
                if improved_prompts and improved_prompts.strip():
                    # 개선된 프롬프트에서 TASK_TYPE, TASK_DESCRIPTION, 시스템 프롬프트, 유저 프롬프트 분리
                    improved_prompts = improved_prompts.strip()
                    
                    # TASK_TYPE 추출 (여러 형태 지원)
                    task_type_patterns = ["TASK_TYPE:", "Task Type:", "Task type:"]
                    task_description_patterns = ["TASK_DESCRIPTION:", "Task Description:", "Task description:"]
                    system_prompt_patterns = ["SYSTEM_PROMPT:", "System Prompt:", "System prompt:"]
                    user_prompt_patterns = ["USER_PROMPT:", "User Prompt:", "User prompt:"]
                    
                    def find_pattern(text, patterns):
                        for pattern in patterns:
                            pos = text.find(pattern)
                            if pos != -1:
                                return pos, pattern
                        return -1, None
                    
                    task_type_start, task_type_pattern = find_pattern(improved_prompts, task_type_patterns)
                    task_description_start, task_description_pattern = find_pattern(improved_prompts, task_description_patterns)
                    system_prompt_start, system_prompt_pattern = find_pattern(improved_prompts, system_prompt_patterns)
                    user_prompt_start, user_prompt_pattern = find_pattern(improved_prompts, user_prompt_patterns)
                    
                    self.logger.info(f"🔍 프롬프트 파싱 위치:")
                    self.logger.info(f"   TASK_TYPE 위치: {task_type_start} (패턴: {task_type_pattern})")
                    self.logger.info(f"   TASK_DESCRIPTION 위치: {task_description_start} (패턴: {task_description_pattern})")
                    self.logger.info(f"   SYSTEM_PROMPT 위치: {system_prompt_start} (패턴: {system_prompt_pattern})")
                    self.logger.info(f"   USER_PROMPT 위치: {user_prompt_start} (패턴: {user_prompt_pattern})")
                    
                    if all(pos != -1 for pos in [task_type_start, task_description_start, system_prompt_start, user_prompt_start]):
                        # 이전 프롬프트 저장 (비교용)
                        previous_system_prompt = current_system_prompt
                        previous_user_prompt = current_user_prompt
                        previous_task_type = current_task_type
                        previous_task_description = current_task_description
                        
                        # 새로운 프롬프트 파싱
                        current_task_type = improved_prompts[task_type_start + len(task_type_pattern):task_description_start].strip()
                        current_task_description = improved_prompts[task_description_start + len(task_description_pattern):system_prompt_start].strip()
                        current_system_prompt = improved_prompts[system_prompt_start + len(system_prompt_pattern):user_prompt_start].strip()
                        current_user_prompt = improved_prompts[user_prompt_start + len(user_prompt_pattern):].strip()
                        
                        self.logger.info(f"✅ 프롬프트 파싱 성공!")
                        self.logger.info(f"   새 태스크 타입: {current_task_type}")
                        self.logger.info(f"   새 태스크 설명: {current_task_description[:100]}{'...' if len(current_task_description) > 100 else ''}")
                        self.logger.info(f"   새 시스템 프롬프트: {current_system_prompt[:150]}{'...' if len(current_system_prompt) > 150 else ''}")
                        self.logger.info(f"   새 유저 프롬프트: {current_user_prompt[:150]}{'...' if len(current_user_prompt) > 150 else ''}")
                        
                        # 프롬프트 업데이트 완료 콜백 호출
                        if self.prompt_updated_callback:
                            self.prompt_updated_callback(
                                iteration=iteration + 1,
                                previous_system_prompt=previous_system_prompt,
                                previous_user_prompt=previous_user_prompt,
                                previous_task_type=previous_task_type,
                                previous_task_description=previous_task_description,
                                new_system_prompt=current_system_prompt,
                                new_user_prompt=current_user_prompt,
                                new_task_type=current_task_type,
                                new_task_description=current_task_description,
                                raw_improved_prompts=improved_prompts
                            )
                    else:
                        self.logger.warning(f"❌ 프롬프트 파싱 실패! 필요한 섹션을 찾을 수 없습니다.")
                        self.logger.warning(f"   현재 프롬프트를 그대로 유지합니다.")
                else:
                    self.logger.warning(f"❌ 메타프롬프트 응답이 비어있습니다! 현재 프롬프트를 그대로 유지합니다.")
            
            if self.iteration_callback:
                self.iteration_callback(iteration_result)
        
        return self.iteration_results

    def _get_recent_prompts(self, num_prompts: int = 5) -> List[IterationResult]:
        """최근 프롬프트 결과를 반환합니다."""
        recent_results = self.iteration_results[-num_prompts:] if len(self.iteration_results) >= num_prompts else self.iteration_results
        return [{
            'iteration': result.iteration,
            'system_prompt': result.system_prompt,
            'user_prompt': result.user_prompt,
            'avg_score': result.avg_score,
            'evaluation_details': [test_case.evaluation_details for test_case in result.test_case_results]
        } for result in recent_results]

    def _get_best_prompt(self) -> Dict:
        """최고 성능의 프롬프트를 반환합니다."""
        if not self.iteration_results:
            return None
        
        best_result = max(self.iteration_results, key=lambda x: x.avg_score)
        return {
            'system_prompt': best_result.system_prompt,
            'user_prompt': best_result.user_prompt,
            'avg_score': best_result.avg_score
        }

    def _generate_meta_prompt(self, system_prompt: str, user_prompt: str, recent_prompts: List[Dict], valid_outputs: List[TestCaseResult], task_type: str, task_description: str) -> str:
        """
        메타프롬프트 템플릿을 생성합니다.
        
        Args:
            system_prompt (str): 현재 시스템 프롬프트
            user_prompt (str): 현재 유저 프롬프트
            recent_prompts (List[Dict]): 최근 프롬프트 히스토리
            valid_outputs (List[TestCaseResult]): 전체 평가 케이스
            task_type (str): 현재 테스크 타입
            task_description (str): 현재 테스크 설명
            
        Returns:
            str: 생성된 메타프롬프트 템플릿
        """
        # 케이스를 점수순으로 정렬
        sorted_cases = sorted(valid_outputs, key=lambda x: x.score)
        
        # 상위 3개 케이스 포맷팅
        formatted_top3_cases = "\n\n".join([
            f"[Top Case {i+1}]\n"
            f"Question: {case.question}\n"
            f"Expected Output: {case.expected_output}\n"
            f"Actual Output: {case.actual_output}\n"
            f"Score: {case.score:.2f}\n"
            f"Evaluation Details: {case.evaluation_details}"
            for i, case in enumerate(reversed(sorted_cases[-3:]))  # 상위 3개를 역순으로
        ])
        
        # 하위 3개 케이스 포맷팅
        formatted_bottom2_cases = "\n\n".join([
            f"[Bottom Case {i+1}]\n"
            f"Question: {case.question}\n"
            f"Expected Output: {case.expected_output}\n"
            f"Actual Output: {case.actual_output}\n"
            f"Score: {case.score:.2f}\n"
            f"Evaluation Details: {case.evaluation_details}"
            for i, case in enumerate(sorted_cases[:2])  # 하위 2개
        ])
        
        # 최근 프롬프트 포맷팅
        formatted_recent_prompts = chr(10).join([
            f"Iteration {p['iteration']} (Average Score: {p['avg_score']:.2f}):{chr(10)}"
            f"System Prompt: {p['system_prompt']}{chr(10)}"
            f"User Prompt: {p['user_prompt']}{chr(10)}"
            f"Evaluation Details: {p.get('evaluation_details', 'No evaluation details available')}"
            for p in recent_prompts[:-1]  # 현재 프롬프트를 제외한 최근 3개
        ])
        
        # Format best performing prompt
        best_prompt = self._get_best_prompt()
        if best_prompt:
            formatted_best_prompt = f"""
System: {best_prompt['system_prompt']}
User: {best_prompt['user_prompt']}
Average Score: {best_prompt['avg_score']:.2f}
"""
        else:
            formatted_best_prompt = "No best performing prompt available yet."
        
        # 메타프롬프트 템플릿 생성
        improvement_prompt = self.meta_user_prompt_template.format(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            random_cases=formatted_top3_cases + "\n\n" + formatted_bottom2_cases,  # 상위/하위 케이스 결합
            recent_prompts=formatted_recent_prompts,
            formatted_best_prompt=formatted_best_prompt,
            task_type=task_type,
            task_description=task_description
        )
        
        return improvement_prompt

    def save_results_to_csv(self):
        """결과를 CSV 형식의 문자열로 반환합니다."""
        data = []
        
        # 비용 분석 데이터
        try:
            cost_breakdown = self.get_iteration_cost_breakdown()
            cost_summary = self.get_cost_summary()
            self.logger.info(f"비용 요약 생성 완료: 총 비용 ${cost_summary.get('total_cost', 0.0):.4f}")
            self.logger.info(f"이터레이션별 비용 분석: {len(cost_breakdown)}개 이터레이션")
        except Exception as e:
            self.logger.error(f"비용 분석 데이터 생성 중 오류: {str(e)}")
            cost_breakdown = {}
            cost_summary = {
                'model_stats': {
                    'total_cost': 0.0,
                    'total_tokens': 0,
                    'total_calls': 0,
                    'total_input_tokens': 0,
                    'total_output_tokens': 0,
                    'total_duration': 0.0
                },
                'evaluator_stats': {
                    'total_cost': 0.0,
                    'total_tokens': 0,
                    'total_calls': 0,
                    'total_input_tokens': 0,
                    'total_output_tokens': 0,
                    'total_duration': 0.0
                },
                'meta_prompt_stats': {
                    'total_cost': 0.0,
                    'total_tokens': 0,
                    'total_calls': 0,
                    'total_input_tokens': 0,
                    'total_output_tokens': 0,
                    'total_duration': 0.0
                },
                'total_cost': 0.0,
                'total_tokens': 0,
                'total_duration': 0.0,
                'total_calls': 0
            }
        
        # 각 이터레이션과 테스트 케이스의 결과를 데이터로 변환
        for iteration_result in self.iteration_results:
            iteration_cost_info = cost_breakdown.get(f"iteration_{iteration_result.iteration}", {}) if cost_breakdown else {}
            
            for test_case in iteration_result.test_case_results:
                row = {
                    'Iteration': iteration_result.iteration,
                    'Average Score': iteration_result.avg_score,
                    'Standard Deviation': iteration_result.std_dev,
                    'Top3 Average Score': iteration_result.top3_avg_score,
                    'Best Average Score': iteration_result.best_avg_score,
                    'Best Sample Score': iteration_result.best_sample_score,
                    'Task Type': iteration_result.task_type,
                    'Task Description': iteration_result.task_description,
                    'Test Case': test_case.test_case,
                    'Question': test_case.question,
                    'Expected Output': test_case.expected_output,
                    'Actual Output': test_case.actual_output,
                    'Score': test_case.score,
                    'System Prompt': iteration_result.system_prompt,
                    'User Prompt': iteration_result.user_prompt,
                    'Created At': iteration_result.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                    
                    # 비용 정보 추가 (안전한 접근 방식)
                    'Iteration_Model_Cost': iteration_cost_info.get('model_cost', 0.0),
                    'Iteration_Evaluator_Cost': iteration_cost_info.get('evaluator_cost', 0.0),
                    'Iteration_Meta_Prompt_Cost': iteration_cost_info.get('meta_prompt_cost', 0.0),
                    'Iteration_Total_Cost': iteration_cost_info.get('total_cost', 0.0),
                    'Total_Model_Cost': cost_summary.get('model_stats', {}).get('total_cost', 0.0),
                    'Total_Evaluator_Cost': cost_summary.get('evaluator_stats', {}).get('total_cost', 0.0),
                    'Total_Meta_Prompt_Cost': cost_summary.get('meta_prompt_stats', {}).get('total_cost', 0.0),
                    'Total_Cost': cost_summary.get('total_cost', 0.0),
                    'Total_Tokens': cost_summary.get('total_tokens', 0),
                    'Total_Duration': cost_summary.get('total_duration', 0.0),
                    'Total_Calls': cost_summary.get('total_calls', 0),
                    'Total_Model_Tokens': cost_summary.get('model_stats', {}).get('total_tokens', 0),
                    'Total_Evaluator_Tokens': cost_summary.get('evaluator_stats', {}).get('total_tokens', 0),
                    'Total_Meta_Prompt_Tokens': cost_summary.get('meta_prompt_stats', {}).get('total_tokens', 0),
                    'Total_Model_Calls': cost_summary.get('model_stats', {}).get('total_calls', 0),
                    'Total_Evaluator_Calls': cost_summary.get('evaluator_stats', {}).get('total_calls', 0),
                    'Total_Meta_Prompt_Calls': cost_summary.get('meta_prompt_stats', {}).get('total_calls', 0)
                }
                
                # 카테고리별 점수와 피드백 추가
                if test_case.evaluation_details and 'category_scores' in test_case.evaluation_details:
                    for category, details in test_case.evaluation_details['category_scores'].items():
                        row[f"{category}_Score"] = details['score']
                        row[f"{category}_State"] = details['current_state']
                        row[f"{category}_Action"] = details['improvement_action']
                        row[f"{category}_Weight"] = details['weight']
                
                data.append(row)
        
        # DataFrame 생성 및 CSV 변환
        df = pd.DataFrame(data)
        return df.to_csv(index=False, encoding='utf-8')

    def get_cost_summary(self) -> Dict:
        """전체 비용 요약을 반환합니다."""
        return {
            "model_stats": self.model_stats.copy(),
            "evaluator_stats": self.evaluator_stats.copy(),
            "meta_prompt_stats": self.meta_prompt_stats.copy(),
            "total_cost": self.model_stats["total_cost"] + self.evaluator_stats["total_cost"] + self.meta_prompt_stats["total_cost"],
            "total_tokens": self.model_stats["total_tokens"] + self.evaluator_stats["total_tokens"] + self.meta_prompt_stats["total_tokens"],
            "total_duration": self.model_stats["total_duration"] + self.evaluator_stats["total_duration"] + self.meta_prompt_stats["total_duration"],
            "total_calls": self.model_stats["total_calls"] + self.evaluator_stats["total_calls"] + self.meta_prompt_stats["total_calls"]
        }

    def get_model_stats(self) -> Dict:
        """모델 호출 통계를 반환합니다."""
        return self.model_stats.copy()

    def get_evaluator_stats(self) -> Dict:
        """평가자 호출 통계를 반환합니다."""
        return self.evaluator_stats.copy()

    def get_meta_prompt_stats(self) -> Dict:
        """메타 프롬프트 생성 통계를 반환합니다."""
        return self.meta_prompt_stats.copy()

    def print_cost_summary(self):
        """비용 요약을 콘솔에 출력합니다."""
        summary = self.get_cost_summary()
        
        print("\n=== 비용 및 사용량 요약 ===")
        print(f"총 비용: ${summary['total_cost']:.4f}")
        print(f"총 토큰: {summary['total_tokens']:,}")
        print(f"총 시간: {summary['total_duration']:.2f}초")
        print(f"총 호출: {summary['total_calls']}")
        
        print("\n--- 모델별 상세 정보 ---")
        for model_type in ["model_stats", "evaluator_stats", "meta_prompt_stats"]:
            stats = summary[model_type]
            print(f"\n{stats['type']}:")
            print(f"  호출 횟수: {stats['total_calls']}")
            print(f"  입력 토큰: {stats['total_input_tokens']:,}")
            print(f"  출력 토큰: {stats['total_output_tokens']:,}")
            print(f"  총 토큰: {stats['total_tokens']:,}")
            print(f"  비용: ${stats['total_cost']:.4f}")
            print(f"  시간: {stats['total_duration']:.2f}초")

    def get_iteration_cost_breakdown(self) -> Dict:
        """이터레이션별 비용 분석을 반환합니다."""
        breakdown = {}
        
        try:
            if not self.iteration_results:
                return breakdown
                
            for iteration in range(1, len(self.iteration_results) + 1):
                iteration_cost = {
                    "model_cost": 0.0,
                    "evaluator_cost": 0.0,
                    "meta_prompt_cost": 0.0,
                    "total_cost": 0.0,
                    "model_calls": 0,
                    "evaluator_calls": 0,
                    "meta_prompt_calls": 0,
                    "total_calls": 0
                }
                
                # 각 모델별 이터레이션 통계 수집
                for stats_name, stats in [("model", self.model_stats), ("evaluator", self.evaluator_stats), ("meta_prompt", self.meta_prompt_stats)]:
                    if stats and "calls_by_iteration" in stats and iteration in stats["calls_by_iteration"]:
                        iter_stats = stats["calls_by_iteration"][iteration]
                        iteration_cost[f"{stats_name}_cost"] = iter_stats.get("cost", 0.0)
                        iteration_cost[f"{stats_name}_calls"] = iter_stats.get("calls", 0)
                        iteration_cost["total_cost"] += iter_stats.get("cost", 0.0)
                        iteration_cost["total_calls"] += iter_stats.get("calls", 0)
                
                breakdown[f"iteration_{iteration}"] = iteration_cost
        except Exception as e:
            self.logger.error(f"이터레이션별 비용 분석 중 오류: {str(e)}")
            return {}
        
        return breakdown 

    def reset_stats(self):
        """모든 통계를 초기화합니다."""
        self.model_stats = self._initialize_stats("모델 호출")
        self.evaluator_stats = self._initialize_stats("평가자 호출")
        self.meta_prompt_stats = self._initialize_stats("메타 프롬프트 생성")
        self.iteration_results = []
        self.logger.info("모든 통계가 초기화되었습니다.")

    def export_cost_summary_to_csv(self) -> str:
        """비용 요약을 CSV 형식으로 내보냅니다."""
        summary = self.get_cost_summary()
        breakdown = self.get_iteration_cost_breakdown()
        
        # 전체 요약 데이터
        summary_data = [{
            'Type': 'Total Summary',
            'Total_Cost': summary['total_cost'],
            'Total_Tokens': summary['total_tokens'],
            'Total_Duration': summary['total_duration'],
            'Total_Calls': summary['total_calls'],
            'Model_Cost': summary['model_stats']['total_cost'],
            'Model_Calls': summary['model_stats']['total_calls'],
            'Model_Tokens': summary['model_stats']['total_tokens'],
            'Evaluator_Cost': summary['evaluator_stats']['total_cost'],
            'Evaluator_Calls': summary['evaluator_stats']['total_calls'],
            'Evaluator_Tokens': summary['evaluator_stats']['total_tokens'],
            'Meta_Prompt_Cost': summary['meta_prompt_stats']['total_cost'],
            'Meta_Prompt_Calls': summary['meta_prompt_stats']['total_calls'],
            'Meta_Prompt_Tokens': summary['meta_prompt_stats']['total_tokens']
        }]
        
        # 이터레이션별 데이터 추가
        for iteration_key, iteration_data in breakdown.items():
            summary_data.append({
                'Type': iteration_key.replace('_', ' ').title(),
                'Total_Cost': iteration_data['total_cost'],
                'Total_Tokens': 0,  # 이터레이션별 토큰 정보는 별도로 계산 필요
                'Total_Duration': 0,  # 이터레이션별 시간 정보는 별도로 계산 필요
                'Total_Calls': iteration_data['total_calls'],
                'Model_Cost': iteration_data['model_cost'],
                'Model_Calls': iteration_data['model_calls'],
                'Model_Tokens': 0,
                'Evaluator_Cost': iteration_data['evaluator_cost'],
                'Evaluator_Calls': iteration_data['evaluator_calls'],
                'Evaluator_Tokens': 0,
                'Meta_Prompt_Cost': iteration_data['meta_prompt_cost'],
                'Meta_Prompt_Calls': iteration_data['meta_prompt_calls'],
                'Meta_Prompt_Tokens': 0
            })
        
        df = pd.DataFrame(summary_data)
        return df.to_csv(index=False, encoding='utf-8') 