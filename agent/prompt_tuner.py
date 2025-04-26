from typing import List, Dict, Optional
import logging
from common.api_client import Model
import os
import random
import statistics
import csv
from datetime import datetime
import io
import json

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
        self.evaluation_history: List[Dict] = []
        self.prompt_history: List[Dict] = []
        self.best_prompt: Optional[str] = None
        self.best_avg_score: float = 0.0
        self.best_sample_score: float = 0.0
        self.progress_callback = None
        self.iteration_callback = None
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 프롬프트 파일 경로
        prompts_dir = os.path.join(os.path.dirname(__file__), 'prompts')
        
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
        
        self.results = []  # Add this line to store results
    
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
    
    def _evaluate_output(self, output: str, expected: str, question: str, task_type: str, task_description: str) -> tuple[float, Dict]:
        """
        Evaluate an output using the evaluator model.
        
        Args:
            output (str): The actual output to evaluate
            expected (str): The expected output
            question (str): The original question
            task_type (str): The type of task being evaluated
            task_description (str): The description of the task being evaluated
            
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
            
            # 평가 모델로 평가 수행
            evaluation = self.evaluator.ask(
                question=evaluation_prompt,
                system_prompt=evaluation_system_prompt
            )
            self.logger.info(f"Evaluating output:")
            self.logger.info(f"Question: {question}")
            self.logger.info(f"Actual output: {output}")
            self.logger.info(f"Expected output: {expected}")
            self.logger.info(f"Evaluation: {evaluation}")
            
            # JSON 파싱
            evaluation_data = json.loads(evaluation)
            
            # 최종 점수와 세부 점수 추출
            final_score = float(evaluation_data.get('final_score', 0))
            scores_data = evaluation_data.get('scores', {})
            
            # 평가 상세 정보 구성
            evaluation_details = {
                'final_score': final_score,
                'category_scores': {}
            }
            
            # 각 카테고리별 점수와 피드백 정보 추출
            for category, details in scores_data.items():
                evaluation_details['category_scores'][category] = {
                    'score': float(details.get('score', 0)),
                    'current_state': details.get('current_state', ''),
                    'improvement_action': details.get('improvement_action', '')
                }
            
            self.logger.info(f"Evaluation score: {final_score}")
            self.logger.info(f"Evaluation details: {evaluation_details}")
            
            return final_score, evaluation_details
            
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            self.logger.error(f"Error during evaluation: {str(e)}")
            return 0.0, {'final_score': 0.0, 'category_scores': {}}
    
    def tune_prompt(self, initial_system_prompt: str, initial_user_prompt: str, initial_test_cases: List[Dict], num_iterations: int = 3, score_threshold: Optional[float] = None, evaluation_score_threshold: float = 0.8, use_meta_prompt: bool = True, num_samples: Optional[int] = None) -> List[Dict]:
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
            List[Dict]: List of iteration results, each containing:
                - iteration: iteration number
                - system_prompt: current system prompt
                - user_prompt: current user prompt
                - avg_score: average score for this iteration
                - std_dev: standard deviation for this iteration
                - top3_avg_score: top 3 average score for this iteration
                - best_avg_score: best average score so far
                - best_sample_score: best individual test case score so far
                - best_prompt: best prompt so far
                - outputs: list of outputs for each test case
        """
        current_system_prompt = initial_system_prompt
        current_user_prompt = initial_user_prompt
        best_system_prompt = initial_system_prompt
        best_user_prompt = initial_user_prompt
        best_avg_score = 0.0
        best_sample_score = 0.0
        iteration_results = []
        self.results = []  # 결과 리스트 초기화
        
        # 초기 task_type과 task_description 설정
        current_task_type = "General Task"
        current_task_description = "General task requiring outputs to various questions"
        
        for iteration in range(num_iterations):
            self.logger.info(f"\nIteration {iteration + 1}/{num_iterations}")
            iteration_scores = []
            iteration_outputs = []
            iteration_best_sample_score = 0.0  # 이터레이션별 최고 점수 초기화
            
            # 각 이터레이션마다 랜덤 샘플링
            test_cases = random.sample(initial_test_cases, num_samples) if num_samples is not None and num_samples < len(initial_test_cases) else initial_test_cases
            
            # 테스트 케이스 실행 및 평가
            for i, test_case in enumerate(test_cases):
                self.logger.info(f"\nTest Case {i}/{len(test_cases)}")
                self.logger.info(f"Question: {test_case['question']}")
                
                # 현재 프롬프트로 출력 생성
                output = self.model.ask(test_case['question'], system_prompt=current_system_prompt, user_prompt=current_user_prompt)
                self.logger.info(f"Output: {output}")
                
                # 출력 평가
                score, evaluation_details = self._evaluate_output(
                    output=output,
                    expected=test_case['expected'],
                    question=test_case['question'],
                    task_type=current_task_type,
                    task_description=current_task_description
                )
                self.logger.info(f"Score: {score}")
                self.logger.info(f"Evaluation details: {evaluation_details}")
                
                # 점수와 출력 저장
                iteration_scores.append(score)
                iteration_outputs.append({
                    'question': test_case['question'],
                    'expected': test_case['expected'],
                    'actual': output,
                    'score': score,
                    'evaluation_details': evaluation_details
                })
                
                # 최고 개별 점수 업데이트
                if score is not None and score > iteration_best_sample_score:
                    iteration_best_sample_score = score
                
                # 평가 기록 저장
                self.evaluation_history.append({
                    'iteration': iteration + 1,
                    'test_case': i,
                    'system_prompt': current_system_prompt,
                    'user_prompt': current_user_prompt,
                    'question': test_case['question'],
                    'expected_output': test_case['expected'],
                    'actual_output': output,
                    'score': score,
                    'evaluation_details': evaluation_details
                })
                
                # 상세 결과 저장
                result = {
                    'iteration': iteration + 1,
                    'question': test_case['question'],
                    'expected_output': test_case['expected'],
                    'actual_output': output,
                    'score': score,
                    'evaluation_details': evaluation_details
                }
                self.results.append(result)
                
                # 프로그레스 바 업데이트
                if self.progress_callback:
                    self.progress_callback(iteration + 1, i + 1)
            
            # iteration이 끝난 후 평균 점수 계산
            valid_scores = [score for score in iteration_scores if score is not None]
            if valid_scores:
                avg_score = sum(valid_scores) / len(valid_scores)
                # 표준편차 계산
                std_dev = statistics.stdev(valid_scores) if len(valid_scores) > 1 else 0.0
                # top3 평균 점수 계산
                top3_scores = sorted(valid_scores, reverse=True)[:3]
                top3_avg_score = sum(top3_scores) / len(top3_scores)
            else:
                avg_score = 0.0
                std_dev = 0.0
                top3_avg_score = 0.0
            self.logger.info(f"Iteration {iteration + 1} 평균 점수: {avg_score:.2f}, 표준편차: {std_dev:.2f}, Top3 평균 점수: {top3_avg_score:.2f}")
            
            # 현재까지의 최고 평균 점수와 비교
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                best_system_prompt = current_system_prompt
                best_user_prompt = current_user_prompt
            
            # 현재 이터레이션 결과 저장
            result = {
                'iteration': iteration + 1,
                'system_prompt': current_system_prompt,
                'user_prompt': current_user_prompt,
                'avg_score': avg_score,
                'std_dev': std_dev,  # 표준편차 추가
                'top3_avg_score': top3_avg_score,  # top3 평균 점수 추가
                'best_avg_score': best_avg_score,
                'best_sample_score': iteration_best_sample_score,  # 이터레이션별 최고 점수 사용
                'best_system_prompt': best_system_prompt,
                'best_user_prompt': best_user_prompt,
                'outputs': iteration_outputs,
                'meta_prompt': None,  # 초기값 설정
                'task_type': current_task_type,
                'task_description': current_task_description
            }
            iteration_results.append(result)
            
            # prompt_history 업데이트
            self.prompt_history.append({
                'iteration': iteration + 1,
                'system_prompt': current_system_prompt,
                'user_prompt': current_user_prompt,
                'avg_score': avg_score,
                'evaluation_details': [output['evaluation_details'] for output in iteration_outputs if output['score'] is not None]
            })
            
            # 점수 임계값 체크
            if score_threshold is not None and avg_score >= score_threshold:
                self.logger.info(f"평균 점수가 임계값({score_threshold}) 이상입니다. 튜닝을 종료합니다.")
                if self.progress_callback:
                    self.progress_callback(num_iterations, len(test_cases))

                if self.iteration_callback:
                    self.iteration_callback(result)

                break
            
            # 프롬프트 개선 (평균 점수가 평가 임계값 미만인 경우)
            if use_meta_prompt and avg_score < evaluation_score_threshold:
                self.logger.info("프롬프트 개선 중...")
                
                # 랜덤으로 5개의 케이스 선택 (데이터가 5개 미만이면 전부 선택)
                valid_outputs = [output for output in iteration_outputs if output['score'] is not None]
                
                # 메타프롬프트를 사용하여 현재 프롬프트를 개선
                improvement_prompt = self._generate_meta_prompt(
                    current_system_prompt, 
                    current_user_prompt, 
                    self._get_recent_prompts(), 
                    valid_outputs,
                    current_task_type,
                    current_task_description
                )
                
                # 결과에 메타프롬프트 추가
                result['meta_prompt'] = improvement_prompt
                
                # 메타프롬프트를 사용하여 프롬프트 개선
                improved_prompts = self.meta_prompt_model.ask(
                    question=improvement_prompt,
                    system_prompt=self.meta_system_prompt_template
                )
                if improved_prompts and improved_prompts.strip():
                    # 개선된 프롬프트에서 TASK_TYPE, TASK_DESCRIPTION, 시스템 프롬프트, 유저 프롬프트 분리
                    improved_prompts = improved_prompts.strip()
                    
                    # TASK_TYPE 추출
                    task_type_start = improved_prompts.find("TASK_TYPE:")
                    task_description_start = improved_prompts.find("TASK_DESCRIPTION:")
                    system_prompt_start = improved_prompts.find("SYSTEM_PROMPT:")
                    user_prompt_start = improved_prompts.find("USER_PROMPT:")
                    
                    if all(pos != -1 for pos in [task_type_start, task_description_start, system_prompt_start, user_prompt_start]):
                        current_task_type = improved_prompts[task_type_start + len("TASK_TYPE:"):task_description_start].strip()
                        current_task_description = improved_prompts[task_description_start + len("TASK_DESCRIPTION:"):system_prompt_start].strip()
                        current_system_prompt = improved_prompts[system_prompt_start + len("SYSTEM_PROMPT:"):user_prompt_start].strip()
                        current_user_prompt = improved_prompts[user_prompt_start + len("USER_PROMPT:"):].strip()
                        
                        # 결과에 TASK_TYPE과 TASK_DESCRIPTION 추가
                        result['task_type'] = current_task_type
                        result['task_description'] = current_task_description
                        
                        self.logger.info(f"추출된 TASK_TYPE: {current_task_type}")
                        self.logger.info(f"추출된 TASK_DESCRIPTION: {current_task_description}")
                        self.logger.info(f"개선된 시스템 프롬프트: {current_system_prompt}")
                        self.logger.info(f"개선된 유저 프롬프트: {current_user_prompt}")
                    else:
                        self.logger.warning("프롬프트 개선 결과가 올바른 형식이 아닙니다. 현재 프롬프트를 유지합니다.")
                else:
                    self.logger.warning("프롬프트 개선에 실패했습니다. 현재 프롬프트를 유지합니다.")
            elif use_meta_prompt:
                self.logger.info(f"평균 점수가 평가 임계값({evaluation_score_threshold}) 이상이므로 프롬프트를 개선하지 않습니다.")
                if self.progress_callback:
                    self.progress_callback(num_iterations, len(test_cases))

                if self.iteration_callback:
                    self.iteration_callback(result)

                break
        
            # 콜백 호출 (메타프롬프트 설정 이후)
            if self.iteration_callback:
                self.iteration_callback(result)

        return iteration_results 

    def _get_recent_prompts(self, num_prompts: int = 5) -> List[Dict]:
        """
        최근 num_prompts개의 프롬프트를 반환합니다.
        
        Args:
            num_prompts (int): 반환할 프롬프트의 개수 (default: 3)
            
        Returns:
            List[Dict]: 최근 프롬프트들의 리스트
        """
        return self.prompt_history[-num_prompts:] if len(self.prompt_history) >= num_prompts else self.prompt_history

    def _get_best_prompt(self) -> Dict:
        """
        현재까지의 최고 점수를 가진 프롬프트를 반환합니다.
        
        Returns:
            Dict: 최고 점수를 가진 프롬프트 정보
        """
        if not self.prompt_history:
            return {
                'avg_score': 0.0,
                'system_prompt': self.initial_system_prompt,
                'user_prompt': self.initial_user_prompt
            }
        return max(self.prompt_history, key=lambda x: x['avg_score'])

    def _generate_meta_prompt(self, system_prompt: str, user_prompt: str, recent_prompts: List[Dict], valid_outputs: List[Dict], task_type: str, task_description: str) -> str:
        """
        메타프롬프트 템플릿을 생성합니다.
        
        Args:
            system_prompt (str): 현재 시스템 프롬프트
            user_prompt (str): 현재 유저 프롬프트
            recent_prompts (List[Dict]): 최근 프롬프트 히스토리
            valid_outputs (List[Dict]): 전체 평가 케이스
            task_type (str): 현재 테스크 타입
            task_description (str): 현재 테스크 설명
            
        Returns:
            str: 생성된 메타프롬프트 템플릿
        """
        # 케이스를 점수순으로 정렬
        sorted_cases = sorted(valid_outputs, key=lambda x: x['score'])
        
        # 상위 3개 케이스 포맷팅
        formatted_top3_cases = "\n\n".join([
            f"[Top Case {i+1}]\n"
            f"Question: {case['question']}\n"
            f"Expected Output: {case['expected']}\n"
            f"Actual Output: {case['actual']}\n"
            f"Score: {case['score']:.2f}\n"
            f"Evaluation Details: {case['evaluation_details']}"
            for i, case in enumerate(reversed(sorted_cases[-3:]))  # 상위 3개를 역순으로
        ])
        
        # 하위 3개 케이스 포맷팅
        formatted_bottom2_cases = "\n\n".join([
            f"[Bottom Case {i+1}]\n"
            f"Question: {case['question']}\n"
            f"Expected Output: {case['expected']}\n"
            f"Actual Output: {case['actual']}\n"
            f"Score: {case['score']:.2f}\n"
            f"Evaluation Details: {case['evaluation_details']}"
            for i, case in enumerate(sorted_cases[:2])  # 하위 3개
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
Evaluation Details: {best_prompt.get('evaluation_details', 'No evaluation details available')}
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

    def save_results_to_csv(self, filename=None):
        """Save iteration results to a CSV file.
        
        Args:
            filename (str, optional): The name of the CSV file. If None, a default name will be used.
            
        Returns:
            str: The CSV content as a string
        """
        if not self.results:
            return ""
            
        # CSV 내용 생성
        output = io.StringIO()
        writer = csv.writer(output)
        
        # 헤더 작성
        writer.writerow(['iteration', 'question', 'expected_output', 'actual_output', 'score', 'evaluation_details'])
        
        # 데이터 작성
        for result in self.results:
            writer.writerow([
                result['iteration'],
                result['question'],
                result['expected_output'],
                result['actual_output'],
                result['score'],
                result['evaluation_details']
            ])
        
        return output.getvalue() 