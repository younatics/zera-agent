from typing import List, Dict, Optional
import logging
from common.api_client import Model
import os
import random

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
        self.best_score: float = 0.0
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
        with open(os.path.join(prompts_dir, 'evaluation_prompt.txt'), 'r', encoding='utf-8') as f:
            self.evaluation_prompt_template = f.read()
        
        # 기본 메타프롬프트 로드
        with open(os.path.join(prompts_dir, 'meta_prompt.txt'), 'r', encoding='utf-8') as f:
            self.meta_prompt_template = f.read()
    
    def set_evaluation_prompt(self, prompt_template: str):
        """
        평가 프롬프트 템플릿을 설정합니다.
        
        Args:
            prompt_template (str): 평가 프롬프트 템플릿. {response}와 {expected}를 포함해야 합니다.
        """
        self.evaluation_prompt_template = prompt_template
    
    def set_meta_prompt(self, prompt_template: str):
        """
        메타프롬프트 템플릿을 설정합니다.
        
        Args:
            prompt_template (str): 메타프롬프트 템플릿. {prompt}를 포함해야 합니다.
        """
        self.meta_prompt_template = prompt_template
    
    def _evaluate_response(self, response: str, expected: str, question: str) -> tuple[float, str]:
        """
        Evaluate a response using the evaluator model.
        
        Args:
            response (str): The actual response to evaluate
            expected (str): The expected response
            question (str): The original question
            
        Returns:
            tuple[float, str]: A tuple containing the score and evaluation reason
        """
        try:
            # 평가 프롬프트 생성
            evaluation_prompt = self.evaluation_prompt_template.format(
                response=response,
                expected=expected,
                question=question
            )
            
            # 평가 모델로 평가 수행
            evaluation = self.evaluator.ask(evaluation_prompt)
            self.logger.info(f"Evaluating response:")
            self.logger.info(f"Question: {question}")
            self.logger.info(f"Actual response: {response}")
            self.logger.info(f"Expected response: {expected}")
            self.logger.info(f"Evaluation: {evaluation}")
            
            # 평가 결과에서 점수 추출
            score = float(evaluation.split('\n')[0].strip())
            reason = '\n'.join(evaluation.split('\n')[1:]).strip()
            
            self.logger.info(f"Evaluation score: {score}")
            self.logger.info(f"Evaluation reason: {reason}")
            
            return score, reason
            
        except (ValueError, TypeError):
            # 숫자로 변환할 수 없는 경우 제외
            return None, "점수 추출 실패, (평가 제외)"
    
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
                - best_score: best score so far
                - best_prompt: best prompt so far
                - responses: list of responses for each test case
        """
        current_system_prompt = initial_system_prompt
        current_user_prompt = initial_user_prompt
        best_prompt = initial_system_prompt
        best_score = 0.0
        iteration_results = []
        
        for iteration in range(num_iterations):
            self.logger.info(f"\nIteration {iteration + 1}/{num_iterations}")
            iteration_scores = []
            iteration_responses = []
            
            # 각 이터레이션마다 랜덤 샘플링
            test_cases = random.sample(initial_test_cases, num_samples) if num_samples is not None and num_samples < len(initial_test_cases) else initial_test_cases
            
            # 테스트 케이스 실행 및 평가
            for i, test_case in enumerate(test_cases):
                self.logger.info(f"\nTest Case {i}/{len(test_cases)}")
                self.logger.info(f"Question: {test_case['question']}")
                
                # 현재 프롬프트로 응답 생성
                response = self.model.ask(test_case['question'], system_prompt=current_system_prompt, user_prompt=current_user_prompt)
                self.logger.info(f"Response: {response}")
                
                # 응답 평가
                score, reason = self._evaluate_response(response, test_case['expected'], test_case['question'])
                self.logger.info(f"Score: {score}")
                self.logger.info(f"Evaluation reason: {reason}")
                
                # 점수와 응답 저장
                iteration_scores.append(score)
                iteration_responses.append({
                    'question': test_case['question'],
                    'expected': test_case['expected'],
                    'actual': response,
                    'score': score,
                    'reason': reason
                })
                
                # 평가 기록 저장
                self.evaluation_history.append({
                    'iteration': iteration + 1,
                    'test_case': i,
                    'system_prompt': current_system_prompt,
                    'user_prompt': current_user_prompt,
                    'question': test_case['question'],
                    'expected_answer': test_case['expected'],
                    'actual_answer': response,
                    'score': score,
                    'evaluation_reason': reason
                })
                
                # 프로그레스 바 업데이트
                if self.progress_callback:
                    self.progress_callback(iteration + 1, i + 1)
                
                # 현재까지의 최고 점수와 비교
                if score is not None and (best_score is None or score > best_score):
                    best_score = score
                    best_prompt = current_system_prompt
            
            # iteration이 끝난 후 평균 점수 계산
            valid_scores = [score for score in iteration_scores if score is not None]
            if valid_scores:
                avg_score = sum(valid_scores) / len(valid_scores)
            else:
                avg_score = 0.0
            self.logger.info(f"Iteration {iteration + 1} 평균 점수: {avg_score:.2f}")
            
            # 현재 이터레이션 결과 저장
            result = {
                'iteration': iteration + 1,
                'system_prompt': current_system_prompt,
                'user_prompt': current_user_prompt,
                'avg_score': avg_score,
                'best_score': best_score,
                'best_prompt': best_prompt,
                'responses': iteration_responses
            }
            iteration_results.append(result)
            
            # 콜백 호출
            if self.iteration_callback:
                self.iteration_callback(result)
            
            # 점수 임계값 체크
            if score_threshold is not None and avg_score >= score_threshold:
                self.logger.info(f"평균 점수가 임계값({score_threshold}) 이상입니다. 튜닝을 종료합니다.")
                if self.progress_callback:
                    self.progress_callback(num_iterations, len(test_cases))
                break
            
            # 프롬프트 개선 (평균 점수가 평가 임계값 미만인 경우)
            if use_meta_prompt and avg_score < evaluation_score_threshold:
                self.logger.info("프롬프트 개선 중...")
                
                # 랜덤으로 5개의 케이스 선택 (데이터가 5개 미만이면 전부 선택)
                valid_responses = [response for response in iteration_responses if response['score'] is not None]
                num_cases = min(5, len(valid_responses))
                if num_cases > 0:
                    random_cases = random.sample(valid_responses, num_cases)
                else:
                    self.logger.warning("유효한 평가 결과가 없습니다. 현재 프롬프트를 유지합니다.")
                    continue
                
                # 랜덤 케이스 포맷팅
                formatted_cases = "\n".join([
                    f"Question: {case['question']}\n"
                    f"Expected answer: {case['expected']}\n"
                    f"Actual answer: {case['actual']}\n"
                    f"Score: {case['score']}\n"
                    f"Evaluation reason: {case['reason']}\n"
                    for case in random_cases
                ])
                
                # 메타프롬프트를 사용하여 현재 프롬프트를 개선
                improvement_prompt = self.meta_prompt_template.format(
                    system_prompt=current_system_prompt,
                    user_prompt=current_user_prompt,
                    random_cases=formatted_cases,
                    recent_prompts=chr(10).join([
                        f"Iteration {p['iteration']} (Average Score: {p['avg_score']:.2f}):{chr(10)}"
                        f"System Prompt: {p['system_prompt']}{chr(10)}"
                        f"User Prompt: {p['user_prompt']}"
                        for p in self._get_recent_prompts()[:-1]  # 현재 프롬프트를 제외한 최근 3개
                    ]),
                    best_prompt_score=self._get_best_prompt()['avg_score'],
                    best_system_prompt=self._get_best_prompt()['system_prompt'],
                    best_user_prompt=self._get_best_prompt()['user_prompt']
                )
                improved_prompts = self.meta_prompt_model.ask("", system_prompt=improvement_prompt)
                if improved_prompts and improved_prompts.strip():
                    # 개선된 프롬프트에서 시스템 프롬프트와 유저 프롬프트 분리
                    improved_prompts = improved_prompts.strip()
                    system_prompt_start = improved_prompts.find("SYSTEM_PROMPT:")
                    user_prompt_start = improved_prompts.find("USER_PROMPT:")
                    
                    if system_prompt_start != -1 and user_prompt_start != -1:
                        current_system_prompt = improved_prompts[system_prompt_start + len("SYSTEM_PROMPT:"):user_prompt_start].strip()
                        current_user_prompt = improved_prompts[user_prompt_start + len("USER_PROMPT:"):].strip()
                        self.logger.info(f"개선된 시스템 프롬프트: {current_system_prompt}")
                        self.logger.info(f"개선된 유저 프롬프트: {current_user_prompt}")
                    else:
                        self.logger.warning("프롬프트 개선 결과가 올바른 형식이 아닙니다. 현재 프롬프트를 유지합니다.")
                        # 현재 프롬프트를 유지하고 계속 진행
                else:
                    self.logger.warning("프롬프트 개선에 실패했습니다. 현재 프롬프트를 유지합니다.")
                    # 현재 프롬프트를 유지하고 계속 진행
            elif use_meta_prompt:
                self.logger.info(f"평균 점수가 평가 임계값({evaluation_score_threshold}) 이상이므로 프롬프트를 개선하지 않습니다.")
                if self.progress_callback:
                    self.progress_callback(num_iterations, len(test_cases))
                break
        
        return iteration_results 

    def _get_recent_prompts(self, num_prompts: int = 3) -> List[Dict]:
        """
        최근 num_prompts개의 프롬프트를 반환합니다.
        
        Args:
            num_prompts (int): 반환할 프롬프트의 개수 (default: 3)
            
        Returns:
            List[Dict]: 최근 프롬프트들의 리스트
        """
        return self.evaluation_history[-num_prompts:] if len(self.evaluation_history) >= num_prompts else self.evaluation_history

    def _get_best_prompt(self) -> Dict:
        """
        현재까지의 최고 점수를 가진 프롬프트를 반환합니다.
        
        Returns:
            Dict: 최고 점수를 가진 프롬프트 정보
        """
        if not self.evaluation_history:
            return {
                'avg_score': 0.0,
                'system_prompt': self.initial_system_prompt,
                'user_prompt': self.initial_user_prompt
            }
        return max(self.evaluation_history, key=lambda x: x['score']) 