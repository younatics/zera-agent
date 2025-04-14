from typing import List, Dict, Optional
import logging
from common.api_client import Model
import os

class PromptTuner:
    """
    A class for automatically fine-tuning system prompts for LLMs.
    """
    
    def __init__(self, model_name: str = "solar", evaluator_model_name: str = "solar", meta_prompt_model_name: str = "solar"):
        """
        Initialize the PromptTuner with specific models.
        
        Args:
            model_name (str): The name of the model to use for tuning (default: "solar")
            evaluator_model_name (str): The name of the model to use for evaluation (default: "solar")
            meta_prompt_model_name (str): The name of the model to use for meta prompt generation (default: "solar")
        """
        self.model = Model(model_name)
        self.evaluator = Model(evaluator_model_name)
        self.meta_prompt_model = Model(meta_prompt_model_name)
        self.evaluation_history: List[Dict] = []
        self.best_prompt: Optional[str] = None
        self.best_score: float = 0.0
        self.progress_callback = None
        self.iteration_callback = None
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 프롬프트 파일 경로
        prompts_dir = os.path.join(os.path.dirname(__file__), 'prompts')
        
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
    
    def _evaluate_response(self, response: str, expected: str, question: str) -> float:
        """
        Evaluate a single response against the expected output using the evaluator model.
        
        Args:
            response (str): The model's response
            expected (str): The expected output
            question (str): The question that was asked
            
        Returns:
            float: Score between 0.0 and 1.0
        """
        self.logger.info("Evaluating response:")
        self.logger.info(f"Question: {question}")
        self.logger.info(f"Actual response: {response}")
        self.logger.info(f"Expected response: {expected}")
        
        evaluation_prompt = self.evaluation_prompt_template.format(
            question=question,
            response=response,
            expected=expected
        )
        
        try:
            evaluation_result = self.evaluator.ask(evaluation_prompt).strip()
            # 평가 결과에서 점수와 이유를 분리
            score_str = evaluation_result.split('\n')[0]  # 첫 번째 줄은 점수
            reason = '\n'.join(evaluation_result.split('\n')[1:])  # 나머지는 평가 이유
            score = float(score_str)
            self.logger.info(f"Evaluation score: {score}")
            self.logger.info(f"Evaluation reason: {reason}")
            return round(score, 2), reason
        except (ValueError, TypeError):
            # 숫자로 변환할 수 없는 경우 기본 평가 방식 사용
            self.logger.warning("Failed to get valid score from evaluator, using fallback evaluation method")
            response = response.lower()
            expected = expected.lower()
            question = question.lower()
            
            # 기본 키워드 기반 평가
            key_phrases = [
                "ai", "assistant", "help", "안녕하세요", "도와드릴까요",
                "어떻게", "무엇을", "필요하신가요"
            ]
            matches = sum(1 for phrase in key_phrases if phrase in response)
            fallback_score = min(1.0, matches / len(key_phrases))
            self.logger.info(f"Fallback evaluation score: {fallback_score}")
            return fallback_score, "키워드 기반 기본 평가"
    
    def evaluate_prompt(self, prompt: str, test_cases: List[Dict]) -> Dict:
        """
        Evaluate a system prompt using a set of test cases.
        
        Args:
            prompt (str): The system prompt to evaluate
            test_cases (List[Dict]): List of test cases, each containing 'question' and 'expected'
            
        Returns:
            Dict: Evaluation results including total score and detailed responses
        """
        total_score = 0.0
        responses = []
        
        for test_case in test_cases:
            response = self.model.ask(test_case['question'], system_prompt=prompt)
            score, reason = self._evaluate_response(response, test_case['expected'], test_case['question'])
            total_score += score
            
            responses.append({
                'question': test_case['question'],
                'expected': test_case['expected'],
                'actual': response,
                'score': score,
                'reason': reason
            })
        
        avg_score = total_score / len(test_cases)
        
        # 현재까지의 최고 점수와 비교
        if avg_score > self.best_score:
            self.best_score = avg_score
            self.best_prompt = prompt
        
        return {
            'avg_score': avg_score,
            'best_score': self.best_score,
            'best_prompt': self.best_prompt,
            'responses': responses
        }
    
    def generate_variations(self, prompt: str, num_variations: int = 3) -> List[str]:
        """
        Generate variations of a given prompt using the model itself.
        
        Args:
            prompt (str): The original prompt
            num_variations (int): Number of variations to generate
            
        Returns:
            List[str]: List of prompt variations including the original
        """
        variations = [prompt]  # Always include the original prompt
        
        # 지정된 수만큼 변형 생성
        for _ in range(num_variations - 1):  # -1 because we already have the original
            variation = self.meta_prompt_model.ask(self.meta_prompt_template.format(prompt=prompt))
            if variation and variation.strip():
                variations.append(variation.strip())
        
        return variations
    
    def tune_prompt(self, initial_prompt: str, test_cases: List[Dict], num_iterations: int = 3, score_threshold: Optional[float] = None, evaluation_score_threshold: float = 0.8, use_meta_prompt: bool = True) -> List[Dict]:
        """
        Tune a system prompt using a set of test cases.
        
        Args:
            initial_prompt (str): The initial system prompt
            test_cases (List[Dict]): List of test cases, each containing 'question' and 'expected'
            num_iterations (int): Number of iterations to perform
            score_threshold (Optional[float]): Threshold to stop tuning if average score exceeds this value
            evaluation_score_threshold (float): Threshold to trigger prompt improvement
            use_meta_prompt (bool): Whether to use meta prompt for improvement
            
        Returns:
            List[Dict]: List of iteration results, each containing:
                - iteration: iteration number
                - prompt: current prompt
                - avg_score: average score for this iteration
                - best_score: best score so far
                - best_prompt: best prompt so far
                - responses: list of responses for each test case
        """
        current_prompt = initial_prompt
        best_prompt = initial_prompt
        best_score = 0.0
        iteration_results = []
        
        for iteration in range(num_iterations):
            self.logger.info(f"\nIteration {iteration + 1}/{num_iterations}")
            iteration_scores = []
            iteration_responses = []
            
            # 각 테스트 케이스에 대해 순차적으로 평가
            for i, test_case in enumerate(test_cases):
                self.logger.info(f"\nTest Case {i + 1}/{len(test_cases)}")
                self.logger.info(f"Question: {test_case['question']}")
                
                # 현재 프롬프트로 응답 생성
                response = self.model.ask(test_case['question'], system_prompt=current_prompt)
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
                    'test_case': i + 1,
                    'prompt': current_prompt,
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
                if score > best_score:
                    best_score = score
                    best_prompt = current_prompt
            
            # iteration이 끝난 후 평균 점수 계산
            avg_score = sum(iteration_scores) / len(iteration_scores)
            self.logger.info(f"Iteration {iteration + 1} 평균 점수: {avg_score:.2f}")
            
            # 현재 이터레이션 결과 저장
            result = {
                'iteration': iteration + 1,
                'prompt': current_prompt,
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
                
                # 최고/최저 점수 케이스 찾기
                best_case_idx = iteration_scores.index(max(iteration_scores))
                worst_case_idx = iteration_scores.index(min(iteration_scores))
                
                best_case = iteration_responses[best_case_idx]
                worst_case = iteration_responses[worst_case_idx]
                
                # 메타프롬프트를 사용하여 현재 프롬프트를 개선
                improvement_prompt = self.meta_prompt_template.format(
                    prompt=current_prompt,
                    best_score=best_case['score'],
                    best_question=best_case['question'],
                    best_expected=best_case['expected'],
                    best_actual=best_case['actual'],
                    best_reason=best_case['reason'],
                    worst_score=worst_case['score'],
                    worst_question=worst_case['question'],
                    worst_expected=worst_case['expected'],
                    worst_actual=worst_case['actual'],
                    worst_reason=worst_case['reason']
                )
                improved_prompt = self.meta_prompt_model.ask("", system_prompt=improvement_prompt)
                if improved_prompt and improved_prompt.strip():
                    current_prompt = improved_prompt.strip()
                    self.logger.info(f"개선된 프롬프트: {current_prompt}")
                else:
                    self.logger.warning("프롬프트 개선에 실패했습니다. 튜닝을 종료합니다.")
                    if self.progress_callback:
                        self.progress_callback(num_iterations, len(test_cases))
                    break
            elif use_meta_prompt:
                self.logger.info(f"평균 점수가 평가 임계값({evaluation_score_threshold}) 이상이므로 프롬프트를 개선하지 않습니다.")
                if self.progress_callback:
                    self.progress_callback(num_iterations, len(test_cases))
                break
        
        return iteration_results 