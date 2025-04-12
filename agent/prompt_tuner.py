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
    
    def _evaluate_response(self, response: str, expected: str) -> float:
        """
        Evaluate a single response against the expected output using the evaluator model.
        
        Args:
            response (str): The model's response
            expected (str): The expected output
            
        Returns:
            float: Score between 0.0 and 1.0
        """
        self.logger.info("Evaluating response:")
        self.logger.info(f"Actual response: {response}")
        self.logger.info(f"Expected response: {expected}")
        
        evaluation_prompt = self.evaluation_prompt_template.format(
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
            return score, reason
        except (ValueError, TypeError):
            # 숫자로 변환할 수 없는 경우 기본 평가 방식 사용
            self.logger.warning("Failed to get valid score from evaluator, using fallback evaluation method")
            response = response.lower()
            expected = expected.lower()
            
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
            test_cases (List[Dict]): List of test cases, each containing 'input' and 'expected_output'
            
        Returns:
            Dict: Evaluation results including total score and detailed responses
        """
        total_score = 0.0
        detailed_responses = []
        
        for test_case in test_cases:
            response = self.model.ask(test_case['input'], system_message=prompt)
            score, reason = self._evaluate_response(response, test_case['expected_output'])
            total_score += score
            
            detailed_responses.append({
                'input': test_case['input'],
                'response': response,
                'expected': test_case['expected_output'],
                'score': score,
                'evaluation_reason': reason
            })
        
        return {
            'total_score': total_score / len(test_cases),
            'detailed_responses': detailed_responses
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
    
    def tune_prompt(self, initial_prompt: str, test_cases: List[Dict], num_iterations: int = 3, score_threshold: Optional[float] = None, evaluation_score_threshold: float = 0.8, use_meta_prompt: bool = True) -> str:
        current_prompt = initial_prompt
        best_prompt = initial_prompt
        best_score = 0.0
        
        for iteration in range(num_iterations):
            self.logger.info(f"\nIteration {iteration + 1}/{num_iterations}")
            
            # 각 테스트 케이스에 대해 순차적으로 평가하고 프롬프트를 조정
            for i, test_case in enumerate(test_cases):
                self.logger.info(f"\nTest Case {i + 1}/{len(test_cases)}")
                self.logger.info(f"Question: {test_case['question']}")
                
                # 현재 프롬프트로 응답 생성
                response = self.model.ask(test_case['question'], current_prompt)
                self.logger.info(f"Response: {response}")
                
                # 응답 평가
                score, reason = self._evaluate_response(response, test_case['expected'])
                self.logger.info(f"Score: {score}")
                self.logger.info(f"Evaluation reason: {reason}")
                
                # 프로그레스 바 업데이트
                if self.progress_callback:
                    self.progress_callback(iteration + 1, i + 1)
                
                # 평가 결과를 바탕으로 프롬프트 조정 (프롬프트 개선이 켜져있을 때만)
                if use_meta_prompt and score < evaluation_score_threshold:  # 점수가 score_threshold 보다 낮은 경우
                    self.logger.info("프롬프트 개선 중...")
                    # 메타프롬프트를 사용하여 현재 프롬프트를 개선
                    improvement_prompt = self.meta_prompt_template.format(
                        prompt=current_prompt,
                        question=test_case['question'],
                        expected=test_case['expected'],
                        evaluation_reason=reason
                    )
                    improved_prompt = self.meta_prompt_model.ask("", improvement_prompt)
                    current_prompt = improved_prompt
                    self.logger.info(f"개선된 프롬프트: {current_prompt}")
                
                # 현재까지의 최고 점수와 비교
                if score > best_score:
                    best_score = score
                    best_prompt = current_prompt
                    self.logger.info(f"새로운 최고 점수: {best_score}")
                
                # 평가 기록 저장
                self.evaluation_history.append({
                    'iteration': iteration + 1,
                    'test_case': i + 1,
                    'prompt': current_prompt,
                    'question': test_case['question'],
                    'response': response,
                    'expected': test_case['expected'],
                    'score': score,
                    'evaluation_reason': reason
                })
                
                # score_threshold가 None이 아니고, 점수가 임계값 이상이면 중단
                if score_threshold is not None and score >= score_threshold:
                    self.logger.info(f"{score_threshold} 이상의 점수({score})를 달성하여 iteration을 중단합니다.")
                    return best_prompt
        
        return best_prompt 