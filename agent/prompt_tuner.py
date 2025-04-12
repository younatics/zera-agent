from typing import List, Dict, Optional
import logging
from common.api_client import Model
import os

class PromptTuner:
    """
    A class for automatically fine-tuning system prompts for LLMs.
    """
    
    def __init__(self, model_name: str = "solar", evaluator_model_name: str = "solar"):
        """
        Initialize the PromptTuner with specific models.
        
        Args:
            model_name (str): The name of the model to use for tuning (default: "solar")
            evaluator_model_name (str): The name of the model to use for evaluation (default: "claude")
        """
        self.model = Model(model_name)
        self.evaluator = Model(evaluator_model_name)
        self.evaluation_history: List[Dict] = []
        self.best_prompt: Optional[str] = None
        self.best_score: float = 0.0
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
            score_str = self.evaluator.ask(evaluation_prompt).strip()
            score = float(score_str)
            self.logger.info(f"Evaluation score: {score}")
            return score
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
            return fallback_score
    
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
            score = self._evaluate_response(response, test_case['expected_output'])
            total_score += score
            
            detailed_responses.append({
                'input': test_case['input'],
                'response': response,
                'expected': test_case['expected_output'],
                'score': score
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
            variation = self.model.ask(self.meta_prompt_template.format(prompt=prompt))
            if variation and variation.strip():
                variations.append(variation.strip())
        
        return variations
    
    def tune(self, initial_prompt: str, test_cases: List[Dict], iterations: int = 5) -> List[Dict]:
        """
        Tune a system prompt through multiple iterations.
        
        Args:
            initial_prompt (str): The initial system prompt
            test_cases (List[Dict]): List of test cases for evaluation
            iterations (int): Number of tuning iterations
            
        Returns:
            List[Dict]: List of results for each iteration, containing:
                - prompt: The prompt used in this iteration
                - avg_score: Average score across all test cases
                - best_score: Best score among test cases
                - worst_score: Worst score among test cases
                - detailed_responses: Detailed responses for each test case
        """
        current_prompt = initial_prompt
        self.best_prompt = initial_prompt
        self.best_score = 0.0
        results = []
        
        for i in range(iterations):
            # Generate variations of the current prompt
            variations = self.generate_variations(current_prompt)
            
            iteration_results = []
            # Evaluate each variation
            for variation in variations:
                evaluation_result = self.evaluate_prompt(variation, test_cases)
                score = evaluation_result['total_score']
                
                # Update best prompt if this variation performs better
                if score > self.best_score:
                    self.best_score = score
                    self.best_prompt = variation
                
                # Record evaluation history
                self.evaluation_history.append({
                    'iteration': i,
                    'prompt': variation,
                    'score': score,
                    'detailed_responses': evaluation_result['detailed_responses']
                })
                
                # Collect scores for this variation
                scores = [r['score'] for r in evaluation_result['detailed_responses']]
                iteration_results.append({
                    'prompt': variation,
                    'avg_score': score,
                    'best_score': max(scores),
                    'worst_score': min(scores),
                    'detailed_responses': evaluation_result['detailed_responses']
                })
            
            # Add the best result from this iteration
            best_iteration_result = max(iteration_results, key=lambda x: x['avg_score'])
            results.append(best_iteration_result)
            
            # Update current prompt to the best performing one
            current_prompt = best_iteration_result['prompt']
        
        return results 