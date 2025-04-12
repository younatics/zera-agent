from typing import List, Dict, Optional
import os
from openai import OpenAI
from anthropic import Anthropic
from common.api_client import Model

class PromptTuner:
    """
    A class for automatically fine-tuning system prompts for LLMs.
    """
    
    def __init__(self, model_name: str = "solar"):
        """
        Initialize the PromptTuner with a specific model.
        
        Args:
            model_name (str): The name of the model to use for tuning (default: "solar")
        """
        self.model = Model(model_name)
        self.client = self._create_client(model_name)
        self.evaluation_history: List[Dict] = []
        self.best_prompt: Optional[str] = None
        self.best_score: float = 0.0
    
    def _create_client(self, model_name: str):
        """Create appropriate client based on model name."""
        if model_name == "claude":
            return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        elif model_name == "solar":
            return OpenAI(
                api_key=os.getenv("SOLAR_API_KEY"),
                base_url="https://api.upstage.ai/v1"
            )
        else:  # gpt4o
            return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def evaluate_prompt(self, prompt: str, test_cases: List[Dict]) -> float:
        """
        Evaluate a system prompt using a set of test cases.
        
        Args:
            prompt (str): The system prompt to evaluate
            test_cases (List[Dict]): List of test cases, each containing 'input' and 'expected_output'
            
        Returns:
            float: The evaluation score (0.0 to 1.0)
        """
        total_score = 0.0
        
        for test_case in test_cases:
            response = self.model.ask(test_case['input'], system_message=prompt)
            # TODO: Implement proper response evaluation
            # For now, we'll use a simple string matching
            score = 1.0 if response == test_case['expected_output'] else 0.0
            total_score += score
        
        return total_score / len(test_cases)
    
    def generate_variations(self, prompt: str) -> List[str]:
        """
        Generate variations of a given prompt.
        
        Args:
            prompt (str): The original prompt
            
        Returns:
            List[str]: List of prompt variations
        """
        # TODO: Implement prompt variation generation
        # For now, return the original prompt
        return [prompt]
    
    def tune(self, initial_prompt: str, test_cases: List[Dict], iterations: int = 5) -> str:
        """
        Tune a system prompt through multiple iterations.
        
        Args:
            initial_prompt (str): The initial system prompt
            test_cases (List[Dict]): List of test cases for evaluation
            iterations (int): Number of tuning iterations
            
        Returns:
            str: The best performing prompt
        """
        current_prompt = initial_prompt
        self.best_prompt = initial_prompt
        self.best_score = self.evaluate_prompt(initial_prompt, test_cases)
        
        for i in range(iterations):
            # Generate variations of the current prompt
            variations = self.generate_variations(current_prompt)
            
            # Evaluate each variation
            for variation in variations:
                score = self.evaluate_prompt(variation, test_cases)
                
                # Update best prompt if this variation performs better
                if score > self.best_score:
                    self.best_score = score
                    self.best_prompt = variation
                
                # Record evaluation history
                self.evaluation_history.append({
                    'iteration': i,
                    'prompt': variation,
                    'score': score
                })
            
            # Update current prompt to the best performing one
            current_prompt = self.best_prompt
        
        return self.best_prompt 