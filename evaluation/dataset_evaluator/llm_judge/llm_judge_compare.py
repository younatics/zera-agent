import json
import pandas as pd
from pathlib import Path
import logging
from openai import OpenAI
import anthropic
import time
from typing import List, Dict, Tuple
import numpy as np
from scipy import stats
import os
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API client setup
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
solar_api_key = os.getenv("SOLAR_API_KEY")

if not openai_api_key and not anthropic_api_key and not solar_api_key:
    raise ValueError("API key is not set.")

openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None
anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key) if anthropic_api_key else None
solar_client = OpenAI(
    api_key=solar_api_key,
    base_url="https://api.upstage.ai/v1"
) if solar_api_key else None

class LLMJudge:
    def __init__(self, model_type: str = "openai"):
        """
        Args:
            model_type (str): Model type to use ("openai", "anthropic", "solar")
        """
        self.model_type = model_type
        self.openai_client = openai_client
        self.anthropic_client = anthropic_client
        self.solar_client = solar_client
        
        self.system_prompt = """You are an expert evaluator specialized in comparing AI-generated summaries. 
Your task is to carefully analyze two summaries and determine which one is better based on the following criteria:

1. Content Coverage:
   - Does the summary capture all key points from the original text?
   - Are important details preserved while eliminating redundancy?

2. Coherence and Flow:
   - Is the summary logically structured and easy to follow?
   - Do the sentences connect smoothly?

3. Conciseness:
   - Is the summary appropriately concise without losing essential information?
   - Are there any unnecessary repetitions or verbose expressions?

4. Language Quality:
   - Is the language clear and professional?
   - Are there any grammatical or syntactical errors?

5. Originality Preservation:
   - Does the summary maintain the original author's tone and style?
   - Are the main arguments and conclusions accurately represented?

Please compare the two summaries (A and B) and choose the better one. 
You must select one summary as better. Do not respond with "both are equally good" or similar.

Your response should be in the format:
Winner: [A/B]
Reason: [Brief explanation of why the chosen summary is better]

Focus on objective evaluation criteria and provide specific examples from the text to support your decision."""

    def compare_responses(self, question: str, response_a: str, response_b: str) -> Tuple[str, str]:
        user_prompt = f"""Original Text: {question}

Summary A: {response_a}

Summary B: {response_b}

Please evaluate these two summaries and determine which one is better. 
Your response should clearly indicate the winner (A or B) and provide a detailed explanation of your choice."""

        try:
            if self.model_type == "openai" and self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0
                )
                result = response.choices[0].message.content.strip()
            elif self.model_type == "anthropic" and self.anthropic_client:
                response = self.anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    max_tokens=2000,
                    temperature=0.0,
                    system=self.system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ]
                )
                result = response.content[0].text.strip()
            elif self.model_type == "solar" and self.solar_client:
                response = self.solar_client.chat.completions.create(
                    model="solar-pro",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0                
                    )
                result = response.choices[0].message.content.strip()
            else:
                raise ValueError(f"Unsupported model type: {self.model_type} or API client not initialized")
            
            # Parse Winner information from first line
            first_line = result.split('\n')[0].strip()
            if first_line.startswith('Winner:'):
                winner = first_line.split(':')[1].strip()
                if winner in ['A', 'B']:
                    return winner, result
            return None, result
        except Exception as e:
            logger.error(f"Error in comparing responses: {e}")
            return None, str(e)

def load_results(file_path: str) -> dict:
    """Load results from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_responses(results: dict) -> List[Dict]:
    """Extract questions and responses from results."""
    samples = []
    for sample in results['samples']:
        samples.append({
            'question': sample.get('question', ''),
            'response': sample.get('model_response', ''),
            'actual_answer': sample.get('actual_answer', '')
        })
    return samples

def calculate_statistical_significance(wins: List[int], total: int) -> float:
    """Calculate statistical significance."""
    if total == 0:
        return 1.0
    p_value = stats.binomtest(sum(wins), total, p=0.5).pvalue
    return p_value

def main():
    # Result file paths
    zera_path = "evaluation/results/zera_score.json"
    baseline_path = "evaluation/results/base_score.json"
    
    # Load results
    zera_results = load_results(zera_path)
    baseline_results = load_results(baseline_path)
    
    # Extract responses
    zera_samples = extract_responses(zera_results)[:10]  # Limit to 10
    baseline_samples = extract_responses(baseline_results)[:10]  # Limit to 10
    
    # Initialize LLM Judge (using Claude)
    judge = LLMJudge()
    
    # Store comparison results
    comparison_results = []
    wins = []
    
    # Variables for intermediate result saving
    batch_size = 50  # Save results every 50
    output_path = Path("evaluation/llm_judge/comparison_results_10.csv")
    
    # Compare each sample
    for i, (zera, baseline) in enumerate(zip(zera_samples, baseline_samples)):
        logger.info(f"Comparing sample {i+1}/{len(zera_samples)}")
        
        winner, reason = judge.compare_responses(
            baseline['question'],
            zera['response'],
            baseline['response']
        )
        
        if winner:
            wins.append(1 if winner == 'A' else 0)
            comparison_results.append({
                'question': baseline['question'],
                'zera_response': zera['response'],
                'baseline_response': baseline['response'],
                'winner': winner,
                'reason': reason
            })
        
        # Save intermediate results
        if (i + 1) % batch_size == 0 or (i + 1) == len(zera_samples):
            results_df = pd.DataFrame(comparison_results)
            results_df.to_csv(output_path, index=False)
            logger.info(f"Saved intermediate results after {i+1} samples")
            
            # Output intermediate statistics
            total_comparisons = len(wins)
            zera_wins = sum(wins)
            baseline_wins = total_comparisons - zera_wins
            p_value = calculate_statistical_significance(wins, total_comparisons)
            
            print(f"\nProgress: {i+1}/{len(zera_samples)} samples")
            print(f"ZERA wins: {zera_wins} ({zera_wins/total_comparisons*100:.1f}%)")
            print(f"Baseline wins: {baseline_wins} ({baseline_wins/total_comparisons*100:.1f}%)")
            print(f"p-value: {p_value:.4f}")
        
        # Set appropriate wait time considering API call limits
        time.sleep(1)
    
    # Output final results
    total_comparisons = len(wins)
    zera_wins = sum(wins)
    baseline_wins = total_comparisons - zera_wins
    p_value = calculate_statistical_significance(wins, total_comparisons)
    
    print("\nFinal A/B Test Results (100 samples):")
    print(f"Total comparisons: {total_comparisons}")
    print(f"ZERA wins: {zera_wins} ({zera_wins/total_comparisons*100:.1f}%)")
    print(f"Baseline wins: {baseline_wins} ({baseline_wins/total_comparisons*100:.1f}%)")
    print(f"p-value: {p_value:.4f}")
    
    print(f"\nResults saved to {output_path}.")

if __name__ == "__main__":
    main() 