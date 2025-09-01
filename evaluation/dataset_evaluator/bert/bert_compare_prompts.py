from bert_score import score
import pandas as pd
import json
import logging
from pathlib import Path

# Logging setup
logging.basicConfig(level=logging.INFO)

def load_results(file_path: str) -> dict:
    """Load results from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_responses(results: dict) -> tuple:
    """Extract model responses and actual answers from results."""
    model_responses = []
    references = []
    
    for sample in results['samples']:
        model_responses.append(sample['model_response'])
        references.append(sample['actual_answer'])
    
    return model_responses, references

def main():
    # Result file paths
    zera_path = "evaluation/bert/zera_score.json"
    baseline_path = "evaluation/bert/base_score.json"
    
    # Load results
    zera_results = load_results(zera_path)
    baseline_results = load_results(baseline_path)
    
    # Extract responses
    zera_responses, zera_refs = extract_responses(zera_results)
    baseline_responses, baseline_refs = extract_responses(baseline_results)
    
    # Calculate BERTScore
    P1, R1, F1_zera = score(zera_responses, zera_refs, lang="en", verbose=True)
    P2, R2, F1_base = score(baseline_responses, baseline_refs, lang="en", verbose=True)
    
    # Output results
    print("\nZERA Prompt Evaluation Results:")
    print(f"Precision: {P1.mean():.3f}")
    print(f"Recall: {R1.mean():.3f}")
    print(f"F1: {F1_zera.mean():.3f}")
    
    print("\nBaseline Prompt Evaluation Results:")
    print(f"Precision: {P2.mean():.3f}")
    print(f"Recall: {R2.mean():.3f}")
    print(f"F1: {F1_base.mean():.3f}")
    
    # Compare results
    print("\nPrompt Comparison:")
    print(f"Precision difference: {P1.mean() - P2.mean():.3f}")
    print(f"Recall difference: {R1.mean() - R2.mean():.3f}")
    print(f"F1 difference: {F1_zera.mean() - F1_base.mean():.3f}")
    
    # Save results as DataFrame
    results_df = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1'],
        'ZERA': [P1.mean(), R1.mean(), F1_zera.mean()],
        'Baseline': [P2.mean(), R2.mean(), F1_base.mean()],
        'Difference': [
            P1.mean() - P2.mean(),
            R1.mean() - R2.mean(),
            F1_zera.mean() - F1_base.mean()
        ]
    })
    
    # Save results
    output_path = Path("evaluation/bert/comparison_results.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}.")

if __name__ == "__main__":
    main() 