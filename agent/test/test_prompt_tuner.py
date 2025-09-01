import csv
import os
from ..prompt_tuner import PromptTuner

def load_test_cases_from_csv(csv_file):
    test_cases = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_case = {
                'question': row['question'],
                'expected': row['expected_answer']
            }
            test_cases.append(test_case)
    return test_cases

def test_prompt_tuner():
    print("Starting prompt tuner test")
    
    # Load test cases from CSV file
    csv_file = "input.csv"  # input.csv in project root
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found")
        return
    
    test_cases = load_test_cases_from_csv(csv_file)
    print(f"\nLoaded {len(test_cases)} test cases from {csv_file}")
    
    # Initial prompts
    initial_system_prompt = "You are a helpful AI assistant."
    initial_user_prompt = "Be polite and concise in your responses."
    
    # Create PromptTuner instance (using solar as default)
    tuner = PromptTuner()
    
    # Execute prompt tuning
    print("\n=== Starting Prompt Tuning ===")
    iteration_results = tuner.tune_prompt(
        initial_system_prompt=initial_system_prompt,
        initial_user_prompt=initial_user_prompt,
        initial_test_cases=test_cases,
        num_iterations=3
    )
    
    # Output results
    print("\n=== Tuning Results ===")
    best_result = max(iteration_results, key=lambda x: x.avg_score)
    print(f"Best average score: {best_result.best_avg_score}")
    print(f"Best individual score: {best_result.best_sample_score}")
    print(f"Optimal system prompt: {best_result.system_prompt}")
    print(f"Optimal user prompt: {best_result.user_prompt}")
    
    print("\n=== Evaluation Records ===")
    for result in iteration_results:
        print(f"Iteration {result.iteration}:")
        print(f"Average score: {result.avg_score}")
        print(f"Standard deviation: {result.std_dev}")
        print(f"Top3 average score: {result.top3_avg_score}")
        print(f"System prompt: {result.system_prompt}")
        print(f"User prompt: {result.user_prompt}\n")

if __name__ == "__main__":
    test_prompt_tuner() 