import json
from evaluation.dataset_evaluator.bbh_evaluator import BBHEvaluator

# Read original file
with open('evaluation/results/BBHEvaluator_gpt-3.5-turbo_20250520_090406.json', 'r') as f:
    data = json.load(f)

# Initialize BBH evaluator (using registered model name)
evaluator = BBHEvaluator(model_name="gpt4o", model_version="gpt4o")

# Re-evaluate each sample
correct = 0
for sample in data['samples']:
    is_correct = evaluator.evaluate_response(sample['model_response'], {'answer': sample['actual_answer']})
    sample['is_correct'] = is_correct
    if is_correct:
        correct += 1

# Update total correct count
data['correct'] = correct

# Save to new file
with open('evaluation/results/BBHEvaluator_gpt-3.5-turbo_20250520_090406_new.json', 'w') as f:
    json.dump(data, f, indent=2) 