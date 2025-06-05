import json
from evaluation.dataset_evaluator.bbh_evaluator import BBHEvaluator

# 원본 파일 읽기
with open('evaluation/results/BBHEvaluator_gpt-3.5-turbo_20250520_090406.json', 'r') as f:
    data = json.load(f)

# BBH 평가기 초기화 (등록된 모델명 사용)
evaluator = BBHEvaluator(model_name="gpt4o", model_version="gpt4o")

# 각 샘플 재평가
correct = 0
for sample in data['samples']:
    is_correct = evaluator.evaluate_response(sample['model_response'], {'answer': sample['actual_answer']})
    sample['is_correct'] = is_correct
    if is_correct:
        correct += 1

# 전체 정답 수 업데이트
data['correct'] = correct

# 새로운 파일로 저장
with open('evaluation/results/BBHEvaluator_gpt-3.5-turbo_20250520_090406_new.json', 'w') as f:
    json.dump(data, f, indent=2) 