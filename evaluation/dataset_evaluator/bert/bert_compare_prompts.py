from bert_score import score
import pandas as pd
import json
import logging
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO)

def load_results(file_path: str) -> dict:
    """JSON 파일에서 결과를 로드합니다."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_responses(results: dict) -> tuple:
    """결과에서 모델 응답과 실제 답변을 추출합니다."""
    model_responses = []
    references = []
    
    for sample in results['samples']:
        model_responses.append(sample['model_response'])
        references.append(sample['actual_answer'])
    
    return model_responses, references

def main():
    # 결과 파일 경로
    zera_path = "evaluation/bert/zera_score.json"
    baseline_path = "evaluation/bert/base_score.json"
    
    # 결과 로드
    zera_results = load_results(zera_path)
    baseline_results = load_results(baseline_path)
    
    # 응답 추출
    zera_responses, zera_refs = extract_responses(zera_results)
    baseline_responses, baseline_refs = extract_responses(baseline_results)
    
    # BERTScore 계산
    P1, R1, F1_zera = score(zera_responses, zera_refs, lang="en", verbose=True)
    P2, R2, F1_base = score(baseline_responses, baseline_refs, lang="en", verbose=True)
    
    # 결과 출력
    print("\nZERA 프롬프트 평가 결과:")
    print(f"Precision: {P1.mean():.3f}")
    print(f"Recall: {R1.mean():.3f}")
    print(f"F1: {F1_zera.mean():.3f}")
    
    print("\nBaseline 프롬프트 평가 결과:")
    print(f"Precision: {P2.mean():.3f}")
    print(f"Recall: {R2.mean():.3f}")
    print(f"F1: {F1_base.mean():.3f}")
    
    # 결과 비교
    print("\n프롬프트 비교:")
    print(f"Precision 차이: {P1.mean() - P2.mean():.3f}")
    print(f"Recall 차이: {R1.mean() - R2.mean():.3f}")
    print(f"F1 차이: {F1_zera.mean() - F1_base.mean():.3f}")
    
    # 결과를 DataFrame으로 저장
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
    
    # 결과 저장
    output_path = Path("evaluation/bert/comparison_results.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\n결과가 {output_path}에 저장되었습니다.")

if __name__ == "__main__":
    main() 