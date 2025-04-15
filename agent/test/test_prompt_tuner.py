import csv
import os
from ..prompt_tuner import PromptTuner

def load_test_cases_from_csv(csv_file):
    test_cases = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_case = {
                'input': row['question'],
                'expected_output': row['expected_answer']
            }
            test_cases.append(test_case)
    return test_cases

def test_prompt_tuner():
    print("프롬프트 튜너 테스트 시작")
    
    # CSV 파일에서 테스트 케이스 로드
    csv_file = "input.csv"  # 프로젝트 루트의 input.csv
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found")
        return
    
    test_cases = load_test_cases_from_csv(csv_file)
    print(f"\nLoaded {len(test_cases)} test cases from {csv_file}")
    
    # 초기 프롬프트
    initial_prompt = "You are a helpful AI assistant. Be polite and concise in your responses."
    
    # PromptTuner 인스턴스 생성 (기본값으로 solar 사용)
    tuner = PromptTuner()
    
    # 프롬프트 튜닝 실행
    print("\n=== 프롬프트 튜닝 시작 ===")
    best_prompt = tuner.tune(initial_prompt, test_cases, iterations=3)
    
    # 결과 출력
    print("\n=== 튜닝 결과 ===")
    print(f"최고 점수: {tuner.best_score}")
    print(f"최적 프롬프트: {best_prompt}")
    
    print("\n=== 평가 기록 ===")
    for record in tuner.evaluation_history:
        print(f"Iteration {record['iteration']}: Score {record['score']}")
        print(f"System Prompt: {record['system_prompt']}")
        print(f"User Prompt: {record['user_prompt']}\n")

if __name__ == "__main__":
    test_prompt_tuner() 