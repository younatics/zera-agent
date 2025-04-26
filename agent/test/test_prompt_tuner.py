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
    print("프롬프트 튜너 테스트 시작")
    
    # CSV 파일에서 테스트 케이스 로드
    csv_file = "input.csv"  # 프로젝트 루트의 input.csv
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found")
        return
    
    test_cases = load_test_cases_from_csv(csv_file)
    print(f"\nLoaded {len(test_cases)} test cases from {csv_file}")
    
    # 초기 프롬프트
    initial_system_prompt = "You are a helpful AI assistant."
    initial_user_prompt = "Be polite and concise in your responses."
    
    # PromptTuner 인스턴스 생성 (기본값으로 solar 사용)
    tuner = PromptTuner()
    
    # 프롬프트 튜닝 실행
    print("\n=== 프롬프트 튜닝 시작 ===")
    iteration_results = tuner.tune_prompt(
        initial_system_prompt=initial_system_prompt,
        initial_user_prompt=initial_user_prompt,
        initial_test_cases=test_cases,
        num_iterations=3
    )
    
    # 결과 출력
    print("\n=== 튜닝 결과 ===")
    best_result = max(iteration_results, key=lambda x: x.avg_score)
    print(f"최고 평균 점수: {best_result.best_avg_score}")
    print(f"최고 개별 점수: {best_result.best_sample_score}")
    print(f"최적 시스템 프롬프트: {best_result.system_prompt}")
    print(f"최적 유저 프롬프트: {best_result.user_prompt}")
    
    print("\n=== 평가 기록 ===")
    for result in iteration_results:
        print(f"Iteration {result.iteration}:")
        print(f"평균 점수: {result.avg_score}")
        print(f"표준편차: {result.std_dev}")
        print(f"Top3 평균 점수: {result.top3_avg_score}")
        print(f"시스템 프롬프트: {result.system_prompt}")
        print(f"유저 프롬프트: {result.user_prompt}\n")

if __name__ == "__main__":
    test_prompt_tuner() 