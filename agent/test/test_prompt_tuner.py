from ..prompt_tuner import PromptTuner

def test_prompt_tuner():
    print("프롬프트 튜너 테스트 시작")
    
    # 테스트 케이스 정의
    test_cases = [
        {
            'input': '안녕하세요',
            'expected_output': '안녕하세요! 저는 AI 어시스턴트입니다. 어떻게 도와드릴까요?'
        },
        {
            'input': '오늘 날씨가 어때요?',
            'expected_output': '죄송합니다. 저는 날씨 정보를 제공할 수 없습니다.'
        }
    ]
    
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
        print(f"Prompt: {record['prompt']}\n")

if __name__ == "__main__":
    test_prompt_tuner() 