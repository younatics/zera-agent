from api_client import Model
import os
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# API 키 설정
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
solar_client = OpenAI(
    api_key=os.getenv("SOLAR_API_KEY"),
    base_url="https://api.upstage.ai/v1"
)

def test_model_initialization():
    print("\n=== 모델 초기화 테스트 ===")
    try:
        # 올바른 모델 이름으로 테스트
        gpt_model = Model("gpt4o")
        print(f"GPT 모델 초기화 성공: {gpt_model.name}")
        
        claude_model = Model("claude")
        print(f"Claude 모델 초기화 성공: {claude_model.name}")
        
        solar_model = Model("solar")
        print(f"Solar Pro 모델 초기화 성공: {solar_model.name}")
        
        # 잘못된 모델 이름으로 테스트
        try:
            wrong_model = Model("invalid_model")
        except ValueError as e:
            print(f"잘못된 모델 이름 테스트 성공: {str(e)}")
            
    except Exception as e:
        print(f"테스트 실패: {str(e)}")

def test_available_models():
    print("\n=== 사용 가능한 모델 목록 테스트 ===")
    try:
        available_models = Model.get_available_models()
        print(f"사용 가능한 모델: {available_models}")
    except Exception as e:
        print(f"테스트 실패: {str(e)}")

def test_model_question():
    print("\n=== 모델 질문 테스트 ===")
    try:
        # 간단한 테스트 질문
        test_prompt = "안녕하세요! 당신은 누구인가요?"
        
        # GPT 모델 테스트
        print("\n[GPT 모델 테스트]")
        gpt_model = Model("gpt4o")
        gpt_answer = gpt_model.ask(openai_client, test_prompt)
        print(f"GPT 응답: {gpt_answer}")
        
        # Claude 모델 테스트
        print("\n[Claude 모델 테스트]")
        claude_model = Model("claude")
        claude_answer = claude_model.ask(anthropic_client, test_prompt)
        print(f"Claude 응답: {claude_answer}")
        
        # Solar Pro 모델 테스트
        print("\n[Solar Pro 모델 테스트]")
        solar_model = Model("solar")
        solar_answer = solar_model.ask(solar_client, test_prompt)
        print(f"Solar Pro 응답: {solar_answer}")
        
    except Exception as e:
        print(f"테스트 실패: {str(e)}")

if __name__ == "__main__":
    print("API 클라이언트 테스트 시작")
    test_model_initialization()
    test_available_models()
    test_model_question()
    print("\n테스트 완료") 