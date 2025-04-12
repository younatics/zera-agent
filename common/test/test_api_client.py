from ..api_client import Model
import os
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 테스트 질문 정의
TEST_QUESTION = "안녕하세요! 당신은 누구인가요?"

# API 키 설정
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
solar_client = OpenAI(
    api_key=os.getenv("SOLAR_API_KEY"),
    base_url="https://api.upstage.ai/v1"
)

def test_model_initialization():
    print("\n=== 모델 초기화 테스트 ===")
    
    # GPT 모델 테스트
    try:
        model = Model("gpt4o")
        print("GPT 모델 초기화 성공:", model.name)
    except Exception as e:
        print("GPT 모델 초기화 실패:", str(e))
    
    # Claude 모델 테스트
    try:
        model = Model("claude")
        print("Claude 모델 초기화 성공:", model.name)
    except Exception as e:
        print("Claude 모델 초기화 실패:", str(e))
    
    # Solar Pro 모델 테스트
    try:
        model = Model("solar")
        print("Solar Pro 모델 초기화 성공:", model.name)
    except Exception as e:
        print("Solar Pro 모델 초기화 실패:", str(e))
    
    # 잘못된 모델 이름 테스트
    try:
        model = Model("invalid_model")
    except Exception as e:
        print("잘못된 모델 이름 테스트 성공:", str(e))

def test_available_models():
    print("\n=== 사용 가능한 모델 목록 테스트 ===")
    models = Model.get_available_models()
    print("사용 가능한 모델:", models)

def test_model_responses():
    print("\n=== 모델 질문 테스트 ===")
    
    # GPT 모델 테스트
    print("\n[GPT 모델 테스트]")
    gpt_model = Model("gpt4o")
    gpt_response = gpt_model.ask(TEST_QUESTION)
    print("GPT 응답:", gpt_response)
    
    # Claude 모델 테스트
    print("\n[Claude 모델 테스트]")
    claude_model = Model("claude")
    claude_response = claude_model.ask(TEST_QUESTION)
    print("Claude 응답:", claude_response)
    
    # Solar Pro 모델 테스트
    print("\n[Solar Pro 모델 테스트]")
    solar_model = Model("solar")
    solar_response = solar_model.ask(TEST_QUESTION)
    print("Solar Pro 응답:", solar_response)

if __name__ == "__main__":
    print("API 클라이언트 테스트 시작")
    test_model_initialization()
    test_available_models()
    test_model_responses()
    print("\n테스트 완료") 