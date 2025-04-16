import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# API 키 확인
keys = ['SOLAR_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY']

print("API 키 확인:")
for key in keys:
    value = os.getenv(key)
    if value:
        print(f"{key}: {'*' * len(value)} (설정됨)")
    else:
        print(f"{key}: (설정되지 않음)") 