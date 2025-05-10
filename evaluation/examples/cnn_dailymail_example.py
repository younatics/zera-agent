"""
CNN/DailyMail 데이터셋 평가 예제

이 예제는 CNN/DailyMail 데이터셋을 사용하여 모델의 뉴스 요약 능력을 평가합니다.
기존 프롬프트와 제라 프롬프트를 동일한 샘플에 대해 비교 평가합니다.
"""

from evaluation.base.main import main
import sys

def run_cnn_dailymail_example(model="gpt4o", model_version="gpt-3.5-turbo"):
    # 명령줄 인자 설정
    sys.argv = [
        "example.py",
        "--dataset", "cnn_dailymail",
        "--model", model,
        "--model_version", model_version,
        # 기존 프롬프트
        # "--base_system_prompt", "You are a summarization assistant. Summarize the following article in 2–3 sentences, focusing on the main idea.",
        # "--base_user_prompt", "Article:",
        # 제라 프롬프트
        "--zera_system_prompt", "Read thoroughly and reason clearly about the provided text to first identify key explicit details. After logically extracting and determining these facts, present your summary strictly as concise, factual bullet points.",
        "--zera_user_prompt", "Summarize the provided text into concise bullet points. Include only key explicit details: names, ages, numbers, dates, specific locations, and clearly mentioned events. Omit any interpretations, assumptions, or generalizations.\n\nExample:\n\nText:\n\"England and Wales Cricket Board managing director Paul Downton insists he retains 'every faith' in coach Peter Moores despite England's humiliating exit at the World Cup on Monday. A 15-run defeat to Bangladesh saw England crash out in the group stages of the one-day tournament after a dismal campaign that included four defeats in five matches. Moores' tactics and team selection have come under heavy scrutiny since he was appointed head coach 11 months ago but Downton insists the former Lancashire coach remains the right man for the job.\"\n\nExpected Summary:\n- England exited World Cup at group stage after 15-run defeat to Bangladesh.\n- England lost four out of five matches in the tournament.\n- Coach Peter Moores appointed England head coach 11 months ago.\n- ECB managing director Paul Downton expresses 'every faith' in Moores despite criticism.\n\nArticle:",
        "--num_samples", "1000",
        # 모델 파라미터
        # "--temperature", "0.2",  # 더 결정적인 응답을 위해 낮은 temperature 사용
        # "--top_p", "0.9"
    ]
    
    # 평가 실행
    main()

if __name__ == "__main__":
    run_cnn_dailymail_example() 