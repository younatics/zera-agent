"""
SamSum 데이터셋 평가 예제

이 예제는 SamSum 데이터셋을 사용하여 모델의 대화 요약 능력을 평가합니다.
기존 프롬프트와 제라 프롬프트를 동일한 샘플에 대해 비교 평가합니다.
"""

from evaluation.base.main import main
import sys

def run_samsum_example(model="claude", model_version="claude-3-sonnet-20240229"):
    # 명령줄 인자 설정
    sys.argv = [
        "example.py",
        "--dataset", "samsum",
        "--model", model,
        "--model_version", model_version,
        # 기존 프롬프트
        "--base_system_prompt", "You are a summarization assistant. Summarize the following article in 2–3 sentences, focusing on the main idea.",
        "--base_user_prompt", "Article:",
        # 제라 프롬프트
        "--zera_system_prompt", "You are an AI assistant adept at accurately summarizing short conversations. Focus solely on explicitly mentioned factual details such as people's names, specific items, tasks to perform, exact locations, precise time references, and explicit instructions. Strictly avoid speculation, inference, humor, or assumptions about unstated motivations or implicit meanings. Provide summaries that are concise, factual, and explicitly reflect only the provided conversation.",
        "--zera_user_prompt", """Summarize the following conversation explicitly, accurately, and concisely. Clearly state only explicitly mentioned information and include specific people, items, explicit tasks requested, exact locations, and precise instructions or timelines. Do not speculate or infer unstated emotions, motivations, or beliefs.

Examples:

Conversation:  
Lisa: Can you pick up something from the pharmacy on your way home?  
John: Sure, which pharmacy?  
Lisa: The one next to Starbucks on 5th street.  
John: Okay. What do you need?  
Lisa: Allergy medication, the exact same one you bought last month.  
John: Got it. Anything else?  
Lisa: No, that's all. Thanks!

Answer:  
Lisa asks John to pick up allergy medication at the pharmacy next to Starbucks on 5th street, specifying it must be the exact same kind he bought last month.

Conversation:  
Taylor: I think I left my power bank on your place yesterday  
Owen: My brother was using it  
Taylor: Can you please ask him to bring it to me, I really need it  
Owen: He is not at home right now  
Taylor: But I am going at my grandpa's and i need it badly  
Owen: Dont worry, you can use mine. will be in front of your house in half an hour  
Taylor: Thanks :)

Answer:  
Taylor left his power bank at Owen's place yesterday and urgently needs it, but Owen's brother was using it and is currently not at home. Owen offers to bring his own power bank to Taylor's house in half an hour.

Now summarize this conversation explicitly and concisely:

Conversation:  
Alice: Hi, dear, you still at the office.  
Rob: I am. Why?  
Alice: Good. On your way back stop at the mall please.  
Rob: Any particular reason?  
Alice: Yeah. I want you to go this store, next to the H&M.  
Rob: The one with baby stuff:0?!  
Alice: No! The one on the other side.  
Rob: Frankly, I don't recall it.  
Alice: Oh, it's got all sorts of frames, pictures and mirrors.  
Rob: Yeah, I remember now. What do I want from there?  
Alice: Look for a mirror. Just like the one we had in the hall.  
Rob: What d'you mean we had? What happened?  
Alice: It sort of broke.  
Rob: Just like that? All by itself?  
Alice: More or less, yes.  
Rob: Doesn't it supposed to mean that we are out of luck for like 7 years.  
Alice: Exactly:(!  
Alice: So be quick and buy one looking exactly the same.  
Rob: You mean, so the luck doesn't even notice?  
Alice: You got. I knew I married a smart guy;)!

Answer:

TASK_HINTS:
  - Explicitly identify people, clearly stated locations, explicitly requested items or tasks, and timelines.
  - Avoid speculation, inference, humor, or emotional interpretation not explicitly mentioned.
  - Double-check exact locations explicitly stated to avoid confusion or misreporting.
  - Preserve explicit ordering of requested tasks and instructions.""",
        "--num_samples", "500",
        # # 모델 파라미터
        # "--temperature", "0.2",  # 더 결정적인 응답을 위해 낮은 temperature 사용
        # "--top_p", "0.9"
    ]
    
    # 평가 실행
    main()

if __name__ == "__main__":
    run_samsum_example() 