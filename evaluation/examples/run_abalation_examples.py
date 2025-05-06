import sys
from evaluation.base.main import main as eval_main

def run_gsm8k_fewshot_ablation(model="gpt4o", model_version="gpt-3.5-turbo"):
    sys.argv = [
        "gsm8k_example.py",
        "--dataset", "gsm8k",
        "--model", model,
        "--model_version", model_version,
        "--base_system_prompt", "Provide the final answer prefixed with '####'.",
        "--base_user_prompt", "Question:\n",
        "--base_num_shots", "1",
        "--zera_system_prompt", "Provide the final answer prefixed with '####'.",
        "--zera_user_prompt", "Question:\n",
        "--zera_num_shots", "5",
        "--num_samples", "1319",
        
    ]
    eval_main()

def run_gsm8k_prompt_ablation(model="gpt4o", model_version="gpt-3.5-turbo"):
    ## 위에 예시만 뺸 것, 밑에 추론만 뺸것
    sys.argv = [
        "gsm8k_example.py",
        "--dataset", "gsm8k",
        "--model", model,
        "--model_version", model_version,
        "--base_system_prompt", "You are a logical reasoning assistant. First reason through the problem naturally and clearly—ignoring formatting. Only at the final stage, concisely summarize critical numeric calculations using the designated \"<<calculation=result>>\" notation, and clearly report your final numeric answer.",
        "--base_user_prompt", "Solve the following problem step-by-step with clear, logical reasoning. Afterward, briefly present each critical calculation step explicitly using the \"<<calculation=result>>\" notation, concluding with your final numeric answer clearly marked after \"####\".\n\nNow solve this problem:\n",
        "--zera_system_prompt", "You are a math assistant. For each problem, present critical numeric calculations using the designated \"<<calculation=result>>\" notation, and clearly report your final numeric answer.",
        "--zera_user_prompt", "Solve the following problem. Use the same format as shown in the example below.\n\nExample:\nQuestion: Sara buys 4 bouquets of roses, each bouquet has 12 roses. She gives away 9 roses. How many roses does Sara have left?\n\nCalculations:\nTotal roses bought: 4 * 12 = <<4*12=48>>\nRoses remaining: 48 - 9 = <<48-9=39>>\n\n#### 39\n\nNow solve this problem:\n",        
        "--num_samples", "1319"
    ]
    eval_main()


def run_bbh_fewshot_ablation(model="gpt4o", model_version="gpt-3.5-turbo"):
    sys.argv = [
        "bbh_example.py",
        "--dataset", "bbh",
        "--model", model,
        "--model_version", model_version,
        "--base_system_prompt", "Answer the following question.",
        "--base_user_prompt", "Question:",
        "--base_num_shots", "1",
        "--zera_system_prompt", "Answer the following question.",
        "--zera_user_prompt", "Question:",
        "--zera_num_shots", "5",
        "--num_samples", "1000"
    ]
    eval_main()

def run_bbh_prompt_ablation(model="gpt4o", model_version="gpt-3.5-turbo"):
    ## 위에 예시만 뺸 것, 밑에 추론만 뺸것

    sys.argv = [
        "bbh_example.py",
        "--dataset", "bbh",
        "--model", model,
        "--model_version", model_version,
        "--base_system_prompt", """You are a logical reasoning expert. Clearly reason each question step-by-step in natural, explicit language. Upon completing your analysis, distinctly separate it from your final concise answer, which must strictly follow the provided formatting instructions.

Solve these logical reasoning problems by explicitly thinking through them step-by-step before providing your final answer.
""",
        "--base_user_prompt", "Now, begin solving.",
        "--zera_system_prompt", """You are a logical expert. Upon completing your analysis, distinctly separate it from your final concise answer, which must strictly follow the provided formatting instructions.
""",
        "--zera_user_prompt", """
        Solve these logical problems:
        
        Examples:

Question: Sort alphabetically: horse dolphin cat bird
bird cat dolphin horse

Question: Jim scored higher than Sam. Sam scored higher than Eve. Who scored lowest?
Options:
(A) Jim
(B) Sam
(C) Eve
(C)

Question: Check validity:
"No cars can fly. All Toyotas are cars. Therefore, no Toyotas can fly."
Options:
(A) valid
(B) invalid
(A)

Now, begin solving.""",
        "--num_samples", "1000"
    ]
    eval_main()

def run_cnn_fewshot_dailymail_ablation(model="gpt4o", model_version="gpt-3.5-turbo"):
    sys.argv = [
        "example.py",
        "--dataset", "cnn_dailymail",
        "--model", model,
        "--model_version", model_version,
        "--base_system_prompt", "You are a summarization assistant. Summarize the following article in 2–3 sentences, focusing on the main idea.",
        "--base_user_prompt", "Article:",
        "--base_num_shots", "1",
        "--zera_system_prompt", "You are a summarization assistant. Summarize the following article in 2–3 sentences, focusing on the main idea.",
        "--zera_user_prompt", "Article:",
        "--zera_num_shots", "5",
        "--num_samples", "1000",
    ]
    eval_main()

def run_cnn_prompt_dailymail_ablation(model="gpt4o", model_version="gpt-3.5-turbo"):
    ## 위에 예시만 뺸 것, 밑에 추론만 뺸것

    sys.argv = [
        "example.py",
        "--dataset", "cnn_dailymail",
        "--model", model,
        "--model_version", model_version,
        "--base_system_prompt", "Read thoroughly and reason clearly about the provided text to first identify key explicit details. After logically extracting and determining these facts, present your summary strictly as concise, factual bullet points.",
        "--base_user_prompt", "Summarize the provided text into concise bullet points. Include only key explicit details: names, ages, numbers, dates, specific locations, and clearly mentioned events. Omit any interpretations, assumptions, or generalizations.\nArticle:",
        "--zera_system_prompt", "Read thoroughly about the provided text to first identify key explicit details. After logically extracting and determining these facts, present your summary strictly as concise, factual bullet points.",
        "--zera_user_prompt", """Summarize the provided text into concise bullet points. Include only key explicit details: names, ages, numbers, dates, specific locations, and clearly mentioned events. Omit any interpretations, assumptions, or generalizations.

Example:

Text:
"England and Wales Cricket Board managing director Paul Downton insists he retains 'every faith' in coach Peter Moores despite England's humiliating exit at the World Cup on Monday. A 15-run defeat to Bangladesh saw England crash out in the group stages of the one-day tournament after a dismal campaign that included four defeats in five matches. Moores' tactics and team selection have come under heavy scrutiny since he was appointed head coach 11 months ago but Downton insists the former Lancashire coach remains the right man for the job."

Expected Summary:
- England exited World Cup at group stage after 15-run defeat to Bangladesh.
- England lost four out of five matches in the tournament.
- Coach Peter Moores appointed England head coach 11 months ago.
- ECB managing director Paul Downton expresses 'every faith' in Moores despite criticism.
        """,
        "--num_samples", "1000",
    ]
    eval_main()


def run_mbpp_fewshot_ablation(model="gpt4o", model_version="gpt-3.5-turbo"):
    sys.argv = [
        "mbpp_example.py",
        "--dataset", "mbpp",
        "--model", model,
        "--model_version", model_version,
        "--base_system_prompt", "Write a Python function that satisfies the following specification.",
        "--base_user_prompt", "Problem:",
        "--base_num_shots", "1",
        "--zera_system_prompt", "Write a Python function that satisfies the following specification.",
        "--zera_user_prompt", "Problem:",
        "--zera_num_shots", "5",
        "--num_samples", "1000",
    ]
    eval_main()

def run_mbpp_prompt_ablation(model="gpt4o", model_version="gpt-3.5-turbo"):
    ## 위에 예시만 뺸 것, 밑에 추론만 뺸것

    sys.argv = [
        "mbpp_example.py",
        "--dataset", "mbpp",
        "--model", model,
        "--model_version", model_version,
        "--base_system_prompt", "You are an expert Python assistant, clearly reasoning through programming tasks before succinctly providing the final solution. Your answers must include clean, accurate Python code, with brief optional explanations or tests afterward only if they enhance clarity.",
        "--base_user_prompt", "Answer the following Python programming question clearly and concisely. Provide your complete solution as Python code. If helpful for clarity, you may briefly add an explanation or practical test cases after your code.\nProblem:",
        "--zera_system_prompt", "You are an expert Python assistant, programming tasks before succinctly providing the final solution. Your answers must include clean, accurate Python code, with brief optional explanations or tests afterward only if they enhance clarity.",
        "--zera_user_prompt", """Answer the following Python programming question clearly and concisely. Provide your complete solution as Python code. If helpful for clarity, you may briefly add an explanation or practical test cases after your code.

Example:

Question: Write a Python function to check whether all list elements are unique.

```python def all_unique(test_list):     return len(test_list) == len(set(test_list))```

(Return value is True if elements are unique, otherwise False.)
        
        """,
        "--num_samples", "1000",
    ]
    eval_main()

def run_mmlu_pro_fewshot_ablation(model="gpt4o", model_version="gpt-3.5-turbo"):
    sys.argv = [
        "mmlu_pro_example.py",
        "--dataset", "mmlu_pro",
        "--model", model,
        "--model_version", model_version,
        "--base_system_prompt", "Answer with only the letter of the correct choice.",
        "--base_user_prompt", "Question:",
        "--base_num_shots", "1",
        "--zera_system_prompt", "Answer with only the letter of the correct choice.",
        "--zera_user_prompt", "Question:",
        "--zera_num_shots", "5",
        "--num_samples", "1000"
    ]
    eval_main()

def run_mmlu_pro_prompt_ablation(model="gpt4o", model_version="gpt-3.5-turbo"):
        ## 위에 예시만 뺸 것, 밑에 추론만 뺸것
    sys.argv = [
        "mmlu_pro_example.py",
        "--dataset", "mmlu_pro",
        "--model", model,
        "--model_version", model_version,
        "--base_system_prompt", "You are an expert logical reasoning assistant. Carefully and naturally reason through each problem step-by-step. Keep your explanations brief, clear, and logical. Only after completing your reasoning, state your final choice strictly as the option letter enclosed in parentheses.",
        "--base_user_prompt", "Solve the following multiple-choice questions by reasoning concisely and logically step by step. Clearly explain the key steps that lead directly to your conclusion. Conclude by stating your final answer strictly as one letter in parentheses,\nQuestion:",
        "--zera_system_prompt", "You are an expert logical assistant. Carefully and naturally think through each problem. Keep your explanations brief, clear, and logical. Only after completing your reasoning, state your final choice strictly as the option letter enclosed in parentheses.",
        "--zera_user_prompt", "Solve the following multiple-choice questions by reasoning concisely and logically. Conclude by stating your final answer strictly as one letter in parentheses, e.g., \"(D)\".\n\nExample 1:\n\nQuestion: A microwave oven operates at 120 volts and draws a current of 2 amperes. How many watts of electrical power does it use?\n\nChoices:\nA. 120 W\nB. 240 W\nC. 480 W\n\nThe correct answer is (B).\n\nExample 2:\n\nQuestion: According to Moore's \"ideal utilitarianism\", the right action is the one producing the greatest amount of:\n\nChoices:\nA. wealth\nB. virtue\nC. fairness\nD. pleasure\nE. peace\nF. justice\nG. happiness\nH. power\nI. good\nJ. knowledge\n\nThe correct answer is (I).\n\nQuestion:",
        "--num_samples", "1000"
    ]
    eval_main()


def run_fewshot_ablation(model="gpt4o", model_version="gpt-3.5-turbo"):
    print("\n===== gsm8k fewshot ablation 평가 실행 =====")
    run_gsm8k_fewshot_ablation(model, model_version)
    print("\n===== bbh fewshot ablation 평가 실행 =====")
    run_bbh_fewshot_ablation(model, model_version)
    # print("\n===== cnn_dailymail fewshot ablation 평가 실행 =====")
    # run_cnn_fewshot_dailymail_ablation(model, model_version)
    print("\n===== mbpp fewshot ablation 평가 실행 =====")
    run_mbpp_fewshot_ablation(model, model_version)
    print("\n===== mmlu_pro fewshot ablation 평가 실행 =====")
    run_mmlu_pro_fewshot_ablation(model, model_version)

def run_prompt_ablation(model="gpt4o", model_version="gpt-3.5-turbo"):
    print("\n===== gsm8k prompt ablation 평가 실행 =====")
    run_gsm8k_prompt_ablation(model, model_version)
    print("\n===== bbh prompt ablation 평가 실행 =====")
    run_bbh_prompt_ablation(model, model_version)
    print("\n===== cnn_dailymail prompt ablation 평가 실행 =====")
    run_cnn_prompt_dailymail_ablation(model, model_version)
    print("\n===== mbpp prompt ablation 평가 실행 =====")
    run_mbpp_prompt_ablation(model, model_version)
    print("\n===== mmlu_pro prompt ablation 평가 실행 =====")
    run_mmlu_pro_prompt_ablation(model, model_version)


def main():
    model = "local"
    model_version = "/data/project/private/kyle/hf_models/Meta-Llama-3-70B-Instruct"
    # model = "gpt4o"
    # model_version = "gpt-3.5-turbo"

    run_fewshot_ablation(model, model_version)
    run_prompt_ablation(model, model_version)


if __name__ == "__main__":
    main() 