import streamlit as st
import pandas as pd
import plotly.express as px
from prompt_tuner import PromptTuner
import os
import logging
import plotly.graph_objects as go
from dataset.mmlu_dataset import MMLUDataset

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Prompt Tuning Visualizer", layout="wide")

st.title("Prompt Tuning Dashboard")

# 모델 정보 정의
MODEL_INFO = {
    "solar": {
        "name": "Solar",
        "description": "Upstage의 Solar 모델",
        "version": "solar-pro"
    },
    "gpt4o": {
        "name": "GPT-4",
        "description": "OpenAI의 GPT-4 모델",
        "version": "gpt-4"
    },
    "claude": {
        "name": "Claude",
        "description": "Anthropic의 Claude 3 Sonnet",
        "version": "claude-3-sonnet-20240229"
    }
}

# 프롬프트 파일 로드
prompts_dir = os.path.join(os.path.dirname(__file__), 'prompts')
with open(os.path.join(prompts_dir, 'initial_prompt.txt'), 'r', encoding='utf-8') as f:
    DEFAULT_INITIAL_PROMPT = f.read()
with open(os.path.join(prompts_dir, 'evaluation_prompt.txt'), 'r', encoding='utf-8') as f:
    DEFAULT_EVALUATION_PROMPT = f.read()
with open(os.path.join(prompts_dir, 'meta_prompt.txt'), 'r', encoding='utf-8') as f:
    DEFAULT_META_PROMPT = f.read()

# MMLU 데이터셋 인스턴스 생성
mmlu_dataset = MMLUDataset()

# 사이드바에서 파라미터 설정
with st.sidebar:
    st.header("튜닝 파라미터")
    iterations = st.slider("반복 횟수", min_value=1, max_value=10, value=1)
    
    # 프롬프트 개선 사용 토글
    use_meta_prompt = st.toggle(
        "프롬프트 개선 사용", 
        value=True, 
        help="메타 프롬프트를 사용하여 프롬프트를 개선합니다. 비활성화하면 초기 프롬프트를 사용합니다."
    )
    
    # 평가 프롬프트 점수 임계값 설정 (프롬프트 개선이 켜져있을 때만 활성화)
    evaluation_threshold = st.slider(
        "평가 프롬프트 점수 임계값",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
        step=0.1,
        disabled=not use_meta_prompt,
        help="이 점수 미만이면 프롬프트를 개선합니다. 프롬프트 개선이 켜져있을 때만 사용 가능합니다."
    )
    
    # 점수 임계값 적용 여부 토글 (프롬프트 개선이 켜져있을 때만 활성화)
    use_threshold = st.toggle(
        "점수 임계값 적용",
        value=True,
        disabled=not use_meta_prompt,
        help="이 옵션이 켜져있으면 점수가 임계값 이상일 때 반복을 중단합니다. 프롬프트 개선이 켜져있을 때만 사용 가능합니다."
    )
    
    # 점수 임계값 슬라이더 (점수 임계값 적용이 꺼져있거나 프롬프트 개선이 꺼져있을 때는 비활성화)
    score_threshold = st.slider(
        "점수 임계값",
        min_value=0.0,
        max_value=1.0,
        value=0.9,
        step=0.05,
        disabled=not (use_threshold and use_meta_prompt),
        help="이 점수 이상이면 반복을 중단합니다. 점수 임계값 적용과 프롬프트 개선이 모두 켜져있을 때만 사용 가능합니다."
    )
    
    # 모델 선택
    model_name = st.selectbox(
        "모델 선택",
        options=list(MODEL_INFO.keys()),
        format_func=lambda x: f"{MODEL_INFO[x]['name']} ({MODEL_INFO[x]['version']})",
        help="프롬프트 튜닝에 사용할 모델을 선택하세요."
    )
    st.caption(MODEL_INFO[model_name]['description'])
    
    # 메타 프롬프트 모델 선택
    meta_prompt_model = st.selectbox(
        "메타 프롬프트 모델 선택",
        options=list(MODEL_INFO.keys()),
        format_func=lambda x: f"{MODEL_INFO[x]['name']} ({MODEL_INFO[x]['version']})",
        help="메타 프롬프트 생성에 사용할 모델을 선택하세요."
    )
    st.caption(MODEL_INFO[meta_prompt_model]['description'])

    # 평가 모델 선택
    evaluator_model = st.selectbox(
        "평가 모델 선택",
        options=list(MODEL_INFO.keys()),
        format_func=lambda x: f"{MODEL_INFO[x]['name']} ({MODEL_INFO[x]['version']})",
        help="응답 평가에 사용할 모델을 선택하세요."
    )
    st.caption(MODEL_INFO[evaluator_model]['description'])

# 프롬프트 설정
with st.expander("초기 프롬프트 설정", expanded=False):
    initial_prompt = st.text_area(
        "프롬프트",
        value=DEFAULT_INITIAL_PROMPT,
        height=100,
        help="튜닝을 시작할 초기 프롬프트를 입력하세요."
    )

# 메타프롬프트 설정
with st.expander("메타프롬프트 설정", expanded=False):
    meta_prompt = st.text_area(
        "메타프롬프트 입력",
        value=DEFAULT_META_PROMPT,
        height=300,
        help="""프롬프트 변형을 생성할 때 사용하는 프롬프트를 입력하세요.
다음 변수들이 사용됩니다:
- {prompt}: 원본 프롬프트
- {question}: 테스트 케이스의 질문
- {expected}: 기대하는 응답
- {evaluation_reason}: 평가 모델이 내린 평가의 이유

평가 이유를 바탕으로 프롬프트를 개선하는 방향성을 제시할 수 있습니다."""
    )

# 평가 프롬프트 설정
with st.expander("평가 프롬프트 설정", expanded=False):
    evaluation_prompt = st.text_area(
        "평가 프롬프트 입력",
        value=DEFAULT_EVALUATION_PROMPT,
        height=300,
        help="""응답을 평가할 때 사용하는 프롬프트를 입력하세요.
{response}와 {expected}는 실제 응답과 기대 응답으로 대체됩니다."""
    )

# 데이터셋 선택
st.header("Dataset Selection")
dataset_type = st.radio(
    "Select Dataset Type",
    ["CSV", "MMLU"],
    horizontal=True
)

if dataset_type == "CSV":
    csv_file = st.file_uploader("Upload CSV file", type=['csv'])
else:
    subject = st.selectbox(
        "Select MMLU Subject",
        mmlu_dataset.subjects,
        index=0
    )
    split = st.selectbox(
        "Select Data Split",
        ["validation", "test"],
        index=0
    )

# 데이터셋 로드 및 테스트 케이스 생성
test_cases = []
if dataset_type == "MMLU":
    try:
        # MMLU 데이터셋 로드
        subject_data = mmlu_dataset.get_subject_data(subject)
        data = subject_data[split]
        
        # 데이터 표시
        st.write(f"총 예제 수: {len(data)}")
        
        # 테스트 케이스 생성 및 데이터프레임 생성
        display_data = []
        for item in data:
            # 선택지를 문자열로 변환
            choices_str = "\n".join([f"{i+1}. {choice}" for i, choice in enumerate(item['choices'])])
            question = f"{item['question']}\n\nChoices:\n{choices_str}"
            expected = str(item['answer'] + 1)  # 0-based를 1-based로 변환
            
            test_cases.append({
                'question': question,
                'expected': expected
            })
            
            # 표시용 데이터 추가
            display_data.append({
                'question': question,
                'expected_answer': expected
            })
        
        # 전체 데이터 표시
        st.write("데이터셋 내용:")
        st.dataframe(pd.DataFrame(display_data))
            
    except Exception as e:
        st.error(f"MMLU 데이터셋 로드 중 오류 발생: {str(e)}")
        st.stop()
else:
    if csv_file is not None:
        try:
            # CSV 파일을 읽을 때 더 유연한 파싱 옵션 사용
            df = pd.read_csv(csv_file, 
                            encoding='utf-8',
                            on_bad_lines='skip',  # 문제가 있는 줄은 건너뛰기
                            quoting=1,  # 모든 필드를 따옴표로 감싸기
                            escapechar='\\')  # 이스케이프 문자 설정
            
            # 데이터프레임이 비어있는지 확인
            if df.empty:
                st.error("CSV 파일이 비어있습니다. 올바른 데이터가 포함된 CSV 파일을 업로드하세요.")
                st.stop()
            
            # 데이터프레임 표시
            st.write("업로드된 데이터:")
            st.dataframe(df)
            
            # 컬럼 이름 확인 및 매핑
            required_columns = ['question', 'expected_answer']
            available_columns = df.columns.tolist()
            
            # 필수 컬럼이 있는지 확인
            missing_columns = [col for col in required_columns if col not in available_columns]
            if missing_columns:
                st.error(f"CSV 파일에 다음 컬럼이 필요합니다: {', '.join(missing_columns)}")
                st.info("CSV 파일은 'question'과 'expected_answer' 컬럼을 포함해야 합니다.")
                st.stop()
            
            # 테스트 케이스 생성
            for _, row in df.iterrows():
                test_cases.append({
                    'question': row['question'],
                    'expected': row['expected_answer']
                })
        except Exception as e:
            st.error(f"CSV 파일 로드 중 오류 발생: {str(e)}")
            st.info("CSV 파일이 올바른 형식인지 확인하세요. 파일이 비어있거나, 인코딩이 UTF-8이 아닐 수 있습니다.")
            st.stop()
    else:
        st.info("CSV 파일을 업로드하거나 MMLU 데이터셋을 선택하세요.")
        st.stop()

# 튜닝 시작 버튼
if st.button("프롬프트 튜닝 시작", type="primary"):
    # API 키 확인
    required_keys = {
        "solar": "SOLAR_API_KEY",
        "gpt4o": "OPENAI_API_KEY",
        "claude": "ANTHROPIC_API_KEY"
    }
    
    missing_keys = []
    for model in [model_name, evaluator_model]:
        key = required_keys[model]
        if not os.getenv(key):
            missing_keys.append(f"{MODEL_INFO[model]['name']} ({key})")
    
    if missing_keys:
        st.error(f"다음 API 키가 필요합니다: {', '.join(missing_keys)}")
        st.info("API 키를 .env 파일에 설정하세요.")
    else:
        # PromptTuner 인스턴스 생성
        tuner = PromptTuner(
            model_name=model_name,
            evaluator_model_name=evaluator_model,
            meta_prompt_model_name=meta_prompt_model
        )
        tuner.set_evaluation_prompt(evaluation_prompt)
        
        # 메타프롬프트가 입력된 경우에만 설정
        if meta_prompt.strip():
            tuner.set_meta_prompt(meta_prompt)
        
        # 프롬프트 튜닝 실행
        with st.spinner("프롬프트 튜닝 중..."):
            # 전체 진행 상황을 위한 프로그레스 바
            progress_bar = st.progress(0)
            total_steps = iterations * len(test_cases)
            
            class ProgressTracker:
                def __init__(self):
                    self.current_step = 0
                    self.progress_text = st.empty()
                
                def update(self, iteration, test_case):
                    self.current_step += 1
                    progress = self.current_step / total_steps
                    progress_bar.progress(progress)
                    self.progress_text.text(f"진행 중: Iteration {iteration}/{iterations}, Test Case {test_case}/{len(test_cases)} ({self.current_step}/{total_steps})")
                
                def complete(self):
                    progress_bar.progress(1.0)
                    self.progress_text.text("완료!")
            
            progress_tracker = ProgressTracker()
            
            # 프로그레스 바 업데이트 콜백 설정
            tuner.progress_callback = lambda i, tc: progress_tracker.update(i, tc)
            
            results = tuner.tune_prompt(
                initial_prompt=initial_prompt,
                test_cases=test_cases,
                num_iterations=iterations,
                score_threshold=score_threshold if use_threshold else None,
                evaluation_score_threshold=evaluation_threshold,
                use_meta_prompt=use_meta_prompt
            )
            
            progress_tracker.complete()
            
            # 결과 표시
            st.success("프롬프트 튜닝 완료!")
            
            # 최종 프롬프트 표시
            st.subheader("최종 프롬프트")
            st.code(results, language="text")
            
            # 평가 기록 표시
            st.subheader("평가 기록")
            evaluation_history = tuner.evaluation_history
            
            # 평가 기록을 데이터프레임으로 변환
            history_df = pd.DataFrame(evaluation_history)
            
            # 컬럼 순서 변경
            history_df = history_df[['iteration', 'test_case', 'prompt', 'question', 'expected_answer', 'actual_answer', 'score', 'evaluation_reason']]
            
            # 점수를 소수점 두자리까지만 표시
            history_df['score'] = history_df['score'].round(2)
            
            # 최고 점수 하이라이트
            def highlight_max_row(s):
                max_score = history_df['score'].max()
                is_max_row = history_df['score'] == max_score
                return ['background-color: #e6ffe6' if is_max_row[i] else '' for i in range(len(s))]
            
            # 상세 평가 기록 표시
            st.dataframe(history_df.style.apply(highlight_max_row)) 