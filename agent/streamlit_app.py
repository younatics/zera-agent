import streamlit as st
import pandas as pd
import plotly.express as px
from prompt_tuner import PromptTuner
import os
import logging
import plotly.graph_objects as go

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

# 사이드바에서 파라미터 설정
with st.sidebar:
    st.header("튜닝 파라미터")
    iterations = st.slider("반복 횟수", min_value=1, max_value=10, value=3)
    
    # 모델 선택
    model_name = st.selectbox(
        "모델 선택",
        options=list(MODEL_INFO.keys()),
        format_func=lambda x: f"{MODEL_INFO[x]['name']} ({MODEL_INFO[x]['version']})",
        help="프롬프트 튜닝에 사용할 모델을 선택하세요."
    )
    st.caption(MODEL_INFO[model_name]['description'])
    
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
{prompt}는 원본 프롬프트로 대체됩니다."""
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

# CSV 파일 업로드
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        # CSV 파일을 읽을 때 더 유연한 파싱 옵션 사용
        df = pd.read_csv(uploaded_file, 
                        encoding='utf-8',
                        on_bad_lines='skip',  # 문제가 있는 줄은 건너뛰기
                        quoting=1,  # 모든 필드를 따옴표로 감싸기
                        escapechar='\\')  # 이스케이프 문자 설정
        
        # 데이터프레임 표시
        st.write("Uploaded Data:")
        st.dataframe(df)
        
        # 테스트 케이스 생성
        test_cases = []
        for _, row in df.iterrows():
            test_cases.append({
                'input': row['question'],
                'expected_output': row['expected_answer']
            })
        
        # 튜닝 시작 버튼
        if st.button("Start Prompt Tuning", type="primary"):
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
                # 프롬프트 튜너 초기화 및 실행
                tuner = PromptTuner(model_name=model_name, evaluator_model_name=evaluator_model)
                tuner.set_evaluation_prompt(evaluation_prompt)
                
                # 메타프롬프트가 입력된 경우에만 설정
                if meta_prompt.strip():
                    tuner.set_meta_prompt(meta_prompt)
                
                with st.spinner("프롬프트 튜닝 중..."):
                    results = tuner.tune(initial_prompt, test_cases, iterations=iterations)
                
                # 결과 표시
                st.header("프롬프트 튜닝 결과")
                
                # 최고의 결과 표시
                best_result = max(results, key=lambda x: x['avg_score'])
                best_prompt = best_result['prompt']
                
                # 모든 iteration의 결과를 하나의 DataFrame으로 통합
                all_results = []
                for i, record in enumerate(results):
                    for j, response in enumerate(record['detailed_responses']):
                        all_results.append({
                            'Iteration': i + 1,
                            '프롬프트': record['prompt'],
                            '평균 점수': record['avg_score'],
                            '테스트 케이스': j + 1,
                            '질문': response['input'],
                            '기대 응답': response['expected'],
                            '실제 응답': response['response'],
                            '점수': response['score']
                        })
                
                df_all = pd.DataFrame(all_results)
                
                # 최적 프롬프트를 강조하기 위한 스타일 함수
                def highlight_best_prompt(row):
                    if row['프롬프트'] == best_prompt:
                        return ['background-color: #e6ffe6'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(
                    df_all.style.apply(highlight_best_prompt, axis=1),
                    column_config={
                        "Iteration": st.column_config.NumberColumn(
                            "Iteration",
                            width="small"
                        ),
                        "프롬프트": st.column_config.TextColumn(
                            "프롬프트",
                            width="large"
                        ),
                        "평균 점수": st.column_config.NumberColumn(
                            "평균 점수",
                            format="%.2f",
                            width="small"
                        ),
                        "테스트 케이스": st.column_config.NumberColumn(
                            "테스트 케이스",
                            width="small"
                        ),
                        "질문": st.column_config.TextColumn(
                            "질문",
                            width="medium"
                        ),
                        "기대 응답": st.column_config.TextColumn(
                            "기대 응답",
                            width="medium"
                        ),
                        "실제 응답": st.column_config.TextColumn(
                            "실제 응답",
                            width="medium"
                        ),
                        "점수": st.column_config.NumberColumn(
                            "점수",
                            format="%.2f",
                            width="small"
                        )
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
                st.markdown("---")
                
    except Exception as e:
        st.error(f"Error processing CSV file: {str(e)}")
        logger.error(f"Error processing CSV file: {str(e)}") 