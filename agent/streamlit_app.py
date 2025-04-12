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

# 기본 평가 프롬프트
DEFAULT_EVALUATION_PROMPT = """당신은 AI 응답의 품질을 평가하는 전문가입니다. 주어진 응답이 기대하는 응답과 얼마나 잘 일치하는지 평가해주세요.

실제 응답:
{response}

기대하는 응답:
{expected}

다음 기준으로 평가해주세요:
1. 의미적 유사성 (응답이 기대하는 내용을 얼마나 잘 전달하는가)
2. 톤과 스타일 (전문적이고 공손한 톤을 유지하는가)
3. 정보의 정확성 (잘못된 정보를 포함하지 않는가)
4. 응답의 완성도 (필요한 정보를 모두 포함하는가)

0.0에서 1.0 사이의 점수만 출력해주세요. 다른 설명은 하지 마세요.
예시: 0.85"""

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
        value="You are a helpful AI assistant. Be polite and concise in your responses.",
        height=100,
        help="튜닝을 시작할 초기 프롬프트를 입력하세요."
    )

# 평가 프롬프트 설정
with st.expander("평가 프롬프트 설정", expanded=False):
    evaluation_prompt = st.text_area(
        "프롬프트",
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
                
                with st.spinner("프롬프트 튜닝 중..."):
                    results = tuner.tune(initial_prompt, test_cases, iterations=iterations)
                
                # 결과 표시
                st.header("프롬프트 튜닝 결과")
                
                # 최고의 결과 표시
                best_result = max(results, key=lambda x: x['avg_score'])
                st.markdown("### 🏆 최고의 결과")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("평균 점수", f"{best_result['avg_score']:.2f}")
                with col2:
                    st.metric("최고 점수", f"{best_result['best_score']:.2f}")
                with col3:
                    st.metric("최저 점수", f"{best_result['worst_score']:.2f}")
                with col4:
                    st.metric("테스트 케이스 수", len(best_result['detailed_responses']))
                
                st.text_area("최적 프롬프트", value=best_result['prompt'], height=80)
                st.markdown("---")
                
                # Iteration별로 결과 표시
                st.markdown("#### 모든 Iteration 결과")
                
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
                st.dataframe(
                    df_all,
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