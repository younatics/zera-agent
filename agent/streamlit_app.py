import streamlit as st
import pandas as pd
import plotly.express as px
from prompt_tuner import PromptTuner
import os
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Prompt Tuning Visualizer", layout="wide")

st.title("Prompt Tuning Dashboard")

# 사이드바에서 파라미터 설정
with st.sidebar:
    st.header("튜닝 파라미터")
    iterations = st.slider("반복 횟수", min_value=1, max_value=10, value=3)
    model_name = st.selectbox("모델 선택", ["solar", "other_model"], index=0)
    evaluator_model = st.selectbox("평가 모델 선택", ["solar", "other_model"], index=0)

# 초기 프롬프트 입력
initial_prompt = st.text_area(
    "초기 프롬프트 입력",
    value="You are a helpful AI assistant. Be polite and concise in your responses.",
    height=100
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
            # 프롬프트 튜너 초기화 및 실행
            tuner = PromptTuner(model_name=model_name, evaluator_model_name=evaluator_model)
            
            with st.spinner("프롬프트 튜닝 중..."):
                best_prompt = tuner.tune(initial_prompt, test_cases, iterations=iterations)
            
            # 결과 표시
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("최종 결과")
                st.metric("최고 점수", f"{tuner.best_score:.2f}")
                st.text_area("최적 프롬프트", value=best_prompt, height=150)
            
            with col2:
                st.subheader("평가 기록")
                # 평가 기록을 DataFrame으로 변환
                history_df = pd.DataFrame(tuner.evaluation_history)
                
                # 점수 변화 그래프
                fig = px.line(history_df, 
                            x=history_df.index, 
                            y='score',
                            title='점수 변화',
                            labels={'x': 'Iteration', 'y': 'Score'})
                st.plotly_chart(fig, use_container_width=True)
                
    except Exception as e:
        st.error(f"Error processing CSV file: {str(e)}")
        logger.error(f"Error processing CSV file: {str(e)}") 