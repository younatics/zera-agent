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
    iterations = st.slider("반복 횟수", min_value=1, max_value=10, value=1)
    
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
        test_cases = []
        for _, row in df.iterrows():
            test_cases.append({
                'question': row['question'],
                'expected': row['expected_answer']
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
                    
                    progress_tracker = ProgressTracker()
                    
                    # 프로그레스 바 업데이트 콜백 설정
                    tuner.progress_callback = lambda i, tc: progress_tracker.update(i, tc)
                    
                    results = tuner.tune_prompt(initial_prompt, test_cases, num_iterations=iterations)
                    
                    # 프로그레스바와 진행 중 문구 제거
                    progress_bar.empty()
                    progress_tracker.progress_text.empty()
                    
                    # 최적의 프롬프트 표시
                    st.markdown("### 최적의 프롬프트")
                    st.text_area("", value=results, height=80, disabled=True)
                    
                    # 모든 Iteration 결과를 테이블로 표시
                    st.markdown("### 모든 Iteration 결과")
                    all_results = []
                    for result in tuner.evaluation_history:
                        all_results.append({
                            'Iteration': result['iteration'],
                            '프롬프트': result['prompt'],
                            '테스트 케이스': result['test_case'],
                            '질문': result['question'],
                            '기대 응답': result['expected'],
                            '실제 응답': result['response'],
                            '점수': result['score'],
                            '평가 이유': result['evaluation_reason'],
                            '메타프롬프트': tuner.meta_prompt_template.format(
                                prompt=result['prompt'],
                                question=result['question'],
                                expected=result['expected'],
                                evaluation_reason=result['evaluation_reason']
                            )
                        })
                    
                    df_all = pd.DataFrame(all_results)
                    
                    # 최고 점수를 가진 행을 찾기
                    best_score = df_all['점수'].max()
                    best_rows = df_all[df_all['점수'] == best_score]
                    
                    # 최고 점수를 가진 행에 하이라이트 스타일 적용
                    def highlight_best(row):
                        if row['점수'] == best_score:
                            return ['background-color: #2e4053; color: white'] * len(row)
                        return ['background-color: #1a1a1a; color: white'] * len(row)
                    
                    st.dataframe(
                        df_all.style.apply(highlight_best, axis=1),
                        column_config={
                            "Iteration": st.column_config.NumberColumn(
                                "Iteration",
                                help="Iteration 번호",
                                format="%d",
                            ),
                            "프롬프트": st.column_config.TextColumn(
                                "프롬프트",
                                help="사용된 프롬프트",
                                width="medium",
                            ),
                            "테스트 케이스": st.column_config.NumberColumn(
                                "테스트 케이스",
                                help="테스트 케이스 번호",
                                format="%d",
                            ),
                            "질문": st.column_config.TextColumn(
                                "질문",
                                help="테스트 케이스 질문",
                                width="medium",
                            ),
                            "기대 응답": st.column_config.TextColumn(
                                "기대 응답",
                                help="기대하는 응답",
                                width="medium",
                            ),
                            "실제 응답": st.column_config.TextColumn(
                                "실제 응답",
                                help="실제 응답",
                                width="medium",
                            ),
                            "점수": st.column_config.NumberColumn(
                                "점수",
                                help="평가 점수",
                                format="%.2f",
                            ),
                            "평가 이유": st.column_config.TextColumn(
                                "평가 이유",
                                help="평가 모델이 내린 평가의 이유",
                                width="large",
                            ),
                            "메타프롬프트": st.column_config.TextColumn(
                                "메타프롬프트",
                                help="프롬프트 개선에 사용된 메타프롬프트",
                                width="large",
                            ),
                        },
                        hide_index=True,
                    )
                
                st.markdown("---")
                
    except Exception as e:
        st.error(f"Error processing CSV file: {str(e)}")
        logger.error(f"Error processing CSV file: {str(e)}") 