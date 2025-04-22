import streamlit as st
import pandas as pd
import plotly.express as px
from prompt_tuner import PromptTuner
from common.api_client import Model
import os
import logging
import plotly.graph_objects as go
import sys
from dotenv import load_dotenv
import tempfile
import base64
import zipfile
import io

# set_page_config은 반드시 첫 번째 Streamlit 명령어여야 함
st.set_page_config(page_title="Prompt Auto Tuning Agent", layout="wide")

# .env 파일 로드
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(env_path, override=True)

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from dataset.mmlu_dataset import MMLUDataset
from dataset.cnn_dataset import CNNDataset
import numpy as np

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.title("Prompt Tuning Dashboard")

# 모델 정보 정의
MODEL_INFO = Model.get_all_model_info()

# 프롬프트 파일 로드
prompts_dir = os.path.join(os.path.dirname(__file__), 'prompts')
with open(os.path.join(prompts_dir, 'initial_system_prompt.txt'), 'r', encoding='utf-8') as f:
    DEFAULT_SYSTEM_PROMPT = f.read()
with open(os.path.join(prompts_dir, 'initial_user_prompt.txt'), 'r', encoding='utf-8') as f:
    DEFAULT_USER_PROMPT = f.read()
with open(os.path.join(prompts_dir, 'evaluation_system_prompt.txt'), 'r', encoding='utf-8') as f:
    DEFAULT_EVALUATION_SYSTEM_PROMPT = f.read()
with open(os.path.join(prompts_dir, 'evaluation_user_prompt.txt'), 'r', encoding='utf-8') as f:
    DEFAULT_EVALUATION_USER_PROMPT = f.read()
with open(os.path.join(prompts_dir, 'meta_system_prompt.txt'), 'r', encoding='utf-8') as f:
    DEFAULT_META_SYSTEM_PROMPT = f.read()
with open(os.path.join(prompts_dir, 'meta_user_prompt.txt'), 'r', encoding='utf-8') as f:
    DEFAULT_META_USER_PROMPT = f.read()

# MMLU 데이터셋 인스턴스 생성
mmlu_dataset = MMLUDataset()

# 사이드바에서 파라미터 설정
with st.sidebar:
    st.header("튜닝 설정")
    
    # 반복 설정 그룹
    with st.expander("반복 설정", expanded=True):
        iterations = st.slider(
            "반복 횟수", 
            min_value=1, 
            max_value=20, 
            value=3,
            help="프롬프트 튜닝을 수행할 반복 횟수를 설정합니다."
        )
    
    # 프롬프트 개선 설정 그룹
    with st.expander("프롬프트 개선 설정", expanded=True):
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
        
        # 평균 점수 임계값 적용 여부 토글 (프롬프트 개선이 켜져있을 때만 활성화)
        use_threshold = st.toggle(
            "평균 점수 임계값 적용",
            value=True,
            disabled=not use_meta_prompt,
            help="이 옵션이 켜져있으면 평균 점수가 임계값 이상일 때 반복을 중단합니다. 프롬프트 개선이 켜져있을 때만 사용 가능합니다."
        )
        
        # 평균 점수 임계값 슬라이더 (평균 점수 임계값 적용이 꺼져있거나 프롬프트 개선이 꺼져있을 때는 비활성화)
        score_threshold = st.slider(
            "평균 점수 임계값",
            min_value=0.0,
            max_value=1.0,
            value=0.9,
            step=0.05,
            disabled=not (use_threshold and use_meta_prompt),
            help="이 점수 이상이면 반복을 중단합니다. 평균 점수 임계값 적용과 프롬프트 개선이 모두 켜져있을 때만 사용 가능합니다."
        )
    
    # 모델 설정 섹션 구분을 위한 디바이더
    st.divider()
    
    # 모델 설정 그룹
    with st.expander("튜닝 모델 설정", expanded=True):
        # 모델 선택
        model_name = st.selectbox(
            "모델 선택",
            options=list(MODEL_INFO.keys()),
            format_func=lambda x: f"{MODEL_INFO[x]['name']} ({MODEL_INFO[x]['default_version']})",
            help="프롬프트 튜닝에 사용할 모델을 선택하세요."
        )
        st.caption(MODEL_INFO[model_name]['description'])
        
        # 튜닝 모델 버전 선택
        use_custom_tuning_version = st.toggle(
            "커스텀 버전 사용",
            value=False,
            help="튜닝 모델의 기본 버전 대신 커스텀 버전을 사용합니다."
        )
        
        if use_custom_tuning_version:
            tuning_model_version = st.text_input(
                "모델 버전",
                value=MODEL_INFO[model_name]['default_version'],
                help="튜닝에 사용할 모델 버전을 입력하세요."
            )
        else:
            tuning_model_version = None  # 기본 버전을 사용하도록 None으로 설정
    
    # 메타 프롬프트 모델 설정 그룹
    with st.expander("메타 프롬프트 모델 설정", expanded=True):
        # 메타 프롬프트 모델 선택
        meta_prompt_model = st.selectbox(
            "모델 선택",
            options=list(MODEL_INFO.keys()),
            format_func=lambda x: f"{MODEL_INFO[x]['name']} ({MODEL_INFO[x]['default_version']})",
            help="메타 프롬프트 생성에 사용할 모델을 선택하세요."
        )
        st.caption(MODEL_INFO[meta_prompt_model]['description'])
        
        # 메타 프롬프트 모델 버전 선택
        use_custom_meta_version = st.toggle(
            "커스텀 버전 사용",
            value=False,
            help="메타 프롬프트 모델의 기본 버전 대신 커스텀 버전을 사용합니다."
        )
        
        if use_custom_meta_version:
            meta_model_version = st.text_input(
                "모델 버전",
                value=MODEL_INFO[meta_prompt_model]['default_version'],
                help="메타 프롬프트 생성에 사용할 모델 버전을 입력하세요."
            )
        else:
            meta_model_version = None  # 기본 버전을 사용하도록 None으로 설정
    
    # 평가 모델 설정 그룹
    with st.expander("평가 모델 설정", expanded=True):
        # 평가 모델 선택
        evaluator_model = st.selectbox(
            "모델 선택",
            options=list(MODEL_INFO.keys()),
            format_func=lambda x: f"{MODEL_INFO[x]['name']} ({MODEL_INFO[x]['default_version']})",
            help="응답 평가에 사용할 모델을 선택하세요."
        )
        st.caption(MODEL_INFO[evaluator_model]['description'])
        
        # 평가 모델 버전 선택
        use_custom_evaluator_version = st.toggle(
            "커스텀 버전 사용",
            value=False,
            help="평가 모델의 기본 버전 대신 커스텀 버전을 사용합니다."
        )
        
        if use_custom_evaluator_version:
            evaluator_model_version = st.text_input(
                "모델 버전",
                value=MODEL_INFO[evaluator_model]['default_version'],
                help="평가에 사용할 모델 버전을 입력하세요."
            )
        else:
            evaluator_model_version = None  # 기본 버전을 사용하도록 None으로 설정

# PromptTuner 객체 생성
tuner = PromptTuner(
    model_name=model_name,
    evaluator_model_name=evaluator_model,
    meta_prompt_model_name=meta_prompt_model,
    model_version=tuning_model_version,
    evaluator_model_version=evaluator_model_version,
    meta_prompt_model_version=meta_model_version
)

# 프롬프트 설정
with st.expander("초기 프롬프트 설정", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        system_prompt = st.text_area(
            "시스템 프롬프트",
            value=DEFAULT_SYSTEM_PROMPT,
            height=100,
            help="튜닝을 시작할 초기 시스템 프롬프트를 입력하세요."
        )
    with col2:
        user_prompt = st.text_area(
            "사용자 프롬프트",
            value=DEFAULT_USER_PROMPT,
            height=100,
            help="튜닝을 시작할 초기 사용자 프롬프트를 입력하세요."
        )
    
    if st.button("초기 프롬프트 업데이트", key="initial_prompt_update"):
        tuner.set_initial_prompt(system_prompt, user_prompt)
        st.success("초기 프롬프트가 업데이트되었습니다.")

# 메타프롬프트 설정
with st.expander("메타프롬프트 설정", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        meta_system_prompt = st.text_area(
            "메타 시스템 프롬프트",
            value=DEFAULT_META_SYSTEM_PROMPT,
            height=300,
            help="프롬프트 엔지니어의 역할과 책임을 정의하는 시스템 프롬프트를 입력하세요."
        )
    with col2:
        meta_user_prompt = st.text_area(
            "메타 유저 프롬프트",
            value=DEFAULT_META_USER_PROMPT,
            height=300,
            help="프롬프트 개선을 위한 입력 데이터와 출력 형식을 정의하는 유저 프롬프트를 입력하세요."
        )
    
    if st.button("메타 프롬프트 업데이트", key="meta_prompt_update"):
        tuner.set_meta_prompt(meta_system_prompt, meta_user_prompt)
        st.success("메타 프롬프트가 업데이트되었습니다.")

# 평가 프롬프트 설정
with st.expander("평가 프롬프트 설정", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        evaluation_system_prompt = st.text_area(
            "평가 시스템 프롬프트",
            value=DEFAULT_EVALUATION_SYSTEM_PROMPT,
            height=200,
            help="평가 모델의 시스템 프롬프트를 설정합니다."
        )
    with col2:
        evaluation_user_prompt = st.text_area(
            "평가 유저 프롬프트",
            value=DEFAULT_EVALUATION_USER_PROMPT,
            height=200,
            help="평가 모델의 유저 프롬프트를 설정합니다. {question}, {response}, {expected}를 포함해야 합니다."
        )
    
    if st.button("평가 프롬프트 업데이트", key="eval_prompt_update"):
        tuner.set_evaluation_prompt(evaluation_system_prompt, evaluation_user_prompt)
        st.success("평가 프롬프트가 업데이트되었습니다.")

# 데이터셋 처리 공통 함수
def process_dataset(data, dataset_type):
    # 데이터 표시
    total_examples = len(data)
    st.write(f"총 예제 수: {total_examples}")
    
    # 샘플 수 선택
    num_samples = st.slider(
        "Number of random samples to evaluate per iteration",
        min_value=1,
        max_value=min(100, total_examples),  # 최대 샘플 수를 100으로 제한
        value=min(5, total_examples),
        help="각 iteration마다 평가할 랜덤 샘플의 개수를 선택하세요."
    )
    
    # 테스트 케이스 생성 및 데이터프레임 생성
    test_cases = []
    display_data = []
    
    if dataset_type == "MMLU":
        for item in data:
            # 선택지를 문자열로 변환
            choices_str = "\n".join([f"{i+1}. {choice}" for i, choice in enumerate(item['choices'])])
            question = f"{item['question']}\n\nChoices:\n{choices_str}"
            expected = str(item['answer'] + 1)  # 0-based를 1-based로 변환
            
            test_cases.append({
                'question': question,
                'expected': expected
            })
            
            display_data.append({
                'question': question,
                'expected_answer': expected
            })
    elif dataset_type == "CSV":
        # 컬럼 이름 확인 및 매핑
        required_columns = ['question', 'expected_answer']
        available_columns = data.columns.tolist()
        
        # 필수 컬럼이 있는지 확인
        missing_columns = [col for col in required_columns if col not in available_columns]
        if missing_columns:
            st.error(f"CSV 파일에 다음 컬럼이 필요합니다: {', '.join(missing_columns)}")
            st.info("CSV 파일은 'question'과 'expected_answer' 컬럼을 포함해야 합니다.")
            st.stop()
        
        for _, row in data.iterrows():
            test_cases.append({
                'question': row['question'],
                'expected': row['expected_answer']
            })
            
            display_data.append({
                'question': row['question'],
                'expected_answer': row['expected_answer']
            })
    else:  # CNN 데이터셋
        # 데이터를 청크 단위로 처리
        chunk_size = 1000  # 한 번에 처리할 데이터 크기
        for i in range(0, len(data), chunk_size):
            chunk = data.iloc[i:i+chunk_size]
            for _, row in chunk.iterrows():
                test_cases.append({
                    'question': row['input'],
                    'expected': row['expected_answer']
                })
                
                display_data.append({
                    'question': row['input'],
                    'expected_answer': row['expected_answer']
                })
    
    # 전체 데이터 표시
    st.write("데이터셋 내용:")
    st.dataframe(pd.DataFrame(display_data))
    
    return test_cases, num_samples

# 데이터셋 선택
st.header("Dataset Selection")
dataset_type = st.radio(
    "Select Dataset Type",
    ["CSV", "MMLU", "CNN"],
    horizontal=True
)

if dataset_type == "CSV":
    csv_file = st.file_uploader("Upload CSV file", type=['csv'])
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
            
            test_cases, num_samples = process_dataset(df, "CSV")
        except Exception as e:
            st.error(f"CSV 파일 로드 중 오류 발생: {str(e)}")
            st.info("CSV 파일이 올바른 형식인지 확인하세요. 파일이 비어있거나, 인코딩이 UTF-8이 아닐 수 있습니다.")
            st.stop()
    else:
        st.info("CSV 파일을 업로드하거나 MMLU 데이터셋을 선택하세요.")
        st.stop()
elif dataset_type == "CNN":
    # CNN 데이터셋 인스턴스 생성
    cnn_dataset = CNNDataset()
    
    # 데이터셋 선택
    split = st.selectbox(
        "데이터셋 선택",
        ["train", "validation", "test"],
        index=0
    )
    
    # 청크 수 확인
    total_chunks = cnn_dataset.get_num_chunks(split)
    
    if total_chunks == 0:
        st.error(f"{split} 데이터셋에 청크 파일이 없습니다.")
        st.stop()
    
    # 청크 선택
    st.write(f"총 {total_chunks}개의 청크가 있습니다.")
    chunk_index = st.number_input(
        "청크 선택",
        min_value=0,
        max_value=total_chunks-1,
        value=0,
        help="처리할 청크의 인덱스를 선택하세요."
    )
    
    try:
        # 선택된 청크 로드
        data = cnn_dataset.load_data(split, chunk_index)
        test_cases, num_samples = process_dataset(data, "CNN")
        
        # 선택된 청크 정보 표시
        st.info(f"선택된 청크: {chunk_index} ({len(data):,}개 예제)")
    except Exception as e:
        st.error(f"CNN 데이터셋 로드 중 오류 발생: {str(e)}")
        st.stop()
else:
    # MMLU 데이터셋 선택에 '모든 과목' 옵션 추가
    subject_options = ["모든 과목"] + mmlu_dataset.subjects
    subject = st.selectbox(
        "Select MMLU Subject",
        subject_options,
        index=0
    )
    split = st.selectbox(
        "Select Data Split",
        ["validation", "test"],
        index=0
    )
    try:
        if subject == "모든 과목":
            # 모든 과목의 데이터 로드
            all_subjects_data = mmlu_dataset.get_all_subjects_data()
            # 모든 과목의 데이터를 하나의 리스트로 합치기
            data = []
            for subject_data in all_subjects_data.values():
                data.extend(subject_data[split])
        else:
            # 특정 과목의 데이터 로드
            subject_data = mmlu_dataset.get_subject_data(subject)
            data = subject_data[split]
        
        test_cases, num_samples = process_dataset(data, "MMLU")
    except Exception as e:
        st.error(f"MMLU 데이터셋 로드 중 오류 발생: {str(e)}")
        st.stop()

# 튜닝 시작 버튼
if st.button("프롬프트 튜닝 시작", type="primary"):
    # API 키 확인
    required_keys = {
        "solar": "SOLAR_API_KEY",
        "gpt4o": "OPENAI_API_KEY",
        "claude": "ANTHROPIC_API_KEY"
    }
    
    # 사용되는 모델들의 API 키 확인
    used_models = set([model_name, evaluator_model])
    missing_keys = []
    
    for model in used_models:
        key = required_keys[model]
        if not os.getenv(key):
            missing_keys.append(f"{MODEL_INFO[model]['name']} ({key})")
    
    if missing_keys:
        st.error(f"다음 API 키가 필요합니다: {', '.join(missing_keys)}")
        st.info("API 키를 .env 파일에 설정하세요.")
    else:
        # 메타프롬프트가 입력된 경우에만 설정
        if meta_system_prompt.strip() and meta_user_prompt.strip():
            tuner.set_meta_prompt(meta_system_prompt, meta_user_prompt)
        
        # 프로그레스 바 설정
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def progress_callback(iteration, test_case_index):
            # 현재 iteration의 진행도 (0부터 시작)
            iteration_progress = (iteration - 1) / iterations
            # 현재 test case의 진행도 (0부터 시작)
            test_case_progress = test_case_index / num_samples
            # 전체 진행도 계산
            progress = iteration_progress + (test_case_progress / iterations)
            progress_bar.progress(progress)
            status_text.text(f"Iteration {iteration}/{iterations}, Test Case {test_case_index}/{num_samples}")
        
        tuner.progress_callback = progress_callback
        
        # 프롬프트 튜닝 실행
        with st.spinner("프롬프트 튜닝 중..."):
            # 결과를 저장할 리스트
            all_results = []
            
            # 프롬프트 히스토리를 저장할 리스트
            prompt_history = []
            
            # 그래프를 위한 컨테이너 생성 (이터레이션 컨테이너들 위에 위치)
            graph_container = st.container()
            graph_placeholder = graph_container.empty()  # 그래프를 위한 placeholder
            
            # 각 이터레이션마다 결과를 보여주기 위한 컨테이너
            iteration_containers = []
            for i in range(iterations):
                iteration_containers.append(st.container())
            
            # 이터레이션 결과를 위한 컨테이너 생성
            iteration_results_container = st.container()
            
            def iteration_callback(result):
                iteration_idx = result['iteration'] - 1
                
                # 현재 결과를 all_results에 추가
                all_results.append(result)
                
                # 프롬프트 히스토리에 추가
                prompt_history.append({
                    'iteration': result['iteration'],
                    'system_prompt': result['system_prompt'],
                    'user_prompt': result['user_prompt'],
                    'avg_score': result['avg_score'],
                    'best_avg_score': result['best_avg_score'],
                    'best_sample_score': result['best_sample_score']
                })
                
                # 그래프 업데이트
                with graph_placeholder.container():
                    fig = go.Figure()
                    x_values = list(range(1, iteration_idx + 2))
                    avg_scores = [r['avg_score'] for r in all_results]
                    best_avg_scores = [r['best_avg_score'] for r in all_results]
                    best_sample_scores = [r['best_sample_score'] for r in all_results]
                    
                    fig.add_trace(go.Scatter(
                        x=x_values,
                        y=avg_scores,
                        name='평균 점수',
                        mode='lines+markers'
                    ))
                    fig.add_trace(go.Scatter(
                        x=x_values,
                        y=best_avg_scores,
                        name='최고 평균 점수',
                        mode='lines+markers'
                    ))
                    fig.add_trace(go.Scatter(
                        x=x_values,
                        y=best_sample_scores,
                        name='최고 개별 점수',
                        mode='lines+markers'
                    ))
                    # 표준편차 그래프 추가
                    std_devs = [r['std_dev'] for r in all_results]
                    fig.add_trace(go.Scatter(
                        x=x_values,
                        y=std_devs,
                        name='표준편차',
                        mode='lines+markers'
                    ))
                    # top3 평균 점수 그래프 추가
                    top3_scores = [r['top3_avg_score'] for r in all_results]
                    fig.add_trace(go.Scatter(
                        x=x_values,
                        y=top3_scores,
                        name='Top3 평균 점수',
                        mode='lines+markers'
                    ))
                    fig.update_layout(
                        title='점수 추이',
                        xaxis_title='이터레이션',
                        yaxis_title='점수',
                        yaxis_range=[0, 1],
                        xaxis=dict(
                            tickmode='array',
                            tickvals=x_values,
                            ticktext=x_values
                        ),
                        height=300  # 그래프 높이를 300px로 설정
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # 이터레이션 결과 표시
                with iteration_results_container:
                    with iteration_containers[iteration_idx]:
                        st.subheader(f"Iteration {result['iteration']}")
                        
                        # 평균 점수와 최고 점수
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("Average Score", f"{result['avg_score']:.2f}")
                        with col2:
                            st.metric("Best Average Score So Far", f"{result['best_avg_score']:.2f}")
                        with col3:
                            st.metric("Best Sample Score So Far", f"{result['best_sample_score']:.2f}")
                        with col4:
                            st.metric("Standard Deviation", f"{result['std_dev']:.2f}")
                        with col5:
                            st.metric("Top3 Average Score", f"{result['top3_avg_score']:.2f}")
                        
                        # 평가 기록을 데이터프레임으로 변환
                        history_df = pd.DataFrame(result['responses'])
                        
                        # 컬럼 순서 변경 및 필요한 컬럼만 선택
                        history_df = history_df[['question', 'expected', 'actual', 'score', 'reason']]
                        
                        # 컬럼 이름 변경
                        history_df.columns = ['Question', 'Expected Answer', 'Actual Answer', 'Score', 'Evaluation Reason']
                        
                        # 점수를 소수점 두자리까지만 표시
                        history_df['Score'] = history_df['Score'].round(2)
                        
                        # 최고 점수를 가진 행 하이라이트
                        def highlight_max_row(df):
                            try:
                                if df.empty:
                                    return pd.DataFrame('', index=df.index, columns=df.columns)
                                max_score = df['Score'].max()
                                is_max = df['Score'] == max_score
                                
                                # 현재 테마 확인
                                is_dark = st.get_option("theme.base") == "dark"
                                
                                # 테마에 따른 색상 선택
                                if not is_dark:
                                    # 라이트모드: 연한 파란색 배경, 진한 파란색 글자
                                    highlight_style = 'background-color: #E3F2FD; color: #0D47A1'
                                else:
                                    # 다크모드: 어두운 청록색 배경, 밝은 청록색 글자
                                    highlight_style = 'background-color: #006064; color: #80DEEA'
                                
                                # 모든 열에 대해 동일한 스타일 적용
                                styles = np.where(is_max, highlight_style, '')
                                # 스타일을 2D 배열로 확장
                                styles_2d = np.tile(styles.reshape(-1, 1), (1, len(df.columns)))
                                return pd.DataFrame(styles_2d, index=df.index, columns=df.columns)
                            except Exception as e:
                                print(f"하이라이트 오류: {str(e)}")
                                return pd.DataFrame('', index=df.index, columns=df.columns)
                        
                        # 테이블 표시
                        st.dataframe(
                            history_df.style.apply(highlight_max_row, axis=None),
                            hide_index=True
                        )
                        
                        # 현재 프롬프트 표시
                        st.write("Current Prompts:")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("System Prompt:")
                            st.code(result['system_prompt'])
                        with col2:
                            st.write("User Prompt:")
                            st.code(result['user_prompt'])
                        
                        # 메타프롬프트 표시 (expander 사용)
                        with st.expander("Meta Prompt", expanded=False):
                            st.code(result['meta_prompt'])
                        
                        st.divider()
            
            # iteration_callback을 설정
            tuner.iteration_callback = iteration_callback
            
            # 프롬프트 튜닝 실행
            results = tuner.tune_prompt(
                initial_system_prompt=system_prompt,
                initial_user_prompt=user_prompt,
                initial_test_cases=test_cases,
                num_iterations=iterations,
                score_threshold=score_threshold if use_threshold else None,
                evaluation_score_threshold=evaluation_threshold,
                use_meta_prompt=use_meta_prompt,
                num_samples=num_samples
            )
            
            # 최종 결과
            st.success("프롬프트 튜닝 완료!")
            
            if results:  # 결과가 있을 때만 처리
                # 전체 결과에서 가장 높은 평균 점수를 가진 프롬프트 찾기
                best_result = max(results, key=lambda x: x['avg_score'])
                st.write("Final Best Prompt:")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("System Prompt:")
                    st.code(best_result['system_prompt'])
                with col2:
                    st.write("User Prompt:")
                    st.code(best_result['user_prompt'])
                st.write(f"최종 결과: 평균 점수 {best_result['avg_score']:.2f}, 최고 평균 점수 {best_result['best_avg_score']:.2f}, 최고 개별 점수 {best_result['best_sample_score']:.2f}")
                
                # CSV 출력 기능
                df = pd.DataFrame(results)
                df = df[['iteration', 'avg_score', 'best_avg_score', 'best_sample_score', 'system_prompt', 'user_prompt', 'meta_prompt']]
                df.columns = ['Iteration', 'Average Score', 'Best Average Score', 'Best Sample Score', 'System Prompt', 'User Prompt', 'Meta Prompt']
                summary_csv = df.to_csv(index=False)
                
                # 상세 결과 CSV 생성
                detailed_csv = tuner.save_results_to_csv()
                
                # ZIP 파일 생성
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                    zip_file.writestr("prompt_tuning_results.csv", summary_csv)
                    zip_file.writestr("detailed_iteration_results.csv", detailed_csv)
                
                # 다운로드 버튼
                st.download_button(
                    label="결과를 CSV로 저장",
                    data=zip_buffer.getvalue(),
                    file_name="prompt_tuning_results.zip",
                    mime="application/zip"
                )
            else:
                st.warning("튜닝 결과가 없습니다.") 