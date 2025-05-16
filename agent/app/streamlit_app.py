import streamlit as st
import pandas as pd
import plotly.express as px
from agent.core.prompt_tuner import PromptTuner
from agent.common.api_client import Model
import os
import logging
import plotly.graph_objects as go
import sys
from dotenv import load_dotenv
import tempfile
import base64
import zipfile
import io
import time
from typing import List, Dict
import numpy as np
from datetime import datetime
import json
from pathlib import Path

# set_page_config은 반드시 첫 번째 Streamlit 명령어여야 함
st.set_page_config(page_title="Prompt Auto Tuning Agent", layout="wide")

def setup_environment():
    # 이미 환경이 로드되었는지 확인
    if hasattr(setup_environment, 'loaded'):
        return
    
    # Try to load from different possible locations
    env_paths = [
        '.env',  # Current directory
        '../.env',  # Parent directory
        '../../.env',  # Parent's parent directory
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'),  # Project root
        str(Path.home() / '.env'),  # User's home directory
    ]
    
    env_loaded = False
    for env_path in env_paths:
        if os.path.exists(env_path):
            load_dotenv(env_path, override=True)
            print(f"Loaded environment from: {env_path}")
            env_loaded = True
            break
    
    if not env_loaded:
        print("Warning: No .env file found")
    
    # Verify required environment variables
    required_vars = ['OPENAI_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        st.error(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        st.error("Please ensure these variables are set in your .env file or environment")
        st.stop()
    
    # 환경이 로드되었음을 표시
    setup_environment.loaded = True

# Call setup at the start
setup_environment()

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from agent.dataset.mmlu_dataset import MMLUDataset
from agent.dataset.mmlu_pro_dataset import MMLUProDataset
from agent.dataset.cnn_dataset import CNNDataset
from agent.dataset.gsm8k_dataset import GSM8KDataset
from agent.dataset.mbpp_dataset import MBPPDataset
from agent.dataset.xsum_dataset import XSumDataset
from agent.dataset.bbh_dataset import BBHDataset
from agent.dataset.truthfulqa_dataset import TruthfulQADataset
from agent.dataset.hellaswag_dataset import HellaSwagDataset
from agent.dataset.humaneval_dataset import HumanEvalDataset
from agent.dataset.samsum_dataset import SamsumDataset
from agent.dataset.meetingbank_dataset import MeetingBankDataset

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.title("Prompt Tuning Dashboard")

# 모델 정보 정의
MODEL_INFO = Model.get_all_model_info()

# 프롬프트 파일 로드
prompts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompts')
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
# MMLU Pro 데이터셋 인스턴스 생성
mmlu_pro_dataset = MMLUProDataset()

# HellaSwag 데이터셋 인스턴스 생성
hellaswag_dataset = HellaSwagDataset()

# HumanEval 데이터셋 인스턴스 생성
humaneval_dataset = HumanEvalDataset()

# XSum 데이터셋 인스턴스 생성 (한 번만 생성)
xsum_dataset = XSumDataset()
# BBH 데이터셋 인스턴스 생성 (한 번만 생성)
bbh_dataset = BBHDataset()
# TruthfulQA 데이터셋 인스턴스 생성 (한 번만 생성)
truthfulqa_dataset = TruthfulQADataset()

# 사이드바에서 파라미터 설정
with st.sidebar:
    st.header("튜닝 설정")
    
    # 반복 설정 그룹
    with st.expander("반복 설정", expanded=True):
        iterations = st.slider(
            "반복 횟수", 
            min_value=1, 
            max_value=100, 
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
            index=list(MODEL_INFO.keys()).index("local1") if "local1" in MODEL_INFO else 0,  # local1이 있으면 기본값, 없으면 첫 번째
            help="프롬프트 튜닝에 사용할 모델을 선택하세요. (solar_strawberry: Upstage Solar-Strawberry API)"
        )
        st.caption(MODEL_INFO[model_name]['description'])
        
        # 튜닝 모델 버전 선택
        use_custom_tuning_version = st.toggle(
            "커스텀 버전 사용",
            value=False,  # 기본값을 False로 변경
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
            index=list(MODEL_INFO.keys()).index("local1") if "local1" in MODEL_INFO else 0,
            help="메타 프롬프트 생성에 사용할 모델을 선택하세요. (solar_strawberry: Upstage Solar-Strawberry API)"
        )
        st.caption(MODEL_INFO[meta_prompt_model]['description'])
        
        # 메타 프롬프트 모델 버전 선택
        use_custom_meta_version = st.toggle(
            "커스텀 버전 사용",
            value=False,  # 기본값을 False로 변경
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
            index=list(MODEL_INFO.keys()).index("local1") if "local1" in MODEL_INFO else 0,
            help="출력 평가에 사용할 모델을 선택하세요. (solar_strawberry: Upstage Solar-Strawberry API)"
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
            help="평가 모델의 유저 프롬프트를 설정합니다. {question}, {output}, {expected}를 포함해야 합니다."
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
    
    if dataset_type == "Samsum":
        for item in data:
            test_cases.append({
                'question': item['dialogue'],
                'expected': item['summary']
            })
            if len(display_data) < 2000:
                display_data.append({
                    'question': item['dialogue'],
                    'expected_answer': item['summary']
                })
    elif dataset_type == "MeetingBank":
        for item in data:
            test_cases.append({
                'question': item['transcript'],
                'expected': item['summary']
            })
            if len(display_data) < 2000:
                display_data.append({
                    'question': item['transcript'],
                    'expected_answer': item['summary']
                })
    elif dataset_type == "BBH":
        for item in data:
            test_cases.append({
                'question': item['input'],
                'expected': item['target']
            })
            
            if len(display_data) < 2000:  # display_data를 2000개로 제한
                display_data.append({
                    'question': item['input'],
                    'expected_answer': item['target']
                })
    elif dataset_type == "MBPP":
        for item in data:
            test_cases.append({
                'question': item['text'],
                'expected': item['code']
            })
            
            if len(display_data) < 2000:  # display_data를 2000개로 제한
                display_data.append({
                    'question': item['text'],
                    'expected_answer': f"```python\n{item['code']}\n```"
                })
    elif dataset_type == "XSum":
        for item in data:
            test_cases.append({
                'question': item['document'],
                'expected': item['summary']
            })
            
            if len(display_data) < 2000:  # display_data를 2000개로 제한
                display_data.append({
                    'question': item['document'],  # 처음 200자만 표시
                    'expected_answer': item['summary']
                })
    elif dataset_type in ["MMLU", "MMLU Pro"]:
        for item in data:
            # 선택지를 문자열로 변환
            choices_str = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(item['choices'])])
            question = f"{item['question']}\n\nChoices:\n{choices_str}"
            # answer 타입에 따라 expected 처리
            if isinstance(item['answer'], int):
                expected = chr(65 + item['answer'])
            elif isinstance(item['answer'], str) and len(item['answer']) == 1 and item['answer'].isalpha():
                expected = item['answer']
            else:
                expected = item['answer']  # 해설 등 기타 문자열
            test_cases.append({
                'question': question,
                'expected': expected
            })
            if len(display_data) < 2000:  # display_data를 2000개로 제한
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
            
            if len(display_data) < 2000:  # display_data를 2000개로 제한
                display_data.append({
                    'question': row['question'],
                    'expected_answer': row['expected_answer']
                })
    elif dataset_type == "CNN":
        for item in data:
            normalized_expected = ' '.join(
                line.strip()
                for line in item['expected_answer'].split('\n')
                if line.strip() and not line.strip().startswith(('-', '*'))
            )
            test_cases.append({
                'question': item['input'],
                'expected': normalized_expected
            })
            
            if len(display_data) < 2000:  # display_data를 2000개로 제한
                display_data.append({
                    'question': item['input'],
                    'expected_answer': normalized_expected
                })
    elif dataset_type == "GSM8K":
        for item in data:
            test_cases.append({
                'question': item['question'],
                'expected': item['answer']
            })
            
            if len(display_data) < 2000:  # display_data를 2000개로 제한
                display_data.append({
                    'question': item['question'],
                    'expected_answer': item['answer']
                })
    elif dataset_type == "TruthfulQA":
        for item in data:
            test_cases.append({
                'question': item['input'],
                'expected': item['target']
            })
            
            if len(display_data) < 2000:  # display_data를 2000개로 제한
                display_data.append({
                    'question': item['input'],
                    'expected_answer': item['target']
                })
    elif dataset_type == "HellaSwag":
        for item in data:
            # 선택지를 문자열로 변환
            choices_str = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(item['choices'])])
            question = f"Activity: {item['activity_label']}\nContext: {item['context']}\n\nComplete the context with the most appropriate ending:\n{choices_str}"
            
            test_cases.append({
                'question': question,
                'expected': chr(65 + item['answer'])  # 0-based index를 A, B, C, D로 변환
            })
            
            if len(display_data) < 2000:  # display_data를 2000개로 제한
                display_data.append({
                    'question': question,
                    'expected_answer': chr(65 + item['answer'])
                })
    elif dataset_type == "HumanEval":
        for item in data:
            test_cases.append({
                'question': item['prompt'],
                'expected': item['canonical_solution']
            })
            
            if len(display_data) < 2000:  # display_data를 2000개로 제한
                display_data.append({
                    'question': item['prompt'],
                    'expected_answer': item['canonical_solution']
                })
    elif dataset_type in ["MMLU", "MMLU Pro"]:
        # 선택된 데이터셋에 따라 적절한 데이터셋 인스턴스와 과목 리스트 선택
        if dataset_type == "MMLU":
            dataset = mmlu_dataset
            dataset_name = "MMLU"
        else:  # MMLU Pro
            dataset = mmlu_pro_dataset
            dataset_name = "MMLU Pro"
        
        # 데이터셋 선택에 '모든 과목' 옵션 추가
        subject_options = ["모든 과목"] + dataset.subjects
        subject = st.selectbox(
            f"Select {dataset_name} Subject",
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
                all_subjects_data = dataset.get_all_subjects_data()
                # 모든 과목의 데이터를 하나의 리스트로 합치기
                data = []
                for subject_data in all_subjects_data.values():
                    data.extend(subject_data[split])
            else:
                # 특정 과목의 데이터 로드
                subject_data = dataset.get_subject_data(subject)
                data = subject_data[split]
            
            test_cases, num_samples = process_dataset(data, dataset_type)
        except Exception as e:
            st.error(f"{dataset_name} 데이터셋 로드 중 오류 발생: {str(e)}")
            st.stop()

    # 전체 데이터 표시
    st.write("데이터셋 내용:")
    st.dataframe(pd.DataFrame(display_data))
    
    return test_cases, num_samples

# 데이터셋 선택
st.header("Dataset Selection")
dataset_type = st.radio(
    "Select Dataset Type",
    ["CSV", "MMLU", "MMLU Pro", "CNN", "GSM8K", "MBPP", "BBH", "TruthfulQA", "HellaSwag", "HumanEval", "Samsum", "MeetingBank"],
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
        st.info("CSV 파일을 업로드하거나 다른 데이터셋을 선택하세요.")
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
    
    # 전체 청크 선택 옵션 추가
    use_all_chunks = st.toggle(
        "전체 청크 사용",
        value=False,
        help="모든 청크의 데이터를 로드합니다. 처리 시간이 오래 걸릴 수 있습니다."
    )
    
    try:
        if use_all_chunks:
            # 모든 청크 로드
            data = cnn_dataset.load_all_data(split)
            test_cases, num_samples = process_dataset(data, "CNN")
            
            # 선택된 청크 정보 표시
            st.info(f"전체 청크 로드 완료 ({len(data):,}개 예제)")
        else:
            # 청크 선택
            st.write(f"총 {total_chunks}개의 청크가 있습니다.")
            chunk_index = st.number_input(
                "청크 선택",
                min_value=0,
                max_value=total_chunks-1,
                value=0,
                help="처리할 청크의 인덱스를 선택하세요."
            )
            
            # 선택된 청크 로드
            data = cnn_dataset.load_data(split, chunk_index)
            test_cases, num_samples = process_dataset(data, "CNN")
            
            # 선택된 청크 정보 표시
            st.info(f"선택된 청크: {chunk_index} ({len(data):,}개 예제)")
    except Exception as e:
        st.error(f"CNN 데이터셋 로드 중 오류 발생: {str(e)}")
        st.stop()
elif dataset_type == "GSM8K":
    # GSM8K 데이터셋 인스턴스 생성
    gsm8k_dataset = GSM8KDataset()
    
    # 데이터셋 선택
    split = st.selectbox(
        "데이터셋 선택",
        ["train", "test"],
        index=0
    )
    
    try:
        # 데이터 로드
        data = gsm8k_dataset.load_data(split)
        test_cases, num_samples = process_dataset(data, "GSM8K")
        
        # 데이터셋 정보 표시
        st.info(f"GSM8K {split} 데이터셋: {len(data):,}개 예제")
    except Exception as e:
        st.error(f"GSM8K 데이터셋 로드 중 오류 발생: {str(e)}")
        st.stop()
elif dataset_type == "MBPP":
    # MBPP 데이터셋 인스턴스 생성
    mbpp_dataset = MBPPDataset()
    
    # 데이터셋 선택
    split = st.selectbox(
        "데이터셋 선택",
        ["train", "test", "validation"],
        index=1  # test를 기본값으로 설정
    )
    
    try:
        # 데이터 로드
        data = mbpp_dataset.get_split_data(split)
        test_cases, num_samples = process_dataset(data, "MBPP")
        
        # 데이터셋 정보 표시
        st.info(f"MBPP {split} 데이터셋: {len(data):,}개 예제")
    except Exception as e:
        st.error(f"MBPP 데이터셋 로드 중 오류 발생: {str(e)}")
        st.stop()
elif dataset_type == "BBH":
    # 이미 생성된 BBHDataset 인스턴스를 사용
    try:
        # 카테고리 선택 UI에 "모든 카테고리" 옵션 추가
        bbh_categories = ["모든 카테고리"] + bbh_dataset.get_all_categories()
        selected_category = st.selectbox(
            "BBH 카테고리 선택",
            bbh_categories,
            index=0,
            key="bbh_category_selectbox"
        )
        if selected_category == "모든 카테고리":
            # 전체 데이터 로드
            all_data_dict = bbh_dataset.get_all_data()
            # {"test": [...]} 형태이므로 합쳐서 리스트로 변환
            data = []
            for split_data in all_data_dict.values():
                data.extend(split_data)
            st.info(f"BBH 전체 데이터셋: {len(data):,}개 예제")
        else:
            # 카테고리별 데이터 로드
            data = bbh_dataset.get_category_data(selected_category)
            st.info(f"BBH {selected_category} 카테고리 데이터셋: {len(data):,}개 예제")
        test_cases, num_samples = process_dataset(data, "BBH")
    except Exception as e:
        st.error(f"BBH 데이터셋 로드 중 오류 발생: {str(e)}")
        st.stop()
elif dataset_type == "TruthfulQA":
    # 이미 생성된 TruthfulQADataset 인스턴스를 사용
    try:
        # 데이터 로드
        data = truthfulqa_dataset.get_split_data("test")
        test_cases, num_samples = process_dataset(data, "TruthfulQA")
        
        # 데이터셋 정보 표시
        st.info(f"TruthfulQA 테스트 데이터셋: {len(data):,}개 예제")
    except Exception as e:
        st.error(f"TruthfulQA 데이터셋 로드 중 오류 발생: {str(e)}")
        st.stop()
elif dataset_type == "HellaSwag":
    try:
        # 데이터셋 선택
        split = st.selectbox(
            "데이터셋 선택",
            ["validation", "train"],
            index=0
        )
        
        # 데이터 로드
        data = hellaswag_dataset.get_split_data(split)
        test_cases, num_samples = process_dataset(data, "HellaSwag")
        
        # 데이터셋 정보 표시
        st.info(f"HellaSwag {split} 데이터셋: {len(data):,}개 예제")
    except Exception as e:
        st.error(f"HellaSwag 데이터셋 로드 중 오류 발생: {str(e)}")
        st.stop()
elif dataset_type == "HumanEval":
    try:
        # HumanEval은 test split만 있음
        data = humaneval_dataset.get_split_data("test")
        test_cases, num_samples = process_dataset(data, "HumanEval")
        
        # 데이터셋 정보 표시
        st.info(f"HumanEval 테스트 데이터셋: {len(data):,}개 예제")
    except Exception as e:
        st.error(f"HumanEval 데이터셋 로드 중 오류 발생: {str(e)}")
        st.stop()
elif dataset_type == "Samsum":
    samsum_dataset = SamsumDataset()
    split = st.selectbox(
        "데이터셋 선택",
        ["train", "validation", "test"],
        index=0
    )
    try:
        data = samsum_dataset.get_split_data(split)
        test_cases, num_samples = process_dataset(data, "Samsum")
        st.info(f"Samsum {split} 데이터셋: {len(data):,}개 예제")
    except Exception as e:
        st.error(f"Samsum 데이터셋 로드 중 오류 발생: {str(e)}")
        st.stop()
elif dataset_type == "MeetingBank":
    meetingbank_dataset = MeetingBankDataset()
    split = st.selectbox(
        "데이터셋 선택",
        ["validation", "test"],
        index=0
    )
    try:
        data = meetingbank_dataset.get_split_data(split)
        test_cases, num_samples = process_dataset(data, "MeetingBank")
        st.info(f"MeetingBank {split} 데이터셋: {len(data):,}개 예제")
    except Exception as e:
        st.error(f"MeetingBank 데이터셋 로드 중 오류 발생: {str(e)}")
        st.stop()
elif dataset_type in ["MMLU", "MMLU Pro"]:
    # 선택된 데이터셋에 따라 적절한 데이터셋 인스턴스와 과목 리스트 선택
    if dataset_type == "MMLU":
        dataset = mmlu_dataset
        dataset_name = "MMLU"
    else:  # MMLU Pro
        dataset = mmlu_pro_dataset
        dataset_name = "MMLU Pro"
    # 데이터셋 선택에 '모든 과목' 옵션 추가
    subject_options = ["모든 과목"] + dataset.subjects
    subject = st.selectbox(
        f"Select {dataset_name} Subject",
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
            all_subjects_data = dataset.get_all_subjects_data()
            # 모든 과목의 데이터를 하나의 리스트로 합치기
            data = []
            for subject_data in all_subjects_data.values():
                data.extend(subject_data[split])
        else:
            # 특정 과목의 데이터 로드
            subject_data = dataset.get_subject_data(subject)
            data = subject_data[split]
        test_cases, num_samples = process_dataset(data, dataset_type)
    except Exception as e:
        st.error(f"{dataset_name} 데이터셋 로드 중 오류 발생: {str(e)}")
        st.stop()

class SessionState:
    """
    Streamlit 앱의 세션 상태를 관리하는 클래스
    """
    @staticmethod
    def init_state():
        """세션 상태를 초기화합니다."""
        st.session_state.all_iteration_results = []
        st.session_state.current_iteration = 0
        st.session_state.show_results = False
        st.session_state.tuning_complete = False
        st.session_state.display_container = st.empty()
    
    @staticmethod
    def reset():
        """상태를 초기화합니다."""
        st.session_state.all_iteration_results = []
        st.session_state.current_iteration = 0
        st.session_state.show_results = False
        st.session_state.tuning_complete = False
    
    @staticmethod
    def update_results(result):
        """새로운 결과를 추가합니다."""
        # 로깅 추가
        logging.info(f"Updating results for iteration {result.iteration}")
        
        # 결과가 이미 있는지 확인
        if not hasattr(st.session_state, 'all_iteration_results'):
            st.session_state.all_iteration_results = []
        
        # 같은 iteration의 결과가 있다면 업데이트, 없다면 추가
        existing_result = next(
            (r for r in st.session_state.all_iteration_results if r.iteration == result.iteration),
            None
        )
        
        if existing_result:
            index = st.session_state.all_iteration_results.index(existing_result)
            st.session_state.all_iteration_results[index] = result
            logging.info(f"Updated existing result at index {index}")
        else:
            st.session_state.all_iteration_results.append(result)
            logging.info(f"Added new result, total results: {len(st.session_state.all_iteration_results)}")
        
        st.session_state.current_iteration = result.iteration - 1  # 0-based index
        st.session_state.show_results = True
        logging.info(f"Session state updated: current_iteration={st.session_state.current_iteration}, show_results={st.session_state.show_results}")
    
    @staticmethod
    def get_results():
        """현재 저장된 모든 결과를 반환합니다."""
        if not hasattr(st.session_state, 'all_iteration_results'):
            st.session_state.all_iteration_results = []
        return st.session_state.all_iteration_results
    
    @staticmethod
    def get_current_iteration():
        """현재 선택된 이터레이션을 반환합니다."""
        if not hasattr(st.session_state, 'current_iteration'):
            st.session_state.current_iteration = 0
        return st.session_state.current_iteration
    
    @staticmethod
    def set_current_iteration(iteration):
        """현재 이터레이션을 설정합니다."""
        st.session_state.current_iteration = iteration

class ResultsDisplay:
    """
    결과 표시를 담당하는 클래스
    """
    def __init__(self):
        SessionState.init_state()
        # 메인 컨테이너 초기화
        if 'main_container' not in st.session_state:
            st.session_state.main_container = st.empty()
    
    def display_metrics(self, results, container):
        """성능 지표를 표시합니다."""
        if not results:
            return
        
        # 그래프 데이터 준비
        x_values = [result.iteration for result in results]
        avg_scores = [result.avg_score for result in results]
        best_sample_scores = [result.best_sample_score for result in results]
        std_devs = [result.std_dev for result in results]
        top3_scores = [result.top3_avg_score for result in results]
        
        
        # 카테고리별 평균 점수 계산
        category_scores = {
            'meaning_accuracy': [],
            'completeness': [],
            'expression_style': [],
            'faithfulness': [],
            'conciseness': [],
            'correctness': [],
            'structural_alignment': [],
            'reasoning_quality': []
        }
        
        for result in results:
            iteration_category_scores = {category: [] for category in category_scores.keys()}
            iteration_category_weights = {category: [] for category in category_scores.keys()}  # 각 이터레이션의 가중치
            
            for test_case in result.test_case_results:
                if test_case.evaluation_details and 'category_scores' in test_case.evaluation_details:
                    for category, details in test_case.evaluation_details['category_scores'].items():
                        if category in iteration_category_scores:
                            iteration_category_scores[category].append(details['score'])
                            iteration_category_weights[category].append(details.get('weight', 0.5))  # 가중치 추가
            
            # 각 카테고리의 평균 점수와 가중치 추가
            for category in category_scores:
                scores = iteration_category_scores[category]
                weights = iteration_category_weights[category]
                avg_score = np.mean(scores) if scores else 0
                avg_weight = np.mean(weights) if weights else 0.5
                category_scores[category].append(avg_score)
        
        # 통합 그래프 생성
        fig = go.Figure()
        
        # 카테고리별 점수를 막대 그래프로 추가
        for category in category_scores:
            fig.add_trace(go.Bar(
                x=x_values,
                y=category_scores[category],
                name=category,
                visible=True
            ))
        
        # 주요 성능 지표 트레이스
        fig.add_trace(go.Scatter(
            x=x_values,
            y=avg_scores,
            name='평균 점수',
            mode='lines+markers',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=x_values,
            y=std_devs,
            name='표준편차',
            mode='lines+markers',
            line=dict(color='purple', width=2, dash='dot')
        ))
        
        fig.add_trace(go.Scatter(
            x=x_values,
            y=best_sample_scores,
            name='최고 개별 점수',
            mode='lines+markers',
            line=dict(color='green', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=x_values,
            y=top3_scores,
            name='Top3 평균 점수',
            mode='lines+markers',
            line=dict(color='red', width=2)
        ))
        
        # 그래프 레이아웃 설정
        fig.update_layout(
            title='통합 성능 지표 및 카테고리 분석',
            xaxis_title='이터레이션',
            yaxis_title='점수',
            yaxis_range=[0, 1],
            xaxis=dict(
                tickmode='array',
                tickvals=x_values,
                ticktext=[f"Iteration {x}" for x in x_values]
            ),
            height=600,
            barmode='group',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05
            )
        )
        
        # 그래프 표시
        container.plotly_chart(fig, use_container_width=True)
    
    def display_iteration_details(self, results, container):
        """이터레이션 상세 정보를 표시합니다."""
        if not results:
            container.info("아직 결과가 없습니다.")
            return
        
        # 이터레이션 선택
        total_iterations = len(results)
        if total_iterations > 0:
            # 이터레이션 선택 UI
            current_iteration = SessionState.get_current_iteration()
            
            # 이터레이션 선택을 위한 탭 생성
            tabs = container.tabs([f"Iteration {i+1}" for i in range(total_iterations)])
            selected_iteration = current_iteration
            
            with tabs[selected_iteration]:
                iteration_result = results[selected_iteration]
                
                # 평균 점수와 표준편차 표시
                col1, col2, col3 = st.columns(3)
                col1.metric("Average Score", f"{iteration_result.avg_score:.2f}")
                col2.metric("Standard Deviation", f"{iteration_result.std_dev:.2f}")
                col3.metric("Top 3 Average", f"{iteration_result.top3_avg_score:.2f}")
                
                # Task Type과 Description expander 추가
                with st.expander(f"Task Type ({iteration_result.task_type})", expanded=False):
                    st.markdown("### Task Description")
                    st.code(iteration_result.task_description, language="text")
                
                # 현재 프롬프트 expander 추가
                with st.expander("현재 프롬프트 보기", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### System Prompt")
                        st.code(iteration_result.system_prompt, language="text")
                    with col2:
                        st.markdown("### User Prompt")
                        st.code(iteration_result.user_prompt, language="text")
                
                # 가중치 점수 expander 추가
                with st.expander("현재 가중치 점수 보기", expanded=False):
                    # 가중치 데이터 수집
                    weight_data = []
                    for test_case in iteration_result.test_case_results:
                        if test_case.evaluation_details and 'category_scores' in test_case.evaluation_details:
                            for category, details in test_case.evaluation_details['category_scores'].items():
                                weight = details.get('weight', 0.5)
                                weight_data.append({
                                    'Category': category,
                                    'Weight': weight
                                })
                    
                    if weight_data:
                        # 카테고리별 평균 가중치 계산
                        df = pd.DataFrame(weight_data)
                        avg_weights = df.groupby('Category')['Weight'].mean().round(3)
                        avg_weights = avg_weights.reset_index()
                        avg_weights.columns = ['Category', 'Average Weight']
                        
                        # 데이터프레임 스타일링
                        def highlight_weights(val):
                            color = f'background-color: rgba(255, 99, 71, {val})'
                            return color
                        
                        # 스타일이 적용된 데이터프레임 표시
                        st.write("Category Weights:")
                        styled_df = avg_weights.style.apply(lambda x: [highlight_weights(v) for v in x], subset=['Average Weight'])
                        st.dataframe(styled_df, use_container_width=True)
                
                # 출력 결과를 데이터프레임으로 변환
                outputs_data = []
                for i, test_case in enumerate(iteration_result.test_case_results):
                    row = {
                        'Test Case': i + 1,
                        'Score': f"{test_case.score:.2f}",
                        'Question': test_case.question,
                        'Expected': test_case.expected_output,
                        'Actual': test_case.actual_output,
                        'Evaluation Details': json.dumps(test_case.evaluation_details, ensure_ascii=False, indent=2)
                    }
                    
                    # 카테고리별 점수와 피드백 추가
                    if test_case.evaluation_details and 'category_scores' in test_case.evaluation_details:
                        for category, details in test_case.evaluation_details['category_scores'].items():
                            row[f"{category} Score"] = f"{details['score']:.2f}"
                            row[f"{category} Weight"] = f"{details.get('weight', 1.0):.2f}"  # 가중치 표시 추가
                            row[f"{category} State"] = details['current_state']
                            row[f"{category} Action"] = details['improvement_action']
                    
                    outputs_data.append(row)
                
                # 데이터프레임 생성
                df = pd.DataFrame(outputs_data)
                
                # 데이터프레임 스타일링 함수
                def highlight_rows(df):
                    scores = df['Score'].astype(float)
                    max_score = scores.max()
                    min_score = scores.min()
                    
                    background_colors = pd.DataFrame('', index=df.index, columns=df.columns)
                    
                    max_score_mask = (scores == max_score)
                    min_score_mask = (scores == min_score)
                    
                    background_colors.loc[max_score_mask] = 'background-color: #90EE90'
                    background_colors.loc[min_score_mask] = 'background-color: #FFB6C6'
                    
                    return background_colors
                
                # 스타일이 적용된 데이터프레임 표시
                st.dataframe(
                    df.style.apply(highlight_rows, axis=None),
                    use_container_width=True,
                    height=400
                )
                
                # 메타프롬프트 expander 추가
                if iteration_result.meta_prompt:
                    with st.expander("메타프롬프트 결과 보기", expanded=False):
                        st.code(iteration_result.meta_prompt, language="text")
            
            # 현재 선택된 이터레이션 저장
            SessionState.set_current_iteration(selected_iteration)
    
    def update(self):
        """결과 표시를 업데이트합니다."""
        results = SessionState.get_results()
        if st.session_state.show_results and results:
            # 기존 컨테이너를 비우고 새로운 컨테이너 생성
            with st.session_state.main_container.container():
                st.empty()  # 기존 내용을 지웁니다
                
                # 메트릭스와 상세 정보를 표시할 새로운 컨테이너 생성
                metrics_container = st.container()
                details_container = st.container()
                
                # 메트릭스와 상세 정보 표시
                self.display_metrics(results, metrics_container)
                self.display_iteration_details(results, details_container)

def run_tuning_process():
    """프롬프트 튜닝 프로세스를 실행하고 결과를 시각화합니다."""
    # UI 상태 초기화
    SessionState.init_state()
    results_display = ResultsDisplay()
    
    with st.spinner('프롬프트 튜닝 중...'):
        def iteration_callback(result):
            logging.info(f"Iteration callback called for iteration {result.iteration}")
            SessionState.update_results(result)
            logging.info("Results updated, updating display...")
            results_display.update()
            logging.info("Display updated")
        
        # iteration_callback 설정
        tuner.iteration_callback = iteration_callback
        
        # 프롬프트 튜닝 실행
        tuner.tune_prompt(
            initial_system_prompt=system_prompt,
            initial_user_prompt=user_prompt,
            initial_test_cases=test_cases,
            num_iterations=iterations,
            score_threshold=score_threshold if use_threshold else None,
            evaluation_score_threshold=evaluation_threshold,
            use_meta_prompt=use_meta_prompt,
            num_samples=num_samples
        )
        
        st.session_state.tuning_complete = True
        logging.info("Tuning process completed")
        
        # 최종 결과
        results = SessionState.get_results()
        if results:
            st.success("프롬프트 튜닝 완료!")
            logging.info(f"Final results count: {len(results)}")
            
            # 전체 결과에서 가장 높은 평균 점수를 가진 프롬프트 찾기
            best_result = max(results, key=lambda x: x.avg_score)
            st.write("Final Best Prompt:")
            col1, col2 = st.columns(2)
            with col1:
                st.write("System Prompt:")
                st.code(best_result.system_prompt)
            with col2:
                st.write("User Prompt:")
                st.code(best_result.user_prompt)
            st.write(f"최종 결과: 평균 점수 {best_result.avg_score:.2f}, 최고 평균 점수 {best_result.best_avg_score:.2f}, 최고 개별 점수 {best_result.best_sample_score:.2f}")
            
            # CSV 다운로드 버튼
            try:
                csv_data = tuner.save_results_to_csv()
                st.download_button(
                    label="결과를 CSV 파일로 저장",
                    data=csv_data,
                    file_name=f"prompt_tuning_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_csv"
                )
            except Exception as e:
                st.error(f"CSV 파일 생성 중 오류가 발생했습니다: {str(e)}")
        else:
            st.warning("튜닝 결과가 없습니다.")

# 튜닝 시작 버튼
if st.button("프롬프트 튜닝 시작", type="primary"):
    # 세션 상태 초기화
    SessionState.reset()
    
    # API 키 확인
    required_keys = {
        "solar": "SOLAR_API_KEY",
        "gpt4o": "OPENAI_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
        "local1": None,  # local1 모델은 API 키가 필요하지 않음
        "local2": None,   # local2 모델은 API 키가 필요하지 않음
        "solar_strawberry": "SOLAR_STRAWBERRY_API_KEY",  # 추가
    }
    
    # 사용되는 모델들의 API 키 확인
    used_models = set([model_name, evaluator_model])
    missing_keys = []
    
    for model in used_models:
        key = required_keys[model]
        if key and not os.getenv(key):  # key가 None이 아닌 경우에만 API 키 확인
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
        run_tuning_process() 