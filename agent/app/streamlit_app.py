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

# set_page_config must be the first Streamlit command
st.set_page_config(page_title="Prompt Auto Tuning Agent", layout="wide")

def setup_environment():
    # Check if environment is already loaded
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
    
    # Mark that environment is loaded
    setup_environment.loaded = True

# Call setup at the start
setup_environment()

# Add project root directory to Python path
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

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.title("Prompt Tuning Dashboard")

# Define model information
MODEL_INFO = Model.get_all_model_info()

# Load prompt files
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

# Create MMLU dataset instance
mmlu_dataset = MMLUDataset()
# Create MMLU Pro dataset instance
mmlu_pro_dataset = MMLUProDataset()

# Create HellaSwag dataset instance
hellaswag_dataset = HellaSwagDataset()

# Create HumanEval dataset instance
humaneval_dataset = HumanEvalDataset()

# Create XSum dataset instance (create only once)
xsum_dataset = XSumDataset()
# Create BBH dataset instance (create only once)
bbh_dataset = BBHDataset()
# Create TruthfulQA dataset instance (create only once)
truthfulqa_dataset = TruthfulQADataset()

# Parameter settings in sidebar
with st.sidebar:
    st.header("Tuning Settings")
    
    # Iteration settings group
    with st.expander("Iteration Settings", expanded=True):
        iterations = st.slider(
            "Number of Iterations", 
            min_value=1, 
            max_value=100, 
            value=3,
            help="Set the number of iterations for prompt tuning."
        )
    
    # Prompt improvement settings group
    with st.expander("Prompt Improvement Settings", expanded=True):
        # Toggle for using prompt improvement
        use_meta_prompt = st.toggle(
            "Use Prompt Improvement", 
            value=True, 
            help="Use meta prompt to improve the prompt. Disable to use initial prompt."
        )
        
        # Evaluation prompt score threshold setting (only active when prompt improvement is enabled)
        evaluation_threshold = st.slider(
            "Evaluation Prompt Score Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.1,
            disabled=not use_meta_prompt,
            help="Improve the prompt if the score is below this threshold. Only available when prompt improvement is enabled."
        )
        
        # Toggle for applying average score threshold (only active when prompt improvement is enabled)
        use_threshold = st.toggle(
            "Apply Average Score Threshold",
            value=True,
            disabled=not use_meta_prompt,
            help="Stop the iteration if the average score is above the threshold. Only available when prompt improvement is enabled."
        )
        
        # Average score threshold slider (disabled when average score threshold is off or prompt improvement is off)
        score_threshold = st.slider(
            "Average Score Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.9,
            step=0.05,
            disabled=not (use_threshold and use_meta_prompt),
            help="Stop the iteration if the average score is above this threshold. Available only when average score threshold is enabled and prompt improvement is enabled."
        )
    
    # Divider for model settings section
    st.divider()
    
    # Model settings group
    with st.expander("Tuning Model Settings", expanded=True):
        # Model selection
        model_name = st.selectbox(
            "Model Selection",
            options=list(MODEL_INFO.keys()),
            format_func=lambda x: f"{MODEL_INFO[x]['name']} ({MODEL_INFO[x]['default_version']})",
            index=list(MODEL_INFO.keys()).index("local1") if "local1" in MODEL_INFO else 0,  # Use local1 as default if available, otherwise first
            help="Select the model to use for prompt tuning. (solar_strawberry: Upstage Solar-Strawberry API)"
        )
        st.caption(MODEL_INFO[model_name]['description'])
        
        # Tuning model version selection
        use_custom_tuning_version = st.toggle(
            "Use Custom Version",
            value=False,  # Changed default value to False
            help="Use a custom version instead of the default version for the tuning model."
        )
        
        if use_custom_tuning_version:
            tuning_model_version = st.text_input(
                "Model Version",
                value=MODEL_INFO[model_name]['default_version'],
                help="Enter the model version to use for tuning."
            )
        else:
            tuning_model_version = None  # Use default version by setting to None
    
    # Meta prompt model settings group
    with st.expander("Meta Prompt Model Settings", expanded=True):
        # Meta prompt model selection
        meta_prompt_model = st.selectbox(
            "Model Selection",
            options=list(MODEL_INFO.keys()),
            format_func=lambda x: f"{MODEL_INFO[x]['name']} ({MODEL_INFO[x]['default_version']})",
            # index=list(MODEL_INFO.keys()).index("local1") if "local1" in MODEL_INFO else 0,
            help="Select the model to use for meta prompt generation. (solar_strawberry: Upstage Solar-Strawberry API)"
        )
        st.caption(MODEL_INFO[meta_prompt_model]['description'])
        
        # Meta prompt model version selection
        use_custom_meta_version = st.toggle(
            "Use Custom Version",
            value=False,  # Changed default value to False
            help="Use a custom version instead of the default version for the meta prompt model."
        )
        
        if use_custom_meta_version:
            meta_model_version = st.text_input(
                "Model Version",
                value=MODEL_INFO[meta_prompt_model]['default_version'],
                help="Enter the model version to use for meta prompt generation."
            )
        else:
            meta_model_version = None  # Use default version by setting to None
    
    # Evaluation model settings group
    with st.expander("Evaluation Model Settings", expanded=True):
        # Evaluation model selection
        evaluator_model = st.selectbox(
            "Model Selection",
            options=list(MODEL_INFO.keys()),
            format_func=lambda x: f"{MODEL_INFO[x]['name']} ({MODEL_INFO[x]['default_version']})",
            index=list(MODEL_INFO.keys()).index("local1") if "local1" in MODEL_INFO else 0,
            help="Select the model to use for output evaluation. (solar_strawberry: Upstage Solar-Strawberry API)"
        )
        st.caption(MODEL_INFO[evaluator_model]['description'])
        
        # Evaluation model version selection
        use_custom_evaluator_version = st.toggle(
            "Use Custom Version",
            value=False,
            help="Use a custom version instead of the default version for the evaluation model."
        )
        
        if use_custom_evaluator_version:
            evaluator_model_version = st.text_input(
                "Model Version",
                value=MODEL_INFO[evaluator_model]['default_version'],
                help="Enter the model version to use for evaluation."
            )
        else:
            evaluator_model_version = None  # Use default version by setting to None

# Create PromptTuner object
tuner = PromptTuner(
    model_name=model_name,
    evaluator_model_name=evaluator_model,
    meta_prompt_model_name=meta_prompt_model,
    model_version=tuning_model_version,
    evaluator_model_version=evaluator_model_version,
    meta_prompt_model_version=meta_model_version
)

# Prompt settings
with st.expander("Initial Prompt Settings", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        system_prompt = st.text_area(
            "System Prompt",
            value=DEFAULT_SYSTEM_PROMPT,
            height=100,
            help="Enter the initial system prompt to start tuning."
        )
    with col2:
        user_prompt = st.text_area(
            "User Prompt",
            value=DEFAULT_USER_PROMPT,
            height=100,
            help="Enter the initial user prompt to start tuning."
        )
    
    if st.button("Update Initial Prompt", key="initial_prompt_update"):
        tuner.set_initial_prompt(system_prompt, user_prompt)
        st.success("Initial prompt updated.")

# Meta prompt settings
with st.expander("Meta Prompt Settings", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        meta_system_prompt = st.text_area(
            "Meta System Prompt",
            value=DEFAULT_META_SYSTEM_PROMPT,
            height=300,
            help="Enter the system prompt that defines the role and responsibility of the prompt engineer."
        )
    with col2:
        meta_user_prompt = st.text_area(
            "Meta User Prompt",
            value=DEFAULT_META_USER_PROMPT,
            height=300,
            help="Enter the user prompt that defines the input data and output format for prompt improvement."
        )
    
    if st.button("Update Meta Prompt", key="meta_prompt_update"):
        tuner.set_meta_prompt(meta_system_prompt, meta_user_prompt)
        st.success("Meta prompt updated.")

# Evaluation prompt settings
with st.expander("Evaluation Prompt Settings", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        evaluation_system_prompt = st.text_area(
            "Evaluation System Prompt",
            value=DEFAULT_EVALUATION_SYSTEM_PROMPT,
            height=200,
            help="Set the system prompt for the evaluation model."
        )
    with col2:
        evaluation_user_prompt = st.text_area(
            "Evaluation User Prompt",
            value=DEFAULT_EVALUATION_USER_PROMPT,
            height=200,
            help="Set the user prompt for the evaluation model. It must include {question}, {output}, {expected}."
        )
    
    if st.button("Update Evaluation Prompt", key="eval_prompt_update"):
        tuner.set_evaluation_prompt(evaluation_system_prompt, evaluation_user_prompt)
        st.success("Evaluation prompt updated.")

# Common dataset processing function
def process_dataset(data, dataset_type):
    # Display data
    total_examples = len(data)
    st.write(f"Total examples: {total_examples}")
    
    # Sample count selection
    num_samples = st.slider(
        "Number of random samples to evaluate per iteration",
        min_value=1,
        max_value=min(100, total_examples),  # Limit maximum sample count to 100
        value=min(5, total_examples),
        help="Select the number of random samples to evaluate per iteration."
    )
    
    # Create test cases and dataframe
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
            
            if len(display_data) < 2000:  # Limit display_data to 2000 items
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
            
            if len(display_data) < 2000:  # Limit display_data to 2000 items
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
            
            if len(display_data) < 2000:  # Limit display_data to 2000 items
                display_data.append({
                    'question': item['document'],  # Show only first 200 characters
                    'expected_answer': item['summary']
                })
    elif dataset_type in ["MMLU", "MMLU Pro"]:
        for item in data:
                        # Convert choices to string
            choices_str = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(item['choices'])])
            question = f"{item['question']}\n\nChoices:\n{choices_str}"
            # Process expected based on answer type
            if isinstance(item['answer'], int):
                expected = chr(65 + item['answer'])
            elif isinstance(item['answer'], str) and len(item['answer']) == 1 and item['answer'].isalpha():
                expected = item['answer']
            else:
                expected = item['answer']  # Explanation and other strings
            test_cases.append({
                'question': question,
                'expected': expected
            })
            if len(display_data) < 2000:  # Limit display_data to 2000 items
                display_data.append({
                    'question': question,
                    'expected_answer': expected
                })
    elif dataset_type == "CSV":
        # Check column names and mapping
        required_columns = ['question', 'expected_answer']
        available_columns = data.columns.tolist()
        
        # Check if required columns exist
        missing_columns = [col for col in required_columns if col not in available_columns]
        if missing_columns:
            st.error(f"CSV file requires the following columns: {', '.join(missing_columns)}")
            st.info("CSV file must include 'question' and 'expected_answer' columns.")
            st.stop()
        
        for _, row in data.iterrows():
            test_cases.append({
                'question': row['question'],
                'expected': row['expected_answer']
            })
            
            if len(display_data) < 2000:  # Limit display_data to 2000 items
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
            
            if len(display_data) < 2000:  # Limit display_data to 2000 items
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
            
            if len(display_data) < 2000:  # Limit display_data to 2000 items
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
            
            if len(display_data) < 2000:  # Limit display_data to 2000 items
                display_data.append({
                    'question': item['input'],
                    'expected_answer': item['target']
                })
    elif dataset_type == "HellaSwag":
        for item in data:
            # Convert choices to string
            choices_str = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(item['choices'])])
            question = f"Activity: {item['activity_label']}\nContext: {item['context']}\n\nComplete the context with the most appropriate ending:\n{choices_str}"
            
            test_cases.append({
                'question': question,
                'expected': chr(65 + item['answer'])  # Convert 0-based index to A, B, C, D
            })
            
            if len(display_data) < 2000:  # Limit display_data to 2000 items
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
            
            if len(display_data) < 2000:  # Limit display_data to 2000 items
                display_data.append({
                    'question': item['prompt'],
                    'expected_answer': item['canonical_solution']
                })
    elif dataset_type in ["MMLU", "MMLU Pro"]:
        # Select appropriate dataset instance and subject list based on selected dataset
        if dataset_type == "MMLU":
            dataset = mmlu_dataset
            dataset_name = "MMLU"
        else:  # MMLU Pro
            dataset = mmlu_pro_dataset
            dataset_name = "MMLU Pro"
        
        # Add 'All Subjects' option to dataset selection
        subject_options = ["All Subjects"] + dataset.subjects
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
            if subject == "All Subjects":
                # Load data from all subjects
                all_subjects_data = dataset.get_all_subjects_data()
                # Combine data from all subjects into one list
                data = []
                for subject_data in all_subjects_data.values():
                    data.extend(subject_data[split])
            else:
                # Load data from specific subject
                subject_data = dataset.get_subject_data(subject)
                data = subject_data[split]
            
            test_cases, num_samples = process_dataset(data, dataset_type)
        except Exception as e:
            st.error(f"{dataset_name} dataset loading error: {str(e)}")
            st.stop()

    # Display full dataset
    st.write("Dataset Content:")
    st.dataframe(pd.DataFrame(display_data))
    
    return test_cases, num_samples

    # Dataset selection
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
            # Use more flexible parsing options when reading CSV file
            df = pd.read_csv(csv_file, 
                            encoding='utf-8',
                            on_bad_lines='skip',  # Skip problematic lines
                            quoting=1,  # Wrap all fields in quotes
                            escapechar='\\')  # Set escape character
            
            # Check if dataframe is empty
            if df.empty:
                st.error("CSV file is empty. Please upload a CSV file with correct data.")
                st.stop()
            
            test_cases, num_samples = process_dataset(df, "CSV")
        except Exception as e:
            st.error(f"CSV file loading error: {str(e)}")
            st.info("Please check if the CSV file is in the correct format. It might be empty or have incorrect encoding.")
            st.stop()
    else:
        st.info("Please upload a CSV file or select another dataset.")
        st.stop()
elif dataset_type == "CNN":
    # Create CNN dataset instance
    cnn_dataset = CNNDataset()
    
    # Dataset selection
    split = st.selectbox(
        "Dataset Selection",
        ["train", "validation", "test"],
        index=0
    )
    
    # Check chunk count
    total_chunks = cnn_dataset.get_num_chunks(split)
    
    if total_chunks == 0:
        st.error(f"No chunk files found for {split} dataset.")
        st.stop()
    
    # Add option to select all chunks
    use_all_chunks = st.toggle(
        "Use All Chunks",
        value=False,
        help="Load data from all chunks. This may take a long time to process."
    )
    
    try:
        if use_all_chunks:
            # Load all chunks
            data = cnn_dataset.load_all_data(split)
            test_cases, num_samples = process_dataset(data, "CNN")
            
            # Display selected chunk information
            st.info(f"All chunks loaded ({len(data):,} examples)")
        else:
            # Chunk selection
            st.write(f"Total {total_chunks} chunks available.")
            chunk_index = st.number_input(
                "Select Chunk",
                min_value=0,
                max_value=total_chunks-1,
                value=0,
                help="Select the index of the chunk to process."
            )
            
            # Load selected chunk
            data = cnn_dataset.load_data(split, chunk_index)
            test_cases, num_samples = process_dataset(data, "CNN")
            
            # Display selected chunk information
            st.info(f"Selected chunk: {chunk_index} ({len(data):,} examples)")
    except Exception as e:
        st.error(f"CNN dataset loading error: {str(e)}")
        st.stop()
elif dataset_type == "GSM8K":
    # Create GSM8K dataset instance
    gsm8k_dataset = GSM8KDataset()
    
    # Dataset selection
    split = st.selectbox(
        "Dataset Selection",
        ["train", "test"],
        index=0
    )
    
    try:
        # Load data
        data = gsm8k_dataset.load_data(split)
        test_cases, num_samples = process_dataset(data, "GSM8K")
        
        # Display dataset information
        st.info(f"GSM8K {split} dataset: {len(data):,} examples")
    except Exception as e:
        st.error(f"GSM8K dataset loading error: {str(e)}")
        st.stop()
elif dataset_type == "MBPP":
    # Create MBPP dataset instance
    mbpp_dataset = MBPPDataset()
    
    # Dataset selection
    split = st.selectbox(
        "Dataset Selection",
        ["train", "test", "validation"],
        index=1  # Set test as default
    )
    
    try:
        # Load data
        data = mbpp_dataset.get_split_data(split)
        test_cases, num_samples = process_dataset(data, "MBPP")
        
        # Display dataset information
        st.info(f"MBPP {split} dataset: {len(data):,} examples")
    except Exception as e:
        st.error(f"MBPP dataset loading error: {str(e)}")
        st.stop()
elif dataset_type == "BBH":
    # Use already created BBHDataset instance
    try:
        # Add "All Categories" option to category selection UI
        bbh_categories = ["All Categories"] + bbh_dataset.get_all_categories()
        selected_category = st.selectbox(
            "Select BBH Category",
            bbh_categories,
            index=0,
            key="bbh_category_selectbox"
        )
        if selected_category == "All Categories":
            # Load full dataset
            all_data_dict = bbh_dataset.get_all_data()
            # {"test": [...]} format, so combine into list
            data = []
            for split_data in all_data_dict.values():
                data.extend(split_data)
            st.info(f"BBH full dataset: {len(data):,} examples")
        else:
            # Load data by category
            data = bbh_dataset.get_category_data(selected_category)
            st.info(f"BBH {selected_category} category dataset: {len(data):,} examples")
        test_cases, num_samples = process_dataset(data, "BBH")
    except Exception as e:
        st.error(f"BBH dataset loading error: {str(e)}")
        st.stop()
elif dataset_type == "TruthfulQA":
    # Use already created TruthfulQADataset instance
    try:
        # Load data
        data = truthfulqa_dataset.get_split_data("test")
        test_cases, num_samples = process_dataset(data, "TruthfulQA")
        
        # Display dataset information
        st.info(f"TruthfulQA test dataset: {len(data):,} examples")
    except Exception as e:
        st.error(f"TruthfulQA dataset loading error: {str(e)}")
        st.stop()
elif dataset_type == "HellaSwag":
    try:
        # Dataset selection
        split = st.selectbox(
            "Dataset Selection",
            ["validation", "train"],
            index=0
        )
        
        # Load data
        data = hellaswag_dataset.get_split_data(split)
        test_cases, num_samples = process_dataset(data, "HellaSwag")
        
        # Display dataset information
        st.info(f"HellaSwag {split} dataset: {len(data):,} examples")
    except Exception as e:
        st.error(f"HellaSwag dataset loading error: {str(e)}")
        st.stop()
elif dataset_type == "HumanEval":
    try:
        # HumanEval only has test split
        data = humaneval_dataset.get_split_data("test")
        test_cases, num_samples = process_dataset(data, "HumanEval")
        
        # Display dataset information
        st.info(f"HumanEval test dataset: {len(data):,} examples")
    except Exception as e:
        st.error(f"HumanEval dataset loading error: {str(e)}")
        st.stop()
elif dataset_type == "Samsum":
    samsum_dataset = SamsumDataset()
    split = st.selectbox(
        "Dataset Selection",
        ["train", "validation", "test"],
        index=0
    )
    try:
        data = samsum_dataset.get_split_data(split)
        test_cases, num_samples = process_dataset(data, "Samsum")
        st.info(f"Samsum {split} dataset: {len(data):,} examples")
    except Exception as e:
        st.error(f"Samsum dataset loading error: {str(e)}")
        st.stop()
elif dataset_type == "MeetingBank":
    meetingbank_dataset = MeetingBankDataset()
    split = st.selectbox(
        "Dataset Selection",
        ["validation", "test"],
        index=0
    )
    try:
        data = meetingbank_dataset.get_split_data(split)
        test_cases, num_samples = process_dataset(data, "MeetingBank")
        st.info(f"MeetingBank {split} dataset: {len(data):,} examples")
    except Exception as e:
        st.error(f"MeetingBank dataset loading error: {str(e)}")
        st.stop()
elif dataset_type in ["MMLU", "MMLU Pro"]:
    # Select appropriate dataset instance and subject list based on selected dataset
    if dataset_type == "MMLU":
        dataset = mmlu_dataset
        dataset_name = "MMLU"
    else:  # MMLU Pro
        dataset = mmlu_pro_dataset
        dataset_name = "MMLU Pro"
    # Add 'All Subjects' option to dataset selection
    subject_options = ["All Subjects"] + dataset.subjects
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
        if subject == "All Subjects":
            # Load data from all subjects
            all_subjects_data = dataset.get_all_subjects_data()
            # Combine data from all subjects into one list
            data = []
            for subject_data in all_subjects_data.values():
                data.extend(subject_data[split])
        else:
            # Load data from specific subject
            subject_data = dataset.get_subject_data(subject)
            data = subject_data[split]
        test_cases, num_samples = process_dataset(data, dataset_type)
    except Exception as e:
        st.error(f"{dataset_name} dataset loading error: {str(e)}")
        st.stop()

class SessionState:
    """
    Class to manage session state for Streamlit app
    """
    @staticmethod
    def init_state():
        """Initialize session state."""
        st.session_state.all_iteration_results = []
        st.session_state.current_iteration = 0
        st.session_state.show_results = False
        st.session_state.tuning_complete = False
        st.session_state.display_container = st.empty()
    
    @staticmethod
    def reset():
        """Reset state."""
        st.session_state.all_iteration_results = []
        st.session_state.current_iteration = 0
        st.session_state.show_results = False
        st.session_state.tuning_complete = False
    
    @staticmethod
    def update_results(result):
        """Add new result."""
        # Add logging
        logging.info(f"Updating results for iteration {result.iteration}")
        
        # Check if result already exists
        if not hasattr(st.session_state, 'all_iteration_results'):
            st.session_state.all_iteration_results = []
        
        # Update if same iteration result exists, otherwise add new
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
        """Return all currently saved results."""
        if not hasattr(st.session_state, 'all_iteration_results'):
            st.session_state.all_iteration_results = []
        return st.session_state.all_iteration_results
    
    @staticmethod
    def get_current_iteration():
        """Return currently selected iteration."""
        if not hasattr(st.session_state, 'current_iteration'):
            st.session_state.current_iteration = 0
        return st.session_state.current_iteration
    
    @staticmethod
    def set_current_iteration(iteration):
        """Set current iteration."""
        st.session_state.current_iteration = iteration

class ResultsDisplay:
    """
    Class responsible for displaying results
    """
    def __init__(self):
        SessionState.init_state()
        # Initialize main container
        if 'main_container' not in st.session_state:
            st.session_state.main_container = st.empty()
    
    def display_metrics(self, results, container):
        """Display performance metrics."""
        if not results:
            return
        
        # Prepare graph data
        x_values = [result.iteration for result in results]
        avg_scores = [result.avg_score for result in results]
        best_sample_scores = [result.best_sample_score for result in results]
        std_devs = [result.std_dev for result in results]
        top3_scores = [result.top3_avg_score for result in results]
        
        
        # Calculate average scores by category
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
            iteration_category_weights = {category: [] for category in category_scores.keys()}  # Weights for each iteration
            
            for test_case in result.test_case_results:
                if test_case.evaluation_details and 'category_scores' in test_case.evaluation_details:
                    for category, details in test_case.evaluation_details['category_scores'].items():
                        if category in iteration_category_scores:
                            iteration_category_scores[category].append(details['score'])
                            iteration_category_weights[category].append(details.get('weight', 0.5))  # Add weight
            
            # Add average score and weight for each category
            for category in category_scores:
                scores = iteration_category_scores[category]
                weights = iteration_category_weights[category]
                avg_score = np.mean(scores) if scores else 0
                avg_weight = np.mean(weights) if weights else 0.5
                category_scores[category].append(avg_score)
        
        # Create integrated graph
        fig = go.Figure()
        
        # Add category scores as bar graph
        for category in category_scores:
            fig.add_trace(go.Bar(
                x=x_values,
                y=category_scores[category],
                name=category,
                visible=True
            ))
        
        # Main performance indicator traces
        fig.add_trace(go.Scatter(
            x=x_values,
            y=avg_scores,
            name='Average Score',
            mode='lines+markers',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=x_values,
            y=std_devs,
            name='Standard Deviation',
            mode='lines+markers',
            line=dict(color='purple', width=2, dash='dot')
        ))
        
        fig.add_trace(go.Scatter(
            x=x_values,
            y=best_sample_scores,
            name='Best Individual Score',
            mode='lines+markers',
            line=dict(color='green', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=x_values,
            y=top3_scores,
            name='Top3 Average Score',
            mode='lines+markers',
            line=dict(color='red', width=2)
        ))
        
        # Set graph layout
        fig.update_layout(
            title='Integrated Performance Metrics and Category Analysis',
            xaxis_title='Iteration',
            yaxis_title='Score',
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
        
        # Display graph
        container.plotly_chart(fig, use_container_width=True)
    
    def display_iteration_details(self, results, container):
        """Display iteration details."""
        if not results:
            container.info("No results yet.")
            return
        
        # Iteration selection
        total_iterations = len(results)
        if total_iterations > 0:
            # Iteration selection UI
            current_iteration = SessionState.get_current_iteration()
            
            # Create tabs for iteration selection
            tabs = container.tabs([f"Iteration {i+1}" for i in range(total_iterations)])
            selected_iteration = current_iteration
            
            with tabs[selected_iteration]:
                iteration_result = results[selected_iteration]
                
                # Display average score and standard deviation
                col1, col2, col3 = st.columns(3)
                col1.metric("Average Score", f"{iteration_result.avg_score:.2f}")
                col2.metric("Standard Deviation", f"{iteration_result.std_dev:.2f}")
                col3.metric("Top 3 Average", f"{iteration_result.top3_avg_score:.2f}")
                
                # Add Task Type and Description expander
                with st.expander(f"Task Type ({iteration_result.task_type})", expanded=False):
                    st.markdown("### Task Description")
                    st.code(iteration_result.task_description, language="text")
                
                # Add current prompt expander
                with st.expander("View Current Prompt", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### System Prompt")
                        st.code(iteration_result.system_prompt, language="text")
                    with col2:
                        st.markdown("### User Prompt")
                        st.code(iteration_result.user_prompt, language="text")
                
                # Add weight score expander
                with st.expander("View Current Weight Scores", expanded=False):
                    # Collect weight data
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
                        # Calculate average weight by category
                        df = pd.DataFrame(weight_data)
                        avg_weights = df.groupby('Category')['Weight'].mean().round(3)
                        avg_weights = avg_weights.reset_index()
                        avg_weights.columns = ['Category', 'Average Weight']
                        
                        # Style dataframe
                        def highlight_weights(val):
                            color = f'background-color: rgba(255, 99, 71, {val})'
                            return color
                        
                        # Display styled dataframe
                        st.write("Category Weights:")
                        styled_df = avg_weights.style.apply(lambda x: [highlight_weights(v) for v in x], subset=['Average Weight'])
                        st.dataframe(styled_df, use_container_width=True)
                
                # Convert output results to dataframe
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
                    
                    # Add category scores and feedback
                    if test_case.evaluation_details and 'category_scores' in test_case.evaluation_details:
                        for category, details in test_case.evaluation_details['category_scores'].items():
                            row[f"{category} Score"] = f"{details['score']:.2f}"
                            row[f"{category} Weight"] = f"{details.get('weight', 1.0):.2f}"  # Add weight display
                            row[f"{category} State"] = details['current_state']
                            row[f"{category} Action"] = details['improvement_action']
                    
                    outputs_data.append(row)
                
                # Create dataframe
                df = pd.DataFrame(outputs_data)
                
                # Dataframe styling function
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
                
                # Display styled dataframe
                st.dataframe(
                    df.style.apply(highlight_rows, axis=None),
                    use_container_width=True,
                    height=400
                )
                
                # Add meta prompt expander
                if iteration_result.meta_prompt:
                    with st.expander("View Meta Prompt Results", expanded=False):
                        st.code(iteration_result.meta_prompt, language="text")
            
            # Save currently selected iteration
            SessionState.set_current_iteration(selected_iteration)
    
    def update(self):
        """Update result display."""
        results = SessionState.get_results()
        if st.session_state.show_results and results:
            # Clear existing container and create new one
            with st.session_state.main_container.container():
                st.empty()  # Clear existing content
                
                # Create new container for metrics and details
                metrics_container = st.container()
                details_container = st.container()
                
                # Display metrics and details
                self.display_metrics(results, metrics_container)
                self.display_iteration_details(results, details_container)

def run_tuning_process():
    """Run prompt tuning process and visualize results."""
    # Initialize UI state
    SessionState.init_state()
    results_display = ResultsDisplay()
    
    with st.spinner('Tuning prompts...'):
        def iteration_callback(result):
            logging.info(f"Iteration callback called for iteration {result.iteration}")
            SessionState.update_results(result)
            logging.info("Results updated, updating display...")
            results_display.update()
            logging.info("Display updated")
        
        # Set iteration_callback
        tuner.iteration_callback = iteration_callback
        
        # Execute prompt tuning
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
        
        # Final results
        results = SessionState.get_results()
        if results:
            st.success("Prompt tuning completed!")
            logging.info(f"Final results count: {len(results)}")
            
            # Display cost summary
            st.header("💰 Cost and Usage Summary")
            cost_summary = tuner.get_cost_summary()
            
            # Display overall cost information as metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Cost", f"${cost_summary['total_cost']:.4f}")
            with col2:
                st.metric("Total Tokens", f"{cost_summary['total_tokens']:,}")
            with col3:
                st.metric("Total Time", f"{cost_summary['total_duration']:.1f} seconds")
            with col4:
                st.metric("Total Calls", f"{cost_summary['total_calls']}")
            
            # Model-wise detailed cost information
            with st.expander("Model-wise Detailed Cost Information", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("🤖 Model Calls")
                    model_stats = cost_summary['model_stats']
                    st.write(f"Total Calls: {model_stats['total_calls']}")
                    st.write(f"Input Tokens: {model_stats['total_input_tokens']:,}")
                    st.write(f"Output Tokens: {model_stats['total_output_tokens']:,}")
                    st.write(f"Total Tokens: {model_stats['total_tokens']:,}")
                    st.write(f"Cost: ${model_stats['total_cost']:.4f}")
                    st.write(f"Time: {model_stats['total_duration']:.2f} seconds")
                
                with col2:
                    st.subheader("�� Evaluator Calls")
                    eval_stats = cost_summary['evaluator_stats']
                    st.write(f"Total Calls: {eval_stats['total_calls']}")
                    st.write(f"Input Tokens: {eval_stats['total_input_tokens']:,}")
                    st.write(f"Output Tokens: {eval_stats['total_output_tokens']:,}")
                    st.write(f"Total Tokens: {eval_stats['total_tokens']:,}")
                    st.write(f"Cost: ${eval_stats['total_cost']:.4f}")
                    st.write(f"Time: {eval_stats['total_duration']:.2f} seconds")
                
                with col3:
                    st.subheader("🔧 Meta Prompt Generation")
                    meta_stats = cost_summary['meta_prompt_stats']
                    st.write(f"Total Calls: {meta_stats['total_calls']}")
                    st.write(f"Input Tokens: {meta_stats['total_input_tokens']:,}")
                    st.write(f"Output Tokens: {meta_stats['total_output_tokens']:,}")
                    st.write(f"Total Tokens: {meta_stats['total_tokens']:,}")
                    st.write(f"Cost: ${meta_stats['total_cost']:.4f}")
                    st.write(f"Time: {meta_stats['total_duration']:.2f} seconds")
            
            # Iteration-wise cost analysis
            iteration_breakdown = tuner.get_iteration_cost_breakdown()
            if iteration_breakdown:
                with st.expander("Iteration-wise Cost Analysis", expanded=False):
                    # Convert iteration-wise cost data to dataframe
                    breakdown_data = []
                    for iteration_key, data in iteration_breakdown.items():
                        breakdown_data.append({
                            'Iteration': iteration_key.replace('iteration_', ''),
                            'Model Cost': f"${data['model_cost']:.4f}",
                            'Evaluator Cost': f"${data['evaluator_cost']:.4f}",
                            'Meta Prompt Cost': f"${data['meta_prompt_cost']:.4f}",
                            'Total Cost': f"${data['total_cost']:.4f}",
                            'Model Calls': data['model_calls'],
                            'Evaluator Calls': data['evaluator_calls'],
                            'Meta Prompt Calls': data['meta_prompt_calls'],
                            'Total Calls': data['total_calls']
                        })
                    
                    if breakdown_data:
                        df_breakdown = pd.DataFrame(breakdown_data)
                        st.dataframe(df_breakdown, use_container_width=True)
            
            # Find prompt with highest average score from all results
            st.header("🏆 Best Prompt")
            best_result = max(results, key=lambda x: x.avg_score)
            st.write("Final Best Prompt:")
            col1, col2 = st.columns(2)
            with col1:
                st.write("System Prompt:")
                st.code(best_result.system_prompt)
            with col2:
                st.write("User Prompt:")
                st.code(best_result.user_prompt)
            st.write(f"Final Result: Average Score {best_result.avg_score:.2f}, Best Average Score {best_result.best_avg_score:.2f}, Best Individual Score {best_result.best_sample_score:.2f}")
            
            # Download buttons
            st.header("📥 Download Results")
            col1, col2 = st.columns(2)
            
            with col1:
                # Full results (including cost information) CSV download
                try:
                    csv_data = tuner.save_results_to_csv()
                    st.download_button(
                        label="📊 Download Full Results (Cost Included) CSV",
                        data=csv_data,
                        file_name=f"prompt_tuning_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="download_full_csv",
                        help="Detailed results per test case and cost information"
                    )
                except Exception as e:
                    st.error(f"Error generating full results CSV file: {str(e)}")
            
            with col2:
                # Cost summary only CSV download
                try:
                    cost_csv_data = tuner.export_cost_summary_to_csv()
                    st.download_button(
                        label="💰 Download Cost Summary CSV",
                        data=cost_csv_data,
                        file_name=f"cost_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="download_cost_csv",
                        help="Model-wise, iteration-wise cost summary data"
                    )
                except Exception as e:
                    st.error(f"Error generating cost summary CSV file: {str(e)}")
            
            # Also output cost summary to console (for developers)
            tuner.print_cost_summary()
        else:
            st.warning("No tuning results.")

# Start tuning button
if st.button("Start Prompt Tuning", type="primary"):
    # Initialize session state
    SessionState.reset()
    
    # Check API keys
    required_keys = {
        "solar": "SOLAR_API_KEY",
        "gpt4o": "OPENAI_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
        "local1": None,  # local1 model doesn't need API key
        "local2": None,   # local2 model doesn't need API key
        "solar_strawberry": "SOLAR_STRAWBERRY_API_KEY",  # Added
    }
    
    # Check API keys for used models
    used_models = set([model_name, evaluator_model])
    missing_keys = []
    
    for model in used_models:
        key = required_keys[model]
        if key and not os.getenv(key):  # Only check API key if key is not None
            missing_keys.append(f"{MODEL_INFO[model]['name']} ({key})")
    
    if missing_keys:
        st.error(f"The following API keys are required: {', '.join(missing_keys)}")
        st.info("Please set these keys in your .env file.")
    else:
        # Only set if meta prompt is entered
        if meta_system_prompt.strip() and meta_user_prompt.strip():
            tuner.set_meta_prompt(meta_system_prompt, meta_user_prompt)
        
        # Set progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def progress_callback(iteration, test_case_index):
            # Current iteration progress (starting from 0)
            iteration_progress = (iteration - 1) / iterations
            # Current test case progress (starting from 0)
            test_case_progress = test_case_index / num_samples
            # Calculate total progress
            progress = iteration_progress + (test_case_progress / iterations)
            progress_bar.progress(progress)
            status_text.text(f"Iteration {iteration}/{iterations}, Test Case {test_case_index}/{num_samples}")
        
        tuner.progress_callback = progress_callback
        
        # Execute prompt tuning
        run_tuning_process() 