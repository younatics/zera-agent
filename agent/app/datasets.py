from __future__ import annotations

import csv
from typing import Any, List, Tuple

import pandas as pd
import streamlit as st

from agent.app.app_logic import build_test_cases


DATASET_TYPES = [
    "CSV",
    "MMLU",
    "MMLU Pro",
    "CNN",
    "GSM8K",
    "MBPP",
    "BBH",
    "TruthfulQA",
    "HellaSwag",
    "HumanEval",
    "Samsum",
    "MeetingBank",
    "XSum",
]


@st.cache_resource
def get_mmlu_dataset():
    from agent.dataset.mmlu_dataset import MMLUDataset

    return MMLUDataset()


@st.cache_resource
def get_mmlu_pro_dataset():
    from agent.dataset.mmlu_pro_dataset import MMLUProDataset

    return MMLUProDataset()


@st.cache_resource
def get_cnn_dataset():
    from agent.dataset.cnn_dataset import CNNDataset

    return CNNDataset()


@st.cache_resource
def get_gsm8k_dataset():
    from agent.dataset.gsm8k_dataset import GSM8KDataset

    return GSM8KDataset()


@st.cache_resource
def get_mbpp_dataset():
    from agent.dataset.mbpp_dataset import MBPPDataset

    return MBPPDataset()


@st.cache_resource
def get_bbh_dataset():
    from agent.dataset.bbh_dataset import BBHDataset

    return BBHDataset()


@st.cache_resource
def get_truthfulqa_dataset():
    from agent.dataset.truthfulqa_dataset import TruthfulQADataset

    return TruthfulQADataset()


@st.cache_resource
def get_hellaswag_dataset():
    from agent.dataset.hellaswag_dataset import HellaSwagDataset

    return HellaSwagDataset()


@st.cache_resource
def get_humaneval_dataset():
    from agent.dataset.humaneval_dataset import HumanEvalDataset

    return HumanEvalDataset()


@st.cache_resource
def get_samsum_dataset():
    from agent.dataset.samsum_dataset import SamsumDataset

    return SamsumDataset()


@st.cache_resource
def get_meetingbank_dataset():
    from agent.dataset.meetingbank_dataset import MeetingBankDataset

    return MeetingBankDataset()


@st.cache_resource
def get_xsum_dataset():
    from agent.dataset.xsum_dataset import XSumDataset

    return XSumDataset()


def process_dataset(data: Any, dataset_type: str) -> Tuple[List[dict], int]:
    total_examples = len(data)
    if total_examples == 0:
        st.error("Dataset is empty.")
        st.stop()

    st.write(f"Total examples: {total_examples}")
    num_samples = st.slider(
        "Number of random samples to evaluate per iteration",
        min_value=1,
        max_value=min(100, total_examples),
        value=min(5, total_examples),
        help="Select the number of random samples to evaluate per iteration.",
    )

    try:
        test_cases, display_data = build_test_cases(data, dataset_type)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    st.write("Dataset Content:")
    st.dataframe(pd.DataFrame(display_data), use_container_width=True)
    return test_cases, num_samples


def _load_csv_dataset() -> Tuple[List[dict], int]:
    csv_file = st.file_uploader("Upload CSV file", type=["csv"])
    if csv_file is None:
        st.info("Please upload a CSV file or select another dataset.")
        st.stop()

    try:
        dataframe = pd.read_csv(
            csv_file,
            encoding="utf-8",
            on_bad_lines="skip",
            quoting=csv.QUOTE_ALL,
            escapechar="\\",
        )
    except Exception as exc:
        st.error(f"CSV file loading error: {exc}")
        st.info("Please check whether the CSV file has the expected columns and encoding.")
        st.stop()

    if dataframe.empty:
        st.error("CSV file is empty. Please upload a CSV file with data.")
        st.stop()

    return process_dataset(dataframe, "CSV")


def _load_cnn_dataset() -> Tuple[List[dict], int]:
    dataset = get_cnn_dataset()
    split = st.selectbox("Dataset Selection", ["train", "validation", "test"], index=0)
    total_chunks = dataset.get_num_chunks(split)

    if total_chunks == 0:
        st.error(f"No chunk files found for {split} dataset.")
        st.stop()

    use_all_chunks = st.toggle(
        "Use All Chunks",
        value=False,
        help="Load data from all chunks. This may take a long time.",
    )

    if use_all_chunks:
        data = dataset.load_all_data(split)
        st.info(f"All chunks loaded ({len(data):,} examples)")
    else:
        st.write(f"Total {total_chunks} chunks available.")
        chunk_index = st.number_input(
            "Select Chunk",
            min_value=0,
            max_value=total_chunks - 1,
            value=0,
            help="Select the index of the chunk to process.",
        )
        data = dataset.load_data(split, int(chunk_index))
        st.info(f"Selected chunk: {chunk_index} ({len(data):,} examples)")

    return process_dataset(data, "CNN")


def _load_bbh_dataset() -> Tuple[List[dict], int]:
    dataset = get_bbh_dataset()
    categories = ["All Categories"] + dataset.get_all_categories()
    selected_category = st.selectbox(
        "Select BBH Category",
        categories,
        index=0,
        key="bbh_category_selectbox",
    )

    if selected_category == "All Categories":
        all_data = dataset.get_all_data()
        data = [item for split_data in all_data.values() for item in split_data]
        st.info(f"BBH full dataset: {len(data):,} examples")
    else:
        data = dataset.get_category_data(selected_category)
        st.info(f"BBH {selected_category} category dataset: {len(data):,} examples")

    return process_dataset(data, "BBH")


def _load_mmlu_dataset(dataset_type: str) -> Tuple[List[dict], int]:
    if dataset_type == "MMLU":
        dataset = get_mmlu_dataset()
        dataset_name = "MMLU"
    else:
        dataset = get_mmlu_pro_dataset()
        dataset_name = "MMLU Pro"

    subject = st.selectbox(
        f"Select {dataset_name} Subject",
        ["All Subjects"] + dataset.subjects,
        index=0,
    )
    split = st.selectbox("Select Data Split", ["validation", "test"], index=0)

    if subject == "All Subjects":
        all_subjects_data = dataset.get_all_subjects_data()
        data = [
            item
            for subject_data in all_subjects_data.values()
            for item in subject_data[split]
        ]
    else:
        data = dataset.get_subject_data(subject)[split]

    return process_dataset(data, dataset_type)


def _load_split_dataset(dataset_type: str) -> Tuple[List[dict], int]:
    loaders = {
        "GSM8K": (get_gsm8k_dataset, ["train", "test"], 0),
        "MBPP": (get_mbpp_dataset, ["train", "test", "validation"], 1),
        "HellaSwag": (get_hellaswag_dataset, ["validation", "train"], 0),
        "Samsum": (get_samsum_dataset, ["train", "validation", "test"], 0),
        "MeetingBank": (get_meetingbank_dataset, ["validation", "test"], 0),
        "XSum": (get_xsum_dataset, ["train", "validation", "test"], 0),
    }
    get_dataset, splits, default_index = loaders[dataset_type]
    split = st.selectbox("Dataset Selection", splits, index=default_index)
    dataset = get_dataset()
    data = dataset.get_split_data(split)
    st.info(f"{dataset_type} {split} dataset: {len(data):,} examples")
    return process_dataset(data, dataset_type)


def _load_fixed_test_dataset(dataset_type: str) -> Tuple[List[dict], int]:
    loaders = {
        "TruthfulQA": get_truthfulqa_dataset,
        "HumanEval": get_humaneval_dataset,
    }
    dataset = loaders[dataset_type]()
    data = dataset.get_split_data("test")
    st.info(f"{dataset_type} test dataset: {len(data):,} examples")
    return process_dataset(data, dataset_type)


def render_dataset_selection() -> Tuple[List[dict], int]:
    st.header("Dataset Selection")
    dataset_type = st.radio(
        "Select Dataset Type",
        DATASET_TYPES,
        horizontal=True,
    )

    try:
        if dataset_type == "CSV":
            return _load_csv_dataset()
        if dataset_type == "CNN":
            return _load_cnn_dataset()
        if dataset_type == "BBH":
            return _load_bbh_dataset()
        if dataset_type in {"MMLU", "MMLU Pro"}:
            return _load_mmlu_dataset(dataset_type)
        if dataset_type in {"TruthfulQA", "HumanEval"}:
            return _load_fixed_test_dataset(dataset_type)
        return _load_split_dataset(dataset_type)
    except Exception as exc:
        st.error(f"{dataset_type} dataset loading error: {exc}")
        st.stop()
