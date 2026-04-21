from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import streamlit as st


@dataclass(frozen=True)
class PromptInputs:
    system_prompt: str
    user_prompt: str
    meta_system_prompt: str
    meta_user_prompt: str
    evaluation_system_prompt: str
    evaluation_user_prompt: str


def render_prompt_controls(templates: Dict[str, str], tuner) -> PromptInputs:
    with st.expander("Initial Prompt Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            system_prompt = st.text_area(
                "System Prompt",
                value=templates["system"],
                height=100,
                help="Enter the initial system prompt to start tuning.",
            )
        with col2:
            user_prompt = st.text_area(
                "User Prompt",
                value=templates["user"],
                height=100,
                help="Enter the initial user prompt to start tuning.",
            )

        if st.button("Update Initial Prompt", key="initial_prompt_update"):
            tuner.set_initial_prompt(system_prompt, user_prompt)
            st.success("Initial prompt updated.")

    with st.expander("Meta Prompt Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            meta_system_prompt = st.text_area(
                "Meta System Prompt",
                value=templates["meta_system"],
                height=300,
                help="Enter the system prompt for the prompt improvement model.",
            )
        with col2:
            meta_user_prompt = st.text_area(
                "Meta User Prompt",
                value=templates["meta_user"],
                height=300,
                help="Enter the user prompt template for prompt improvement.",
            )

        if st.button("Update Meta Prompt", key="meta_prompt_update"):
            tuner.set_meta_prompt(meta_system_prompt, meta_user_prompt)
            st.success("Meta prompt updated.")

    with st.expander("Evaluation Prompt Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            evaluation_system_prompt = st.text_area(
                "Evaluation System Prompt",
                value=templates["evaluation_system"],
                height=200,
                help="Set the system prompt for the evaluation model.",
            )
        with col2:
            evaluation_user_prompt = st.text_area(
                "Evaluation User Prompt",
                value=templates["evaluation_user"],
                height=200,
                help="Set the user prompt. It must include {question}, {output}, {expected}.",
            )

        if st.button("Update Evaluation Prompt", key="eval_prompt_update"):
            tuner.set_evaluation_prompt(evaluation_system_prompt, evaluation_user_prompt)
            st.success("Evaluation prompt updated.")

    return PromptInputs(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        meta_system_prompt=meta_system_prompt,
        meta_user_prompt=meta_user_prompt,
        evaluation_system_prompt=evaluation_system_prompt,
        evaluation_user_prompt=evaluation_user_prompt,
    )
