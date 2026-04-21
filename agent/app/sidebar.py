from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import streamlit as st


@dataclass(frozen=True)
class AppSettings:
    iterations: int
    use_meta_prompt: bool
    evaluation_threshold: float
    use_threshold: bool
    score_threshold: float
    model_name: str
    tuning_model_version: Optional[str]
    meta_prompt_model: str
    meta_model_version: Optional[str]
    evaluator_model: str
    evaluator_model_version: Optional[str]


def _default_index(options: list[str], preferred: str | None) -> int:
    if preferred and preferred in options:
        return options.index(preferred)
    return 0


def _format_model(model_info: Dict[str, Dict[str, str]], model_name: str) -> str:
    info = model_info[model_name]
    return f"{info['name']} ({info['default_version']})"


def _render_model_settings(
    title: str,
    model_info: Dict[str, Dict[str, str]],
    key_prefix: str,
    default_model: str | None = None,
) -> Tuple[str, Optional[str]]:
    options = list(model_info.keys())

    with st.expander(title, expanded=True):
        model_name = st.selectbox(
            "Model Selection",
            options=options,
            format_func=lambda name: _format_model(model_info, name),
            index=_default_index(options, default_model),
            key=f"{key_prefix}_model_name",
            help="Select the model for this stage.",
        )
        st.caption(model_info[model_name]["description"])

        use_custom_version = st.toggle(
            "Use Custom Version",
            value=False,
            key=f"{key_prefix}_custom_version_enabled",
            help="Use a custom version instead of the model default.",
        )
        if not use_custom_version:
            return model_name, None

        model_version = st.text_input(
            "Model Version",
            value=model_info[model_name]["default_version"],
            key=f"{key_prefix}_model_version",
            help="Enter the model version for this stage.",
        )
        return model_name, model_version


def render_sidebar(model_info: Dict[str, Dict[str, str]]) -> AppSettings:
    default_local_model = "local1" if "local1" in model_info else None

    with st.sidebar:
        st.header("Tuning Settings")

        with st.expander("Iteration Settings", expanded=True):
            iterations = st.slider(
                "Number of Iterations",
                min_value=1,
                max_value=100,
                value=3,
                help="Set the number of prompt tuning iterations.",
            )

        with st.expander("Prompt Improvement Settings", expanded=True):
            use_meta_prompt = st.toggle(
                "Use Prompt Improvement",
                value=True,
                help="Use a meta prompt to improve prompts between iterations.",
            )
            evaluation_threshold = st.slider(
                "Evaluation Prompt Score Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.8,
                step=0.1,
                disabled=not use_meta_prompt,
                help="Improve the prompt when the score is below this threshold.",
            )
            use_threshold = st.toggle(
                "Apply Average Score Threshold",
                value=True,
                disabled=not use_meta_prompt,
                help="Stop early when the average score reaches the threshold.",
            )
            score_threshold = st.slider(
                "Average Score Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.9,
                step=0.05,
                disabled=not (use_threshold and use_meta_prompt),
                help="Average score threshold for early stopping.",
            )

        st.divider()

        model_name, tuning_model_version = _render_model_settings(
            "Tuning Model Settings",
            model_info,
            key_prefix="tuning",
            default_model=default_local_model,
        )
        meta_prompt_model, meta_model_version = _render_model_settings(
            "Meta Prompt Model Settings",
            model_info,
            key_prefix="meta",
        )
        evaluator_model, evaluator_model_version = _render_model_settings(
            "Evaluation Model Settings",
            model_info,
            key_prefix="evaluator",
            default_model=default_local_model,
        )

    return AppSettings(
        iterations=iterations,
        use_meta_prompt=use_meta_prompt,
        evaluation_threshold=evaluation_threshold,
        use_threshold=use_threshold,
        score_threshold=score_threshold,
        model_name=model_name,
        tuning_model_version=tuning_model_version,
        meta_prompt_model=meta_prompt_model,
        meta_model_version=meta_model_version,
        evaluator_model=evaluator_model,
        evaluator_model_version=evaluator_model_version,
    )
