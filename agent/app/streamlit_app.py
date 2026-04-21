import logging

import streamlit as st

st.set_page_config(page_title="Prompt Auto Tuning Agent", layout="wide")

from agent.app.config import load_prompt_templates, missing_api_keys, setup_environment
from agent.app.datasets import render_dataset_selection
from agent.app.prompt_controls import PromptInputs, render_prompt_controls
from agent.app.results_display import ResultsDisplay, SessionState, display_final_results
from agent.app.sidebar import AppSettings, render_sidebar
from agent.common.api_client import Model
from agent.core.prompt_tuner import PromptTuner


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_tuner(settings: AppSettings) -> PromptTuner:
    return PromptTuner(
        model_name=settings.model_name,
        evaluator_model_name=settings.evaluator_model,
        meta_prompt_model_name=settings.meta_prompt_model,
        model_version=settings.tuning_model_version,
        evaluator_model_version=settings.evaluator_model_version,
        meta_prompt_model_version=settings.meta_model_version,
    )


def configure_tuner(tuner: PromptTuner, prompts: PromptInputs) -> None:
    tuner.set_initial_prompt(prompts.system_prompt, prompts.user_prompt)
    tuner.set_evaluation_prompt(
        prompts.evaluation_system_prompt,
        prompts.evaluation_user_prompt,
    )
    if prompts.meta_system_prompt.strip() and prompts.meta_user_prompt.strip():
        tuner.set_meta_prompt(prompts.meta_system_prompt, prompts.meta_user_prompt)


def validate_api_keys(settings: AppSettings) -> bool:
    used_models = {settings.model_name, settings.evaluator_model}
    if settings.use_meta_prompt:
        used_models.add(settings.meta_prompt_model)

    missing_keys = missing_api_keys(used_models)
    if not missing_keys:
        return True

    st.error(f"The following API keys are required: {', '.join(missing_keys)}")
    st.info("Please set these keys in your .env file.")
    return False


def run_tuning_process(
    tuner: PromptTuner,
    prompts: PromptInputs,
    settings: AppSettings,
    test_cases: list[dict],
    num_samples: int,
) -> None:
    SessionState.init_state()
    results_display = ResultsDisplay()

    with st.spinner("Tuning prompts..."):
        def iteration_callback(result):
            logger.info("Iteration callback called for iteration %s", result.iteration)
            SessionState.update_results(result)
            results_display.update()

        tuner.iteration_callback = iteration_callback
        tuner.tune_prompt(
            initial_system_prompt=prompts.system_prompt,
            initial_user_prompt=prompts.user_prompt,
            initial_test_cases=test_cases,
            num_iterations=settings.iterations,
            score_threshold=(
                settings.score_threshold
                if settings.use_meta_prompt and settings.use_threshold
                else None
            ),
            evaluation_score_threshold=settings.evaluation_threshold,
            use_meta_prompt=settings.use_meta_prompt,
            num_samples=num_samples,
        )

        st.session_state.tuning_complete = True
        logger.info("Tuning process completed")
        display_final_results(tuner, SessionState.get_results())


def render_app() -> None:
    setup_environment()
    st.title("Prompt Tuning Dashboard")

    model_info = Model.get_all_model_info()
    settings = render_sidebar(model_info)
    tuner = build_tuner(settings)
    prompts = render_prompt_controls(load_prompt_templates(), tuner)
    test_cases, num_samples = render_dataset_selection()

    if not st.button("Start Prompt Tuning", type="primary"):
        return

    SessionState.reset()
    configure_tuner(tuner, prompts)

    if not validate_api_keys(settings):
        return

    progress_bar = st.progress(0)
    status_text = st.empty()

    def progress_callback(iteration: int, test_case_index: int) -> None:
        iteration_progress = (iteration - 1) / settings.iterations
        test_case_progress = test_case_index / max(num_samples, 1)
        progress = min(iteration_progress + (test_case_progress / settings.iterations), 1.0)
        progress_bar.progress(progress)
        status_text.text(
            f"Iteration {iteration}/{settings.iterations}, "
            f"Test Case {test_case_index}/{num_samples}"
        )

    tuner.progress_callback = progress_callback
    run_tuning_process(tuner, prompts, settings, test_cases, num_samples)


render_app()
