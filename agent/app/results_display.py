from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Iterable, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


logger = logging.getLogger(__name__)


class SessionState:
    """Manage Streamlit session state for live prompt tuning results."""

    @staticmethod
    def init_state() -> None:
        if "all_iteration_results" not in st.session_state:
            st.session_state.all_iteration_results = []
        if "current_iteration" not in st.session_state:
            st.session_state.current_iteration = 0
        if "show_results" not in st.session_state:
            st.session_state.show_results = False
        if "tuning_complete" not in st.session_state:
            st.session_state.tuning_complete = False

    @staticmethod
    def reset() -> None:
        st.session_state.all_iteration_results = []
        st.session_state.current_iteration = 0
        st.session_state.show_results = False
        st.session_state.tuning_complete = False

    @staticmethod
    def update_results(result) -> None:
        SessionState.init_state()
        results = st.session_state.all_iteration_results
        existing_index = next(
            (index for index, item in enumerate(results) if item.iteration == result.iteration),
            None,
        )

        if existing_index is None:
            results.append(result)
        else:
            results[existing_index] = result

        st.session_state.current_iteration = max(result.iteration - 1, 0)
        st.session_state.show_results = True
        logger.info("Stored result for iteration %s", result.iteration)

    @staticmethod
    def get_results() -> List:
        SessionState.init_state()
        return st.session_state.all_iteration_results

    @staticmethod
    def get_current_iteration() -> int:
        SessionState.init_state()
        return st.session_state.current_iteration

    @staticmethod
    def set_current_iteration(iteration: int) -> None:
        st.session_state.current_iteration = iteration


class ResultsDisplay:
    """Render prompt tuning metrics and iteration details."""

    def __init__(self) -> None:
        SessionState.init_state()
        if "main_container" not in st.session_state:
            st.session_state.main_container = st.empty()

    def display_metrics(self, results: Iterable, container) -> None:
        results = list(results)
        if not results:
            return

        x_values = [result.iteration for result in results]
        category_scores = _category_scores_by_iteration(results)

        fig = go.Figure()
        for category, scores in category_scores.items():
            fig.add_trace(go.Bar(x=x_values, y=scores, name=category, visible=True))

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=[result.avg_score for result in results],
                name="Average Score",
                mode="lines+markers",
                line=dict(color="blue", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=[result.std_dev for result in results],
                name="Standard Deviation",
                mode="lines+markers",
                line=dict(color="purple", width=2, dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=[result.best_sample_score for result in results],
                name="Best Individual Score",
                mode="lines+markers",
                line=dict(color="green", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=[result.top3_avg_score for result in results],
                name="Top 3 Average Score",
                mode="lines+markers",
                line=dict(color="red", width=2),
            )
        )

        fig.update_layout(
            title="Integrated Performance Metrics and Category Analysis",
            xaxis_title="Iteration",
            yaxis_title="Score",
            yaxis_range=[0, 1],
            xaxis=dict(
                tickmode="array",
                tickvals=x_values,
                ticktext=[f"Iteration {value}" for value in x_values],
            ),
            height=600,
            barmode="group",
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.05),
        )
        container.plotly_chart(fig, width="stretch")

    def display_iteration_details(self, results: Iterable, container) -> None:
        results = list(results)
        if not results:
            container.info("No results yet.")
            return

        current_iteration = min(SessionState.get_current_iteration(), len(results) - 1)
        selected_iteration = container.selectbox(
            "Iteration",
            options=list(range(len(results))),
            index=current_iteration,
            format_func=lambda index: f"Iteration {results[index].iteration}",
            key=f"iteration_select_{len(results)}",
        )
        SessionState.set_current_iteration(selected_iteration)
        with container.container():
            _render_iteration(results[selected_iteration])

    def update(self) -> None:
        results = SessionState.get_results()
        if not (st.session_state.show_results and results):
            return

        with st.session_state.main_container.container():
            st.empty()
            self.display_metrics(results, st.container())
            self.display_iteration_details(results, st.container())


def _category_scores_by_iteration(results: Iterable) -> dict:
    categories = {
        "meaning_accuracy": [],
        "completeness": [],
        "expression_style": [],
        "faithfulness": [],
        "conciseness": [],
        "correctness": [],
        "structural_alignment": [],
        "reasoning_quality": [],
    }

    for result in results:
        iteration_scores = {category: [] for category in categories}
        for test_case in result.test_case_results:
            details = test_case.evaluation_details or {}
            for category, values in details.get("category_scores", {}).items():
                if category in iteration_scores:
                    iteration_scores[category].append(values.get("score", 0))

        for category in categories:
            scores = iteration_scores[category]
            categories[category].append(float(np.mean(scores)) if scores else 0.0)

    return categories


def _render_iteration(iteration_result) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Score", f"{iteration_result.avg_score:.2f}")
    col2.metric("Standard Deviation", f"{iteration_result.std_dev:.2f}")
    col3.metric("Top 3 Average", f"{iteration_result.top3_avg_score:.2f}")

    with st.expander(f"Task Type ({iteration_result.task_type})", expanded=False):
        st.markdown("### Task Description")
        st.code(iteration_result.task_description, language="text")

    with st.expander("View Current Prompt", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### System Prompt")
            st.code(iteration_result.system_prompt, language="text")
        with col2:
            st.markdown("### User Prompt")
            st.code(iteration_result.user_prompt, language="text")

    _render_weight_summary(iteration_result)
    _render_test_case_table(iteration_result)

    if iteration_result.meta_prompt:
        with st.expander("View Meta Prompt Results", expanded=False):
            st.code(iteration_result.meta_prompt, language="text")


def _render_weight_summary(iteration_result) -> None:
    weight_data = []
    for test_case in iteration_result.test_case_results:
        details = test_case.evaluation_details or {}
        for category, values in details.get("category_scores", {}).items():
            weight_data.append(
                {
                    "Category": category,
                    "Weight": values.get("weight", 0.5),
                }
            )

    if not weight_data:
        return

    with st.expander("View Current Weight Scores", expanded=False):
        dataframe = pd.DataFrame(weight_data)
        avg_weights = dataframe.groupby("Category")["Weight"].mean().round(3).reset_index()
        avg_weights.columns = ["Category", "Average Weight"]
        st.write("Category Weights:")
        st.dataframe(avg_weights, width="stretch")


def _render_test_case_table(iteration_result) -> None:
    rows = []
    for index, test_case in enumerate(iteration_result.test_case_results, start=1):
        row = {
            "Test Case": index,
            "Score": f"{test_case.score:.2f}",
            "Question": test_case.question,
            "Expected": test_case.expected_output,
            "Actual": test_case.actual_output,
            "Evaluation Details": json.dumps(
                test_case.evaluation_details,
                ensure_ascii=False,
                indent=2,
            ),
        }

        details = test_case.evaluation_details or {}
        for category, values in details.get("category_scores", {}).items():
            row[f"{category} Score"] = f"{values.get('score', 0):.2f}"
            row[f"{category} Weight"] = f"{values.get('weight', 1.0):.2f}"
            row[f"{category} State"] = values.get("current_state", "")
            row[f"{category} Action"] = values.get("improvement_action", "")

        rows.append(row)

    dataframe = pd.DataFrame(rows)
    st.dataframe(
        dataframe.style.apply(_highlight_score_extremes, axis=None),
        width="stretch",
        height=400,
    )


def _highlight_score_extremes(dataframe: pd.DataFrame) -> pd.DataFrame:
    scores = dataframe["Score"].astype(float)
    colors = pd.DataFrame("", index=dataframe.index, columns=dataframe.columns)
    colors.loc[scores == scores.max()] = "background-color: #90EE90"
    colors.loc[scores == scores.min()] = "background-color: #FFB6C6"
    return colors


def display_final_results(tuner, results: Iterable) -> None:
    results = list(results)
    if not results:
        st.warning("No tuning results.")
        return

    st.success("Prompt tuning completed!")
    _render_cost_summary(tuner)
    _render_best_prompt(results)
    _render_downloads(tuner)
    tuner.print_cost_summary()


def _render_cost_summary(tuner) -> None:
    st.header("Cost and Usage Summary")
    cost_summary = tuner.get_cost_summary()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Cost", f"${cost_summary['total_cost']:.4f}")
    col2.metric("Total Tokens", f"{cost_summary['total_tokens']:,}")
    col3.metric("Total Time", f"{cost_summary['total_duration']:.1f} seconds")
    col4.metric("Total Calls", f"{cost_summary['total_calls']}")

    with st.expander("Model-wise Detailed Cost Information", expanded=False):
        col1, col2, col3 = st.columns(3)
        _render_usage_column(col1, "Model Calls", cost_summary["model_stats"])
        _render_usage_column(col2, "Evaluator Calls", cost_summary["evaluator_stats"])
        _render_usage_column(col3, "Meta Prompt Generation", cost_summary["meta_prompt_stats"])

    iteration_breakdown = tuner.get_iteration_cost_breakdown()
    if not iteration_breakdown:
        return

    with st.expander("Iteration-wise Cost Analysis", expanded=False):
        rows = []
        for iteration_key, data in iteration_breakdown.items():
            rows.append(
                {
                    "Iteration": iteration_key.replace("iteration_", ""),
                    "Model Cost": f"${data['model_cost']:.4f}",
                    "Evaluator Cost": f"${data['evaluator_cost']:.4f}",
                    "Meta Prompt Cost": f"${data['meta_prompt_cost']:.4f}",
                    "Total Cost": f"${data['total_cost']:.4f}",
                    "Model Calls": data["model_calls"],
                    "Evaluator Calls": data["evaluator_calls"],
                    "Meta Prompt Calls": data["meta_prompt_calls"],
                    "Total Calls": data["total_calls"],
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch")


def _render_usage_column(column, title: str, stats: dict) -> None:
    with column:
        st.subheader(title)
        st.write(f"Total Calls: {stats['total_calls']}")
        st.write(f"Input Tokens: {stats['total_input_tokens']:,}")
        st.write(f"Output Tokens: {stats['total_output_tokens']:,}")
        st.write(f"Total Tokens: {stats['total_tokens']:,}")
        st.write(f"Cost: ${stats['total_cost']:.4f}")
        st.write(f"Time: {stats['total_duration']:.2f} seconds")


def _render_best_prompt(results: List) -> None:
    st.header("Best Prompt")
    best_result = max(results, key=lambda result: result.avg_score)
    st.write("Final Best Prompt:")
    col1, col2 = st.columns(2)
    with col1:
        st.write("System Prompt:")
        st.code(best_result.system_prompt)
    with col2:
        st.write("User Prompt:")
        st.code(best_result.user_prompt)
    st.write(
        "Final Result: "
        f"Average Score {best_result.avg_score:.2f}, "
        f"Best Average Score {best_result.best_avg_score:.2f}, "
        f"Best Individual Score {best_result.best_sample_score:.2f}"
    )


def _render_downloads(tuner) -> None:
    st.header("Download Results")
    col1, col2 = st.columns(2)

    with col1:
        try:
            st.download_button(
                label="Download Full Results CSV",
                data=tuner.save_results_to_csv(),
                file_name=f"prompt_tuning_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_full_csv",
                help="Detailed results per test case and cost information",
            )
        except Exception as exc:
            st.error(f"Error generating full results CSV file: {exc}")

    with col2:
        try:
            st.download_button(
                label="Download Cost Summary CSV",
                data=tuner.export_cost_summary_to_csv(),
                file_name=f"cost_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_cost_csv",
                help="Model-wise and iteration-wise cost summary data",
            )
        except Exception as exc:
            st.error(f"Error generating cost summary CSV file: {exc}")
