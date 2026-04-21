#!/usr/bin/env python3
"""Run no-API smoke checks for the tuning script and app support logic."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_command(command: list[str]) -> None:
    print(f"$ {' '.join(command)}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def assert_mock_outputs(output_dir: Path) -> None:
    expected_patterns = [
        "config_bbh_*.json",
        "results_bbh_*.csv",
        "cost_summary_bbh_*.csv",
        "best_prompt_bbh_*.json",
    ]
    missing = [
        pattern
        for pattern in expected_patterns
        if not list(output_dir.glob(pattern))
    ]
    if missing:
        raise AssertionError(f"Mock run did not create expected files: {missing}")


def main() -> None:
    run_command([sys.executable, "-m", "unittest", "discover", "-v"])
    run_command(
        [
            sys.executable,
            "-m",
            "compileall",
            "agent/app",
            "scripts/run_prompt_tuning.py",
            "setup.py",
        ]
    )

    with tempfile.TemporaryDirectory(prefix="zera-no-api-") as tmpdir:
        output_dir = Path(tmpdir)
        run_command(
            [
                sys.executable,
                "scripts/run_prompt_tuning.py",
                "--dataset",
                "bbh",
                "--total_samples",
                "5",
                "--iteration_samples",
                "2",
                "--iterations",
                "2",
                "--mock",
                "--output_dir",
                str(output_dir),
            ]
        )
        assert_mock_outputs(output_dir)

    print("No-API smoke checks passed.")


if __name__ == "__main__":
    main()
