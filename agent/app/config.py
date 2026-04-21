from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable


PROMPT_FILES = {
    "system": "initial_system_prompt.txt",
    "user": "initial_user_prompt.txt",
    "evaluation_system": "evaluation_system_prompt.txt",
    "evaluation_user": "evaluation_user_prompt.txt",
    "meta_system": "meta_system_prompt.txt",
    "meta_user": "meta_user_prompt.txt",
}

REQUIRED_API_KEYS = {
    "solar": "SOLAR_API_KEY",
    "gpt4o": "OPENAI_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "local1": None,
    "local2": None,
    "solar_strawberry": "SOLAR_STRAWBERRY_API_KEY",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_dotenv_file(env_path: Path) -> bool:
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        return False

    load_dotenv(env_path, override=True)
    return True


def setup_environment() -> None:
    if getattr(setup_environment, "loaded", False):
        return

    env_paths = [
        Path(".env"),
        Path("../.env"),
        Path("../../.env"),
        project_root() / ".env",
        Path.home() / ".env",
    ]

    for env_path in env_paths:
        if env_path.exists():
            if _load_dotenv_file(env_path):
                print(f"Loaded environment from: {env_path}")
                break
            print("Warning: python-dotenv is not installed; skipping .env loading")
            break
    else:
        print("Warning: No .env file found")

    setup_environment.loaded = True


def load_prompt_templates() -> Dict[str, str]:
    prompts_dir = project_root() / "agent" / "prompts"
    return {
        name: (prompts_dir / filename).read_text(encoding="utf-8")
        for name, filename in PROMPT_FILES.items()
    }


def missing_api_keys(model_names: Iterable[str]) -> list[str]:
    missing = []
    for model_name in sorted(set(model_names)):
        key = REQUIRED_API_KEYS.get(model_name)
        if key and not os.getenv(key):
            missing.append(key)
    return missing
