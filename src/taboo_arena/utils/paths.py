"""Filesystem path helpers."""

from __future__ import annotations

import os
from pathlib import Path

from platformdirs import user_data_dir

APP_NAME = "taboo-arena"
APP_AUTHOR = "taboo-arena"


def get_app_dir() -> Path:
    """Return the application data directory."""
    env_override = os.getenv("TABOO_ARENA_APP_DIR")
    if env_override:
        return Path(env_override).expanduser().resolve()
    return Path(user_data_dir(APP_NAME, APP_AUTHOR))


def get_dataset_dir() -> Path:
    """Return the dataset cache directory."""
    return get_app_dir() / "datasets"


def get_model_dir() -> Path:
    """Return the model metadata directory."""
    return get_app_dir() / "models"


def get_log_dir() -> Path:
    """Return the log directory."""
    env_override = os.getenv("TABOO_ARENA_LOG_DIR")
    if env_override:
        return Path(env_override).expanduser().resolve()
    return get_app_dir() / "logs"


def get_ui_preferences_path() -> Path:
    """Return the persisted Streamlit UI preferences path."""
    return get_app_dir() / "ui_preferences.yaml"


def get_custom_model_store_path() -> Path:
    """Return the persisted custom model registry path."""
    return get_model_dir() / "custom_models.json"


def ensure_app_dirs() -> None:
    """Create the standard application directories if they are missing."""
    for path in [get_app_dir(), get_dataset_dir(), get_model_dir(), get_log_dir()]:
        path.mkdir(parents=True, exist_ok=True)
