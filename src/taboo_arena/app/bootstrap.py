"""Shared service construction for the UI and CLI."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from taboo_arena.cards import DatasetManager
from taboo_arena.config import AppSettings
from taboo_arena.logging.run_logger import RunLogger
from taboo_arena.models import ModelManager, ModelRegistry


def load_app_settings(config_path: Path | None = None) -> AppSettings:
    """Load application settings from YAML."""
    path = config_path or Path(__file__).resolve().parents[3] / "configs" / "defaults.yaml"
    return AppSettings.from_yaml(path)


def build_registry() -> ModelRegistry:
    """Create a model registry instance."""
    return ModelRegistry()


def build_dataset_manager(settings: AppSettings) -> DatasetManager:
    """Create a dataset manager."""
    return DatasetManager(settings.dataset)


def build_logger(
    log_dir: Path | None = None,
    *,
    console_trace: bool = True,
    event_callback: Callable[[dict[str, Any]], None] | None = None,
) -> RunLogger:
    """Create a run logger."""
    return RunLogger(log_root=log_dir, console_trace=console_trace, event_callback=event_callback)


def build_model_manager(logger: RunLogger | None = None) -> ModelManager:
    """Create a model manager."""
    return ModelManager(logger=logger)
