"""Persistence helpers for Streamlit UI preferences."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from taboo_arena.config import AppSettings, DevicePreference, MemoryPolicy
from taboo_arena.utils.paths import ensure_app_dirs, get_log_dir, get_ui_preferences_path

BenchmarkSelectionMode = Literal["all_eligible", "random_sample"]


class PersistedAppPreferences(BaseModel):
    """User-specific UI settings that should survive Streamlit restarts."""

    source_ref: str = "main"
    language: str = "en"
    max_guess_attempts: int = Field(default=3, ge=1, le=10)
    max_clue_repairs: int = Field(default=3, ge=1, le=10)
    guesser_hidden_retry_budget: int = Field(default=2, ge=0, le=10)
    block_on_uncertain: bool = False
    random_seed: int = 7
    device_preference: DevicePreference = "auto"
    show_hidden_repairs: bool = True
    allow_same_model_for_multiple_roles: bool = True
    debug_show_target: bool = False
    console_trace: bool = True
    log_dir: Path = Field(default_factory=get_log_dir)
    memory_policy: MemoryPolicy = "keep_loaded_if_possible"
    validator_min_token_count: int = Field(default=2, ge=1, le=10)
    validator_check_substring: bool = True
    validator_check_token_match: bool = True
    validator_check_whole_word: bool = True
    validator_check_stemming: bool = True
    validator_check_similarity: bool = True
    validator_similarity_threshold: int = Field(default=88, ge=1, le=100)
    cluer_temperature: float = Field(default=0.4, ge=0.0, le=2.0)
    cluer_top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    cluer_max_tokens: int = Field(default=256, ge=1, le=4096)
    guesser_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    guesser_top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    guesser_max_tokens: int = Field(default=128, ge=1, le=4096)
    judge_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    judge_top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    judge_max_tokens: int = Field(default=384, ge=1, le=4096)
    cluer_model_id: str | None = None
    guesser_model_id: str | None = None
    judge_model_id: str | None = None
    cluer_generation_dirty: bool = False
    guesser_generation_dirty: bool = False
    judge_generation_dirty: bool = False
    cluer_generation_source_model_id: str | None = None
    guesser_generation_source_model_id: str | None = None
    judge_generation_source_model_id: str | None = None
    show_gated_models: bool = False
    benchmark_card_selection_mode: BenchmarkSelectionMode = "all_eligible"
    benchmark_sample_size: int = Field(default=25, ge=1, le=10_000)

    @classmethod
    def from_settings(cls, settings: AppSettings) -> PersistedAppPreferences:
        """Build the default persisted preference payload from app settings."""
        return cls(
            source_ref=settings.dataset.source_ref,
            language=settings.dataset.language,
            max_guess_attempts=settings.run.max_guess_attempts,
            max_clue_repairs=settings.run.max_clue_repairs,
            guesser_hidden_retry_budget=settings.run.guesser_hidden_retry_budget,
            block_on_uncertain=settings.run.block_on_uncertain,
            random_seed=settings.run.random_seed,
            device_preference=settings.run.device_preference,
            show_hidden_repairs=settings.run.show_hidden_repairs,
            allow_same_model_for_multiple_roles=settings.run.allow_same_model_for_multiple_roles,
            debug_show_target=settings.run.debug_show_target,
            console_trace=settings.run.console_trace,
            log_dir=settings.run.log_dir,
            memory_policy=settings.run.memory_policy,
            validator_min_token_count=settings.run.logical_validator.min_token_count,
            validator_check_substring=settings.run.logical_validator.check_substring,
            validator_check_token_match=settings.run.logical_validator.check_token_match,
            validator_check_whole_word=settings.run.logical_validator.check_whole_word,
            validator_check_stemming=settings.run.logical_validator.check_stemming,
            validator_check_similarity=settings.run.logical_validator.check_similarity_to_previous,
            validator_similarity_threshold=settings.run.logical_validator.similarity_threshold,
            cluer_temperature=settings.run.generation.cluer.temperature,
            cluer_top_p=settings.run.generation.cluer.top_p,
            cluer_max_tokens=settings.run.generation.cluer.max_tokens,
            guesser_temperature=settings.run.generation.guesser.temperature,
            guesser_top_p=settings.run.generation.guesser.top_p,
            guesser_max_tokens=settings.run.generation.guesser.max_tokens,
            judge_temperature=settings.run.generation.judge.temperature,
            judge_top_p=settings.run.generation.judge.top_p,
            judge_max_tokens=settings.run.generation.judge.max_tokens,
        )


def load_app_preferences(
    settings: AppSettings,
    *,
    preferences_path: Path | None = None,
) -> PersistedAppPreferences:
    """Load persisted UI preferences and fall back to config defaults when missing."""
    defaults = PersistedAppPreferences.from_settings(settings)
    path = preferences_path or get_ui_preferences_path()
    if not path.exists():
        return defaults

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return defaults
    if not isinstance(payload, dict):
        return defaults

    merged = defaults.model_dump(mode="json")
    for field_name in PersistedAppPreferences.model_fields:
        if field_name in payload:
            merged[field_name] = payload[field_name]
    try:
        return PersistedAppPreferences.model_validate(merged)
    except Exception:
        return defaults


def preferences_from_state(
    state: Any,
    *,
    fallback_settings: AppSettings | None = None,
) -> PersistedAppPreferences:
    """Extract the stable, user-editable subset of Streamlit state."""
    defaults = PersistedAppPreferences.from_settings(fallback_settings or AppSettings())
    payload = defaults.model_dump(mode="json")
    for field_name in PersistedAppPreferences.model_fields:
        if field_name in state:
            payload[field_name] = state[field_name]
    return PersistedAppPreferences.model_validate(payload)


def save_app_preferences(
    preferences: PersistedAppPreferences,
    *,
    preferences_path: Path | None = None,
) -> None:
    """Write persisted UI preferences to disk when they change."""
    ensure_app_dirs()
    path = preferences_path or get_ui_preferences_path()
    rendered = yaml.safe_dump(
        preferences.model_dump(mode="json", exclude_none=True),
        sort_keys=False,
        allow_unicode=False,
    )
    if path.exists() and path.read_text(encoding="utf-8") == rendered:
        return
    path.write_text(rendered, encoding="utf-8")


def persist_session_preferences(
    state: Any,
    *,
    fallback_settings: AppSettings | None = None,
    preferences_path: Path | None = None,
) -> PersistedAppPreferences:
    """Capture the current session-state preferences and save them."""
    preferences = preferences_from_state(state, fallback_settings=fallback_settings)
    save_app_preferences(preferences, preferences_path=preferences_path)
    return preferences
