"""Session-state helpers for the Streamlit app."""

from __future__ import annotations

import random
from typing import Any, cast

from taboo_arena.cards.dataset import LOCAL_BUNDLED_DATASET_DIR, LOCAL_BUNDLED_SOURCE_REF
from taboo_arena.config import AppSettings, GenerationParams, RoleName
from taboo_arena.models.registry import ModelEntry

ROLE_NAMES: tuple[RoleName, ...] = ("cluer", "guesser", "judge")
ROLE_TOKEN_FLOORS: dict[RoleName, int] = {
    "cluer": 256,
    "guesser": 128,
    "judge": 384,
}


def initialize_session_state(state: Any, settings: AppSettings) -> None:
    """Seed Streamlit session state with stable defaults."""
    if "random_seed" not in state:
        state["random_seed"] = random.SystemRandom().randint(1, 2_147_483_647)

    defaults: dict[str, Any] = {
        "show_gated_models": False,
        "current_state": "idle",
        "current_logger": None,
        "current_result": None,
        "current_error_message": None,
        "current_deck": None,
        "live_round_controller": None,
        "selected_card_id": None,
        "stop_requested": False,
        "start_round_clicked": False,
        "start_batch_clicked": False,
        "active_job": None,
        "batch_job": None,
        "selected_categories": [],
        "selected_categories_initialized": False,
        "resource_history": [],
        "resource_last_sample_at": 0.0,
        "transcript_history_events": [],
        "transcript_history_run_ids": [],
        "session_history_round_summaries": [],
        "source_ref": settings.dataset.source_ref,
        "language": settings.dataset.language,
        "max_guess_attempts": settings.run.max_guess_attempts,
        "max_clue_repairs": settings.run.max_clue_repairs,
        "block_on_uncertain": settings.run.block_on_uncertain,
        "random_seed": int(state["random_seed"]),
        "device_preference": settings.run.device_preference,
        "show_hidden_repairs": settings.run.show_hidden_repairs,
        "allow_same_model_for_multiple_roles": settings.run.allow_same_model_for_multiple_roles,
        "debug_show_target": settings.run.debug_show_target,
        "console_trace": settings.run.console_trace,
        "log_dir": str(settings.run.log_dir),
        "memory_policy": settings.run.memory_policy,
        "validator_min_token_count": settings.run.logical_validator.min_token_count,
        "validator_check_substring": settings.run.logical_validator.check_substring,
        "validator_check_token_match": settings.run.logical_validator.check_token_match,
        "validator_check_whole_word": settings.run.logical_validator.check_whole_word,
        "validator_check_stemming": settings.run.logical_validator.check_stemming,
        "validator_check_similarity": settings.run.logical_validator.check_similarity_to_previous,
        "validator_similarity_threshold": settings.run.logical_validator.similarity_threshold,
        "cluer_temperature": settings.run.generation.cluer.temperature,
        "cluer_top_p": settings.run.generation.cluer.top_p,
        "cluer_max_tokens": settings.run.generation.cluer.max_tokens,
        "guesser_temperature": settings.run.generation.guesser.temperature,
        "guesser_top_p": settings.run.generation.guesser.top_p,
        "guesser_max_tokens": settings.run.generation.guesser.max_tokens,
        "judge_temperature": settings.run.generation.judge.temperature,
        "judge_top_p": settings.run.generation.judge.top_p,
        "judge_max_tokens": settings.run.generation.judge.max_tokens,
        "batch_sample_size": 5,
        "batch_repeats_per_card": 1,
        "batch_cluer_ids": [],
        "batch_guesser_ids": [],
        "batch_judge_ids": [],
        "batch_fixed_card_ids": [],
    }
    for role in ROLE_NAMES:
        defaults[f"{role}_generation_dirty"] = False
        defaults[f"{role}_generation_source_model_id"] = None
    for key, value in defaults.items():
        state.setdefault(key, value)
    if (
        LOCAL_BUNDLED_DATASET_DIR.exists()
        and state.get("current_deck") is None
        and str(state.get("language", "en")) == "en"
        and str(state.get("source_ref", "")) == "main"
    ):
        state["source_ref"] = LOCAL_BUNDLED_SOURCE_REF
        state["current_deck"] = None
    if "app_randomizer" not in state:
        state["app_randomizer"] = random.Random(int(state["random_seed"]))


def choose_default_model_id(
    entries: list[ModelEntry],
    *,
    role: RoleName,
    available_vram_gb: float | None,
    prefer_gpu: bool,
) -> str:
    """Pick a sensible initial model for a role based on local memory."""
    role_entries = [entry for entry in entries if role in entry.roles_supported]
    if not role_entries:
        raise ValueError(f"No models are available for role '{role}'.")

    if prefer_gpu and available_vram_gb is not None:
        fitting_entries = [
            entry
            for entry in role_entries
            if entry.estimated_vram_gb is not None and entry.estimated_vram_gb <= available_vram_gb * 0.85
        ]
        if fitting_entries:
            return max(
                fitting_entries,
                key=lambda item: (float(item.estimated_vram_gb or 0.0), item.display_name.lower()),
            ).id

    return min(
        role_entries,
        key=lambda item: (
            item.estimated_vram_gb is None,
            float(item.estimated_vram_gb) if item.estimated_vram_gb is not None else float("inf"),
            item.display_name.lower(),
        ),
    ).id


def sync_selected_model_generation_defaults(
    state: Any,
    selected_models: dict[str, ModelEntry],
    *,
    force: bool = False,
) -> dict[str, str]:
    """Apply selected-model generation defaults, always resyncing when the model changes."""
    applied: dict[str, str] = {}
    for role_name, entry in selected_models.items():
        role = cast(RoleName, role_name)
        source_key = f"{role}_generation_source_model_id"
        dirty_key = f"{role}_generation_dirty"
        recommended = recommended_generation_defaults(role, entry)
        current_values = applied_generation_params(state, role)
        current_matches = generation_params_match(current_values, recommended)
        should_apply = force or state.get(source_key) is None or state.get(source_key) != entry.id or (
            not bool(state.get(dirty_key, False)) and not current_matches
        )
        if not should_apply:
            continue
        state[f"{role}_temperature"] = float(recommended.temperature)
        state[f"{role}_top_p"] = float(recommended.top_p)
        state[f"{role}_max_tokens"] = int(recommended.max_tokens)
        state[source_key] = entry.id
        state[dirty_key] = False
        applied[role] = entry.id
    return applied


def mark_generation_dirty(state: Any, role: RoleName) -> None:
    """Remember that the user manually changed generation settings for a role."""
    state[f"{role}_generation_dirty"] = True


def generation_status_text(state: Any, role: RoleName, entry: ModelEntry) -> str:
    """Describe whether the role uses registry defaults or custom overrides."""
    source_model_id = state.get(f"{role}_generation_source_model_id")
    dirty = bool(state.get(f"{role}_generation_dirty", False))
    current_values = applied_generation_params(state, role)
    recommended = recommended_generation_defaults(role, entry)
    matches_recommended = generation_params_match(current_values, recommended)

    if source_model_id == entry.id and matches_recommended:
        return f"Applied now matches the recommended {role} defaults for {entry.display_name}."
    if source_model_id == entry.id and dirty:
        return f"Applied now uses custom overrides for {entry.display_name}."
    if matches_recommended:
        return f"Applied now matches the recommended {role} defaults for {entry.display_name}."
    return "Applied values differ from the recommendation. Click 'Use recommended defaults' to resync."


def recommended_generation_defaults(role: RoleName, entry: ModelEntry) -> GenerationParams:
    """Return practical role-aware defaults for the selected model."""
    defaults = entry.default_generation_params
    return defaults.model_copy(
        update={
            "max_tokens": max(int(defaults.max_tokens), ROLE_TOKEN_FLOORS[role]),
        }
    )


def applied_generation_params(state: Any, role: RoleName) -> GenerationParams:
    """Return the currently applied generation settings for one role from session state."""
    return GenerationParams(
        temperature=float(state.get(f"{role}_temperature", 0.0)),
        top_p=float(state.get(f"{role}_top_p", 0.1)),
        max_tokens=int(state.get(f"{role}_max_tokens", 16)),
    )


def generation_params_match(left: GenerationParams, right: GenerationParams) -> bool:
    """Return whether two generation parameter sets are effectively identical."""
    return (
        abs(float(left.temperature) - float(right.temperature)) < 1e-9
        and abs(float(left.top_p) - float(right.top_p)) < 1e-9
        and int(left.max_tokens) == int(right.max_tokens)
    )


def prepare_category_selection(state: Any, categories: list[str]) -> list[str]:
    """Normalize the category selection before the pills widget is created."""
    if not state.get("selected_categories_initialized", False):
        state["selected_categories"] = list(categories)
        state["selected_categories_initialized"] = True
        return list(categories)

    current_selection = [item for item in state.get("selected_categories", []) if item in categories]
    if state.get("selected_categories") and not current_selection:
        current_selection = list(categories)
    state["selected_categories"] = current_selection
    return current_selection
