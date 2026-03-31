"""Session-state helpers for the Streamlit app."""

from __future__ import annotations

import random
from typing import Any, cast

from taboo_arena.app.preferences import PersistedAppPreferences
from taboo_arena.cards.dataset import LOCAL_BUNDLED_DATASET_DIR, LOCAL_BUNDLED_SOURCE_REF
from taboo_arena.config import AppSettings, GenerationParams, RoleGenerationSettings, RoleName
from taboo_arena.models.registry import ModelEntry

ROLE_NAMES: tuple[RoleName, ...] = ("cluer", "guesser", "judge")
ROLE_TOKEN_FLOORS: dict[RoleName, int] = {
    "cluer": 256,
    "guesser": 128,
    "judge": 384,
}
DEFAULT_TARGET_VRAM_GB = 12.0
ROLE_GENERATION_FALLBACKS = RoleGenerationSettings()
RECOMMENDED_TEMPERATURES: dict[str, dict[RoleName, float]] = {
    "compact": {"cluer": 0.28, "guesser": 0.14, "judge": 0.04},
    "balanced": {"cluer": 0.34, "guesser": 0.16, "judge": 0.05},
    "full": {"cluer": 0.40, "guesser": 0.18, "judge": 0.07},
    "overflow": {"cluer": 0.38, "guesser": 0.17, "judge": 0.06},
}
RECOMMENDED_TOP_P: dict[str, dict[RoleName, float]] = {
    "compact": {"cluer": 0.82, "guesser": 0.72, "judge": 0.62},
    "balanced": {"cluer": 0.86, "guesser": 0.76, "judge": 0.66},
    "full": {"cluer": 0.90, "guesser": 0.80, "judge": 0.70},
    "overflow": {"cluer": 0.88, "guesser": 0.78, "judge": 0.68},
}
RECOMMENDED_MAX_TOKENS: dict[str, dict[RoleName, int]] = {
    "compact": {"cluer": 640, "guesser": 384, "judge": 512},
    "balanced": {"cluer": 512, "guesser": 320, "judge": 448},
    "full": {"cluer": 448, "guesser": 288, "judge": 384},
    "overflow": {"cluer": 384, "guesser": 256, "judge": 384},
}


def initialize_session_state(
    state: Any,
    settings: AppSettings,
    preferences: PersistedAppPreferences | None = None,
) -> None:
    """Seed Streamlit session state with stable defaults."""
    persisted = preferences or PersistedAppPreferences.from_settings(settings)

    defaults: dict[str, Any] = {
        "show_gated_models": persisted.show_gated_models,
        "current_state": "idle",
        "current_logger": None,
        "current_result": None,
        "current_error_message": None,
        "current_deck": None,
        "live_round_controller": None,
        "selected_card_id": None,
        "stop_requested": False,
        "start_round_clicked": False,
        "start_benchmark_clicked": False,
        "active_job": None,
        "selected_categories": [],
        "selected_categories_initialized": False,
        "resource_history": [],
        "resource_last_sample_at": 0.0,
        "transcript_history_events": [],
        "transcript_history_run_ids": [],
        "session_history_round_summaries": [],
        "benchmark_running": False,
        "benchmark_plan": None,
        "benchmark_progress": None,
        "benchmark_summary": None,
        "benchmark_error": None,
        "benchmark_card_selection_mode": persisted.benchmark_card_selection_mode,
        "benchmark_sample_size": persisted.benchmark_sample_size,
        "source_ref": persisted.source_ref,
        "language": persisted.language,
        "max_guess_attempts": persisted.max_guess_attempts,
        "max_clue_repairs": persisted.max_clue_repairs,
        "guesser_hidden_retry_budget": persisted.guesser_hidden_retry_budget,
        "block_on_uncertain": persisted.block_on_uncertain,
        "random_seed": int(persisted.random_seed),
        "device_preference": persisted.device_preference,
        "show_hidden_repairs": persisted.show_hidden_repairs,
        "allow_same_model_for_multiple_roles": persisted.allow_same_model_for_multiple_roles,
        "debug_show_target": persisted.debug_show_target,
        "console_trace": persisted.console_trace,
        "log_dir": str(persisted.log_dir),
        "memory_policy": persisted.memory_policy,
        "validator_min_token_count": persisted.validator_min_token_count,
        "validator_check_substring": persisted.validator_check_substring,
        "validator_check_token_match": persisted.validator_check_token_match,
        "validator_check_whole_word": persisted.validator_check_whole_word,
        "validator_check_stemming": persisted.validator_check_stemming,
        "validator_check_similarity": persisted.validator_check_similarity,
        "validator_similarity_threshold": persisted.validator_similarity_threshold,
        "cluer_temperature": persisted.cluer_temperature,
        "cluer_top_p": persisted.cluer_top_p,
        "cluer_max_tokens": persisted.cluer_max_tokens,
        "guesser_temperature": persisted.guesser_temperature,
        "guesser_top_p": persisted.guesser_top_p,
        "guesser_max_tokens": persisted.guesser_max_tokens,
        "judge_temperature": persisted.judge_temperature,
        "judge_top_p": persisted.judge_top_p,
        "judge_max_tokens": persisted.judge_max_tokens,
        "cluer_model_id": persisted.cluer_model_id,
        "guesser_model_id": persisted.guesser_model_id,
        "judge_model_id": persisted.judge_model_id,
    }
    for role in ROLE_NAMES:
        defaults[f"{role}_generation_dirty"] = bool(getattr(persisted, f"{role}_generation_dirty"))
        defaults[f"{role}_generation_source_model_id"] = getattr(
            persisted,
            f"{role}_generation_source_model_id",
        )
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
    random_seed = int(state["random_seed"])
    if state.get("app_randomizer_seed") != random_seed or "app_randomizer" not in state:
        state["app_randomizer"] = random.Random(random_seed)
        state["app_randomizer_seed"] = random_seed


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
    target_vram_gb: float | None = None,
) -> dict[str, str]:
    """Apply selected-model generation defaults, always resyncing when the model changes."""
    applied: dict[str, str] = {}
    for role_name, entry in selected_models.items():
        role = cast(RoleName, role_name)
        source_key = f"{role}_generation_source_model_id"
        dirty_key = f"{role}_generation_dirty"
        recommended = recommended_generation_defaults(role, entry, target_vram_gb=target_vram_gb)
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


def generation_status_text(
    state: Any,
    role: RoleName,
    entry: ModelEntry,
    *,
    target_vram_gb: float | None = None,
) -> str:
    """Describe whether the role uses registry defaults or custom overrides."""
    source_model_id = state.get(f"{role}_generation_source_model_id")
    dirty = bool(state.get(f"{role}_generation_dirty", False))
    current_values = applied_generation_params(state, role)
    recommended = recommended_generation_defaults(role, entry, target_vram_gb=target_vram_gb)
    matches_recommended = generation_params_match(current_values, recommended)

    if source_model_id == entry.id and matches_recommended:
        return f"Applied now matches the recommended {role} defaults for {entry.display_name}."
    if source_model_id == entry.id and dirty:
        return f"Applied now uses custom overrides for {entry.display_name}."
    if matches_recommended:
        return f"Applied now matches the recommended {role} defaults for {entry.display_name}."
    return "Applied values differ from the recommendation. Click 'Use recommended defaults' to resync."


def recommended_generation_defaults(
    role: RoleName,
    entry: ModelEntry,
    *,
    target_vram_gb: float | None = None,
) -> GenerationParams:
    """Return practical role-aware defaults tuned for local Taboo play."""
    defaults = entry.default_generation_params
    size_tier = _recommendation_size_tier(entry, target_vram_gb=target_vram_gb)
    return GenerationParams(
        temperature=RECOMMENDED_TEMPERATURES[size_tier][role],
        top_p=RECOMMENDED_TOP_P[size_tier][role],
        max_tokens=max(
            int(defaults.max_tokens),
            ROLE_TOKEN_FLOORS[role],
            RECOMMENDED_MAX_TOKENS[size_tier][role],
        ),
    )


def applied_generation_params(state: Any, role: RoleName) -> GenerationParams:
    """Return the currently applied generation settings for one role from session state."""
    fallback = getattr(ROLE_GENERATION_FALLBACKS, role)
    return GenerationParams(
        temperature=float(state.get(f"{role}_temperature", fallback.temperature)),
        top_p=float(state.get(f"{role}_top_p", fallback.top_p)),
        max_tokens=int(state.get(f"{role}_max_tokens", fallback.max_tokens)),
    )


def generation_params_match(left: GenerationParams, right: GenerationParams) -> bool:
    """Return whether two generation parameter sets are effectively identical."""
    return (
        abs(float(left.temperature) - float(right.temperature)) < 1e-9
        and abs(float(left.top_p) - float(right.top_p)) < 1e-9
        and int(left.max_tokens) == int(right.max_tokens)
    )


def generation_widget_key(role: RoleName, field_name: str) -> str:
    """Return the transient widget key used for generation controls."""
    return f"{role}_{field_name}_input_widget"


def sync_generation_widget_state(state: Any, role: RoleName) -> None:
    """Mirror persisted generation settings into ephemeral widget keys."""
    applied = applied_generation_params(state, role)
    state[generation_widget_key(role, "temperature")] = float(applied.temperature)
    state[generation_widget_key(role, "top_p")] = float(applied.top_p)
    state[generation_widget_key(role, "max_tokens")] = int(applied.max_tokens)


def apply_generation_widget_state(state: Any, role: RoleName) -> None:
    """Commit widget-edited generation values back to the persisted app state."""
    fallback = applied_generation_params(state, role)
    state[f"{role}_temperature"] = float(
        state.get(generation_widget_key(role, "temperature"), fallback.temperature)
    )
    state[f"{role}_top_p"] = float(state.get(generation_widget_key(role, "top_p"), fallback.top_p))
    state[f"{role}_max_tokens"] = int(
        state.get(generation_widget_key(role, "max_tokens"), fallback.max_tokens)
    )
    mark_generation_dirty(state, role)


def _recommendation_size_tier(entry: ModelEntry, *, target_vram_gb: float | None = None) -> str:
    """Map a model estimate to a practical local-VRAM recommendation tier."""
    effective_target = float(target_vram_gb or DEFAULT_TARGET_VRAM_GB)
    estimated_vram_gb = float(entry.estimated_vram_gb or effective_target * 0.65)
    usage_ratio = estimated_vram_gb / max(effective_target, 0.1)
    if usage_ratio <= 0.40:
        return "compact"
    if usage_ratio <= 0.75:
        return "balanced"
    if usage_ratio <= 1.00:
        return "full"
    return "overflow"


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
