from __future__ import annotations

import random

from taboo_arena.app.state import (
    applied_generation_params,
    choose_default_model_id,
    generation_status_text,
    initialize_session_state,
    mark_generation_dirty,
    prepare_category_selection,
    recommended_generation_defaults,
    sync_selected_model_generation_defaults,
)
from taboo_arena.config import AppSettings, GenerationParams
from taboo_arena.models.registry import ModelEntry


def _entry(model_id: str, *, estimated_vram_gb: float, max_tokens: int = 128) -> ModelEntry:
    return ModelEntry(
        id=model_id,
        display_name=model_id,
        backend="transformers_safetensors",
        repo_id=f"repo/{model_id}",
        architecture_family="test",
        chat_template_id="generic_completion",
        supports_system_prompt=True,
        roles_supported=["cluer", "guesser", "judge"],
        languages=["en"],
        estimated_vram_gb=estimated_vram_gb,
        default_generation_params=GenerationParams(
            temperature=0.25 + estimated_vram_gb / 100.0,
            top_p=0.9,
            max_tokens=max_tokens,
        ),
    )


def test_choose_default_model_prefers_largest_fitting_gpu_model() -> None:
    selected_id = choose_default_model_id(
        [
            _entry("small", estimated_vram_gb=4.0),
            _entry("mid", estimated_vram_gb=8.0),
            _entry("too-big", estimated_vram_gb=14.0),
        ],
        role="cluer",
        available_vram_gb=12.0,
        prefer_gpu=True,
    )

    assert selected_id == "mid"


def test_choose_default_model_prefers_smallest_entry_when_gpu_not_preferred() -> None:
    selected_id = choose_default_model_id(
        [
            _entry("small", estimated_vram_gb=4.0),
            _entry("mid", estimated_vram_gb=8.0),
        ],
        role="judge",
        available_vram_gb=12.0,
        prefer_gpu=False,
    )

    assert selected_id == "small"


def test_sync_selected_model_generation_defaults_resets_to_new_model_defaults_on_selection_change() -> None:
    state: dict[str, object] = {}
    initialize_session_state(state, AppSettings())
    first_entry = _entry("model-a", estimated_vram_gb=4.0, max_tokens=111)
    second_entry = _entry("model-b", estimated_vram_gb=8.0, max_tokens=222)

    sync_selected_model_generation_defaults(
        state,
        {"cluer": first_entry, "guesser": first_entry, "judge": first_entry},
    )
    assert state["cluer_max_tokens"] == 256
    assert "recommended cluer defaults" in generation_status_text(state, "cluer", first_entry)

    state["cluer_max_tokens"] = 777
    mark_generation_dirty(state, "cluer")
    sync_selected_model_generation_defaults(
        state,
        {"cluer": second_entry, "guesser": second_entry, "judge": second_entry},
    )

    assert state["cluer_max_tokens"] == 256
    assert "matches the recommended cluer defaults" in generation_status_text(
        state,
        "cluer",
        second_entry,
    )

    sync_selected_model_generation_defaults(
        state,
        {"cluer": second_entry, "guesser": second_entry, "judge": second_entry},
        force=True,
    )

    assert state["cluer_max_tokens"] == 256
    assert state["cluer_generation_dirty"] is False


def test_generation_status_text_reports_actual_applied_values() -> None:
    state: dict[str, object] = {}
    initialize_session_state(state, AppSettings())
    entry = _entry("model-a", estimated_vram_gb=4.0, max_tokens=160)

    sync_selected_model_generation_defaults(state, {"cluer": entry})
    assert "matches the recommended cluer defaults" in generation_status_text(state, "cluer", entry)

    state["cluer_temperature"] = 0.0
    state["cluer_top_p"] = 0.1
    state["cluer_max_tokens"] = 16
    assert "Applied values differ from the recommendation" in generation_status_text(
        state,
        "cluer",
        entry,
    )

    applied = applied_generation_params(state, "cluer")
    assert applied.temperature == 0.0
    assert applied.top_p == 0.1
    assert applied.max_tokens == 16


def test_recommended_generation_defaults_apply_role_token_floors() -> None:
    entry = _entry("model-a", estimated_vram_gb=4.0, max_tokens=160)

    assert recommended_generation_defaults("cluer", entry).max_tokens == 256
    assert recommended_generation_defaults("guesser", entry).max_tokens == 160
    assert recommended_generation_defaults("judge", entry).max_tokens == 384


def test_prepare_category_selection_initializes_and_filters() -> None:
    state: dict[str, object] = {}
    initialize_session_state(state, AppSettings())

    first_selection = prepare_category_selection(state, ["Animals", "People"])
    assert first_selection == ["Animals", "People"]
    assert state["selected_categories_initialized"] is True

    state["selected_categories"] = ["People", "Missing"]
    second_selection = prepare_category_selection(state, ["Animals", "People", "Places"])
    assert second_selection == ["People"]
    assert state["selected_categories"] == ["People"]


def test_initialize_session_state_creates_session_seed_and_rng() -> None:
    state: dict[str, object] = {}

    initialize_session_state(state, AppSettings())

    assert isinstance(state["random_seed"], int)
    assert int(state["random_seed"]) > 0
    assert isinstance(state["app_randomizer"], random.Random)
