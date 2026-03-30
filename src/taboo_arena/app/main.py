"""Single-screen Streamlit UI for taboo-arena."""

from __future__ import annotations

import html
import queue
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd
import streamlit as st

from taboo_arena.analytics import compute_summary_metrics
from taboo_arena.app.bootstrap import (
    build_dataset_manager,
    build_logger,
    build_model_manager,
    build_registry,
    load_app_settings,
)
from taboo_arena.app.jobs import ActiveJob, start_batch_job, start_single_round_job
from taboo_arena.app.session_facade import SessionFacade
from taboo_arena.app.state import (
    applied_generation_params,
    apply_generation_widget_state,
    choose_default_model_id,
    generation_status_text,
    generation_widget_key,
    initialize_session_state,
    prepare_category_selection,
    recommended_generation_defaults,
    sync_generation_widget_state,
    sync_selected_model_generation_defaults,
)
from taboo_arena.app.ui_stats_panel import render_live_round_pulse_inline, render_round_pulse_inline
from taboo_arena.app.ui_theme import apply_theme
from taboo_arena.app.ui_transcript_panel import (
    active_transcript_placeholder,
    archive_current_logger_for_transcript,
    live_logger_events,
    render_transcript_panel_content,
)
from taboo_arena.cards.dataset import (
    LOCAL_BUNDLED_DATASET_DIR,
    LOCAL_BUNDLED_SOURCE_REF,
    LOCAL_BUNDLED_SOURCE_REPO,
)
from taboo_arena.config import AppSettings, BackendName, GenerationParams, RoleName
from taboo_arena.logging.run_logger import RunLogger
from taboo_arena.logging.schemas import RoundSummaryRecord
from taboo_arena.models import ModelEntry, ModelRegistry
from taboo_arena.models.manager import ModelManager
from taboo_arena.utils.normalization import slugify
from taboo_arena.utils.system import (
    RuntimeDiagnostics,
    SystemMemorySnapshot,
    get_memory_snapshot,
    get_runtime_diagnostics,
    get_system_load_sample,
)

st.set_page_config(page_title="taboo-arena", layout="wide")

RESOURCE_HISTORY_LIMIT = 40
LIVE_PANEL_REFRESH_SECONDS = 0.5
RESOURCE_SAMPLE_INTERVAL_SECONDS = 1.0


@dataclass(slots=True)
class RoleConsoleState:
    """UI state for one role card in the compact dashboard."""

    mode: Literal["sleep", "idle", "thinking"]
    note: str


@st.cache_resource
def get_persistent_model_manager() -> ModelManager:
    """Keep loaded models alive across Streamlit reruns when possible."""
    return build_model_manager()


def _active_run_present() -> bool:
    return st.session_state.active_job is not None


def _live_logger_events(current_logger: RunLogger) -> list[dict[str, Any]]:
    """Compatibility wrapper for existing tests and local helpers."""
    return live_logger_events(current_logger)


def _active_transcript_placeholder(current_logger: RunLogger | None) -> str:
    """Compatibility wrapper for existing tests and local helpers."""
    return active_transcript_placeholder(current_logger)


def _render_live_dashboard(
    *,
    registry: ModelRegistry,
    model_manager: ModelManager,
) -> None:
    session = SessionFacade(st.session_state)
    _poll_active_job_updates()
    settings = _settings_from_session(load_app_settings())
    deck = _ensure_deck(settings)
    _prepare_card_selection_state(deck)
    current_logger = session.current_logger
    if current_logger is not None:
        model_manager.logger = current_logger
    runtime_diagnostics = get_runtime_diagnostics()
    memory_snapshot = get_memory_snapshot()
    _record_system_load_sample()

    _render_header(runtime_diagnostics, current_logger)
    selected_models = _render_top_bar(
        registry,
        model_manager,
        memory_snapshot,
        current_logger,
    )
    target_vram_gb = memory_snapshot.vram_total_gb or memory_snapshot.vram_available_gb
    sync_selected_model_generation_defaults(
        st.session_state,
        selected_models,
        target_vram_gb=target_vram_gb,
    )
    session.start_batch_clicked = False
    if session.current_error_message:
        st.error(str(session.current_error_message))
    _render_main_body(deck, selected_models, model_manager)
    if _finalize_active_job_if_needed():
        st.rerun()


def main() -> None:
    """Render the single-screen app."""
    initialize_session_state(st.session_state, load_app_settings())
    apply_theme()

    registry = build_registry()
    model_manager = get_persistent_model_manager()
    _render_live_dashboard(registry=registry, model_manager=model_manager)

def _settings_from_session(settings: AppSettings) -> AppSettings:
    settings.dataset.source_ref = str(st.session_state.source_ref)
    settings.dataset.language = str(st.session_state.language)
    settings.run.max_guess_attempts = int(st.session_state.max_guess_attempts)
    settings.run.max_clue_repairs = int(st.session_state.max_clue_repairs)
    settings.run.block_on_uncertain = bool(st.session_state.block_on_uncertain)
    settings.run.random_seed = int(st.session_state.random_seed)
    settings.run.device_preference = st.session_state.device_preference
    settings.run.show_hidden_repairs = bool(st.session_state.show_hidden_repairs)
    settings.run.allow_same_model_for_multiple_roles = bool(
        st.session_state.allow_same_model_for_multiple_roles
    )
    settings.run.debug_show_target = bool(st.session_state.debug_show_target)
    settings.run.console_trace = bool(st.session_state.console_trace)
    settings.run.log_dir = Path(str(st.session_state.log_dir)).expanduser()
    settings.run.memory_policy = st.session_state.memory_policy
    settings.run.logical_validator.min_token_count = int(st.session_state.validator_min_token_count)
    settings.run.logical_validator.check_substring = bool(st.session_state.validator_check_substring)
    settings.run.logical_validator.check_token_match = bool(st.session_state.validator_check_token_match)
    settings.run.logical_validator.check_whole_word = bool(st.session_state.validator_check_whole_word)
    settings.run.logical_validator.check_stemming = bool(st.session_state.validator_check_stemming)
    settings.run.logical_validator.check_similarity_to_previous = bool(
        st.session_state.validator_check_similarity
    )
    settings.run.logical_validator.similarity_threshold = int(
        st.session_state.validator_similarity_threshold
    )
    settings.run.generation.cluer = GenerationParams(
        temperature=float(st.session_state.cluer_temperature),
        top_p=float(st.session_state.cluer_top_p),
        max_tokens=int(st.session_state.cluer_max_tokens),
    )
    settings.run.generation.guesser = GenerationParams(
        temperature=float(st.session_state.guesser_temperature),
        top_p=float(st.session_state.guesser_top_p),
        max_tokens=int(st.session_state.guesser_max_tokens),
    )
    settings.run.generation.judge = GenerationParams(
        temperature=float(st.session_state.judge_temperature),
        top_p=float(st.session_state.judge_top_p),
        max_tokens=int(st.session_state.judge_max_tokens),
    )
    return settings


def _ensure_deck(settings: AppSettings) -> Any:
    bundled_source_expected = (
        settings.dataset.language == "en"
        and LOCAL_BUNDLED_DATASET_DIR.exists()
        and settings.dataset.source_ref in {LOCAL_BUNDLED_SOURCE_REF, "main"}
    )
    if st.session_state.current_deck is not None:
        current = st.session_state.current_deck
        if current.metadata.source_ref == settings.dataset.source_ref and (
            not bundled_source_expected or current.metadata.source_repo == LOCAL_BUNDLED_SOURCE_REPO
        ):
            return current
    dataset_manager = build_dataset_manager(settings)
    try:
        deck = dataset_manager.ensure_dataset(source_ref=settings.dataset.source_ref)
    except Exception as exc:
        st.error(f"Dataset error: {exc}")
        st.stop()
    st.session_state.current_deck = deck
    if st.session_state.selected_card_id is None and deck.cards:
        st.session_state.selected_card_id = _pick_random_card_id(deck.cards)
    return deck


def _record_system_load_sample() -> None:
    now = time.monotonic()
    last_sample_at = float(st.session_state.get("resource_last_sample_at", 0.0))
    if st.session_state.get("resource_history") and now - last_sample_at < RESOURCE_SAMPLE_INTERVAL_SECONDS:
        return
    history = list(st.session_state.get("resource_history", []))
    sample = get_system_load_sample()
    history.append(
        {
            "cpu_percent": sample.cpu_percent,
            "ram_percent": sample.ram_percent,
            "gpu_percent": sample.gpu_percent,
            "vram_percent": sample.vram_percent,
        }
    )
    st.session_state.resource_history = history[-RESOURCE_HISTORY_LIMIT:]
    st.session_state.resource_last_sample_at = now


def _render_header(runtime: RuntimeDiagnostics, logger: RunLogger | None) -> None:
    gpu_label = runtime.system_gpu_name or "local runtime"
    if runtime.system_gpu_present and runtime.any_backend_gpu_ready():
        subtitle = f"{gpu_label} ready. Compact local arena for cluer, guesser, and judge."
    elif runtime.system_gpu_present:
        subtitle = f"{gpu_label} detected, but the current backend mix is partly CPU-bound."
    else:
        subtitle = "Compact single-machine benchmark dashboard."

    left_col, right_col = st.columns([1.5, 1.2], vertical_alignment="center")
    with left_col:
        st.markdown(
            "".join(
                [
                    "<div class='hero-strip'>",
                    "<div class='hero-title'>taboo-arena</div>",
                    f"<div class='hero-subtitle'>{html.escape(subtitle)}</div>",
                    "</div>",
                ]
            ),
            unsafe_allow_html=True,
        )
    with right_col:
        if _active_run_present():
            _render_live_resource_cards()
        else:
            _render_resource_cards()


def _render_resource_cards() -> None:
    history = list(st.session_state.get("resource_history", []))
    cpu_col, gpu_col = st.columns(2)
    with cpu_col:
        _render_resource_card(
            primary_label="CPU",
            primary_value=_latest_history_value(history, "cpu_percent"),
            secondary_label="RAM",
            secondary_value=_latest_history_value(history, "ram_percent"),
            primary_series=_history_series(history, "cpu_percent"),
            secondary_series=_history_series(history, "ram_percent"),
        )
    with gpu_col:
        _render_resource_card(
            primary_label="GPU",
            primary_value=_latest_history_value(history, "gpu_percent"),
            secondary_label="VRAM",
            secondary_value=_latest_history_value(history, "vram_percent"),
            primary_series=_history_series(history, "gpu_percent"),
            secondary_series=_history_series(history, "vram_percent"),
        )


@st.fragment(run_every=LIVE_PANEL_REFRESH_SECONDS)
def _render_live_resource_cards() -> None:
    _poll_active_job_updates()
    _record_system_load_sample()
    _render_resource_cards()


def _render_resource_card(
    *,
    primary_label: str,
    primary_value: float | None,
    secondary_label: str,
    secondary_value: float | None,
    primary_series: list[float],
    secondary_series: list[float],
) -> None:
    svg = _sparkline_svg(
        primary_series=primary_series,
        secondary_series=secondary_series,
    )
    primary_text = "n/a" if primary_value is None else f"{primary_value:.0f}%"
    secondary_text = "n/a" if secondary_value is None else f"{secondary_value:.0f}%"
    st.markdown(
        "".join(
            [
                "<div class='resource-card'>",
                "<div class='resource-body'>",
                f"<div class='resource-spark'>{svg}</div>",
                "<div class='resource-values'>",
                f"<span>{html.escape(primary_label)}</span><strong>{html.escape(primary_text)}</strong>",
                f"<span>{html.escape(secondary_label)}</span><strong>{html.escape(secondary_text)}</strong>",
                "</div>",
                "</div>",
                "</div>",
            ]
        ),
        unsafe_allow_html=True,
    )


def _history_series(history: list[dict[str, Any]], key: str) -> list[float]:
    if not history:
        return [0.0, 0.0]
    values: list[float] = []
    last_known = 0.0
    for sample in history:
        raw_value = sample.get(key)
        if raw_value is None:
            values.append(last_known)
            continue
        last_known = float(raw_value)
        values.append(last_known)
    return values[-RESOURCE_HISTORY_LIMIT:]


def _latest_history_value(history: list[dict[str, Any]], key: str) -> float | None:
    for sample in reversed(history):
        raw_value = sample.get(key)
        if raw_value is not None:
            return float(raw_value)
    return None


def _sparkline_svg(
    *,
    primary_series: list[float],
    secondary_series: list[float],
) -> str:
    width = 300
    height = 52
    primary_path = _sparkline_path(primary_series, width=width, height=height)
    secondary_path = _sparkline_path(secondary_series, width=width, height=height)
    return (
        f"<svg viewBox='0 0 {width} {height}' preserveAspectRatio='none' "
        "xmlns='http://www.w3.org/2000/svg' aria-hidden='true'>"
        f"<path d='{secondary_path}' fill='none' stroke='#2ca25f' stroke-width='3' stroke-linecap='round'/>"
        f"<path d='{primary_path}' fill='none' stroke='#d74e3a' stroke-width='3' stroke-linecap='round'/>"
        "</svg>"
    )


def _sparkline_path(values: list[float], *, width: int, height: int) -> str:
    if not values:
        return f"M 0 {height / 2:.2f} L {width:.2f} {height / 2:.2f}"
    if len(values) == 1:
        y_value = height - (max(0.0, min(values[0], 100.0)) / 100.0) * height
        return f"M 0 {y_value:.2f} L {width:.2f} {y_value:.2f}"

    points: list[str] = []
    step = width / (len(values) - 1)
    for index, raw_value in enumerate(values):
        clamped = max(0.0, min(float(raw_value), 100.0))
        x_value = step * index
        y_value = height - (clamped / 100.0) * height
        prefix = "M" if index == 0 else "L"
        points.append(f"{prefix} {x_value:.2f} {y_value:.2f}")
    return " ".join(points)


def _compact_identifier(value: str) -> str:
    if value in {"", "n/a"}:
        return value
    return value if len(value) <= 16 else f"{value[:8]}...{value[-6:]}"


def _render_runtime_banner(
    runtime: RuntimeDiagnostics,
    memory_snapshot: SystemMemorySnapshot,
) -> None:
    gpu_name = runtime.system_gpu_name or "Unknown GPU"
    vram_summary = (
        f"VRAM total={_format_gb(memory_snapshot.vram_total_gb)}; "
        f"currently free={_format_gb(memory_snapshot.vram_available_gb)}"
    )
    torch_summary = (
        f"Transformers runtime: torch {runtime.torch_version or 'not installed'}; "
        f"CUDA built={runtime.torch_cuda_built}; CUDA available={runtime.torch_cuda_available}"
    )
    gguf_summary = (
        "llama.cpp runtime: "
        f"installed={runtime.llama_cpp_installed}; gpu_offload={runtime.llama_cpp_gpu_offload}"
    )

    if runtime.system_gpu_present and runtime.any_backend_gpu_ready():
        st.success(
            f"System GPU detected: {gpu_name}. "
            f"Available backend acceleration: transformers={runtime.transformers_gpu_ready()}, "
            f"gguf={runtime.gguf_gpu_ready()}."
        )
        st.caption(f"{vram_summary} | {torch_summary} | {gguf_summary}")
        return

    if runtime.system_gpu_present:
        st.warning(
            f"System GPU detected: {gpu_name}, but the current app environment is CPU-only "
            "for the installed model backends."
        )
        st.caption(
            f"{vram_summary} | {torch_summary} | {gguf_summary}. "
            "The app can only use GPU when the installed backend builds support CUDA/offload."
        )
        return

    st.info("No supported GPU runtime was detected in this environment.")
    st.caption(f"{vram_summary} | {torch_summary} | {gguf_summary}")


def _render_top_bar(
    registry: ModelRegistry,
    model_manager: ModelManager,
    memory_snapshot: SystemMemorySnapshot,
    logger: RunLogger | None,
) -> dict[str, ModelEntry]:
    entries = registry.list_entries(show_gated=bool(st.session_state.show_gated_models))
    if not entries:
        st.error("No model registry entries are available.")
        st.stop()

    default_ids = _ensure_default_model_ids(entries, memory_snapshot)
    top_left, top_mid, top_right = st.columns(3)

    selected_models: dict[str, ModelEntry] = {}
    target_vram_gb = memory_snapshot.vram_total_gb or memory_snapshot.vram_available_gb
    for label, key, container in [
        ("Cluer", "cluer_model_id", top_left),
        ("Guesser", "guesser_model_id", top_mid),
        ("Judge", "judge_model_id", top_right),
    ]:
        role = cast(RoleName, label.lower())
        role_entries = registry.list_entries(
            show_gated=bool(st.session_state.show_gated_models),
            role=role,
        )
        selected_id = st.session_state.get(key, default_ids[label.lower()])
        option_ids = [entry.id for entry in role_entries]
        if selected_id not in option_ids:
            selected_id = option_ids[0]
            st.session_state[key] = selected_id
        with container:
            with st.container(border=True):
                initial_state = _build_role_console_state(
                    role=role,
                    entry=registry.get(selected_id),
                    model_manager=model_manager,
                    logger=logger,
                )
                st.markdown(
                    _role_header_html(role=role, mode=initial_state.mode),
                    unsafe_allow_html=True,
                )
                picker_col, settings_col = st.columns([1, 0.1], vertical_alignment="center")
                with picker_col:
                    chosen_id = st.selectbox(
                        f"{label} model",
                        options=option_ids,
                        index=option_ids.index(selected_id),
                        format_func=lambda item_id: registry.get(item_id).display_name,
                        key=f"{key}_widget",
                        width="stretch",
                        label_visibility="collapsed",
                    )
                st.session_state[key] = chosen_id
                entry = registry.get(chosen_id)
                sync_selected_model_generation_defaults(
                    st.session_state,
                    {role: entry},
                    target_vram_gb=target_vram_gb,
                )
                with settings_col:
                    role_settings_clicked = st.button(
                        "⚙",
                        key=f"{role}_settings_button",
                        help=f"Open {label.lower()} model settings",
                        width="content",
                    )
                if role_settings_clicked:
                    _render_role_settings_dialog(
                        role=role,
                        entry=entry,
                        model_manager=model_manager,
                        target_vram_gb=target_vram_gb,
                    )
                selected_models[label.lower()] = entry
                console_state = _build_role_console_state(
                    role=role,
                    entry=entry,
                    model_manager=model_manager,
                    logger=logger,
                )
                st.caption(_model_status_text(entry, model_manager))
                st.caption(console_state.note)
    return selected_models


def _ensure_default_model_ids(
    entries: list[ModelEntry],
    memory_snapshot: SystemMemorySnapshot,
) -> dict[str, str]:
    defaults = {}
    prefer_gpu = st.session_state.device_preference != "cpu"
    available_vram_gb = memory_snapshot.vram_total_gb or memory_snapshot.vram_available_gb
    for role in ["cluer", "guesser", "judge"]:
        defaults[role] = choose_default_model_id(
            entries,
            role=cast(RoleName, role),
            available_vram_gb=available_vram_gb,
            prefer_gpu=prefer_gpu,
        )
    for role in ["cluer", "guesser", "judge"]:
        key = f"{role}_model_id"
        st.session_state.setdefault(key, defaults[role])
    return defaults


def _model_status_text(entry: ModelEntry, model_manager: ModelManager) -> str:
    size_text = (
        f"Model est. VRAM: {entry.estimated_vram_gb:.1f} GB"
        if entry.estimated_vram_gb is not None
        else "Model est. VRAM: unknown"
    )
    if model_manager.inspect_cache(entry):
        return f"Install status: cached | {size_text}"
    if entry.gated:
        return f"Install status: gated, download on use | {size_text}"
    return f"Install status: download on use | {size_text}"


@st.dialog("Model settings")
def _render_role_settings_dialog(
    *,
    role: RoleName,
    entry: ModelEntry,
    model_manager: ModelManager,
    target_vram_gb: float | None,
) -> None:
    label = role.capitalize()
    sync_selected_model_generation_defaults(
        st.session_state,
        {role: entry},
        target_vram_gb=target_vram_gb,
    )
    defaults = recommended_generation_defaults(role, entry, target_vram_gb=target_vram_gb)
    st.markdown(f"**{label} model**")
    st.caption(entry.display_name)
    st.caption(_model_status_text(entry, model_manager))
    st.caption(
        "Recommended defaults: "
        f"temp {defaults.temperature:.2f}, top_p {defaults.top_p:.2f}, max_tokens {defaults.max_tokens}"
    )
    applied = applied_generation_params(st.session_state, role)
    st.caption(
        "Applied now: "
        f"temp {applied.temperature:.2f}, top_p {applied.top_p:.2f}, max_tokens {applied.max_tokens}"
    )
    st.caption(generation_status_text(st.session_state, role, entry, target_vram_gb=target_vram_gb))
    if st.button(
        "Use recommended defaults",
        key=f"{role}_dialog_apply_defaults",
        width="stretch",
    ):
        sync_selected_model_generation_defaults(
            st.session_state,
            {role: entry},
            force=True,
            target_vram_gb=target_vram_gb,
        )
        sync_generation_widget_state(st.session_state, role)
        st.rerun()
    _render_generation_inputs(label, role, entry, st, target_vram_gb=target_vram_gb)


@st.dialog("Arena settings")
def _render_app_settings_dialog(
    *,
    selected_models: dict[str, ModelEntry],
    model_manager: ModelManager,
) -> None:
    device_options = ["auto", "cpu", "cuda"]
    memory_policy_options = [
        "keep_loaded_if_possible",
        "keep_cpu_offloaded_if_possible",
        "sequential_load_unload",
    ]
    settings_col_a, settings_col_b = st.columns(2)
    with settings_col_a:
        max_guess_attempts = st.number_input(
            "Guess attempts",
            min_value=1,
            max_value=10,
            value=int(st.session_state.max_guess_attempts),
            step=1,
            format="%d",
        )
        max_clue_repairs = st.number_input(
            "Max clue repairs",
            min_value=1,
            max_value=10,
            value=int(st.session_state.max_clue_repairs),
            step=1,
            format="%d",
        )
        block_on_uncertain = st.checkbox(
            "Block on uncertain",
            value=bool(st.session_state.block_on_uncertain),
        )
        device_preference = st.selectbox(
            "Device preference",
            options=device_options,
            index=device_options.index(str(st.session_state.device_preference)),
        )
        memory_policy = st.selectbox(
            "Memory policy",
            options=memory_policy_options,
            index=memory_policy_options.index(str(st.session_state.memory_policy)),
        )
    with settings_col_b:
        show_hidden_repairs = st.checkbox(
            "Show hidden repairs",
            value=bool(st.session_state.show_hidden_repairs),
        )
        allow_same_model_for_multiple_roles = st.checkbox(
            "Allow same model for multiple roles",
            value=bool(st.session_state.allow_same_model_for_multiple_roles),
        )
        show_gated_models = st.checkbox(
            "Show gated models",
            value=bool(st.session_state.show_gated_models),
        )
        console_trace = st.checkbox(
            "Console trace to terminal",
            value=bool(st.session_state.console_trace),
        )
        debug_show_target = st.checkbox(
            "Debug: show target word",
            value=bool(st.session_state.debug_show_target),
        )

    with st.expander("Validator rules", expanded=False):
        validator_col_a, validator_col_b = st.columns(2)
        with validator_col_a:
            validator_min_token_count = st.number_input(
                "Validator min tokens",
                min_value=1,
                max_value=10,
                value=int(st.session_state.validator_min_token_count),
                step=1,
                format="%d",
            )
            validator_check_substring = st.checkbox(
                "Check substring",
                value=bool(st.session_state.validator_check_substring),
            )
            validator_check_token_match = st.checkbox(
                "Check token match",
                value=bool(st.session_state.validator_check_token_match),
            )
            validator_check_whole_word = st.checkbox(
                "Check whole word",
                value=bool(st.session_state.validator_check_whole_word),
            )
        with validator_col_b:
            validator_similarity_threshold = st.slider(
                "Similarity threshold",
                min_value=50,
                max_value=100,
                value=int(st.session_state.validator_similarity_threshold),
            )
            validator_check_stemming = st.checkbox(
                "Check stemming",
                value=bool(st.session_state.validator_check_stemming),
            )
            validator_check_similarity = st.checkbox(
                "Check clue similarity",
                value=bool(st.session_state.validator_check_similarity),
            )

    st.session_state.max_guess_attempts = int(max_guess_attempts)
    st.session_state.max_clue_repairs = int(max_clue_repairs)
    st.session_state.block_on_uncertain = bool(block_on_uncertain)
    st.session_state.device_preference = cast(Any, device_preference)
    st.session_state.memory_policy = cast(Any, memory_policy)
    st.session_state.show_hidden_repairs = bool(show_hidden_repairs)
    st.session_state.allow_same_model_for_multiple_roles = bool(allow_same_model_for_multiple_roles)
    st.session_state.show_gated_models = bool(show_gated_models)
    st.session_state.console_trace = bool(console_trace)
    st.session_state.debug_show_target = bool(debug_show_target)
    st.session_state.validator_min_token_count = int(validator_min_token_count)
    st.session_state.validator_check_substring = bool(validator_check_substring)
    st.session_state.validator_check_token_match = bool(validator_check_token_match)
    st.session_state.validator_check_whole_word = bool(validator_check_whole_word)
    st.session_state.validator_similarity_threshold = int(validator_similarity_threshold)
    st.session_state.validator_check_stemming = bool(validator_check_stemming)
    st.session_state.validator_check_similarity = bool(validator_check_similarity)

    _render_memory_fit_summary(
        selected_models=selected_models,
        model_manager=model_manager,
        memory_snapshot=get_memory_snapshot(),
    )


def _build_role_console_state(
    *,
    role: RoleName,
    entry: ModelEntry,
    model_manager: ModelManager,
    logger: RunLogger | None,
) -> RoleConsoleState:
    loaded = model_manager.is_loaded(
        entry,
        device_preference=st.session_state.device_preference,
    )
    mode: Literal["sleep", "idle", "thinking"] = "idle" if loaded else "sleep"
    note = "Loaded and waiting for the next turn." if loaded else "Model not loaded yet."

    logger_events = [] if logger is None else logger.snapshot_events()
    if logger_events:
        last_event = logger_events[-1]
        event_type = str(last_event.get("event_type", ""))
        event_model_id = str(last_event.get("model_id", ""))
        active_role = _active_role_from_event(last_event)

        if event_type == "model_download_started" and event_model_id == entry.id:
            mode = "sleep"
            note = "Downloading model files from Hugging Face..."
        elif event_type == "model_load_started" and event_model_id == entry.id:
            mode = "sleep"
            note = "Loading weights into RAM / VRAM..."
        elif event_type in {"model_download_finished", "model_load_finished"} and event_model_id == entry.id:
            mode = "idle"
            note = "Model is ready."
        elif active_role == role and _active_run_present():
            mode = "thinking"
            note = _thinking_note_for_role(role, last_event)
        elif loaded:
            note = "Loaded and ready."
        elif model_manager.inspect_cache(entry):
            note = "Cached locally. It will wake on demand."
        else:
            note = "Sleeping until first use."

    return RoleConsoleState(
        mode=mode,
        note=note,
    )


def _active_role_from_event(event: dict[str, Any]) -> RoleName | None:
    state = str(event.get("state", ""))
    if state == "generating_clue":
        return "cluer"
    if state == "llm_validation":
        return "judge"
    if state == "generating_guess":
        return "guesser"
    return None


def _thinking_note_for_role(role: RoleName, event: dict[str, Any]) -> str:
    if role == "cluer":
        return "Composing a clue draft..."
    if role == "judge":
        return "Reviewing the clue for taboo violations..."
    return "Turning the clue history into a guess..."


def _role_header_html(*, role: RoleName, mode: Literal["sleep", "idle", "thinking"]) -> str:
    role_names = {
        "cluer": "Cluer",
        "guesser": "Guesser",
        "judge": "Judge",
    }
    avatars = {
        "cluer": "🗣️",
        "guesser": "🕵️",
        "judge": "⚖️",
    }
    return "".join(
        [
            "<div class='role-head'>",
            "<div class='role-meta'>",
            f"<div class='role-name'>{html.escape(role_names[role])}</div>",
            f"<div class='role-state'>{html.escape(mode)}</div>",
            "</div>",
            (
                f"<div class='role-avatar avatar-{role} avatar-{mode}'>"
                f"{html.escape(avatars[role])}</div>"
            ),
            "</div>",
        ]
    )


def _render_advanced_settings(
    registry: ModelRegistry,
    deck: Any,
    selected_models: dict[str, ModelEntry],
    model_manager: ModelManager,
    memory_snapshot: SystemMemorySnapshot,
) -> None:
    target_vram_gb = memory_snapshot.vram_total_gb or memory_snapshot.vram_available_gb
    with st.expander("Advanced settings", expanded=False):
        settings_col_a, settings_col_b, settings_col_c = st.columns(3)
        with settings_col_a:
            st.number_input("Max clue repairs", min_value=1, max_value=10, key="max_clue_repairs")
            st.checkbox("block_on_uncertain", key="block_on_uncertain")
            st.selectbox("Language", options=["en"], key="language")
            st.checkbox("Show gated models", key="show_gated_models")
            if st.session_state.show_gated_models:
                st.caption("Gated entries may require a Hugging Face token and approved access.")
            st.selectbox(
                "Memory policy",
                options=[
                    "keep_loaded_if_possible",
                    "keep_cpu_offloaded_if_possible",
                    "sequential_load_unload",
                ],
                key="memory_policy",
            )
        with settings_col_b:
            st.selectbox("Device preference", options=["auto", "cpu", "cuda"], key="device_preference")
            st.checkbox("Show hidden repairs", key="show_hidden_repairs")
            st.checkbox(
                "Allow same model for multiple roles",
                key="allow_same_model_for_multiple_roles",
            )
            st.checkbox("Console trace to terminal", key="console_trace")
            st.checkbox("Debug: show target word", key="debug_show_target")
            st.text_input("Dataset source_ref", key="source_ref")
        with settings_col_c:
            st.text_input("Log directory", key="log_dir")
            st.number_input(
                "Validator min tokens",
                min_value=1,
                max_value=10,
                key="validator_min_token_count",
            )
            st.slider(
                "Similarity threshold",
                min_value=50,
                max_value=100,
                key="validator_similarity_threshold",
            )
            st.checkbox("Check substring", key="validator_check_substring")
            st.checkbox("Check token match", key="validator_check_token_match")
            st.checkbox("Check whole word", key="validator_check_whole_word")
            st.checkbox("Check stemming", key="validator_check_stemming")
            st.checkbox("Check clue similarity", key="validator_check_similarity")

        _render_memory_fit_summary(
            selected_models=selected_models,
            model_manager=model_manager,
            memory_snapshot=memory_snapshot,
        )

        st.markdown("**Per-role generation**")
        action_col, _ = st.columns([1.2, 4.8])
        with action_col:
            if st.button(
                "Use selected role/model defaults",
                key="apply_selected_model_defaults",
                width="stretch",
            ):
                sync_selected_model_generation_defaults(
                    st.session_state,
                    selected_models,
                    force=True,
                    target_vram_gb=target_vram_gb,
                )
        cluer_col, guesser_col, judge_col = st.columns(3)
        _render_generation_inputs(
            "Cluer",
            "cluer",
            selected_models["cluer"],
            cluer_col,
            target_vram_gb=target_vram_gb,
        )
        _render_generation_inputs(
            "Guesser",
            "guesser",
            selected_models["guesser"],
            guesser_col,
            target_vram_gb=target_vram_gb,
        )
        _render_generation_inputs(
            "Judge",
            "judge",
            selected_models["judge"],
            judge_col,
            target_vram_gb=target_vram_gb,
        )

        with st.expander("Batch lab (parked for now)", expanded=False):
            batch_col_a, batch_col_b, batch_col_c = st.columns(3)
            all_entries = registry.list_entries(show_gated=bool(st.session_state.show_gated_models))
            option_ids = [entry.id for entry in all_entries]
            with batch_col_a:
                st.multiselect(
                    "Batch cluer models",
                    options=option_ids,
                    default=st.session_state.batch_cluer_ids or [st.session_state.cluer_model_id],
                    format_func=lambda item_id: registry.get(item_id).display_name,
                    key="batch_cluer_ids",
                )
                st.number_input(
                    "Batch sample size",
                    min_value=1,
                    max_value=max(1, len(deck.cards)),
                    key="batch_sample_size",
                )
            with batch_col_b:
                st.multiselect(
                    "Batch guesser models",
                    options=option_ids,
                    default=st.session_state.batch_guesser_ids or [st.session_state.guesser_model_id],
                    format_func=lambda item_id: registry.get(item_id).display_name,
                    key="batch_guesser_ids",
                )
                st.number_input(
                    "Repeats per card",
                    min_value=1,
                    max_value=20,
                    key="batch_repeats_per_card",
                )
            with batch_col_c:
                st.multiselect(
                    "Batch judge models",
                    options=option_ids,
                    default=st.session_state.batch_judge_ids or [st.session_state.judge_model_id],
                    format_func=lambda item_id: registry.get(item_id).display_name,
                    key="batch_judge_ids",
                )
                st.multiselect(
                    "Fixed card subset",
                    options=[card.id for card in deck.cards],
                    default=st.session_state.batch_fixed_card_ids,
                    key="batch_fixed_card_ids",
                )

        st.markdown("**Custom model entry**")
        custom_col_a, custom_col_b, custom_col_c = st.columns(3)
        with custom_col_a:
            repo_id = st.text_input("Custom repo_id", key="custom_repo_id")
            display_name = st.text_input("Custom display_name", key="custom_display_name")
            backend = st.selectbox(
                "Custom backend",
                options=["transformers_safetensors", "llama_cpp_gguf"],
                key="custom_backend",
            )
            architecture_family = st.text_input("Architecture family", key="custom_architecture_family")
        with custom_col_b:
            revision = st.text_input("Revision", key="custom_revision")
            gguf_filename = st.text_input("GGUF filename", key="custom_filename")
            tokenizer_repo = st.text_input("Tokenizer repo", key="custom_tokenizer_repo")
            chat_template_id = st.selectbox(
                "Chat template",
                options=[
                    "qwen_chatml",
                    "mistral_inst",
                    "llama3_chat",
                    "gemma_chat",
                    "phi_chat",
                    "generic_completion",
                ],
                key="custom_chat_template_id",
            )
            prompt_profile_id = st.selectbox(
                "Prompt profile",
                options=["compact_small", "standard", "strict_judge"],
                key="custom_prompt_profile_id",
            )
        with custom_col_c:
            supports_system_prompt = st.checkbox(
                "Supports system prompt",
                value=True,
                key="custom_supports_system_prompt",
            )
            estimated_vram_gb = st.number_input(
                "Estimated model VRAM GB",
                min_value=0.0,
                value=4.0,
                step=0.5,
                key="custom_estimated_vram_gb",
            )
            st.caption(
                f"Detected system VRAM: {_format_gb(memory_snapshot.vram_total_gb)} total, "
                f"{_format_gb(memory_snapshot.vram_available_gb)} currently free. "
                "This field is only for the custom model's own footprint estimate."
            )
            roles_supported = st.multiselect(
                "Roles supported",
                options=["cluer", "guesser", "judge"],
                default=["cluer", "guesser", "judge"],
                key="custom_roles_supported",
            )
            gated = st.checkbox("Gated / auth required", key="custom_gated")
            prompt_override_id = st.text_input("Prompt override id", key="custom_prompt_override_id")

        if st.button("Add custom model", key="add_custom_model"):
            try:
                registry.add_custom_entry(
                    ModelEntry(
                        id=slugify(display_name or repo_id),
                        display_name=display_name or repo_id,
                        backend=cast(BackendName, backend),
                        repo_id=repo_id,
                        revision=revision or None,
                        filename=gguf_filename or None,
                        tokenizer_repo=tokenizer_repo or None,
                        architecture_family=architecture_family or "custom",
                        chat_template_id=chat_template_id,
                        prompt_profile_id=prompt_profile_id,
                        prompt_override_id=prompt_override_id or None,
                        supports_system_prompt=supports_system_prompt,
                        roles_supported=cast(
                            list[RoleName],
                            roles_supported or ["cluer", "guesser", "judge"],
                        ),
                        languages=["en"],
                        estimated_vram_gb=float(estimated_vram_gb),
                        requires_hf_auth=bool(gated),
                        gated=bool(gated),
                        source="custom",
                    )
                )
                st.success("Custom model saved. It will appear in the selectors on the next rerun.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not save custom model: {exc}")


def _render_generation_inputs(
    label: str,
    key_prefix: str,
    entry: ModelEntry,
    container: Any | None = None,
    *,
    target_vram_gb: float | None = None,
) -> None:
    if container is not None and hasattr(container, "__enter__") and hasattr(container, "__exit__"):
        with container:
            _render_generation_inputs_body(
                label=label,
                key_prefix=key_prefix,
                entry=entry,
                target_vram_gb=target_vram_gb,
            )
        return
    _render_generation_inputs_body(
        label=label,
        key_prefix=key_prefix,
        entry=entry,
        target_vram_gb=target_vram_gb,
    )


def _render_generation_inputs_body(
    *,
    label: str,
    key_prefix: str,
    entry: ModelEntry,
    target_vram_gb: float | None = None,
) -> None:
    """Render generation controls in the current Streamlit context."""
    st.markdown(f"**{label}**")
    role = cast(RoleName, key_prefix)
    defaults = recommended_generation_defaults(role, entry, target_vram_gb=target_vram_gb)
    sync_generation_widget_state(st.session_state, role)
    st.caption(
        "Recommended defaults: "
        f"temp {defaults.temperature:.2f}, top_p {defaults.top_p:.2f}, max_tokens {defaults.max_tokens}"
    )
    st.caption(generation_status_text(st.session_state, role, entry, target_vram_gb=target_vram_gb))
    st.slider(
        "temperature",
        min_value=0.0,
        max_value=1.5,
        step=0.05,
        key=generation_widget_key(role, "temperature"),
        on_change=_apply_generation_widget_change,
        args=(role,),
    )
    st.slider(
        "top_p",
        min_value=0.1,
        max_value=1.0,
        step=0.05,
        key=generation_widget_key(role, "top_p"),
        on_change=_apply_generation_widget_change,
        args=(role,),
    )
    st.number_input(
        "max_tokens",
        min_value=16,
        max_value=1024,
        key=generation_widget_key(role, "max_tokens"),
        on_change=_apply_generation_widget_change,
        args=(role,),
    )


def _render_memory_fit_summary(
    *,
    selected_models: dict[str, ModelEntry],
    model_manager: ModelManager,
    memory_snapshot: SystemMemorySnapshot,
) -> None:
    assessment = model_manager.assess_memory(
        list(selected_models.values()),
        requested_policy=st.session_state.memory_policy,
        device_preference=st.session_state.device_preference,
    )
    st.markdown("**Runtime fit**")
    fit_items = [
        ("VRAM total", _format_gb(memory_snapshot.vram_total_gb, decimals=1)),
        ("VRAM free", _format_gb(memory_snapshot.vram_available_gb, decimals=1)),
        ("Models est.", _format_gb(assessment.estimated_vram_gb, decimals=1)),
        ("Peak est.", _format_gb(assessment.peak_active_vram_gb, decimals=1)),
    ]
    fit_cards = "".join(
        (
            "<div class='fit-card'>"
            f"<div class='fit-card-label'>{html.escape(label)}</div>"
            f"<div class='fit-card-value'>{html.escape(value)}</div>"
            "</div>"
        )
        for label, value in fit_items
    )
    st.markdown(f"<div class='fit-grid'>{fit_cards}</div>", unsafe_allow_html=True)

    summary_text = (
        f"Requested policy: `{assessment.requested_policy}`. "
        f"Recommended: `{assessment.recommended_policy}`. {assessment.reason}"
    )
    if assessment.fits_requested_policy:
        st.info(summary_text)
    else:
        st.warning(summary_text)


def _apply_generation_widget_change(role: RoleName) -> None:
    apply_generation_widget_state(st.session_state, role)


def _format_gb(value: float | None, *, decimals: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{decimals}f} GB"


def _render_backstage(
    registry: ModelRegistry,
    deck: Any,
    selected_models: dict[str, ModelEntry],
    model_manager: ModelManager,
    runtime_diagnostics: RuntimeDiagnostics,
    memory_snapshot: SystemMemorySnapshot,
) -> None:
    current_logger = cast(RunLogger | None, st.session_state.current_logger)
    current_card = _selected_card(deck)
    with st.expander("Workbench + diagnostics", expanded=False):
        _render_runtime_banner(runtime_diagnostics, memory_snapshot)
        _render_status_strip(current_logger)
        _render_live_activity_panel()
        if current_card is not None:
            st.markdown("**Current card metadata**")
            st.json(
                {
                    "id": current_card.id,
                    "category": current_card.category_label or current_card.source_category,
                    "difficulty": current_card.difficulty,
                    "source_repo": current_card.source_repo,
                    "source_ref": current_card.source_ref,
                    "source_commit": current_card.source_commit,
                    "language": current_card.lang,
                }
            )
    _render_advanced_settings(registry, deck, selected_models, model_manager, memory_snapshot)


def _render_status_strip(logger: RunLogger | None) -> None:
    metrics = _status_metrics(logger)
    cols = st.columns(6)
    labels = [
        ("run id", metrics["run_id"]),
        ("round id", metrics["round_id"]),
        ("guess attempt", metrics["attempt"]),
        ("clue repair", metrics["repair"]),
        ("state", metrics["state"]),
        ("latency", metrics["latency"]),
    ]
    for col, (label, value) in zip(cols, labels, strict=True):
        with col:
            st.markdown(
                f"<div class='metric-label'>{label}</div><div class='metric-value'>{value}</div>",
                unsafe_allow_html=True,
            )


def _status_metrics(logger: RunLogger | None) -> dict[str, str]:
    if logger is None:
        return {
            "run_id": "n/a",
            "round_id": "n/a",
            "attempt": f"0 / {int(st.session_state.max_guess_attempts)}",
            "repair": "0",
            "state": st.session_state.current_state,
            "latency": "n/a",
        }
    events = logger.snapshot_events()
    if not events:
        return {
            "run_id": "n/a",
            "round_id": "n/a",
            "attempt": f"0 / {int(st.session_state.max_guess_attempts)}",
            "repair": "0",
            "state": st.session_state.current_state,
            "latency": "n/a",
        }
    last_event = events[-1]
    summary = compute_summary_metrics(logger.snapshot_round_summaries(), events)
    latency_dict = summary.get("average_latency_per_role", {})
    latency_text = ", ".join(f"{role}:{value}ms" for role, value in latency_dict.items()) or "n/a"
    attempt = str(last_event.get("attempt_no", 0))
    attempt = (
        f"{attempt} / {int(st.session_state.max_guess_attempts)}"
        if attempt != "0"
        else f"0 / {int(st.session_state.max_guess_attempts)}"
    )
    return {
        "run_id": logger.run_id,
        "round_id": str(last_event.get("round_id", "n/a")),
        "attempt": attempt,
        "repair": str(last_event.get("clue_repair_no", 0)),
        "state": str(last_event.get("state", st.session_state.current_state)),
        "latency": latency_text,
    }


def _render_live_activity_panel() -> None:
    job = cast(ActiveJob | None, st.session_state.active_job)
    if job is None:
        return
    logger = job.logger
    logger_events = logger.snapshot_events()
    last_event = logger_events[-1] if logger_events else {}
    state = str(last_event.get("state", "starting"))
    event_type = str(last_event.get("event_type", "job_started"))
    elapsed_seconds = max(time.time() - job.started_at, 0.0)
    label = f"Run in progress: {state}"
    if job.completed and job.error_message is None:
        label = "Run finished"
    elif job.completed and job.error_message is not None:
        label = "Run failed"

    with st.status(label, expanded=True):
        st.write(f"Mode: {job.kind}")
        st.write(f"Latest event: `{event_type}`")
        st.write(f"Elapsed: {elapsed_seconds:.1f}s")
        if event_type == "model_load_started":
            st.info("Loading cached model weights into RAM/VRAM. This can take a while on first use.")
        elif event_type == "model_download_started":
            st.info("Downloading a missing model from Hugging Face.")
        elif state in {"generating_clue", "generating_guess", "llm_validation", "logical_validation"}:
            st.info("The round is actively generating or validating output.")
        if logger_events:
            st.dataframe(
                pd.DataFrame(logger_events[-8:]),
                width="stretch",
                hide_index=True,
            )


def _render_main_body(
    deck: Any,
    selected_models: dict[str, ModelEntry],
    model_manager: ModelManager,
) -> None:
    left_col, middle_col, right_col = st.columns([2, 3, 2], gap="small")
    filtered_cards = _filtered_cards(deck)
    selected_card = _selected_card(deck)

    with left_col:
        with st.container(border=True, key="bottom_card_panel", height="stretch"):
            st.markdown("<div class='section-heading'>Taboo card</div>", unsafe_allow_html=True)
            card_showcase_col, card_action_col = st.columns([0.9, 0.1], gap="small", vertical_alignment="center")
            with card_showcase_col:
                if selected_card is None:
                    st.warning("Enable at least one category to reveal a card.")
                else:
                    st.markdown(_game_card_html(selected_card), unsafe_allow_html=True)
            random_disabled = not filtered_cards
            with card_action_col:
                if st.button(
                    "↻",
                    key="random_card_icon",
                    help="Pick a random allowed card",
                    width="stretch",
                    disabled=random_disabled,
                ):
                    st.session_state.selected_card_id = _pick_random_card_id(filtered_cards)
                    st.rerun()

            st.markdown("<div class='section-heading'>Categories</div>", unsafe_allow_html=True)
            categories = _category_options(deck)
            st.pills(
                "Categories",
                options=categories,
                selection_mode="multi",
                key="selected_categories",
                width="stretch",
                label_visibility="collapsed",
            )
            if not st.session_state.selected_categories:
                st.caption("Turn on at least one category to allow random selection and running.")

            control_bottom_left, control_bottom_right = st.columns([1, 0.12], vertical_alignment="center")
            run_disabled = selected_card is None or _active_run_present()
            st.session_state.start_round_clicked = control_bottom_left.button(
                "Start",
                width="stretch",
                disabled=run_disabled,
            )
            control_bottom_left.caption(f"Seed {int(st.session_state.random_seed)}")
            app_settings_clicked = control_bottom_right.button(
                "⚙",
                key="open_app_settings",
                help="Open arena settings",
                width="stretch",
            )
            if app_settings_clicked:
                _render_app_settings_dialog(
                    selected_models=selected_models,
                    model_manager=model_manager,
                )

    settings = _settings_from_session(load_app_settings())
    registry = build_registry()
    _maybe_process_actions(settings, registry, model_manager, deck, selected_models)

    session = SessionFacade(st.session_state)
    logger = session.current_logger
    current_result = session.current_result

    with middle_col:
        with st.container(border=True, key="bottom_transcript_panel", height="stretch"):
            st.markdown("<div class='section-heading'>Chat / transcript</div>", unsafe_allow_html=True)
            transcript_slot = st.empty()
            if not _active_run_present():
                with transcript_slot.container():
                    render_transcript_panel_content(
                        session=session,
                        logger=logger,
                        current_result=current_result,
                        active_run_present=False,
                    )

    with right_col:
        with st.container(border=True, key="bottom_metrics_panel", height="stretch"):
            st.markdown("<div class='section-heading'>Metrics</div>", unsafe_allow_html=True)
            metrics_slot = st.empty()
            if not _active_run_present():
                with metrics_slot.container():
                    render_round_pulse_inline(
                        session=session,
                        logger=logger,
                        selected_models=selected_models,
                        model_manager=model_manager,
                        current_events=[] if logger is None else live_logger_events(logger),
                    )
    if _active_run_present():
        _render_live_round_panels(
            transcript_slot=transcript_slot,
            metrics_slot=metrics_slot,
            selected_models=selected_models,
            model_manager=model_manager,
        )


@st.fragment(run_every=LIVE_PANEL_REFRESH_SECONDS)
def _render_live_round_panels(
    *,
    transcript_slot: Any,
    metrics_slot: Any,
    selected_models: dict[str, ModelEntry],
    model_manager: ModelManager,
) -> None:
    session = SessionFacade(st.session_state)
    _poll_active_job_updates()
    logger = session.current_logger
    current_result = session.current_result
    with transcript_slot.container():
        render_transcript_panel_content(
            session=session,
            logger=logger,
            current_result=current_result,
            active_run_present=True,
        )
    with metrics_slot.container():
        render_live_round_pulse_inline(
            session=session,
            logger=logger,
            selected_models=selected_models,
            model_manager=model_manager,
            current_events=[] if logger is None else live_logger_events(logger),
        )
    if _finalize_active_job_if_needed():
        st.rerun()

def _compact_prompt_preview(prompt_text: str) -> str:
    normalized = prompt_text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned_lines: list[str] = []
    previous_blank = False
    for raw_line in normalized.split("\n"):
        line = re.sub(r"[ \t]+$", "", raw_line).strip()
        if not line:
            if previous_blank:
                continue
            cleaned_lines.append("")
            previous_blank = True
            continue
        cleaned_lines.append(line)
        previous_blank = False
    return "\n".join(cleaned_lines).strip()


def _category_options(deck: Any) -> list[str]:
    return sorted({str(card.category_label or card.source_category) for card in deck.cards})


def _prepare_card_selection_state(deck: Any) -> None:
    selected_categories = prepare_category_selection(st.session_state, _category_options(deck))
    allowed = set(selected_categories)
    allowed_cards = [
        card
        for card in deck.cards
        if (card.category_label or card.source_category) in allowed
    ]
    if allowed_cards and st.session_state.selected_card_id not in {card.id for card in allowed_cards}:
        st.session_state.selected_card_id = _pick_random_card_id(allowed_cards)


def _filtered_cards(deck: Any) -> list[Any]:
    allowed = set(st.session_state.selected_categories)
    return [
        card
        for card in deck.cards
        if (card.category_label or card.source_category) in allowed
    ]


def _selected_card(deck: Any) -> Any | None:
    filtered_cards = _filtered_cards(deck)
    if not filtered_cards:
        return None
    for card in filtered_cards:
        if card.id == st.session_state.selected_card_id:
            return card
    st.session_state.selected_card_id = _pick_random_card_id(filtered_cards)
    for card in filtered_cards:
        if card.id == st.session_state.selected_card_id:
            return card
    return filtered_cards[0]


def _pick_random_card_id(cards: list[Any]) -> str:
    if not cards:
        raise ValueError("Cannot pick a random card from an empty list.")
    rng = _session_rng()
    return str(rng.choice(cards).id)


def _session_rng() -> random.Random:
    return cast(random.Random, st.session_state.app_randomizer)


def _display_card(deck: Any, logger: RunLogger | None) -> Any | None:
    """Prefer the live round card while a session is active, else show the selected card."""
    if logger is not None:
        latest_card_id = _latest_logger_card_id(logger)
        if latest_card_id:
            for card in deck.cards:
                if card.id == latest_card_id:
                    return card
    return _selected_card(deck)


def _latest_logger_card_id(logger: RunLogger) -> str | None:
    """Return the newest card id mentioned by the run logger."""
    for event in reversed(logger.snapshot_events()):
        card_id = str(event.get("card_id", "")).strip()
        if card_id:
            return card_id
    return None


def _game_card_html(card: Any) -> str:
    taboo_html = "".join(
        f"<div class='game-card-taboo'>{html.escape(str(word))}</div>"
        for word in card.taboo_hard
    )
    kind = "Phrase" if " " in str(card.target).strip() else "Word"
    subtitle = f"{card.category_label or card.source_category} • {card.difficulty or 'standard'}"
    return "".join(
        [
            "<div class='game-card-shell'>",
            "<div class='game-card-top'>",
            "<div class='game-card-title'>Taboo</div>",
            f"<div class='game-card-target'>{html.escape(str(card.target))}</div>",
            f"<div class='game-card-kind'>{html.escape(kind)}</div>",
            "</div>",
            f"<div class='game-card-taboo-stack'>{taboo_html}</div>",
            f"<div class='game-card-footer'>{html.escape(subtitle)}</div>",
            "</div>",
        ]
    )

def _maybe_process_actions(
    settings: AppSettings,
    registry: ModelRegistry,
    model_manager: ModelManager,
    deck: Any,
    selected_models: dict[str, ModelEntry],
) -> None:
    session = SessionFacade(st.session_state)
    if st.session_state.start_round_clicked:
        if not _validate_model_selection(selected_models):
            return
        if _active_run_present():
            st.warning("A run is already in progress.")
            return
        current_card = _selected_card(deck)
        if current_card is None:
            st.error("Enable at least one category before starting a round.")
            return
        archive_current_logger_for_transcript(session)
        logger = build_logger(
            settings.run.log_dir,
            console_trace=settings.run.console_trace,
        )
        logger.emit("app_started", state="idle")
        model_manager.logger = logger
        session.current_logger = logger
        st.session_state.current_state = "generating_clue"
        session.current_error_message = None
        st.session_state.stop_requested = False
        session.current_result = None
        session.active_job = start_single_round_job(
            settings=settings,
            model_manager=model_manager,
            logger=logger,
            card=current_card,
            cluer_entry=selected_models["cluer"],
            guesser_entry=selected_models["guesser"],
            judge_entry=selected_models["judge"],
        )
        st.rerun()

    if st.session_state.start_batch_clicked:
        if _active_run_present():
            st.warning("A run is already in progress.")
            return
        archive_current_logger_for_transcript(session)
        batch_model_ids = {
            "cluer": st.session_state.batch_cluer_ids or [st.session_state.cluer_model_id],
            "guesser": st.session_state.batch_guesser_ids or [st.session_state.guesser_model_id],
            "judge": st.session_state.batch_judge_ids or [st.session_state.judge_model_id],
        }
        if not all(batch_model_ids.values()):
            st.error("Select at least one model for each batch role.")
            return
        logger = build_logger(
            settings.run.log_dir,
            console_trace=settings.run.console_trace,
        )
        logger.emit("app_started", state="batch_running")
        cards = _batch_cards(deck)
        tasks = [
            {
                "card_id": card.id,
                "cluer_model_id": cluer_id,
                "guesser_model_id": guesser_id,
                "judge_model_id": judge_id,
            }
            for cluer_id in batch_model_ids["cluer"]
            for guesser_id in batch_model_ids["guesser"]
            for judge_id in batch_model_ids["judge"]
            for _ in range(int(st.session_state.batch_repeats_per_card))
            for card in cards
        ]
        session.current_logger = logger
        session.current_result = None
        st.session_state.current_state = "batch_running"
        session.current_error_message = None
        st.session_state.stop_requested = False
        session.active_job = start_batch_job(
            settings=settings,
            model_manager=model_manager,
            logger=logger,
            registry=registry,
            tasks=tasks,
            cards_by_id={card.id: card for card in deck.cards},
        )
        st.rerun()


def _batch_cards(deck: Any) -> list[Any]:
    fixed_ids = set(st.session_state.batch_fixed_card_ids)
    if fixed_ids:
        cards = [card for card in deck.cards if card.id in fixed_ids]
    else:
        cards = deck.cards[: int(st.session_state.batch_sample_size)]
    return cards


def _validate_model_selection(selected_models: dict[str, ModelEntry]) -> bool:
    if st.session_state.allow_same_model_for_multiple_roles:
        return True
    ids = [entry.id for entry in selected_models.values()]
    if len(ids) != len(set(ids)):
        st.error("Duplicate role models are disabled by the current advanced setting.")
        return False
    return True


def _poll_active_job_updates() -> bool:
    job = cast(ActiveJob | None, st.session_state.active_job)
    if job is None:
        return False

    changed = False
    update_queue = job.update_queue
    if update_queue is not None:
        while True:
            try:
                message = update_queue.get_nowait()
            except queue.Empty:
                break
            payload_type = str(message.get("type", ""))
            if payload_type == "event":
                event = cast(dict[str, Any], message.get("event", {}))
                if event:
                    job.logger.ingest_event(event)
                    changed = True
            elif payload_type == "summary":
                payload = message.get("summary", {})
                summary = RoundSummaryRecord.model_validate(payload)
                job.logger.ingest_round_summary(summary)
                changed = True
            elif payload_type == "result":
                result = cast(Any, message.get("result"))
                if result is not None:
                    job.result = result
                    changed = True
            elif payload_type == "error":
                job.error_message = str(message.get("error_message", "Unknown error"))
                changed = True
            elif payload_type == "completed":
                job.completed = True
                changed = True

    process = job.process
    if process is not None and not process.is_alive() and not job.completed:
        job.completed = True
        if process.exitcode not in {0, None} and job.error_message is None:
            job.error_message = f"Worker process exited with code {process.exitcode}."
        changed = True

    return changed


def _finalize_active_job_if_needed() -> bool:
    job = cast(ActiveJob | None, st.session_state.active_job)
    if job is None:
        return False
    _poll_active_job_updates()
    if not job.completed:
        return False

    if job.thread is not None:
        job.thread.join(timeout=0.1)
    if job.process is not None:
        job.process.join(timeout=0.1)
    st.session_state.current_logger = job.logger

    if job.error_message is not None:
        st.session_state.current_state = "idle"
        st.session_state.current_error_message = job.error_message
        st.session_state.active_job = None
        return True

    st.session_state.current_error_message = None
    if job.result is not None:
        st.session_state.current_result = job.result
        st.session_state.selected_card_id = job.result.card.id
        st.session_state.current_state = "round_finished"
    elif job.batch_results:
        st.session_state.current_result = job.batch_results[-1]
        st.session_state.current_state = "idle"
    else:
        st.session_state.current_state = "idle"
    st.session_state.stop_requested = False
    st.session_state.active_job = None
    return True


def _autorefresh_active_job_if_needed() -> None:
    return None


if __name__ == "__main__":
    main()
