"""Layered prompt builders shared across roles, profiles, and model families."""

from __future__ import annotations

import json
from typing import Any

from taboo_arena.cards.schemas import CardRecord
from taboo_arena.models.registry import ModelEntry
from taboo_arena.prompts.store import (
    PromptProfileDefinition,
    RoleTaskSpec,
    load_prompt_fragment,
    load_prompt_profile,
    load_role_task_spec,
    render_prompt_override_messages,
)
from taboo_arena.prompts.templates import PromptMessage

DEFAULT_PROMPT_PROFILE_ID = "standard"


def build_cluer_messages(
    card: CardRecord,
    accepted_clues: list[str],
    rejected_clues: list[str],
    wrong_guesses: list[str],
    attempt_no: int,
    repair_no: int,
    allowed_angles: list[str],
    blocked_terms: list[str],
    blocked_prior_clues: list[str],
    blocked_angles: list[str],
    repair_feedback_json: str | None = None,
    *,
    model_entry: ModelEntry | None = None,
) -> list[PromptMessage]:
    """Build prompt messages for structured clue-candidate generation."""
    return _build_role_messages(
        spec_id="cluer_base",
        control_state={
            "attempt_index": attempt_no,
            "retry_index": repair_no,
            "target": card.target,
            "forbidden_words": list(card.taboo_hard),
            "allowed_angles": list(allowed_angles),
            "blocked_terms": list(blocked_terms),
            "blocked_prior_clues": list(blocked_prior_clues),
            "blocked_angles": list(blocked_angles),
            "revision_notes": _parse_jsonish(repair_feedback_json) if repair_feedback_json else {},
            "previous_accepted_clues": list(accepted_clues),
            "previous_rejected_clues": list(rejected_clues),
            "previous_wrong_guesses": list(wrong_guesses),
        },
        model_entry=model_entry,
    )


def build_guesser_messages(
    card: CardRecord,
    current_clue: str,
    accepted_clues: list[str],
    wrong_guesses: list[str],
    attempt_no: int,
    *,
    model_entry: ModelEntry | None = None,
) -> list[PromptMessage]:
    """Build prompt messages for shortlist-style guess generation."""
    prior_clues = accepted_clues[:-1] if accepted_clues else []
    return _build_role_messages(
        spec_id="guesser_base",
        control_state={
            "attempt_index": attempt_no,
            "current_clue": current_clue,
            "prior_accepted_clues": list(prior_clues),
            "prior_wrong_guesses": list(wrong_guesses),
            "target_type_hint": card.category_label or card.source_category,
        },
        model_entry=model_entry,
    )


def build_clue_judge_messages(
    card: CardRecord,
    clue_draft: str,
    accepted_clues: list[str],
    rejected_clues: list[str],
    attempt_no: int,
    *,
    model_entry: ModelEntry | None = None,
) -> list[PromptMessage]:
    """Build prompt messages for clue-rule arbitration."""
    reason_codes = load_prompt_fragment("judge_reason_codes")
    return _build_role_messages(
        spec_id="judge_clue_base",
        control_state={
            "attempt_index": attempt_no,
            "target": card.target,
            "forbidden_words": list(card.taboo_hard),
            "selected_clue": clue_draft,
            "previous_accepted_clues": list(accepted_clues),
            "previous_rejected_clues": list(rejected_clues),
            "allowed_block_reason_codes": list(reason_codes["clue_block"]),
            "allowed_warning_codes": list(reason_codes["clue_warning"]),
        },
        model_entry=model_entry,
    )


def build_guess_judge_messages(
    card: CardRecord,
    guess_text: str,
    attempt_no: int,
    match_status: str,
    match_reason: str,
    candidate_spans: list[str],
    warnings: list[str],
    *,
    model_entry: ModelEntry | None = None,
) -> list[PromptMessage]:
    """Build prompt messages for final visible-guess arbitration."""
    reason_codes = load_prompt_fragment("judge_reason_codes")
    return _build_role_messages(
        spec_id="judge_guess_base",
        control_state={
            "attempt_index": attempt_no,
            "target": card.target,
            "visible_guess": guess_text,
            "deterministic_match_status": match_status,
            "deterministic_match_reason": match_reason,
            "candidate_answer_spans": list(candidate_spans),
            "deterministic_warnings": list(warnings),
            "allowed_reason_codes": list(reason_codes["guess"]),
        },
        model_entry=model_entry,
    )


def build_judge_messages(
    card: CardRecord,
    clue_draft: str,
    accepted_clues: list[str],
    wrong_guesses: list[str],
    attempt_no: int,
    *,
    model_entry: ModelEntry | None = None,
) -> list[PromptMessage]:
    """Compatibility wrapper for the clue-judge prompt builder."""
    return build_clue_judge_messages(
        card=card,
        clue_draft=clue_draft,
        accepted_clues=accepted_clues,
        rejected_clues=[],
        attempt_no=attempt_no,
        model_entry=model_entry,
    )


def _build_role_messages(
    *,
    spec_id: str,
    control_state: dict[str, Any],
    model_entry: ModelEntry | None,
) -> list[PromptMessage]:
    spec = load_role_task_spec(spec_id)
    override_messages = _render_override_if_present(spec, control_state, model_entry=model_entry)
    if override_messages is not None:
        return override_messages

    profile = load_prompt_profile(_prompt_profile_id(model_entry))
    message = PromptMessage(
        role=spec.message_role,
        content=_compose_prompt_text(spec, profile, control_state),
    )
    return [message]


def _render_override_if_present(
    spec: RoleTaskSpec,
    control_state: dict[str, Any],
    *,
    model_entry: ModelEntry | None,
) -> list[PromptMessage] | None:
    override_id = None if model_entry is None else model_entry.prompt_override_id
    if not override_id:
        return None
    override_context = dict(control_state)
    override_context["output_contract_json"] = _output_contract(spec.output_contract_id)
    override_context["profile_id"] = _prompt_profile_id(model_entry)
    return render_prompt_override_messages(override_id, spec.role_id, override_context)


def _compose_prompt_text(
    spec: RoleTaskSpec,
    profile: PromptProfileDefinition,
    control_state: dict[str, Any],
) -> str:
    wording = load_prompt_fragment("wording")
    blocks: list[str] = [spec.intro]

    objective_lines = [*spec.objective_lines]
    if objective_lines:
        blocks.append(_render_section(profile.section_labels.get("objective", "Objective"), objective_lines, profile))

    rule_lines = [
        *[wording[key] for key in spec.required_line_keys],
        *[wording[key] for key in spec.forbidden_line_keys],
        *[wording[key] for key in spec.advisory_line_keys],
    ]
    if rule_lines:
        blocks.append(_render_section(profile.section_labels.get("rules", "Rules"), rule_lines, profile))

    blocks.append(_render_control_state(spec, profile, control_state))
    blocks.append(f"{profile.output_header}\n{_output_contract(spec.output_contract_id)}")
    return "\n\n".join(block for block in blocks if block.strip())


def _render_section(title: str, lines: list[str], profile: PromptProfileDefinition) -> str:
    if profile.include_section_titles:
        rendered_lines = "\n".join(f"- {line}" for line in lines)
        return f"{title}\n{rendered_lines}"
    return "\n".join(lines)


def _render_control_state(
    spec: RoleTaskSpec,
    profile: PromptProfileDefinition,
    control_state: dict[str, Any],
) -> str:
    lines: list[str] = [profile.control_header]
    for field_name in spec.control_field_order:
        rendered_value = _render_control_value(
            control_state.get(field_name),
            limit=profile.list_limits.get(field_name),
            compact=profile.section_style == "compact_control",
        )
        if rendered_value is None:
            continue
        label = profile.control_key_aliases.get(field_name, field_name)
        if profile.section_style == "compact_control":
            lines.append(f"{label}={rendered_value}")
        else:
            lines.append(f"- {label}: {rendered_value}")
    if profile.control_footer:
        lines.append(profile.control_footer)
    return "\n".join(lines)


def _render_control_value(
    value: Any,
    *,
    limit: int | None,
    compact: bool,
) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        trimmed = list(value[:limit]) if limit is not None else list(value)
        return json.dumps(trimmed, ensure_ascii=True)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    text = str(value).strip()
    if not text:
        return None
    if compact:
        return json.dumps(text, ensure_ascii=True) if "\n" in text else text
    return text


def _prompt_profile_id(model_entry: ModelEntry | None) -> str:
    if model_entry is None:
        return DEFAULT_PROMPT_PROFILE_ID
    return model_entry.prompt_profile_id or DEFAULT_PROMPT_PROFILE_ID


def _output_contract(contract_id: str) -> str:
    output_contracts = load_prompt_fragment("output_contracts")
    contract_payload = output_contracts[contract_id]
    strict_json_only = str(output_contracts["strict_json_only"])
    schema_kind = str(contract_payload["schema_kind"])
    schema_example = _render_schema_example(schema_kind)
    return f"{strict_json_only}\n{schema_example}"


def _render_schema_example(schema_kind: str) -> str:
    if schema_kind == "cluer_candidates":
        angle_values = "|".join(str(item) for item in load_prompt_fragment("angle_enum")["values"])
        return json.dumps(
            {
                "candidates": [
                    {
                        "angle": angle_values,
                        "clue": "string",
                    }
                ]
            },
            ensure_ascii=True,
            separators=(",", ":"),
        )
    if schema_kind == "guesser_candidates":
        return json.dumps(
            {
                "guesses": ["string", "string", "string"],
            },
            ensure_ascii=True,
            separators=(",", ":"),
        )
    reason_codes = load_prompt_fragment("judge_reason_codes")
    if schema_kind == "judge_clue":
        return json.dumps(
            {
                "allow": True,
                "block_reason_codes": ["|".join(str(item) for item in reason_codes["clue_block"])],
                "warnings": ["|".join(str(item) for item in reason_codes["clue_warning"])],
                "matched_surface_forms": ["string"],
                "judge_version": "clue_judge_v1",
            },
            ensure_ascii=True,
            separators=(",", ":"),
        )
    if schema_kind == "judge_guess":
        return json.dumps(
            {
                "correct": False,
                "reason_codes": ["|".join(str(item) for item in reason_codes["guess"])],
                "warnings": ["string"],
                "matched_surface_forms": ["string"],
                "judge_version": "guess_judge_v1",
            },
            ensure_ascii=True,
            separators=(",", ":"),
        )
    raise KeyError(f"Unknown output contract schema kind: {schema_kind}")


def _parse_jsonish(value: str) -> Any:
    try:
        return json.loads(value)
    except Exception:
        return value
