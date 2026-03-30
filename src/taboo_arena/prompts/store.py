"""Prompt file loading and rendering."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any, Literal

from jinja2 import Environment, StrictUndefined
from pydantic import BaseModel, Field

from taboo_arena.prompts.templates import PromptMessage


class PromptFileMessage(BaseModel):
    """One message template loaded from a prompt JSON file."""

    role: Literal["system", "user", "assistant"]
    template: str = Field(min_length=1)


class PromptFileDefinition(BaseModel):
    """Prompt definition stored on disk."""

    id: str
    description: str | None = None
    messages: list[PromptFileMessage] = Field(min_length=1)


class RoleTaskSpec(BaseModel):
    """Canonical semantic task spec for one gameplay role."""

    id: str
    role_id: Literal["cluer", "guesser", "judge_clue", "judge_guess"]
    message_role: Literal["system", "user", "assistant"] = "user"
    intro: str = Field(min_length=1)
    objective_lines: list[str] = Field(default_factory=list)
    required_line_keys: list[str] = Field(default_factory=list)
    forbidden_line_keys: list[str] = Field(default_factory=list)
    advisory_line_keys: list[str] = Field(default_factory=list)
    control_field_order: list[str] = Field(default_factory=list)
    output_contract_id: str = Field(min_length=1)


class PromptProfileDefinition(BaseModel):
    """Packaging profile layered on top of role semantics."""

    id: str
    description: str | None = None
    section_style: Literal["compact_control", "sectioned_control"] = "compact_control"
    include_section_titles: bool = True
    section_labels: dict[str, str] = Field(default_factory=dict)
    control_key_aliases: dict[str, str] = Field(default_factory=dict)
    list_limits: dict[str, int] = Field(default_factory=dict)
    output_header: str = "OUTPUT_CONTRACT"
    control_header: str = "CONTROL_STATE"
    control_footer: str = "END_CONTROL_STATE"


_JINJA_ENV = Environment(
    autoescape=False,
    trim_blocks=False,
    lstrip_blocks=False,
    undefined=StrictUndefined,
)


def get_prompt_root() -> Path:
    """Return the repository prompt directory."""
    return Path(__file__).resolve().parents[3] / "prompts"


def get_prompt_path(prompt_id: str) -> Path:
    """Resolve one prompt definition path by id."""
    relative = Path(prompt_id)
    if relative.suffix != ".json":
        relative = relative.with_suffix(".json")
    return get_prompt_root() / relative


@cache
def load_prompt_definition(prompt_id: str) -> PromptFileDefinition:
    """Load and validate a prompt definition from disk."""
    path = get_prompt_path(prompt_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return PromptFileDefinition.model_validate(payload)


def render_prompt_messages(prompt_id: str, context: dict[str, Any]) -> list[PromptMessage]:
    """Render a prompt JSON definition into prompt messages."""
    definition = load_prompt_definition(prompt_id)
    rendered_messages: list[PromptMessage] = []
    for message in definition.messages:
        template = _JINJA_ENV.from_string(message.template)
        rendered_messages.append(
            PromptMessage(
                role=message.role,
                content=template.render(**context),
            )
        )
    return rendered_messages


def get_role_spec_path(spec_id: str) -> Path:
    """Resolve one role task spec path by id."""
    return get_prompt_path(f"roles/{spec_id}")


@cache
def load_role_task_spec(spec_id: str) -> RoleTaskSpec:
    """Load one canonical role task spec."""
    payload = json.loads(get_role_spec_path(spec_id).read_text(encoding="utf-8"))
    return RoleTaskSpec.model_validate(payload)


def get_profile_path(profile_id: str) -> Path:
    """Resolve one prompt-profile path by id."""
    return get_prompt_path(f"profiles/{profile_id}")


@cache
def load_prompt_profile(profile_id: str) -> PromptProfileDefinition:
    """Load one prompt profile definition."""
    payload = json.loads(get_profile_path(profile_id).read_text(encoding="utf-8"))
    return PromptProfileDefinition.model_validate(payload)


def get_fragment_path(fragment_id: str) -> Path:
    """Resolve one shared fragment path by id."""
    return get_prompt_path(f"fragments/{fragment_id}")


@cache
def load_prompt_fragment(fragment_id: str) -> Any:
    """Load one shared prompt fragment payload."""
    return json.loads(get_fragment_path(fragment_id).read_text(encoding="utf-8"))


def get_prompt_override_path(override_id: str, role_id: str) -> Path:
    """Resolve one optional per-model override prompt path."""
    return get_prompt_root() / "overrides" / override_id / f"{role_id}.json"


def render_prompt_override_messages(
    override_id: str,
    role_id: str,
    context: dict[str, Any],
) -> list[PromptMessage] | None:
    """Render one optional per-model override prompt when present."""
    path = get_prompt_override_path(override_id, role_id)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    definition = PromptFileDefinition.model_validate(payload)
    rendered_messages: list[PromptMessage] = []
    for message in definition.messages:
        template = _JINJA_ENV.from_string(message.template)
        rendered_messages.append(
            PromptMessage(
                role=message.role,
                content=template.render(**context),
            )
        )
    return rendered_messages
