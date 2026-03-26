"""Prompt file loading and rendering."""

from __future__ import annotations

import json
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
    return get_prompt_root() / f"{prompt_id}.json"


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
