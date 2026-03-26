from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from taboo_arena.cards.schemas import CardRecord
from taboo_arena.models.backends import GenerationResponse
from taboo_arena.models.registry import ModelEntry


@dataclass(slots=True)
class FakeModelManager:
    responses: dict[str, list[str]]
    calls: list[dict[str, Any]] | None = None
    logger: Any | None = None

    def resolve_runtime_policy(
        self,
        entries: list[ModelEntry],
        requested_policy: str,
        device_preference: str = "auto",
    ) -> str:
        return requested_policy

    def generate(
        self,
        *,
        model_entry: ModelEntry,
        messages: list[Any],
        generation_params: Any,
        runtime_policy: str,
        device_preference: str,
        trace_role: str = "unknown",
        banned_phrases: list[str] | None = None,
    ) -> GenerationResponse:
        if self.calls is not None:
            self.calls.append(
                {
                    "model_id": model_entry.id,
                    "trace_role": trace_role,
                    "banned_phrases": list(banned_phrases or []),
                }
            )
        text = self.responses[model_entry.id].pop(0)
        return GenerationResponse(
            text=text,
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=12.5,
            prompt_template_id=model_entry.chat_template_id,
        )


@pytest.fixture
def sample_card() -> CardRecord:
    return CardRecord(
        id="animals:bear:0001",
        target="Bear",
        taboo_hard=["grizzly", "honey", "pooh"],
        aliases=["Grizzly Bear"],
        source_category="animals",
        source_repo="Kovah/Taboo-Data",
        source_ref="main",
        category_label="Animals",
    )


@pytest.fixture
def sample_model_entry() -> ModelEntry:
    return ModelEntry(
        id="fake-model",
        display_name="Fake Model",
        backend="transformers_safetensors",
        repo_id="fake/repo",
        architecture_family="fake",
        chat_template_id="generic_completion",
        supports_system_prompt=True,
        roles_supported=["cluer", "guesser", "judge"],
        languages=["en"],
    )


@pytest.fixture
def dataset_source_dir(tmp_path: Path) -> Path:
    root = tmp_path / "Taboo-Data-main"
    (root / "src" / "data" / "en").mkdir(parents=True)
    (root / "src" / "data" / "categories.json").write_text(
        '{"animals":"Animals","food":"Food"}',
        encoding="utf-8",
    )
    (root / "src" / "data" / "en" / "animals.json").write_text(
        '{"Bear":["grizzly","honey","pooh"],"Rabbit":["bunny","hop","carrot",""]}',
        encoding="utf-8",
    )
    return root
