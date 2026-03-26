"""Curated and custom model registry support."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from taboo_arena.config import BackendName, GenerationParams, RoleName
from taboo_arena.utils.paths import ensure_app_dirs, get_custom_model_store_path


class ModelEntry(BaseModel):
    """One curated or custom model entry."""

    id: str
    display_name: str
    backend: BackendName
    repo_id: str
    revision: str | None = None
    filename: str | None = None
    tokenizer_repo: str | None = None
    architecture_family: str
    chat_template_id: str
    supports_system_prompt: bool
    stop_tokens: list[str] = Field(default_factory=list)
    roles_supported: list[RoleName]
    languages: list[str]
    estimated_vram_gb: float | None = None
    default_generation_params: GenerationParams = Field(default_factory=GenerationParams)
    requires_hf_auth: bool = False
    gated: bool = False
    notes: str = ""
    source: Literal["curated", "custom"] = "curated"

    @model_validator(mode="after")
    def validate_backend_specific_fields(self) -> ModelEntry:
        """Check backend-specific requirements."""
        if self.backend == "llama_cpp_gguf" and not self.filename:
            raise ValueError("GGUF model entries must provide a filename.")
        return self


class ModelRegistryPayload(BaseModel):
    """Top-level YAML payload for the curated registry."""

    models: list[ModelEntry]


class ModelRegistry:
    """Load curated registry entries plus persisted user-provided custom entries."""

    def __init__(
        self,
        registry_path: Path | None = None,
        custom_store_path: Path | None = None,
    ) -> None:
        self.registry_path = registry_path or Path(__file__).resolve().parents[3] / "model_registry.yaml"
        self.custom_store_path = custom_store_path or get_custom_model_store_path()
        ensure_app_dirs()
        self._curated_entries = self._load_curated_entries()
        self._custom_entries = self._load_custom_entries()

    def list_entries(
        self,
        *,
        show_gated: bool = False,
        role: RoleName | None = None,
        language: str = "en",
    ) -> list[ModelEntry]:
        """Return curated and custom entries filtered for the requested UI context."""
        entries = [*self._curated_entries, *self._custom_entries]
        filtered: list[ModelEntry] = []
        for entry in entries:
            if not show_gated and entry.gated:
                continue
            if role is not None and role not in entry.roles_supported:
                continue
            if language not in entry.languages:
                continue
            filtered.append(entry)
        return sorted(filtered, key=lambda item: (item.source, item.display_name.lower()))

    def get(self, model_id: str) -> ModelEntry:
        """Return one entry by id."""
        for entry in [*self._curated_entries, *self._custom_entries]:
            if entry.id == model_id:
                return entry
        raise KeyError(f"Unknown model id: {model_id}")

    def add_custom_entry(self, entry: ModelEntry) -> ModelEntry:
        """Persist a validated custom model entry."""
        custom_entry = entry.model_copy(update={"source": "custom"})
        existing_ids = {item.id for item in [*self._curated_entries, *self._custom_entries]}
        if custom_entry.id in existing_ids:
            raise ValueError(f"Model id already exists: {custom_entry.id}")
        self._custom_entries.append(custom_entry)
        self._persist_custom_entries()
        return custom_entry

    def custom_entries(self) -> list[ModelEntry]:
        """Return custom entries only."""
        return list(self._custom_entries)

    def _load_curated_entries(self) -> list[ModelEntry]:
        payload = yaml.safe_load(self.registry_path.read_text(encoding="utf-8")) or {}
        parsed = ModelRegistryPayload.model_validate(payload)
        return [entry.model_copy(update={"source": "curated"}) for entry in parsed.models]

    def _load_custom_entries(self) -> list[ModelEntry]:
        if not self.custom_store_path.exists():
            return []
        payload = json.loads(self.custom_store_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("Custom model store must contain a JSON array.")
        return [ModelEntry.model_validate({**item, "source": "custom"}) for item in payload]

    def _persist_custom_entries(self) -> None:
        serializable = [entry.model_dump(mode="json") for entry in self._custom_entries]
        self.custom_store_path.parent.mkdir(parents=True, exist_ok=True)
        self.custom_store_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

