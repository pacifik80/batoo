from __future__ import annotations

from pathlib import Path

import pytest

from taboo_arena.models.registry import ModelEntry, ModelRegistry


def test_registry_parses_curated_entries(tmp_path: Path) -> None:
    custom_store = tmp_path / "custom.json"
    registry = ModelRegistry(custom_store_path=custom_store)
    entries = registry.list_entries()
    assert any(entry.id == "qwen2.5-1.5b-instruct" for entry in entries)


def test_custom_model_entry_requires_gguf_filename() -> None:
    with pytest.raises(ValueError):
        ModelEntry(
            id="broken-gguf",
            display_name="Broken GGUF",
            backend="llama_cpp_gguf",
            repo_id="fake/repo",
            architecture_family="fake",
            chat_template_id="generic_completion",
            supports_system_prompt=True,
            roles_supported=["cluer"],
            languages=["en"],
        )


def test_registry_persists_custom_entries(tmp_path: Path) -> None:
    custom_store = tmp_path / "custom.json"
    registry = ModelRegistry(custom_store_path=custom_store)
    entry = ModelEntry(
        id="custom-model",
        display_name="Custom Model",
        backend="transformers_safetensors",
        repo_id="me/custom",
        architecture_family="custom",
        chat_template_id="generic_completion",
        supports_system_prompt=True,
        roles_supported=["cluer", "guesser", "judge"],
        languages=["en"],
        source="custom",
    )
    registry.add_custom_entry(entry)
    reloaded = ModelRegistry(custom_store_path=custom_store)
    assert reloaded.get("custom-model").source == "custom"

