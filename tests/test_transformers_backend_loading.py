from __future__ import annotations

from pathlib import Path

from taboo_arena.models.backends import (
    _load_pretrained_with_fallback,
    _supports_remote_code_fallback,
)
from taboo_arena.models.manager import ModelManager
from taboo_arena.models.registry import ModelEntry


def _entry() -> ModelEntry:
    return ModelEntry(
        id="phi-like-model",
        display_name="Phi-like Model",
        backend="transformers_safetensors",
        repo_id="fake/repo",
        architecture_family="phi4",
        chat_template_id="phi_chat",
        supports_system_prompt=True,
        roles_supported=["cluer", "guesser", "judge"],
        languages=["en"],
    )


def test_remote_code_fallback_is_disabled_for_local_snapshot_without_python_files(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    assert _supports_remote_code_fallback(str(tmp_path)) is False


def test_local_snapshot_without_python_files_does_not_retry_with_trust_remote_code(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    calls: list[bool] = []

    def _loader(source: str, *, trust_remote_code: bool, **_: object) -> str:
        calls.append(trust_remote_code)
        raise RuntimeError("native load failed")

    try:
        _load_pretrained_with_fallback(
            loader=_loader,
            source=str(tmp_path),
            kwargs={},
        )
    except RuntimeError as exc:
        assert str(exc) == "native load failed"
    else:
        raise AssertionError("Expected the native loader failure to be surfaced.")

    assert calls == [False]


def test_snapshot_download_includes_python_support_files(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_snapshot_download(**kwargs: object) -> str:
        captured.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr("taboo_arena.models.manager.snapshot_download", _fake_snapshot_download)

    manager = ModelManager()
    manager.ensure_model_available(_entry())

    allow_patterns = captured["allow_patterns"]
    assert isinstance(allow_patterns, list)
    assert "*.py" in allow_patterns
    assert "**/*.py" in allow_patterns
