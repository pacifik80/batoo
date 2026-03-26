from __future__ import annotations

import zipfile
from pathlib import Path

from taboo_arena.cards.dataset import DatasetManager
from taboo_arena.cards.schemas import DatasetMetadata
from taboo_arena.config import DatasetSettings


def test_import_from_directory_normalizes_cards(dataset_source_dir: Path, tmp_path: Path) -> None:
    manager = DatasetManager(DatasetSettings(), dataset_root=tmp_path)
    deck = manager.import_from_directory(
        dataset_source_dir,
        metadata=DatasetMetadata(source_repo="Kovah/Taboo-Data", source_ref="main"),
    )
    assert len(deck.cards) == 2
    assert deck.cards[0].target == "Bear"
    assert deck.cards[0].taboo_hard == ["grizzly", "honey", "pooh"]
    assert deck.cards[1].taboo_hard == ["bunny", "hop", "carrot"]


def test_download_and_import_uses_archive(dataset_source_dir: Path, tmp_path: Path, monkeypatch) -> None:
    manager = DatasetManager(DatasetSettings(source_ref="main"), dataset_root=tmp_path)
    archive_path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for path in dataset_source_dir.rglob("*"):
            if path.is_file():
                archive.write(path, arcname=str(path.relative_to(dataset_source_dir.parent)))

    def fake_download(url: str, destination: Path) -> None:
        destination.write_bytes(archive_path.read_bytes())

    monkeypatch.setattr(manager, "_download_file", fake_download)
    monkeypatch.setattr(manager, "_resolve_source_commit", lambda repo, ref: "abc123")
    deck = manager.download_and_import("main")
    cached = manager.load_cached_deck("main")
    assert deck.metadata.source_commit == "abc123"
    assert cached is not None
    assert cached.metadata.card_count == 2
