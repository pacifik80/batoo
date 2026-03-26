from __future__ import annotations

import json
import zipfile
from pathlib import Path

from taboo_arena.cards.dataset import (
    LOCAL_BUNDLED_SOURCE_REF,
    LOCAL_BUNDLED_SOURCE_REPO,
    DatasetManager,
)
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


def test_import_from_local_bundle_uses_root_json_files_only(tmp_path: Path) -> None:
    bundled_root = tmp_path / "taboo_cards_en"
    bundled_root.mkdir()
    (bundled_root / "food_and_drink.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "category": "Food & Drink",
                "language": "en",
                "cards": [
                    {
                        "id": "food_and_drink-001",
                        "target": "Pizza",
                        "taboo": ["cheese", "slice", "crust", "delivery", "oven"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (bundled_root / "misc").mkdir()
    (bundled_root / "misc" / "ignored.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "category": "Ignored",
                "language": "en",
                "cards": [
                    {
                        "id": "ignored-001",
                        "target": "ShouldNotAppear",
                        "taboo": ["one", "two"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    manager = DatasetManager(
        DatasetSettings(source_ref=LOCAL_BUNDLED_SOURCE_REF),
        dataset_root=tmp_path / "cache",
        bundled_source_root=bundled_root,
    )
    deck = manager.ensure_dataset(source_ref=LOCAL_BUNDLED_SOURCE_REF)

    assert len(deck.cards) == 1
    assert deck.cards[0].id == "food_and_drink-001"
    assert deck.cards[0].category_label == "Food & Drink"
    assert deck.cards[0].source_repo == LOCAL_BUNDLED_SOURCE_REPO


def test_bundled_deck_overrides_stale_cached_main(tmp_path: Path) -> None:
    bundled_root = tmp_path / "taboo_cards_en"
    bundled_root.mkdir()
    (bundled_root / "everyday_life.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "category": "Everyday Life",
                "language": "en",
                "cards": [
                    {
                        "id": "everyday_life-001",
                        "target": "Umbrella",
                        "taboo": ["rain", "wet", "carry", "open", "storm"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    settings = DatasetSettings(source_ref="main")
    manager = DatasetManager(
        settings,
        dataset_root=tmp_path / "cache",
        bundled_source_root=bundled_root,
    )
    cached_dir = (tmp_path / "cache") / settings.source_repo.replace("/", "__") / "main"
    cached_dir.mkdir(parents=True)
    (cached_dir / "metadata.json").write_text(
        DatasetMetadata(source_repo=settings.source_repo, source_ref="main", card_count=1).model_dump_json(indent=2),
        encoding="utf-8",
    )
    (cached_dir / "deck.json").write_text(
        json.dumps(
            [
                {
                    "id": "stale-001",
                    "target": "OldCard",
                    "taboo_hard": ["old", "stale"],
                    "aliases": [],
                    "difficulty": None,
                    "lang": "en",
                    "source_category": "stale",
                    "source_repo": settings.source_repo,
                    "source_ref": "main",
                    "source_commit": None,
                    "category_label": "Stale",
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    deck = manager.ensure_dataset(source_ref="main")

    assert len(deck.cards) == 1
    assert deck.cards[0].target == "Umbrella"
    assert deck.cards[0].source_repo == LOCAL_BUNDLED_SOURCE_REPO
