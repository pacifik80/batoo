"""Dataset download and normalization."""

from __future__ import annotations

import json
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from taboo_arena.cards.schemas import CardRecord, DatasetMetadata, ImportedDeck
from taboo_arena.config import DatasetSettings
from taboo_arena.utils.normalization import dedupe_preserve_order, slugify
from taboo_arena.utils.paths import ensure_app_dirs, get_dataset_dir


class DatasetError(RuntimeError):
    """Raised when dataset acquisition or parsing fails."""


class DatasetManager:
    """Download, import, cache, and serve the normalized card deck."""

    def __init__(self, settings: DatasetSettings, dataset_root: Path | None = None) -> None:
        self.settings = settings
        self.dataset_root = dataset_root or get_dataset_dir()
        ensure_app_dirs()

    def ensure_dataset(
        self,
        source_ref: str | None = None,
        logger: Any | None = None,
    ) -> ImportedDeck:
        """Return a cached deck or download and import it when missing."""
        ref = source_ref or self.settings.source_ref
        cached = self.load_cached_deck(ref)
        if cached is not None:
            return cached
        return self.download_and_import(ref, logger=logger)

    def download_and_import(
        self,
        source_ref: str,
        logger: Any | None = None,
    ) -> ImportedDeck:
        """Download the GitHub archive for a ref and import it into normalized cache files."""
        repo_dir = self._ref_dir(source_ref)
        raw_dir = repo_dir / "raw"
        repo_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)
        archive_path = repo_dir / "source.zip"
        archive_url = self._build_archive_url(self.settings.source_repo, source_ref)

        if logger is not None:
            logger.emit(
                "dataset_download_started",
                source_repo=self.settings.source_repo,
                source_ref=source_ref,
                imported_language=self.settings.language,
                state="downloading_dataset",
            )

        try:
            self._download_file(archive_url, archive_path)
        except urllib.error.HTTPError as exc:
            raise DatasetError(f"Failed to download dataset archive: {exc}") from exc
        except urllib.error.URLError as exc:
            raise DatasetError(f"Network error while downloading dataset archive: {exc}") from exc

        source_commit = self._resolve_source_commit(self.settings.source_repo, source_ref)

        if logger is not None:
            logger.emit(
                "dataset_download_finished",
                source_repo=self.settings.source_repo,
                source_ref=source_ref,
                source_commit=source_commit,
                imported_language=self.settings.language,
                state="downloading_dataset",
            )
            logger.emit(
                "deck_import_started",
                source_repo=self.settings.source_repo,
                source_ref=source_ref,
                source_commit=source_commit,
                imported_language=self.settings.language,
                state="downloading_dataset",
            )

        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(temp_dir)

            extracted_roots = [path for path in temp_dir.iterdir() if path.is_dir()]
            if not extracted_roots:
                raise DatasetError("Downloaded dataset archive did not contain a project directory.")
            source_root = extracted_roots[0]

            metadata = DatasetMetadata(
                source_repo=self.settings.source_repo,
                source_ref=source_ref,
                source_commit=source_commit,
                imported_language=self.settings.language,
            )
            deck = self.import_from_directory(source_root, metadata)

        self._write_cache(repo_dir, deck)

        if logger is not None:
            logger.emit(
                "deck_import_finished",
                source_repo=deck.metadata.source_repo,
                source_ref=deck.metadata.source_ref,
                source_commit=deck.metadata.source_commit,
                imported_language=deck.metadata.imported_language,
                state="idle",
            )
        return deck

    def load_cached_deck(self, source_ref: str | None = None) -> ImportedDeck | None:
        """Load a cached normalized deck from disk."""
        ref = source_ref or self.settings.source_ref
        repo_dir = self._ref_dir(ref)
        deck_path = repo_dir / "deck.json"
        meta_path = repo_dir / "metadata.json"
        if not deck_path.exists() or not meta_path.exists():
            return None
        metadata = DatasetMetadata.model_validate_json(meta_path.read_text(encoding="utf-8"))
        raw_cards = json.loads(deck_path.read_text(encoding="utf-8"))
        cards = [CardRecord.model_validate(item) for item in raw_cards]
        return ImportedDeck(metadata=metadata, cards=cards)

    def import_from_directory(self, source_root: Path, metadata: DatasetMetadata) -> ImportedDeck:
        """Import and normalize cards from an unpacked upstream repository checkout."""
        categories_path = source_root / "src" / "data" / "categories.json"
        language_dir = source_root / "src" / "data" / metadata.imported_language
        if not categories_path.exists():
            raise DatasetError(f"Missing categories file at {categories_path}")
        if not language_dir.exists():
            raise DatasetError(f"Missing language directory at {language_dir}")

        categories = json.loads(categories_path.read_text(encoding="utf-8"))
        if not isinstance(categories, dict):
            raise DatasetError("categories.json must decode to an object.")

        cards: list[CardRecord] = []
        for category_file in sorted(language_dir.glob("*.json")):
            category_slug = category_file.stem
            category_label = categories.get(category_slug, category_slug)
            payload = json.loads(category_file.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise DatasetError(f"Expected a JSON object in {category_file}")

            for index, (target, taboo_words_raw) in enumerate(payload.items(), start=1):
                if not isinstance(target, str):
                    continue
                if not isinstance(taboo_words_raw, list):
                    continue
                taboo_words = [word.strip() for word in taboo_words_raw if isinstance(word, str) and word.strip()]
                if len(taboo_words) < 2:
                    continue
                card_id = f"{category_slug}:{slugify(target)}:{index:04d}"
                cards.append(
                    CardRecord(
                        id=card_id,
                        target=target.strip(),
                        taboo_hard=dedupe_preserve_order(taboo_words),
                        aliases=[],
                        difficulty=None,
                        lang=metadata.imported_language,
                        source_category=category_slug,
                        source_repo=metadata.source_repo,
                        source_ref=metadata.source_ref,
                        source_commit=metadata.source_commit,
                        category_label=str(category_label),
                    )
                )

        if not cards:
            raise DatasetError("No English cards were imported from the upstream dataset.")

        metadata.card_count = len(cards)
        return ImportedDeck(metadata=metadata, cards=cards)

    def _ref_dir(self, source_ref: str) -> Path:
        repo_slug = self.settings.source_repo.replace("/", "__")
        return self.dataset_root / repo_slug / slugify(source_ref)

    @staticmethod
    def _build_archive_url(repo: str, ref: str) -> str:
        return f"https://codeload.github.com/{repo}/zip/{ref}"

    @staticmethod
    def _download_file(url: str, destination: Path) -> None:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "taboo-arena/0.1",
                "Accept": "application/vnd.github+json",
            },
        )
        with urllib.request.urlopen(request) as response, destination.open("wb") as output:
            shutil.copyfileobj(response, output)

    @staticmethod
    def _resolve_source_commit(repo: str, ref: str) -> str | None:
        url = f"https://api.github.com/repos/{repo}/commits/{ref}"
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "taboo-arena/0.1",
                "Accept": "application/vnd.github+json",
            },
        )
        try:
            with urllib.request.urlopen(request) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            return None
        sha = payload.get("sha")
        return str(sha) if sha else None

    @staticmethod
    def _write_cache(repo_dir: Path, deck: ImportedDeck) -> None:
        (repo_dir / "metadata.json").write_text(deck.metadata.model_dump_json(indent=2), encoding="utf-8")
        deck_payload = [card.model_dump(mode="json") for card in deck.cards]
        (repo_dir / "deck.json").write_text(json.dumps(deck_payload, indent=2), encoding="utf-8")

