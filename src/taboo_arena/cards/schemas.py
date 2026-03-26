"""Card and dataset schemas."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class CardRecord(BaseModel):
    """Normalized card record used by the benchmark engine."""

    id: str
    target: str
    taboo_hard: list[str]
    aliases: list[str] = Field(default_factory=list)
    difficulty: str | None = None
    lang: str = "en"
    source_category: str
    source_repo: str
    source_ref: str
    source_commit: str | None = None
    category_label: str | None = None


class DatasetMetadata(BaseModel):
    """Metadata describing an imported deck."""

    source_repo: str
    source_ref: str
    source_commit: str | None = None
    import_timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    imported_language: str = "en"
    card_count: int = 0


class ImportedDeck(BaseModel):
    """Imported deck plus metadata."""

    metadata: DatasetMetadata
    cards: list[CardRecord]

