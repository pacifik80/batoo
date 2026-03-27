"""View models for transcript bubble rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

TranscriptTone = Literal["pending", "rejected", "accepted", "judge", "guess", "success", "meta"]
TranscriptAlignment = Literal["left", "center", "right"]


@dataclass(slots=True)
class BubbleDebugField:
    """One labeled debug field inside a bubble section."""

    label: str
    value: str


@dataclass(slots=True)
class BubbleDebugSection:
    """One collapsed debug section inside a transcript bubble."""

    title: str
    summary: str | None = None
    fields: list[BubbleDebugField] = field(default_factory=list)


@dataclass(slots=True)
class BubbleRawArtifact:
    """Optional raw or low-level artifact stored for debug rendering only."""

    label: str
    value: str


@dataclass(slots=True)
class TranscriptMessage:
    """One rendered transcript bubble with public text and hidden debug details."""

    role: str
    label: str
    tone: TranscriptTone
    alignment: TranscriptAlignment
    message_id: str = ""
    public_text: str = ""
    pending_public_text: str | None = None
    status_label: str | None = None
    debug_sections: list[BubbleDebugSection] = field(default_factory=list)
    raw_artifacts: list[BubbleRawArtifact] = field(default_factory=list)
    prompt_text: str | None = None
    prompt_model_id: str | None = None
    prompt_template_id: str | None = None

    @property
    def text(self) -> str:
        """Compatibility alias for older transcript tests/callers."""
        return self.public_text

    @text.setter
    def text(self, value: str) -> None:
        self.public_text = value

    @property
    def status_text(self) -> str | None:
        """Compatibility alias for older transcript tests/callers."""
        return self.status_label

    @status_text.setter
    def status_text(self, value: str | None) -> None:
        self.status_label = value

    @property
    def debug_entries(self) -> list[BubbleDebugField]:
        """Compatibility view flattening all debug sections into one list."""
        return [field for section in self.debug_sections for field in section.fields]


TranscriptDebugEntry = BubbleDebugField
