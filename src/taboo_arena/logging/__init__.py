"""Run logging."""

from taboo_arena.logging.schemas import RoundSummaryRecord

__all__ = ["RoundSummaryRecord", "RunLogger"]


def __getattr__(name: str) -> object:
    """Lazily expose RunLogger to avoid import cycles."""
    if name == "RunLogger":
        from taboo_arena.logging.run_logger import RunLogger

        return RunLogger
    raise AttributeError(name)
