"""Typer CLI for taboo-arena."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from taboo_arena.app.bootstrap import (
    build_dataset_manager,
    build_logger,
    build_model_manager,
    build_registry,
    load_app_settings,
)
from taboo_arena.engine import BatchRunner, BatchSpec, RoundEngine
from taboo_arena.models.backends import ModelRuntimeError
from taboo_arena.models.manager import ModelAuthError
from taboo_arena.utils.runtime_setup import ensure_runtime_environment

app = typer.Typer(no_args_is_help=True)


@app.command("ensure-runtime")
def ensure_runtime(
    json_output: bool = typer.Option(False, "--json", help="Emit the bootstrap summary as JSON."),
) -> None:
    """Repair the project env to use GPU when available, else CPU."""
    repo_root = Path(__file__).resolve().parents[3]
    report = ensure_runtime_environment(repo_root, echo=typer.echo)
    if json_output:
        typer.echo(json.dumps(report.as_dict(), indent=2))


@app.command("download-dataset")
def download_dataset(
    source_ref: str = typer.Option("main", help="Git ref to import from the upstream card repo."),
) -> None:
    """Download and normalize the English deck."""
    settings = load_app_settings()
    logger = build_logger(settings.run.log_dir, console_trace=settings.run.console_trace)
    dataset_manager = build_dataset_manager(settings)
    logger.emit("app_started", state="downloading_dataset")
    deck = dataset_manager.download_and_import(source_ref, logger=logger)
    typer.echo(
        json.dumps(
            {
                "cards": len(deck.cards),
                "source_repo": deck.metadata.source_repo,
                "source_ref": deck.metadata.source_ref,
                "source_commit": deck.metadata.source_commit,
                "run_id": logger.run_id,
            },
            indent=2,
        )
    )


@app.command("list-models")
def list_models(show_gated: bool = typer.Option(False, help="Include gated curated models.")) -> None:
    """List curated and custom model entries."""
    registry = build_registry()
    model_manager = build_model_manager()
    rows = []
    for entry in registry.list_entries(show_gated=show_gated):
        rows.append(
            {
                "id": entry.id,
                "display_name": entry.display_name,
                "backend": entry.backend,
                "gated": entry.gated,
                "cached": model_manager.inspect_cache(entry),
                "roles": ",".join(entry.roles_supported),
            }
        )
    typer.echo(pd.DataFrame(rows).to_string(index=False))


@app.command("download-model")
def download_model(model_id: str = typer.Argument(..., help="Registry model id to download.")) -> None:
    """Download one model into the Hugging Face cache."""
    registry = build_registry()
    settings = load_app_settings()
    logger = build_logger(settings.run.log_dir, console_trace=settings.run.console_trace)
    model_manager = build_model_manager(logger)
    entry = registry.get(model_id)
    try:
        path = model_manager.ensure_model_available(entry)
    except (ModelAuthError, ModelRuntimeError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps({"model_id": model_id, "path": str(path), "run_id": logger.run_id}, indent=2))


@app.command("run-single")
def run_single(
    cluer_model: str = typer.Option(..., help="Model id used for the cluer role."),
    guesser_model: str = typer.Option(..., help="Model id used for the guesser role."),
    judge_model: str = typer.Option(..., help="Model id used for the judge role."),
    card_id: str | None = typer.Option(None, help="Optional specific card id."),
    source_ref: str = typer.Option("main", help="Dataset ref."),
) -> None:
    """Run one benchmark round."""
    settings = load_app_settings()
    settings.dataset.source_ref = source_ref
    logger = build_logger(settings.run.log_dir, console_trace=settings.run.console_trace)
    logger.emit("app_started", state="idle")
    dataset_manager = build_dataset_manager(settings)
    deck = dataset_manager.ensure_dataset(source_ref=source_ref, logger=logger)
    registry = build_registry()
    model_manager = build_model_manager(logger)
    engine = RoundEngine(model_manager=model_manager, logger=logger, settings=settings.run)

    card = next((item for item in deck.cards if item.id == card_id), deck.cards[0])
    try:
        result = engine.play_round(
            card=card,
            cluer_entry=registry.get(cluer_model),
            guesser_entry=registry.get(guesser_model),
            judge_entry=registry.get(judge_model),
        )
    except (ModelAuthError, ModelRuntimeError) as exc:
        logger.emit("error", error_message=str(exc), state="idle")
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        json.dumps(
            {
                "run_id": result.run_id,
                "round_id": result.round_id,
                "card_id": result.card.id,
                "solved": result.solved,
                "solved_on_attempt": result.solved_on_attempt,
                "terminal_reason": result.terminal_reason,
                "log_dir": str(logger.run_dir),
            },
            indent=2,
        )
    )


@app.command("run-batch")
def run_batch(
    cluer_model: list[str] = typer.Option(..., help="One or more cluer model ids."),
    guesser_model: list[str] = typer.Option(..., help="One or more guesser model ids."),
    judge_model: list[str] = typer.Option(..., help="One or more judge model ids."),
    sample_size: int = typer.Option(5, help="Number of cards to sample."),
    repeats_per_card: int = typer.Option(1, help="How many times to replay each selected card."),
    fixed_card_ids: str | None = typer.Option(None, help="Comma-separated fixed card ids."),
    source_ref: str = typer.Option("main", help="Dataset ref."),
    seed: int = typer.Option(7, help="Deterministic batch seed."),
) -> None:
    """Run a batch benchmark inside one log directory."""
    settings = load_app_settings()
    settings.dataset.source_ref = source_ref
    settings.run.random_seed = seed
    logger = build_logger(settings.run.log_dir, console_trace=settings.run.console_trace)
    logger.emit("app_started", state="batch_running")
    dataset_manager = build_dataset_manager(settings)
    deck = dataset_manager.ensure_dataset(source_ref=source_ref, logger=logger)
    registry = build_registry()
    model_manager = build_model_manager(logger)
    engine = RoundEngine(model_manager=model_manager, logger=logger, settings=settings.run)
    batch_runner = BatchRunner(engine, logger)

    if fixed_card_ids:
        requested = {item.strip() for item in fixed_card_ids.split(",") if item.strip()}
        cards = [card for card in deck.cards if card.id in requested]
    else:
        cards = deck.cards[:sample_size]

    try:
        results = batch_runner.run(
            BatchSpec(
                cluer_entries=[registry.get(model_id) for model_id in cluer_model],
                guesser_entries=[registry.get(model_id) for model_id in guesser_model],
                judge_entries=[registry.get(model_id) for model_id in judge_model],
                cards=cards,
                repeats_per_card=repeats_per_card,
                seed=seed,
            )
        )
    except (ModelAuthError, ModelRuntimeError) as exc:
        logger.emit("error", error_message=str(exc), state="idle")
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        json.dumps(
            {
                "run_id": logger.run_id,
                "rounds": len(results),
                "log_dir": str(logger.run_dir),
            },
            indent=2,
        )
    )


@app.command("export-summary")
def export_summary(
    output_path: Path = typer.Option(..., help="Destination CSV path."),
    log_root: Path | None = typer.Option(None, help="Optional log root override."),
) -> None:
    """Concatenate round summaries across run directories."""
    settings = load_app_settings()
    root = (log_root or settings.run.log_dir).resolve()
    frames: list[pd.DataFrame] = []
    for run_dir in root.iterdir() if root.exists() else []:
        rounds_path = run_dir / "rounds.csv"
        if rounds_path.exists():
            frames.append(pd.read_csv(rounds_path))
    summary = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    typer.echo(json.dumps({"rows": int(len(summary)), "output": str(output_path)}, indent=2))


if __name__ == "__main__":
    app()
