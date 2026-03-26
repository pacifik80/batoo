# taboo-arena

`taboo-arena` is a local-first Python app for benchmarking LLMs in a custom Taboo-like setup. It is built around three explicit roles:

- Cluer: sees the target and forbidden words, then proposes a clue
- Guesser: sees accepted clue history and returns exactly one guess
- Judge: checks each clue for rule compliance with both deterministic logic and an LLM pass

This is not a clone of the official board game. It is a benchmarking harness with fixed rules, structured logs, a single Streamlit screen, and matching CLI commands.

## What The App Does

- Runs one round or many rounds locally on one machine
- Lets you choose separate models for Cluer, Guesser, and Judge
- Supports `transformers_safetensors` and `llama_cpp_gguf`
- Loads curated registry models and user-defined custom Hugging Face entries
- Downloads missing models on demand into the Hugging Face cache
- Downloads and normalizes the English Taboo card dataset when missing
- Writes machine-friendly JSONL, CSV, and Parquet logs for external analysis

## Custom Gameplay Rules

Each round uses one normalized card with:

- `id`
- `target`
- `taboo_hard`
- `aliases`
- `difficulty`
- `lang`
- `source_category`
- `source_repo`
- `source_ref`
- `source_commit`

There are exactly 3 guess attempts per round.

For each guess attempt:

1. The Cluer proposes one clue draft.
2. The clue is checked by the deterministic validator.
3. The clue is checked again by the LLM Judge.
4. If the clue fails, the Cluer gets hidden repair attempts.
5. If a clue is accepted, the Guesser returns exactly one guess.
6. If the guess matches the target or an alias, the round succeeds.
7. After 3 failed guesses, the round ends with `max_guess_attempts_reached`.

## Guess Attempts vs Hidden Clue Repairs

These are intentionally different:

- Guess attempts are capped at 3.
- Hidden clue repairs do not consume guess attempts.
- Max clue repairs per guess attempt defaults to 3 and is configurable.
- If no clue is repaired successfully within the repair budget, the round ends with `clue_not_repaired`.

This separation is important for benchmarking. It lets you measure clue quality and clue recoverability independently from guessing skill.

## Why The App Uses Two Judge Layers

Every clue draft is evaluated by:

1. A deterministic logical validator
2. A separate LLM Judge

The logical validator catches direct rule breaks such as exact matches, taboo-word reuse, punctuation-stripped matches, token overlap, whole-word matches, substring matches, alias matches, simple stemming, repeated clues, and clue-too-short failures.

The LLM Judge adds a second pass for suspicious semantic or near-match cases and returns strict JSON only:

- `verdict`
- `reasons`
- `suspicious_terms`
- `confidence`
- `summary`
- `judge_version`

Default merge policy:

- logical fail -> final fail
- logical pass + judge fail -> final fail
- logical pass + judge uncertain -> `pass_with_warning`
- logical pass + judge pass -> final pass

If `block_on_uncertain` is enabled, uncertain judge verdicts also block the clue.

## Dataset Download And Import Flow

The app downloads the upstream card source from:

- `Kovah/Taboo-Data`

It does not vendor the dataset into this repository.

On first use, the dataset manager:

1. Downloads a GitHub archive for the requested ref
2. Unpacks it into the local app data directory
3. Reads `src/data/categories.json`
4. Imports English cards from `src/data/en/*.json`
5. Normalizes the cards into the internal schema
6. Caches `deck.json` and `metadata.json` locally

Stored dataset metadata includes:

- `source_repo`
- `source_ref`
- `source_commit` when GitHub resolves it
- `import_timestamp`
- `imported_language = en`

## Hugging Face Auth And Gated Models

Some curated models are marked as gated. By default they are hidden in the UI.

- Enable `show gated models` to see them
- Set `HF_TOKEN` if a selected model requires authentication
- Gated or auth failures are surfaced as actionable errors instead of crashing the whole app

Example environment file:

```env
HF_TOKEN=hf_your_token_here
TABOO_ARENA_DATASET_REF=main
TABOO_ARENA_DEVICE=auto
```

## Safetensors And GGUF Configuration

Curated starter models live in [`model_registry.yaml`](./model_registry.yaml).

Each entry defines:

- backend
- repo id
- optional revision
- optional GGUF filename
- tokenizer repo
- architecture family
- prompt template id
- system prompt support
- supported roles
- estimated VRAM
- default generation parameters
- gated/auth flags

Supported prompt adapters:

- `qwen_chatml`
- `mistral_inst`
- `llama3_chat`
- `gemma_chat`
- `phi_chat`
- `generic_completion`

Role prompts are stored in the repository-level [`prompts`](./prompts) folder:

- [`cluer.json`](./prompts/cluer.json)
- [`guesser.json`](./prompts/guesser.json)
- [`judge.json`](./prompts/judge.json)

The app loads those JSON files at runtime, so role prompt maintenance now happens there rather than in Python code.

For GGUF models, a specific filename is required. For example, a `Q4_K_M` file can be referenced directly in the registry or in a custom entry.

## Installation

The project targets Python 3.11 and `uv`.

```bash
uv sync
```

If you want the exact validation toolchain too:

```bash
uv sync --extra dev
```

## Run The Streamlit App

The app uses one main screen only.

```bash
uv run --python 3.11 python -m taboo_arena.cli.main ensure-runtime
uv run --python 3.11 streamlit run src/taboo_arena/app/main.py
```

Or on Windows with the bundled launcher:

```powershell
.\scripts\run_app.ps1
```

The Windows launcher prepares the project `.venv` before start:

- syncs the project without clobbering runtime-managed backend packages
- installs CUDA-enabled `torch` automatically when an NVIDIA GPU is visible
- installs `llama-cpp-python` into the same env

If you want the launcher to also attempt a CUDA rebuild for `llama-cpp-python`, set:

```powershell
$env:TABOO_ARENA_BOOTSTRAP_LLAMA_CPP_GPU = "1"
.\scripts\run_app.ps1
```

The screen contains:

- top bar with model selectors and run controls
- a left card column
- a center transcript column
- a right log and metrics column
- an advanced settings expander for repair limits, judge behavior, validator strictness, batch settings, and custom models

## CLI Commands

Download and normalize the dataset:

```bash
uv run taboo-arena download-dataset --source-ref main
```

List curated and custom models:

```bash
uv run taboo-arena list-models
```

Download one model into the local HF cache:

```bash
uv run taboo-arena download-model qwen2.5-1.5b-instruct
```

Run a single round:

```bash
uv run taboo-arena run-single \
  --cluer-model qwen2.5-1.5b-instruct \
  --guesser-model qwen2.5-1.5b-instruct \
  --judge-model qwen2.5-1.5b-instruct
```

Run a batch:

```bash
uv run taboo-arena run-batch \
  --cluer-model qwen2.5-1.5b-instruct \
  --guesser-model qwen2.5-3b-instruct \
  --judge-model phi-4-mini-instruct \
  --sample-size 5 \
  --repeats-per-card 2
```

Export concatenated round summaries across run directories:

```bash
uv run taboo-arena export-summary --output-path logs/all-rounds.csv
```

## Logs

Every run writes machine-friendly artifacts under the configured log directory:

- `events.jsonl`
- `rounds.csv`
- `rounds.parquet`
- `summary.csv`

The logs are meant for external analytics and include:

- role model ids and backends
- attempt and repair counters
- raw and normalized clue or guess text
- logical validator verdicts and matched terms
- LLM judge verdicts and reasons
- merged verdicts and disagreement flags
- latencies
- terminal reasons

## Running Tests And Checks

```bash
uv run --python 3.11 python -m pytest
uvx ruff check .
uv run --python 3.11 python -m mypy src tests
```

## Known Limitations

- Real safetensors execution depends on a CUDA-ready `torch` build plus compatible hardware and drivers.
- GGUF GPU offload on Windows still depends on a working `llama-cpp-python` CUDA build, which can be slower or less reliable to provision than the transformers runtime.
- The UI batch stop flow is cooperative across Streamlit reruns rather than a background service.
- The upstream card dataset does not currently provide aliases or difficulty metadata, so those fields remain empty unless a future source adds them.
- The first version is English-only, though the dataset and settings layer are structured for future extension.
