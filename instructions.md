Build a local Python application called `taboo-arena`.

Goal
Create a single-screen local Streamlit app for running a custom Taboo-like benchmark between three model roles:
1. Cluer: sees the target word and forbidden words and produces a clue
2. Guesser: sees the accepted clue history and tries to guess the target
3. Judge: reviews each clue for rule compliance

This is NOT a clone of the official board game.
This is a custom benchmark mode with exactly 3 guess attempts per round.

Core scope
- local-first, single-machine app
- English-only in v1
- supports separate models for Cluer, Guesser, Judge
- supports both `safetensors` and `gguf`
- supports curated models and arbitrary custom user-provided Hugging Face model entries
- automatic model download when missing
- automatic card dataset download when missing
- machine-friendly logging for external analytics
- one main screen only; do not build additional pages in v1

Non-goals
- no attempt to emulate official Hasbro timing or full rules
- no timer
- no multiplayer/network mode
- no extra screens beyond the single main screen
- no chain-of-thought prompting
- no hidden background services

Gameplay
Each round uses exactly one card with normalized internal schema:
- id
- target
- taboo_hard[]
- aliases[] optional
- difficulty optional
- lang = "en"
- source_category
- source_repo
- source_ref
- source_commit optional

There are exactly 3 guess attempts per round.

Each guess attempt works like this:
1. Cluer receives:
   - target
   - taboo_hard
   - aliases if present
   - history of previous accepted clues
   - history of previous wrong guesses
   - current attempt number
2. Cluer generates one clue draft.
3. The clue draft is evaluated by:
   - deterministic logical validator
   - LLM judge
4. If the clue is rejected, the app asks Cluer to repair/regenerate the clue.
5. Hidden clue repairs do NOT consume one of the 3 guess attempts.
6. Maximum clue repairs per guess attempt is configurable, default = 3.
7. If no accepted clue is produced within the repair budget, the round ends with terminal reason `clue_not_repaired`.
8. If a clue is accepted, Guesser receives:
   - current accepted clue
   - prior accepted clues
   - prior wrong guesses
   - current attempt number
9. Guesser returns exactly one guess.
10. If the guess matches target or alias, the round ends in success.
11. Otherwise continue to the next guess attempt.
12. After 3 failed guesses, the round ends with terminal reason `max_guess_attempts_reached`.

Important constraints
- There is only one gameplay mode.
- Hidden clue repair is always part of the workflow.
- The 3-attempt limit applies only to guessing.
- Logs must record on which repair iteration the clue became accepted, or that it was never accepted.

Dataset / cards
On first run, download the card source from the GitHub repository `Kovah/Taboo-Data`.

Requirements:
- do not vendor the full dataset into this repo
- implement a dataset downloader that:
  - downloads a zip/tar archive for a chosen GitHub ref
  - stores it in app data dir
  - unpacks it
  - imports only English data in v1
  - parses source files under the repo data layout
  - converts them into internal normalized card records
- persist dataset metadata:
  - source_repo
  - source_ref
  - source_commit if available
  - import_timestamp
  - imported_language = en
- expose a CLI command for dataset download/import
- cache imported normalized deck locally so the app does not re-import on every startup

Judge stack
Every clue draft must be evaluated by two layers.

Layer 1: deterministic logical validator
Must check at minimum:
- exact target match, case-insensitive
- exact taboo word match, case-insensitive
- punctuation-stripped matching
- token-level matching
- whole-word matching
- substring / part-of-word matching
- alias matching
- simple stemming / lemmatization
- repeated clue detection
- empty clue
- clue too short
- optional overly-similar-to-previous-clue heuristics

Return structured output:
- verdict: pass | fail
- normalized_text
- violations[]
- matched_terms[]
- validator_version

Layer 2: LLM judge
Separate model role from Cluer and Guesser.
Input:
- target
- taboo_hard
- aliases
- current clue draft
- previous accepted clues
- previous wrong guesses
- current attempt number

Output must be strict JSON only:
- verdict: pass | fail | uncertain
- reasons[]
- suspicious_terms[]
- confidence float 0..1
- summary string
- judge_version

No chain-of-thought. Only final structured verdict.

Judge merge logic
Implement configurable merge policy with these defaults:
- if logical validator verdict = fail => final verdict = fail
- else if LLM judge verdict = fail => final verdict = fail
- else if LLM judge verdict = uncertain => final verdict = pass_with_warning
- else => final verdict = pass

Advanced setting:
- `block_on_uncertain` default = false

If `block_on_uncertain = true`, then `uncertain` should also block the clue.

Always log:
- logical verdict
- logical violations
- llm judge verdict
- llm judge reasons
- merged final verdict
- judge disagreement flag

Model support
Support three independently selected roles:
- Cluer model
- Guesser model
- Judge model

The same model may be used for multiple roles.

Supported backend families in v1:
1. `transformers_safetensors`
2. `llama_cpp_gguf`

Model acquisition
- if a selected model is missing locally, download it automatically from Hugging Face
- use Hugging Face cache
- GGUF entries must support selecting a specific file name such as `Q4_K_M`
- support Hugging Face token via environment variable
- surface gated-access/auth errors clearly in the UI and CLI
- do not crash the whole app on gated access failure; show actionable error state

Model catalog
Implement both:
1. `model_registry.yaml` for curated starter entries
2. manual custom model entry in the UI

Curated entry fields:
- id
- display_name
- backend
- repo_id
- revision optional
- filename optional for GGUF
- tokenizer_repo optional
- architecture_family
- chat_template_id
- supports_system_prompt bool
- roles_supported
- languages
- estimated_vram_gb
- default_generation_params
- requires_hf_auth bool
- gated bool
- notes

Custom entry fields:
- backend
- repo_id
- optional revision
- optional gguf filename
- optional tokenizer_repo
- chat_template_id
- supports_system_prompt
- estimated_vram_gb optional
- role assignment

Registry requirements
- include a checkbox `show gated models`
- by default hide curated entries marked `gated: true`
- when enabled, show gated models and display a short warning about Hugging Face access requirements
- seed the curated registry with English-capable instruct model families appropriate for local testing:
  - Qwen2.5 Instruct (small to medium sizes)
  - Phi-4-mini-instruct
  - Mistral-7B-Instruct-v0.3
  - Gemma 3 instruct variants
  - Llama 3.1 instruct variants

Prompt formatting
Registry-driven prompt formatting is mandatory.
Do not assume a universal chat format.

Implement chat template adapters for at least:
- qwen_chatml
- mistral_inst
- llama3_chat
- gemma_chat
- phi_chat
- generic_completion fallback

Each model entry must specify:
- chat_template_id
- whether system prompts are supported
- stop tokens if required
- role formatting rules

Memory management
Implement `ModelManager` that can:
- inspect whether models are already cached
- download missing models
- load and unload models
- estimate whether selected models fit in available VRAM / RAM
- support both one-model-for-all-roles and three-different-models scenarios
- support mixed backend scenarios, e.g. one GGUF role and one safetensors role

Memory policies:
- `keep_loaded_if_possible`
- `sequential_load_unload`

Default behavior:
- try to keep loaded if resources allow
- otherwise fall back automatically to sequential load/unload

Main UI: one screen only
Build a single Streamlit screen with this layout.

Top bar
- Cluer model selector
- Guesser model selector
- Judge model selector
- install/download status for each selected model
- checkbox `show gated models`
- button `Start Round`
- button `Start Batch Run`
- button `Stop`
- advanced settings expander

Main body: 3 columns

Left column: stylized card
- show card in a visually distinct card-like container
- display:
  - card id
  - category
  - difficulty if present
  - taboo words
  - source metadata
- target should be visible only in explicit debug mode
- controls:
  - random card
  - pick specific card
  - reload dataset metadata

Center column: chat / transcript
- show Cluer messages
- show Judge feedback
- show Guesser guesses
- show hidden repair sub-iterations as collapsible blocks
- visually separate guess attempts
- show final result banner:
  - solved on attempt 1/2/3
  - failed after 3 guesses
  - clue_not_repaired

Right column: round log + running statistics
- event timeline for current run
- overall score and aggregate metrics
- clue compliance stats
- guessing success stats
- judge disagreement stats
- export current run button
- latest errors/warnings

Top status strip
- current run id
- current round id
- current guess attempt number out of 3
- current clue repair number
- current state:
  - idle
  - downloading_dataset
  - downloading_model
  - generating_clue
  - logical_validation
  - llm_validation
  - repairing_clue
  - generating_guess
  - round_finished
  - batch_running
  - stopped
- rolling latency summaries by role

Advanced settings
- max clue repairs per guess attempt (default 3)
- block_on_uncertain
- temperature / top_p / max_tokens per role
- random seed
- language selector fixed to English in v1 but architected for future extension
- strictness toggles for logical validator
- log directory
- device preference
- show/hide hidden repair messages
- allow same model for multiple roles toggle
- source_ref for dataset import

Logging
Write machine-friendly logs intended for external analytics.

Required output files:
- `events.jsonl`
- `rounds.csv`
- `rounds.parquet`
- `summary.csv`

Required event types:
- app_started
- dataset_download_started
- dataset_download_finished
- deck_import_started
- deck_import_finished
- model_download_started
- model_download_finished
- round_started
- clue_draft_generated
- logical_validation_completed
- llm_validation_completed
- clue_repair_requested
- clue_accepted
- guess_generated
- round_finished
- batch_started
- batch_finished
- stopped
- error

Event fields
Include where relevant:
- run_id
- batch_id optional
- round_id
- card_id
- timestamp
- state
- source_repo
- source_ref
- source_commit optional
- imported_language
- cluer_model_id
- guesser_model_id
- judge_model_id
- cluer_backend
- guesser_backend
- judge_backend
- attempt_no
- clue_repair_no
- latency_ms
- prompt_tokens
- completion_tokens
- clue_text_raw
- clue_text_normalized
- guess_text_raw
- guess_text_normalized
- logical_verdict
- logical_violations[]
- logical_matched_terms[]
- llm_judge_verdict
- llm_judge_reasons[]
- llm_judge_suspicious_terms[]
- llm_judge_confidence
- final_judge_verdict
- judge_disagreement
- prompt_template_id
- seed
- terminal_reason optional
- error_message optional

Round summary fields
- run_id
- round_id
- card_id
- target
- solved bool
- solved_on_attempt nullable
- total_guess_attempts_used
- total_clue_repairs
- first_clue_passed_without_repair bool
- clue_repaired_successfully bool
- clue_not_repaired bool
- terminal_reason
- cluer_model_id
- guesser_model_id
- judge_model_id
- total_latency_ms

Metrics
Compute and display at minimum:
- rounds_played
- solve_rate_within_3
- solve_on_attempt_1_rate
- solve_on_attempt_2_rate
- solve_on_attempt_3_rate
- average_wrong_guesses_before_success
- first_draft_clue_pass_rate
- repaired_clue_success_rate
- clue_total_failure_rate
- average_repairs_per_round
- logical_fail_rate
- llm_judge_fail_rate
- judge_disagreement_rate
- average_latency_per_role

Role analytics
Make it easy to answer:
- which models are best at clueing
- which are best at guessing
- which models fail clue rules most often
- which judges are strictest
- where logical validator and LLM judge disagree most

Batch mode
Batch mode must exist but stay inside the same main screen.

Requirements:
- run many rounds automatically
- support combinations of selected Cluer x Guesser x Judge models
- support fixed card subsets or random samples
- support repeated runs per card
- support deterministic seeds
- write same logging artifacts as single-round mode
- export pairwise/triple-role summaries suitable for external analytics

CLI
Also expose a CLI:
- `download-dataset`
- `list-models`
- `download-model`
- `run-single`
- `run-batch`
- `export-summary`

Implementation stack
- Python 3.11
- uv
- streamlit
- pydantic
- typer
- pandas
- pyarrow
- pytest
- ruff
- mypy
- jinja2
- rapidfuzz
- nltk or spacy for morphology
- transformers
- huggingface_hub
- llama-cpp-python or local llama.cpp server integration

Engineering requirements
- modular architecture
- Windows-friendly
- single-GPU-friendly
- graceful fallback on OOM / gated access / missing token / bad custom model entry
- clear user-visible error states
- type hints on public APIs
- docstrings on public APIs
- config-driven behavior
- no hidden telemetry
- no chain-of-thought storage

Suggested repo structure
taboo-arena/
  README.md
  pyproject.toml
  model_registry.yaml
  .env.example
  configs/
  src/taboo_arena/
    app/
    cards/
    models/
    prompts/
    judge/
    engine/
    logging/
    analytics/
    cli/
    utils/
  tests/

Tests
Add tests for:
- dataset download/import
- card normalization
- model registry parsing
- custom model entry validation
- logical validator
- judge merge logic
- round engine
- hidden repair loop
- logging outputs
- batch runner smoke test

README
README must explain:
- project purpose
- custom gameplay rules
- difference between guessing attempts and hidden clue repairs
- why both logical and LLM judging are used
- dataset download/import flow
- Hugging Face auth and gated model behavior
- how safetensors and GGUF models are configured
- how to run single rounds
- how to run batch mode
- where logs are written
- known limitations

Acceptance criteria
The implementation is complete when:
- I can select Cluer, Guesser, and Judge in the single main screen
- I can use the same model for all roles or different models per role
- missing models auto-download
- missing English card dataset auto-downloads and imports
- every clue draft is checked by both logical validator and LLM judge
- both judge outputs and the merged verdict are logged
- hidden clue repairs are capped at 3 by default
- only guessing is capped at 3 attempts
- the UI matches the requested top bar + 3-column layout
- logs are written to JSONL/CSV/Parquet
- batch mode runs from the same screen
- tests pass
- README is complete