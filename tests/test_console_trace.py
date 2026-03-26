from __future__ import annotations

from pathlib import Path

from taboo_arena.logging.run_logger import RunLogger


def test_run_logger_console_trace_outputs_events_and_prompt(capsys, tmp_path: Path) -> None:
    logger = RunLogger(log_root=tmp_path, console_trace=True)
    logger.emit("round_started", state="generating_clue", round_id="round_1")
    logger.trace_prompt(
        role="cluer",
        model_id="fake-model",
        prompt_template_id="generic_completion",
        prompt="USER: Say something",
        generation_params={"temperature": 0.4},
    )
    logger.trace_response(
        role="cluer",
        model_id="fake-model",
        text="forest giant",
        latency_ms=12.5,
        prompt_tokens=10,
        completion_tokens=3,
    )

    output = capsys.readouterr().out
    assert "[taboo-arena:event]" in output
    assert "[taboo-arena:prompt]" in output
    assert "USER: Say something" in output
    assert "[taboo-arena:response]" in output
    assert "forest giant" in output
