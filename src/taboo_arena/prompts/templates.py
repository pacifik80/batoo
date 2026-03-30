"""Registry-driven chat prompt rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

RoleLabel = Literal["system", "user", "assistant"]


@dataclass(slots=True)
class PromptMessage:
    """A rendered prompt message."""

    role: RoleLabel
    content: str


@dataclass(slots=True)
class RenderedPrompt:
    """Prompt text ready for a backend."""

    prompt: str
    prompt_template_id: str
    stop_tokens: list[str]
    add_special_tokens: bool = True


def render_prompt(
    template_id: str,
    messages: list[PromptMessage],
    *,
    supports_system_prompt: bool,
    stop_tokens: list[str] | None = None,
    tokenizer: Any | None = None,
) -> RenderedPrompt:
    """Render a role-based prompt into the format expected by a specific model family."""
    effective_messages = _coerce_system_prompt(messages, supports_system_prompt)
    rendered, add_special_tokens = _render_prompt_text(
        template_id,
        effective_messages,
        tokenizer=tokenizer,
    )
    return RenderedPrompt(
        prompt=rendered,
        prompt_template_id=template_id if template_id in TEMPLATE_RENDERERS else "generic_completion",
        stop_tokens=list(stop_tokens or []),
        add_special_tokens=add_special_tokens,
    )


def _render_prompt_text(
    template_id: str,
    messages: list[PromptMessage],
    *,
    tokenizer: Any | None = None,
) -> tuple[str, bool]:
    tokenizer_rendered = _render_with_tokenizer_chat_template(messages, tokenizer=tokenizer)
    if tokenizer_rendered is not None:
        return tokenizer_rendered, False
    renderer = TEMPLATE_RENDERERS.get(template_id, _render_generic_completion)
    return renderer(messages), True


def _render_with_tokenizer_chat_template(
    messages: list[PromptMessage],
    *,
    tokenizer: Any | None,
) -> str | None:
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return None
    chat_messages = [
        {
            "role": message.role,
            "content": message.content,
        }
        for message in messages
    ]
    try:
        rendered = tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return None
    return str(rendered)


def _coerce_system_prompt(messages: list[PromptMessage], supports_system_prompt: bool) -> list[PromptMessage]:
    if supports_system_prompt:
        return messages
    system_text = "\n\n".join(message.content for message in messages if message.role == "system").strip()
    if not system_text:
        return [message for message in messages if message.role != "system"]

    coerced: list[PromptMessage] = []
    system_consumed = False
    for message in messages:
        if message.role == "system":
            continue
        if not system_consumed and message.role == "user":
            coerced.append(PromptMessage(role="user", content=f"Instructions:\n{system_text}\n\n{message.content}"))
            system_consumed = True
            continue
        coerced.append(message)
    if not system_consumed:
        coerced.insert(0, PromptMessage(role="user", content=f"Instructions:\n{system_text}"))
    return coerced


def _render_qwen_chatml(messages: list[PromptMessage]) -> str:
    segments: list[str] = []
    if not messages or messages[0].role != "system":
        segments.append(
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>"
        )
    for message in messages:
        segments.append(f"<|im_start|>{message.role}\n{message.content}\n<|im_end|>")
    segments.append("<|im_start|>assistant\n")
    return "\n".join(segments)


def _render_mistral_inst(messages: list[PromptMessage]) -> str:
    system_parts = [message.content for message in messages if message.role == "system"]
    system_text = "\n\n".join(system_parts).strip()
    rendered: list[str] = []
    for message in messages:
        if message.role == "system":
            continue
        if message.role == "user":
            content = message.content
            if system_text:
                content = f"{system_text}\n\n{content}"
                system_text = ""
            rendered.append(f"[INST] {content} [/INST]")
        else:
            rendered.append(f"{message.content}</s>")
    if not rendered:
        rendered.append("[INST] [/INST]")
    if not rendered[-1].endswith("[/INST]"):
        rendered.append("[INST] [/INST]")
    return "<s>" + "".join(rendered)


def _render_llama3_chat(messages: list[PromptMessage]) -> str:
    parts = ["<|begin_of_text|>"]
    for message in messages:
        parts.append(
            f"<|start_header_id|>{message.role}<|end_header_id|>\n\n{message.content}<|eot_id|>"
        )
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


def _render_gemma_chat(messages: list[PromptMessage]) -> str:
    parts: list[str] = ["<bos>"]
    for message in messages:
        role = "model" if message.role == "assistant" else "user"
        if message.role == "system":
            role = "user"
        parts.append(f"<start_of_turn>{role}\n{message.content}<end_of_turn>\n")
    parts.append("<start_of_turn>model\n")
    return "".join(parts)


def _render_phi_chat(messages: list[PromptMessage]) -> str:
    parts: list[str] = []
    for message in messages:
        parts.append(f"<|{message.role}|>{message.content}<|end|>")
    parts.append("<|assistant|>")
    return "".join(parts)


def _render_generic_completion(messages: list[PromptMessage]) -> str:
    lines = [f"{message.role.upper()}: {message.content}" for message in messages]
    lines.append("ASSISTANT:")
    return "\n\n".join(lines)


TEMPLATE_RENDERERS = {
    "qwen_chatml": _render_qwen_chatml,
    "mistral_inst": _render_mistral_inst,
    "llama3_chat": _render_llama3_chat,
    "gemma_chat": _render_gemma_chat,
    "phi_chat": _render_phi_chat,
    "generic_completion": _render_generic_completion,
}
