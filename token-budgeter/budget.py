from __future__ import annotations

import tiktoken

WINDOWS = {
    "gpt-5.5": 1_000_000,
    "claude-sonnet-4-6": 1_000_000,
    "qwen3.5-8b": 32_768,
    "gemini-2.5-flash": 1_000_000,
}


def count_tokens(text: str, model: str) -> int:
    # Real tokenizer, not a len(text)//4 estimate (which is ~20% off
    # on code and non-English text).
    # Accurate for OpenAI models. The cl100k_base fallback is an
    # approximation for anything tiktoken doesn't recognize.
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)
    return len(tokens)


def budget(model: str, system_prompt: str, reserve_answer: int = 1024) -> int:
    window = WINDOWS[model]
    system_prompt_tokens = count_tokens(system_prompt, model)
    return max(0, window - system_prompt_tokens - reserve_answer)


def chunks_that_fit(chunks: list[str], model: str, system_prompt: str) -> list[str]:
    remaining_budget = budget(model, system_prompt)
    kept: list[str] = []
    for chunk in chunks:
        cost = count_tokens(chunk, model)
        if cost > remaining_budget:
            break
        kept.append(chunk)
        remaining_budget -= cost
    return kept
