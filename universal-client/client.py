from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from openai import OpenAI
from anthropic import Anthropic
from google import genai


type ProviderKind = Literal["openai-compatible", "anthropic", "google"]
type ProviderName = Literal["local", "openai", "anthropic", "google"]


@dataclass
class Provider:
    kind: ProviderKind
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None


PROVIDERS: dict[ProviderName, Provider] = {
    "local": Provider(
        "openai-compatible", os.getenv("OLLAMA_BASE_URL"), "ollama", "qwen3:8b"
    ),
    "openai": Provider(
        "openai-compatible", None, os.getenv("OPENAI_API_KEY"), "gpt-5.5"
    ),
    "anthropic": Provider(
        "anthropic", None, os.getenv("ANTHROPIC_API_KEY"), "claude-sonnet-4-6"
    ),
    "google": Provider("google", None, os.getenv("GOOGLE_API_KEY"), "gemini-3.5-flash"),
}


def complete(
    prompt: str, provider: ProviderName = "local", temperature: float = 0.2
) -> str:
    p = PROVIDERS[provider]
    if p.kind == "openai-compatible":
        client = OpenAI(base_url=p.base_url, api_key=p.api_key)
        response = client.chat.completions.create(
            model=p.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content
    if p.kind == "anthropic":
        ac = Anthropic(api_key=p.api_key)
        msg = ac.messages.create(
            model=p.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=1024,
            temperature=temperature,
        )
        return msg.content[0].text
    if p.kind == "google":
        gg = genai.Client(api_key=p.api_key)
        response = gg.models.generate_content(
            model=p.model,
            contents=[prompt],
            config={
                "temperature": temperature,
            },
        )
        return response.text or ""
    raise ValueError(f"Unknown provider: {p.kind}")
