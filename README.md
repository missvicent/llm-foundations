# llm-foundations

Foundational building blocks for working with LLMs in Python. Focuses on the primitives every LLM application sits on top of:

1. **Universal client** — one `complete(prompt, provider)` that runs across OpenAI, Anthropic, Google, Ollama, and MLX. A single adapter covers every OpenAI-compatible endpoint; Anthropic and Google get thin native branches. Local-first via Ollama so nothing requires an API key to run.
2. **Token budgeter** — count tokens before sending, fit prompts to a model's context window, and trim conversation history under a budget. The piece that prevents silent truncation and surprise bills.
3. **Extraction** — turn freeform model output into typed structured data (JSON / Pydantic schemas), with validation and a single retry loop. The bridge between an LLM and the rest of a program.

Each component is small, standalone, and readable top-to-bottom.

## Requirements

- Python 3.12 (managed by `uv`)
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- [Ollama](https://ollama.com) for the local provider (no API keys required to start)
- A local model: `ollama pull qwen3.5:8b`
- Ollama running: `ollama serve` (defaults to `http://localhost:11434`)

Python dependencies (declared in `pyproject.toml`):

- `openai` — OpenAI, Ollama, MLX, and any OpenAI-compatible endpoint
- `anthropic` — Claude
- `google-genai` — Gemini

## Run

```bash
uv run python client.py
```

Default provider is `"local"` (Ollama), so no API keys are needed for the first run.

## Cloud providers (optional)

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
```

## MLX on Mac (optional)

Run an MLX-compatible server on `http://localhost:8080/v1` to use the `"mlx"` provider.
