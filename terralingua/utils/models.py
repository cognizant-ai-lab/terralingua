"""Single source of truth for LLM models exposed across the project.

Both the dashboard model picker (served via /api/models) and the analyst
context/output token tables (in terralingua.utils.llm_utils) derive from MODELS here.
Keep additions here so the frontend list and backend lookup stay in sync.
"""

from typing import TypedDict


class Model(TypedDict, total=False):
    id: str
    label: str
    provider: str
    context: int
    long_context: int | None
    output: int
    aliases: list[str]


MODELS: list[Model] = [
    {
        "id": "claude-opus-4-7",
        "label": "Claude Opus 4.7",
        "provider": "anthropic",
        "context": 200_000,
        "long_context": 1_000_000,
        "output": 32_000,
        "aliases": [],
    },
    {
        "id": "claude-sonnet-4-6",
        "label": "Claude Sonnet 4.6",
        "provider": "anthropic",
        "context": 200_000,
        "long_context": 1_000_000,
        "output": 32_000,
        "aliases": [],
    },
    {
        "id": "claude-sonnet-4-5",
        "label": "Claude Sonnet 4.5",
        "provider": "anthropic",
        "context": 200_000,
        "long_context": 1_000_000,
        "output": 16_000,
        "aliases": ["claude-sonnet-4-5-20250929"],
    },
    {
        "id": "claude-haiku-4-5",
        "label": "Claude Haiku 4.5",
        "provider": "anthropic",
        "context": 200_000,
        "long_context": None,
        "output": 16_000,
        "aliases": ["claude-haiku-4-5-20251001"],
    },
    {
        "id": "gpt-5.5",
        "label": "GPT-5.5",
        "provider": "openai",
        "context": 400_000,
        "long_context": None,
        "output": 128_000,
        "aliases": [],
    },
    {
        "id": "gpt-5.5-pro",
        "label": "GPT-5.5 pro",
        "provider": "openai",
        "context": 400_000,
        "long_context": None,
        "output": 128_000,
        "aliases": [],
    },
    {
        "id": "gpt-5.4",
        "label": "GPT-5.4",
        "provider": "openai",
        "context": 400_000,
        "long_context": None,
        "output": 128_000,
        "aliases": [],
    },
    {
        "id": "gpt-5.4-pro",
        "label": "GPT-5.4 pro",
        "provider": "openai",
        "context": 400_000,
        "long_context": None,
        "output": 128_000,
        "aliases": [],
    },
    {
        "id": "gpt-5.4-mini",
        "label": "GPT-5.4 mini",
        "provider": "openai",
        "context": 400_000,
        "long_context": None,
        "output": 128_000,
        "aliases": [],
    },
    {
        "id": "gpt-5.4-nano",
        "label": "GPT-5.4 nano",
        "provider": "openai",
        "context": 400_000,
        "long_context": None,
        "output": 128_000,
        "aliases": [],
    },
    {
        "id": "gpt-5",
        "label": "GPT-5",
        "provider": "openai",
        "context": 400_000,
        "long_context": None,
        "output": 128_000,
        "aliases": [],
    },
    {
        "id": "gpt-5-mini",
        "label": "GPT-5 mini",
        "provider": "openai",
        "context": 400_000,
        "long_context": None,
        "output": 128_000,
        "aliases": [],
    },
    {
        "id": "gpt-5-nano",
        "label": "GPT-5 nano",
        "provider": "openai",
        "context": 400_000,
        "long_context": None,
        "output": 128_000,
        "aliases": [],
    },
    {
        "id": "gpt-4.1",
        "label": "GPT-4.1",
        "provider": "openai",
        "context": 1_047_576,
        "long_context": None,
        "output": 32_768,
        "aliases": [],
    },
    {
        "id": "gpt-4.1-mini",
        "label": "GPT-4.1 mini",
        "provider": "openai",
        "context": 1_047_576,
        "long_context": None,
        "output": 32_768,
        "aliases": [],
    },
]

_PROVIDER_LABELS = {"anthropic": "Anthropic", "openai": "OpenAI"}


def _all_keys(m: Model) -> list[str]:
    return [m["id"], *m.get("aliases", [])]


def context_table() -> dict[str, dict[str, int]]:
    """{model_id: {"base": int, "long"?: int}} — keyed by canonical id and aliases."""
    out: dict[str, dict[str, int]] = {}
    for m in MODELS:
        entry: dict[str, int] = {"base": m["context"]}
        if m.get("long_context"):
            entry["long"] = m["long_context"]  # type: ignore[assignment]
        for k in _all_keys(m):
            out[k] = entry
    return out


def output_table() -> dict[str, int]:
    """{model_id: max_output_tokens} — keyed by canonical id and aliases."""
    out: dict[str, int] = {}
    for m in MODELS:
        for k in _all_keys(m):
            out[k] = m["output"]
    return out


def models_for_dashboard() -> dict[str, list[dict[str, str]]]:
    """Grouped by provider, ordered as declared in MODELS. Frontend-friendly shape."""
    grouped: dict[str, list[dict[str, str]]] = {}
    for m in MODELS:
        grouped.setdefault(m["provider"], []).append(
            {"value": m["id"], "label": m["label"]}
        )
    return {
        "providers": [
            {"id": p, "label": _PROVIDER_LABELS.get(p, p), "models": models}
            for p, models in grouped.items()
        ]
    }
