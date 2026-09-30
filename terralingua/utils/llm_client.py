import ast
import asyncio
import json
import re
from dataclasses import dataclass
from typing import Dict, List

import litellm
from dotenv import find_dotenv, load_dotenv
from litellm import ModelResponse
from litellm.exceptions import BadRequestError

load_dotenv(find_dotenv(usecwd=True), override=True)

litellm.suppress_debug_info = True

_LONG_CTX_HEADERS = {"anthropic-beta": "context-1m-2025-08-07"}


@dataclass
class Response:
    content: str | None | Dict
    input_tokens: int
    output_tokens: int
    cost: float = 0.0


def _response_cost(resp: ModelResponse) -> float:
    """Best-effort USD cost of a single litellm response.

    litellm attaches its own cost calculation to ``_hidden_params["response_cost"]``;
    it already accounts for the model's pricing and Anthropic prompt-cache
    read/write tokens. Fall back to ``completion_cost()`` and finally to 0.0 for
    models litellm can't price (e.g. local vLLM endpoints)."""
    hidden = getattr(resp, "_hidden_params", None) or {}
    cost = hidden.get("response_cost")
    if cost is not None:
        return float(cost)
    try:
        return float(litellm.completion_cost(completion_response=resp))
    except Exception:
        return 0.0


def _remove_thinking_tags(text: str) -> str:
    if "</think>" in text:
        return text.partition("</think>")[2]
    return text


def _extract_json(
    text: str, has_response_format: bool
) -> tuple[str | None, str | None]:
    if has_response_format:
        return text, None
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        return json_match.group(1), None
    json_match = re.search(r"(\{.*\})", text, re.DOTALL)
    if json_match:
        return json_match.group(1), None
    return None, "Error: No valid JSON found in response"


def _cache_system_messages(messages: list[dict]) -> list[dict]:
    """Wrap plain-string system message content in an Anthropic cache_control block."""
    result = []
    for msg in messages:
        if msg["role"] == "system" and isinstance(msg.get("content"), str):
            result.append(
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": msg["content"],
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            )
        else:
            result.append(msg)
    return result


class LLMClient:
    """LLM client for structured JSON workflows (anthropologist, analysts).

    Wraps litellm.completion with retry logic, JSON extraction, and
    Anthropic prompt-caching support.
    """

    def __init__(self, client: str = "anthropic", long_context: bool = False):
        if client not in ("openai", "anthropic"):
            raise ValueError(f"Unsupported client: {client}")
        self.provider = client
        self.long_context = long_context

    async def _acompletion(
        self,
        model: str,
        messages: list[dict],
        chat_parameters: dict,
        api_key: str | None = None,
    ) -> tuple[str, int, int]:
        """Single LLM round-trip via litellm.acompletion(). All retry / JSON-
        parsing / error-reprompting lives one layer up in get_response_async."""
        kwargs: dict = {"model": model, "messages": messages, **chat_parameters}
        if self.provider == "anthropic":
            kwargs["model"] = f"anthropic/{model}"
            kwargs["messages"] = _cache_system_messages(messages)
            if self.long_context:
                kwargs["extra_headers"] = _LONG_CTX_HEADERS
        if api_key is not None:
            kwargs["api_key"] = api_key
        resp: ModelResponse = await litellm.acompletion(**kwargs)  # type: ignore[assignment]
        text = resp.choices[0].message.content.strip()
        return text, resp.usage.prompt_tokens, resp.usage.completion_tokens

    async def get_response_async(
        self,
        model: str,
        messages: list[dict],
        chat_parameters: dict | None = None,
        max_retries: int = 10,
        enable_error_reprompting: bool = True,
        track_tokens: bool = True,
        output_json: bool = True,
        api_key: str | None = None,
    ) -> Response:
        """Async LLM call with retry, JSON extraction, and error reprompting.

        Single source of truth for the retry/parse logic. The sync wrapper
        ``get_response`` defers to this via ``asyncio.run`` so the two paths
        cannot drift."""
        if chat_parameters is None:
            chat_parameters = {}

        token_counter = {"input": 0, "output": 0}
        messages_copy = list(messages)
        has_response_format = "response_format" in chat_parameters

        for trial in range(max_retries):
            try:
                text, input_tokens, output_tokens = await self._acompletion(
                    model, messages_copy, chat_parameters, api_key=api_key
                )

                if track_tokens:
                    token_counter["input"] += input_tokens
                    token_counter["output"] += output_tokens

                text = _remove_thinking_tags(text)

                if not output_json:
                    return Response(
                        content=text,
                        input_tokens=token_counter["input"],
                        output_tokens=token_counter["output"],
                    )

                try:
                    return Response(
                        content=json.loads(text),
                        input_tokens=token_counter["input"],
                        output_tokens=token_counter["output"],
                    )
                except json.JSONDecodeError:
                    pass

                json_str, error_msg = _extract_json(text, has_response_format)
                if json_str:
                    try:
                        return Response(
                            content=json.loads(json_str),
                            input_tokens=token_counter["input"],
                            output_tokens=token_counter["output"],
                        )
                    except json.JSONDecodeError:
                        try:
                            parsed = ast.literal_eval(json_str)
                            return Response(
                                content=parsed,
                                input_tokens=token_counter["input"],
                                output_tokens=token_counter["output"],
                            )
                        except (ValueError, SyntaxError) as e:
                            error_msg = f"JSON parsing error: {str(e)} \n Response was: {json_str}"
                    except Exception as e:
                        error_msg = f"Unexpected error parsing JSON: {str(e)} \n Response was: {json_str}"

                if not enable_error_reprompting:
                    raise ValueError(error_msg)

                messages_copy.extend(
                    [
                        {"role": "assistant", "content": text},
                        {
                            "role": "user",
                            "content": f"{error_msg}\nPlease provide a valid JSON response.",
                        },
                    ]
                )

            except (BadRequestError, ValueError):
                raise
            except Exception as e:
                if trial == max_retries - 1:
                    raise Exception(
                        f"Failed after {max_retries} retries. Last error: {str(e)}"
                    )

        raise Exception(f"Failed to get valid response after {max_retries} retries")

    def get_response(
        self,
        model: str,
        messages: list[dict],
        chat_parameters: dict | None = None,
        max_retries: int = 10,
        enable_error_reprompting: bool = True,
        track_tokens: bool = True,
        output_json: bool = True,
        api_key: str | None = None,
    ) -> Response:
        """Sync shim around ``get_response_async``.

        Kept during migration so existing sync callers keep working. Internally
        spins up a one-shot event loop via ``asyncio.run``; cannot be invoked
        from inside a running event loop (use ``get_response_async`` there).
        Will be removed once every caller has migrated to async."""
        return asyncio.run(
            self.get_response_async(
                model=model,
                messages=messages,
                chat_parameters=chat_parameters,
                max_retries=max_retries,
                enable_error_reprompting=enable_error_reprompting,
                track_tokens=track_tokens,
                output_json=output_json,
                api_key=api_key,
            )
        )


class AgentClient:
    """LLM client for environment agents.

    Thin wrapper around litellm supporting OpenAI, Anthropic, and local
    OpenAI-compatible endpoints (vLLM). Handles prompt caching and
    provider-specific model prefixing automatically. Auto mode preserves an
    explicit provider/model identifier for LiteLLM routing.
    """

    def __init__(self, provider: str = "openai", **kwargs):
        if provider not in ("openai", "anthropic", "auto"):
            raise ValueError(f"Unsupported provider: {provider}")
        self.provider = provider
        self._extra_kwargs = kwargs  # base_url, api_key for local vLLM endpoints

    def _build_kwargs(self, messages: list[dict], chat_params: dict) -> dict:
        model = chat_params.get("model", "")
        params = {k: v for k, v in chat_params.items() if k != "model"}

        if self.provider == "anthropic":
            # reasoning_effort stays in params: litellm maps it to Anthropic
            # adaptive thinking + output_config.effort (Opus 4.6+/Sonnet 4.6).
            model = f"anthropic/{model}"
            messages = _cache_system_messages(messages)
        elif self._extra_kwargs.get("base_url"):
            # Local OpenAI-compatible endpoint (vLLM) — prefix required by litellm
            model = f"openai/{model}"

        return {"model": model, "messages": messages, **params, **self._extra_kwargs}

    def get_response(
        self, messages: List[Dict[str, str]], chat_params: Dict
    ) -> Response:
        resp: ModelResponse = litellm.completion(
            **self._build_kwargs(messages, chat_params)
        )  # type: ignore[assignment]
        return Response(
            content=resp.choices[0].message.content,
            input_tokens=resp.usage.prompt_tokens or 0,
            output_tokens=resp.usage.completion_tokens or 0,
            cost=_response_cost(resp),
        )

    async def get_response_async(
        self, messages: List[Dict[str, str]], chat_params: Dict
    ) -> Response:
        resp: ModelResponse = await litellm.acompletion(
            **self._build_kwargs(messages, chat_params)
        )  # type: ignore[assignment]
        return Response(
            content=resp.choices[0].message.content,
            input_tokens=resp.usage.prompt_tokens or 0,
            output_tokens=resp.usage.completion_tokens or 0,
            cost=_response_cost(resp),
        )
