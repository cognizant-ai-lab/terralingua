import logging
import re
from itertools import cycle
from typing import Tuple

import requests

from terralingua.utils.llm_client import AgentClient

log = logging.getLogger(__name__)

MODEL_MAP = {
    "o4-mini": "o4-mini",
    "o3-mini": "o3-mini",
    "gpt-5.1": "gpt-5.1",
    "gpt-5-mini": "gpt-5-mini",
    "QWEN2.5": "Qwen/Qwen2.5-32B-Instruct",
    "QWEN3": "Qwen/Qwen3-32B",
    "DeepSeek-R1-32": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "DeepSeek-R1-70": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "claude-sonnet-4-6": "claude-sonnet-4-6",
    "claude-haiku-4-5": "claude-haiku-4-5",
    "claude-opus-4-8": "claude-opus-4-8",
    "claude-opus-5-5": "claude-opus-5-5",
    "claude-opus-5.5": "claude-opus-5-5",
    "claude-sonnet-5": "claude-sonnet-5",
}


class LLMRouter:
    def __init__(
        self,
        model_short: str,
        ports: Tuple[int] | None,
        instances: int | None = None,
        *,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ):
        self.model_name, self.provider = self.resolve_model(model_short)
        # Per-experiment reasoning/token overrides (from AgentConfig). Unset knobs
        # leave each model's built-in default in place. Stored on self so they are
        # re-applied when refresh() rebuilds clients.
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self._helper_routers: dict[tuple[str, str], LLMRouter] = {}
        self.refresh(ports, instances)

    @staticmethod
    def resolve_model(model: str) -> tuple[str, str]:
        """Resolve registered aliases, native IDs, and explicit provider prefixes."""
        resolved = MODEL_MAP.get(model, model)
        local_models = {MODEL_MAP[key] for key in (
            "QWEN2.5", "QWEN3", "DeepSeek-R1-32", "DeepSeek-R1-70",
        )}
        if resolved in local_models:
            return resolved, "local"
        for provider in ("anthropic", "openai"):
            prefix = provider + "/"
            if resolved.startswith(prefix) and resolved[len(prefix):]:
                if provider == "openai":
                    return resolved, provider
                model_id = resolved[len(prefix):]
                return MODEL_MAP.get(model_id, model_id), provider
        if "/" in resolved and all(resolved.split("/", 1)):
            return resolved, "auto"
        if resolved.startswith("claude-"):
            return resolved, "anthropic"
        if resolved.startswith(("gpt-", "chatgpt-")) or re.fullmatch(r"o\d+(?:[-.].*)?", resolved):
            return resolved, "openai"
        raise ValueError(
            f"Unknown model: {model}. Use a registered alias or a provider/model identifier."
        )

    def next_for_model(self, model: str):
        """Select an independent helper client with that model's defaults."""
        resolved = self.resolve_model(model)
        helper = self._helper_routers.get(resolved)
        if helper is None:
            helper = LLMRouter(model, ports=self.ports, instances=1)
            self._helper_routers[resolved] = helper
        return helper.next()

    def _apply_overrides(self, chat_params: dict, provider: str) -> None:
        """Overlay the per-experiment reasoning/token knobs onto a model's default
        chat params. max_tokens caps output for any provider. reasoning_effort is
        handed to litellm, which maps it to the provider's reasoning control:
        OpenAI's reasoning_effort, or Anthropic adaptive thinking +
        output_config.effort for Opus 4.6+/Sonnet 4.6 (requires litellm>=1.90.2).
        Local vLLM models emit their own <think> reasoning and take no such param."""
        if self.max_tokens is not None:
            chat_params["max_tokens"] = self.max_tokens
        if self.reasoning_effort is not None and provider in ("openai", "anthropic"):
            chat_params["reasoning_effort"] = self.reasoning_effort

    def refresh(self, ports=None, instances=None):
        self.ports = ports
        self._helper_routers.clear()
        if self.provider == "local":
            if ports is None:
                raise ValueError(
                    f"Ports must be specified for local model {self.model_name}"
                )
            self.clients = self._discover_local(ports)
        else:
            if instances is None:
                raise ValueError(
                    f"Instances must be specified for remote model {self.model_name}"
                )
            self.clients = [self._build_remote_client() for _ in range(instances)]
        self.cycle = cycle(self.clients)

    def _discover_local(self, ports):
        available = []
        for p in ports:
            try:
                r = requests.get(f"http://127.0.0.1:{p}/v1/models", timeout=2)
                if r.status_code != 200:
                    continue
                data = r.json().get("data", [{}])[0]
                log.info(f"Port {p}: hosting model {data.get('id')}")

                try:
                    if data.get("id") == self.model_name:
                        available.append(self._build_local_client(p))
                except Exception as e:
                    log.warning(
                        f"Error building client for model {self.model_name} on port {p}: {e}"
                    )
                    continue
            except requests.exceptions.RequestException:
                log.debug(f"Port {p}: no response")
                continue

        if not available:
            raise RuntimeError(f"No VLLM ports hosting {self.model_name}")

        return available

    def _build_local_client(self, port):
        if self.model_name == MODEL_MAP["QWEN2.5"]:
            llm_client = AgentClient(
                base_url=f"http://127.0.0.1:{port}/v1", api_key="EMPTY"
            )
            llm_chat_params = {
                "model": "Qwen/Qwen2.5-32B-Instruct",
                "response_format": {"type": "json_object"},
                "temperature": 1,
            }
        elif self.model_name == MODEL_MAP["QWEN3"]:
            llm_client = AgentClient(
                base_url=f"http://127.0.0.1:{port}/v1", api_key="EMPTY"
            )
            llm_chat_params = {
                "model": "Qwen/Qwen3-32B",
                "response_format": {"type": "json_object"},
                "temperature": 1,
                "max_tokens": 256,  # Limit output to 256 tokens
            }
        elif self.model_name == MODEL_MAP["DeepSeek-R1-32"]:
            llm_client = AgentClient(
                base_url=f"http://127.0.0.1:{port}/v1", api_key="EMPTY"
            )
            llm_chat_params = {
                "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                "temperature": 1,
                "post_prompt": "NOTE: do NOT spend too much time and tokens reasoning.",
            }
        elif self.model_name == MODEL_MAP["DeepSeek-R1-70"]:
            llm_client = AgentClient(
                base_url=f"http://127.0.0.1:{port}/v1", api_key="EMPTY"
            )
            llm_chat_params = {
                "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
                "temperature": 1,
                "post_prompt": "NOTE: do NOT spend too much time and tokens reasoning.",
            }
        else:
            raise ValueError(f"Unsupported local model: {self.model_name}.")
        self._apply_overrides(llm_chat_params, "local")
        return llm_client, llm_chat_params

    def _build_remote_client(self):
        if self.provider == "anthropic":
            llm_client = AgentClient(provider="anthropic")
            llm_chat_params = {"model": self.model_name, "max_tokens": 20000}
        elif self.provider in ("openai", "auto"):
            llm_client = AgentClient(provider=self.provider)
            llm_chat_params = {"model": self.model_name}
            if self.provider == "openai":
                llm_chat_params["response_format"] = {"type": "json_object"}
                model_id = self.model_name.removeprefix("openai/")
                if model_id in ("o3-mini", "gpt-5.1", "gpt-5-mini"):
                    llm_chat_params["reasoning_effort"] = "low"
        else:
            raise ValueError(f"Unsupported model: {self.model_name}.")
        self._apply_overrides(llm_chat_params, self.provider)
        return llm_client, llm_chat_params

    def next(self):
        return next(self.cycle)
