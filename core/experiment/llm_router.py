from itertools import cycle
from typing import Tuple

import requests

from core.utils.llm_client import AgentClient

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
}


class LLMRouter:
    def __init__(
        self, model_short: str, ports: Tuple[int] | None, instances: int | None = None
    ):
        self.model_name = MODEL_MAP.get(model_short)
        if self.model_name is None:
            raise ValueError(f"Unknown model: {model_short}")
        self.refresh(ports, instances)

    def refresh(self, ports=None, instances=None):
        if "qwen" in self.model_name.lower() or "deepseek" in self.model_name.lower():  # type: ignore
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
                if r.status_code == 200:
                    data = r.json().get("data", [{}])[0]
                    print(f"Port {p}: Hosting model - {data.get('id')}")

                try:
                    if data.get("id") == self.model_name:
                        available.append(self._build_local_client(p))
                except Exception as e:
                    print(
                        f"Error building client for model {self.model_name} on port {p} \n {e}"
                    )
                    continue
            except requests.exceptions.RequestException:
                print(f"Port {p}: No response")
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
        return llm_client, llm_chat_params

    def _build_remote_client(self):
        if self.model_name == "o4-mini":
            llm_client = AgentClient(provider="openai")
            llm_chat_params = {
                "model": "o4-mini",
                "response_format": {"type": "json_object"},
            }
        elif self.model_name == "o3-mini":
            llm_client = AgentClient(provider="openai")
            llm_chat_params = {
                "model": "o3-mini",
                "response_format": {"type": "json_object"},
                "reasoning_effort": "low",
            }
        elif self.model_name == "gpt-5.1":
            llm_client = AgentClient(provider="openai")
            llm_chat_params = {
                "model": "gpt-5.1",
                "response_format": {"type": "json_object"},
                "reasoning_effort": "low",
            }
        elif self.model_name == "gpt-5-mini":
            llm_client = AgentClient(provider="openai")
            llm_chat_params = {
                "model": "gpt-5-mini",
                "response_format": {"type": "json_object"},
                "reasoning_effort": "low",
            }
        elif "claude" in str(self.model_name).lower():
            llm_client = AgentClient(provider="anthropic")
            llm_chat_params = {
                "model": self.model_name,
                "max_tokens": 4096,
            }
        else:
            raise ValueError(f"Unsupported model: {self.model_name}.")
        return llm_client, llm_chat_params

    def next(self):
        return next(self.cycle)
