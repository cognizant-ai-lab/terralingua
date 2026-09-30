"""Helper requests use their selected model's route and output defaults."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest

from terralingua.experiment import llm_router as router_module
from terralingua.experiment.llm_router import MODEL_MAP, LLMRouter
from terralingua.utils.llm_client import AgentClient


@pytest.fixture
def sent(monkeypatch):
    requests = []

    async def complete(client, *, messages, chat_params):
        requests.append(client._build_kwargs(messages, chat_params))
        return SimpleNamespace(content="HIT")

    monkeypatch.setattr(AgentClient, "get_response_async", complete)
    return requests


def router_for(agent_model, ports=None):
    return LLMRouter(agent_model, ports=ports, instances=1, max_tokens=91, reasoning_effort="max")


def call(router, model):
    client, chat_params = router.next_for_model(model)
    reply = asyncio.run(client.get_response_async(
        messages=[{"role": "user", "content": "Inspect this result"}], chat_params=chat_params,
    ))
    return reply.content


def local_ports(monkeypatch, models):
    calls = []

    def get(url, timeout):
        port = urlparse(url).port
        calls.append(port)
        model = models[port]
        if model is None:
            return SimpleNamespace(status_code=503)
        return SimpleNamespace(status_code=200, json=lambda: {"data": [{"id": model}]})

    monkeypatch.setattr(router_module.requests, "get", get)
    return calls


@pytest.mark.parametrize("agent_model,helper_model,wire_model,cap,effort", [
    ("claude-opus-5-5", "gpt-5-mini", "gpt-5-mini", None, "low"),
    ("gpt-5-mini", "claude-haiku-4-5", "anthropic/claude-haiku-4-5", 20000, None),
    ("gpt-5-mini", "gpt-5-mini", "gpt-5-mini", None, "low"),
])
def test_explicit_helper_owns_provider_and_defaults(sent, agent_model, helper_model,
                                                    wire_model, cap, effort):
    router = router_for(agent_model)
    original = deepcopy(router.clients[0][1])
    assert call(router, helper_model) == "HIT"
    request = sent[-1]
    assert request["model"] == wire_model
    assert request.get("max_tokens") == cap
    assert request.get("reasoning_effort") == effort
    assert "base_url" not in request and "api_key" not in request
    assert router.clients[0][1] == original


@pytest.mark.parametrize("helper_model,wire_model,cap", [
    ("claude-haiku-4-5-20251001", "anthropic/claude-haiku-4-5-20251001", 20000),
    ("gpt-4o-mini", "gpt-4o-mini", None),
    ("openai/custom-model", "openai/custom-model", None),
    ("anthropic/claude-opus-5.5", "anthropic/claude-opus-5-5", 20000),
    ("gemini/gemini-2.5-pro", "gemini/gemini-2.5-pro", None),
])
def test_native_and_provider_qualified_helper_ids_remain_supported(sent, helper_model, wire_model, cap):
    call(router_for("claude-opus-5-5"), helper_model)
    assert sent[-1]["model"] == wire_model
    assert sent[-1].get("max_tokens") == cap
    assert "base_url" not in sent[-1]


def test_cloud_helper_does_not_inherit_local_agent_endpoint(monkeypatch, sent):
    local_ports(monkeypatch, {9101: MODEL_MAP["QWEN3"]})
    call(router_for("QWEN3", ports=(9101,)), "claude-haiku-4-5")
    assert sent[-1]["model"] == "anthropic/claude-haiku-4-5"
    assert sent[-1]["max_tokens"] == 20000
    assert "base_url" not in sent[-1] and "api_key" not in sent[-1]


@pytest.mark.parametrize("agent_model", ["claude-opus-5-5", "QWEN2.5"])
def test_local_helper_discovers_its_own_endpoint_and_skips_failed_ports(monkeypatch, sent, agent_model):
    calls = local_ports(monkeypatch, {
        9101: MODEL_MAP["QWEN2.5"], 9102: MODEL_MAP["QWEN3"], 9103: None,
    })
    router = router_for(agent_model, ports=(9101, 9102, 9103))
    for _ in range(2):
        call(router, "QWEN3")
        request = sent[-1]
        assert request["model"] == "openai/Qwen/Qwen3-32B"
        assert request["base_url"] == "http://127.0.0.1:9102/v1"
        assert request["api_key"] == "EMPTY"
        assert request["max_tokens"] == 256
        assert "reasoning_effort" not in request
    # Each relevant router discovers once; helper reuse does not probe again.
    assert calls.count(9102) == (2 if agent_model == "QWEN2.5" else 1)


def test_helper_cache_normalizes_aliases_and_refreshes_local_endpoints(monkeypatch):
    local_ports(monkeypatch, {9101: MODEL_MAP["QWEN3"], 9102: MODEL_MAP["QWEN3"]})
    router = LLMRouter("gpt-5-mini", ports=(9101,), instances=1, max_tokens=91)
    first, params = router.next_for_model("QWEN3")
    same, _ = router.next_for_model(MODEL_MAP["QWEN3"])
    assert first is same and params["max_tokens"] == 256
    router.refresh(ports=(9102,), instances=1)
    refreshed, _ = router.next_for_model("QWEN3")
    assert refreshed is not first
    assert refreshed._extra_kwargs["base_url"] == "http://127.0.0.1:9102/v1"


def test_ambiguous_helper_name_fails_before_any_provider_call(sent):
    with pytest.raises(ValueError, match="provider/model"):
        call(router_for("gpt-5-mini"), "ambiguous-model")
    assert sent == []
