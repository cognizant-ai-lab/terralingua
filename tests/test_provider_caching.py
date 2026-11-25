"""Cache boundaries survive provider/model switches and SDK serialization."""

import copy

import pytest
from litellm import ModelResponse

from terralingua.experiment.llm_router import LLMRouter
from terralingua.utils.llm_client import AgentClient, _cache_usage


@pytest.mark.parametrize(
    "provider,model,key",
    [
        ("anthropic", "claude-sonnet-5-5", "cache_control"),
        ("anthropic", "anthropic/claude-sonnet-5-5", "cache_control"),
        ("auto", "anthropic/claude-sonnet-5-5", "cache_control"),
        ("openai", "gpt-6.1-terra", "prompt_cache_breakpoint"),
        ("auto", "openai/gpt-6.1-terra", "prompt_cache_breakpoint"),
    ],
)
def test_instruction_blocks_and_explicit_routes(provider, model, key):
    messages = [
        {"role": "system", "content": "Common instructions"},
        {
            "role": "developer",
            "content": [
                {"type": "text", "text": "Server instructions"},
                {"type": "text", "text": "MCP definitions"},
            ],
        },
        {"role": "user", "content": "Changing observation"},
        {"role": "system", "content": "Turn-specific correction"},
    ]
    original = copy.deepcopy(messages)
    request = AgentClient(provider)._build_kwargs(messages, {"model": model})
    sent = request["messages"]
    assert key in sent[1]["content"][-1]
    assert key not in sent[1]["content"][0]
    assert sent[0] == original[0]
    assert sent[2:] == original[2:]
    assert messages == original
    assert "anthropic/anthropic/" not in request["model"]


@pytest.mark.parametrize(
    "model,key,marker",
    [
        (
            "anthropic/claude-sonnet-5-5",
            "cache_control",
            {"type": "ephemeral", "ttl": "1h"},
        ),
        (
            "openai/gpt-6.1-terra",
            "prompt_cache_breakpoint",
            {"mode": "explicit", "ttl": "12h"},
        ),
    ],
)
def test_preserves_explicit_cache_settings(model, key, marker):
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "Rules", key: marker},
            ],
        }
    ]
    params = {"model": model, "prompt_cache_options": {"mode": "implicit"}}
    request = AgentClient("auto")._build_kwargs(messages, params)
    assert request["messages"] == messages
    assert request["prompt_cache_options"] == params["prompt_cache_options"]


@pytest.mark.parametrize("argument", ["base_url", "api_base"])
@pytest.mark.parametrize("where", ["client", "params"])
@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://api.openai.com/v1", True),
        ("http://localhost:8000/v1", False),
        ("https://api.openai.com.other.example/v1", False),
    ],
)
def test_openai_endpoint_detection(argument, where, url, expected):
    extra = {argument: url}
    client = AgentClient(**(extra if where == "client" else {}))
    params = {"model": "openai/gpt-6.1-terra"}
    if where == "params":
        params.update(extra)
    request = client._build_kwargs(
        [{"role": "system", "content": "Stable rules"}],
        params,
    )
    assert ("prompt_cache_options" in request) == expected
    assert request["model"] == "openai/gpt-6.1-terra"


@pytest.mark.parametrize(
    "model",
    [
        "claude-opus-5.5",
        "anthropic/claude-opus-5.5",
        "claude-sonnet-5-5",
        "claude-haiku-4-5",
        "gpt-6.1-terra",
        "openai/gpt-6.1-terra",
        "gpt-6.2-astra",
    ],
)
def test_router_applies_caching_for_native_and_qualified_models(model):
    router = LLMRouter(model, ports=None, instances=1)
    client, params = router.next()
    request = client._build_kwargs(
        [{"role": "system", "content": "Instructions"}],
        params,
    )
    block = request["messages"][0]["content"][0]
    key = (
        "cache_control" if client.provider == "anthropic" else "prompt_cache_breakpoint"
    )
    assert key in block


def test_anthropic_usage_fallback_keeps_cache_writes_and_reads():
    response = ModelResponse(
        usage={
            "prompt_tokens": 5010,
            "completion_tokens": 10,
            "cache_read_input_tokens": 4000,
            "cache_creation_input_tokens": 1000,
        }
    )
    assert _cache_usage(response) == {
        "cached_input_tokens": 4000,
        "cache_write_input_tokens": 1000,
    }


@pytest.mark.parametrize("role", ["system", "developer"])
def test_anthropic_breakpoint_reaches_http_request(monkeypatch, role):
    """Use LiteLLM's actual Anthropic adapter with only the HTTP call mocked."""
    import json

    import httpx
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    sent = []

    def handle(request):
        sent.append(json.loads(request.content))
        assert request.url.path == "/v1/messages"
        return httpx.Response(
            200,
            json={
                "id": "msg_cache_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-5-5",
                "content": [{"type": "text", "text": "{}"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "cache_read_input_tokens": 4000,
                    "cache_creation_input_tokens": 1000,
                },
            },
        )

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        response = AgentClient("auto", client=HTTPHandler(client=http)).get_response(
            [
                {"role": role, "content": [{"type": "text", "text": "Rules"}]},
                {"role": "user", "content": "New observation"},
            ],
            {"model": "anthropic/claude-sonnet-5-5", "max_tokens": 128},
        )
    assert sent[0]["system"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in json.dumps(sent[0]["messages"])
    assert response.input_tokens == 5010
    assert response.cached_input_tokens == 4000
    assert response.cache_write_input_tokens == 1000
