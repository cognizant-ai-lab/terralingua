import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import pytest

from terralingua.utils.llm_client import (
    AgentClient,
    LLMClient,
    _cache_system_messages,
    _extract_json,
    _remove_thinking_tags,
)


def _mock_resp(content: str, prompt_tokens: int = 10, completion_tokens: int = 5) -> MagicMock:
    m = MagicMock()
    m.choices[0].message.content = content
    m.usage.prompt_tokens = prompt_tokens
    m.usage.completion_tokens = completion_tokens
    return m


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestRemoveThinkingTags:
    def test_strips_tags(self):
        assert _remove_thinking_tags("thinking\n</think>answer") == "answer"


class TestExtractJson:
    def test_response_format_passthrough(self):
        s, err = _extract_json('{"k": "v"}', has_response_format=True)
        assert s == '{"k": "v"}'
        assert err is None

    def test_markdown_code_block(self):
        text = '```json\n{"key": "value"}\n```'
        s, err = _extract_json(text, has_response_format=False)
        assert s == '{"key": "value"}'
        assert err is None

    def test_bare_json_in_text(self):
        text = 'Here is the result: {"a": 1} done.'
        s, err = _extract_json(text, has_response_format=False)
        assert s is not None
        assert "a" in s
        assert err is None

    def test_no_json_returns_error(self):
        s, err = _extract_json("no json here", has_response_format=False)
        assert s is None
        assert err is not None


class TestCacheSystemMessages:
    def test_wraps_string_system_message(self):
        msgs = [{"role": "system", "content": "be helpful"}, {"role": "user", "content": "hi"}]
        result = _cache_system_messages(msgs)
        assert result[0]["content"] == [
            {"type": "text", "text": "be helpful", "cache_control": {"type": "ephemeral"}}
        ]
        assert result[1] == {"role": "user", "content": "hi"}

    def test_non_system_messages_unchanged(self):
        msgs = [{"role": "user", "content": "hi"}]
        assert _cache_system_messages(msgs) == msgs

    def test_already_list_content_not_double_wrapped(self):
        content = [{"type": "text", "text": "hi"}]
        msgs = [{"role": "system", "content": content}]
        result = _cache_system_messages(msgs)
        assert result[0]["content"] is content


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------

class TestLLMClientInit:
    def test_invalid_provider_raises(self):
        with pytest.raises(ValueError):
            LLMClient(client="gemini")


class TestLLMClientGetResponse:
    @patch("terralingua.utils.llm_client.litellm.acompletion", new_callable=AsyncMock)
    def test_plain_text_output(self, mock_comp):
        mock_comp.return_value = _mock_resp("hello world")
        resp = LLMClient(client="openai").get_response(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}], output_json=False
        )
        assert resp.content == "hello world"
        assert resp.input_tokens == 10
        assert resp.output_tokens == 5

    @patch("terralingua.utils.llm_client.litellm.acompletion", new_callable=AsyncMock)
    def test_json_output_direct(self, mock_comp):
        mock_comp.return_value = _mock_resp('{"k": "v"}')
        resp = LLMClient(client="openai").get_response(
            model="gpt-4o", messages=[], output_json=True
        )
        assert resp.content == {"k": "v"}

    @patch("terralingua.utils.llm_client.litellm.acompletion", new_callable=AsyncMock)
    def test_json_output_from_markdown_block(self, mock_comp):
        mock_comp.return_value = _mock_resp('```json\n{"k": "v"}\n```')
        resp = LLMClient(client="openai").get_response(model="gpt-4o", messages=[], output_json=True)
        assert resp.content == {"k": "v"}

    @patch("terralingua.utils.llm_client.litellm.acompletion", new_callable=AsyncMock)
    def test_json_retry_with_reprompting(self, mock_comp):
        mock_comp.side_effect = [_mock_resp("not json"), _mock_resp('{"k": "v"}')]
        resp = LLMClient(client="openai").get_response(model="gpt-4o", messages=[], output_json=True)
        assert resp.content == {"k": "v"}
        assert mock_comp.call_count == 2

    @patch("terralingua.utils.llm_client.litellm.acompletion", new_callable=AsyncMock)
    def test_disable_reprompting_raises_immediately(self, mock_comp):
        mock_comp.return_value = _mock_resp("not json")
        with pytest.raises(ValueError):
            LLMClient(client="openai").get_response(
                model="gpt-4o", messages=[], output_json=True,
                max_retries=5, enable_error_reprompting=False
            )
        assert mock_comp.call_count == 1

    @patch("terralingua.utils.llm_client.litellm.acompletion", new_callable=AsyncMock)
    def test_max_retries_exhausted_raises(self, mock_comp):
        mock_comp.return_value = _mock_resp("still not json")
        with pytest.raises(Exception, match="Failed"):
            LLMClient(client="openai").get_response(
                model="gpt-4o", messages=[], output_json=True, max_retries=2
            )

    @patch("terralingua.utils.llm_client.litellm.acompletion", new_callable=AsyncMock)
    def test_bad_request_error_reraised(self, mock_comp):
        mock_comp.side_effect = litellm.BadRequestError(
            message="context too long", model="gpt-4o", llm_provider="openai"
        )
        with pytest.raises(litellm.BadRequestError):
            LLMClient(client="openai").get_response(model="gpt-4o", messages=[], output_json=False)

    @patch("terralingua.utils.llm_client.litellm.acompletion", new_callable=AsyncMock)
    def test_thinking_tags_stripped(self, mock_comp):
        mock_comp.return_value = _mock_resp("thinking...\n</think>the answer")
        resp = LLMClient(client="openai").get_response(model="gpt-4o", messages=[], output_json=False)
        assert resp.content == "the answer"

    @patch("terralingua.utils.llm_client.litellm.acompletion", new_callable=AsyncMock)
    def test_token_tracking_accumulates_across_retries(self, mock_comp):
        mock_comp.side_effect = [
            _mock_resp("not json", prompt_tokens=10, completion_tokens=5),
            _mock_resp('{"k": "v"}', prompt_tokens=20, completion_tokens=8),
        ]
        resp = LLMClient(client="openai").get_response(model="gpt-4o", messages=[], output_json=True)
        assert resp.input_tokens == 30
        assert resp.output_tokens == 13

    @patch("terralingua.utils.llm_client.litellm.acompletion", new_callable=AsyncMock)
    def test_anthropic_model_prefixed(self, mock_comp):
        mock_comp.return_value = _mock_resp('{"k": "v"}')
        LLMClient(client="anthropic").get_response(
            model="claude-sonnet-4-6", messages=[{"role": "user", "content": "hi"}]
        )
        assert mock_comp.call_args.kwargs["model"] == "anthropic/claude-sonnet-4-6"

    @patch("terralingua.utils.llm_client.litellm.acompletion", new_callable=AsyncMock)
    def test_anthropic_long_context_header(self, mock_comp):
        mock_comp.return_value = _mock_resp('{"k": "v"}')
        LLMClient(client="anthropic", long_context=True).get_response(
            model="claude-sonnet-4-6", messages=[]
        )
        assert mock_comp.call_args.kwargs.get("extra_headers") == {
            "anthropic-beta": "context-1m-2025-08-07"
        }

    @patch("terralingua.utils.llm_client.litellm.acompletion", new_callable=AsyncMock)
    def test_anthropic_no_long_context_header_by_default(self, mock_comp):
        mock_comp.return_value = _mock_resp('{"k": "v"}')
        LLMClient(client="anthropic", long_context=False).get_response(
            model="claude-sonnet-4-6", messages=[]
        )
        assert "extra_headers" not in mock_comp.call_args.kwargs

    @patch("terralingua.utils.llm_client.litellm.acompletion", new_callable=AsyncMock)
    def test_anthropic_system_message_cached(self, mock_comp):
        mock_comp.return_value = _mock_resp('{"k": "v"}')
        msgs = [{"role": "system", "content": "be helpful"}, {"role": "user", "content": "hi"}]
        LLMClient(client="anthropic").get_response(model="claude-sonnet-4-6", messages=msgs)
        sent_msgs = mock_comp.call_args.kwargs["messages"]
        assert sent_msgs[0]["content"] == [
            {"type": "text", "text": "be helpful", "cache_control": {"type": "ephemeral"}}
        ]


# ---------------------------------------------------------------------------
# AgentClient
# ---------------------------------------------------------------------------

class TestAgentClientInit:
    def test_invalid_provider_raises(self):
        with pytest.raises(ValueError):
            AgentClient(provider="gemini")


class TestAgentClientGetResponse:
    @patch("terralingua.utils.llm_client.litellm.completion")
    def test_openai_model_passthrough(self, mock_comp):
        mock_comp.return_value = _mock_resp("hello")
        resp = AgentClient(provider="openai").get_response(
            messages=[{"role": "user", "content": "hi"}],
            chat_params={"model": "gpt-4o", "temperature": 0.7},
        )
        assert resp.content == "hello"
        kw = mock_comp.call_args.kwargs
        assert kw["model"] == "gpt-4o"
        assert kw["temperature"] == 0.7

    @patch("terralingua.utils.llm_client.litellm.completion")
    def test_anthropic_model_prefixed(self, mock_comp):
        mock_comp.return_value = _mock_resp("hello")
        AgentClient(provider="anthropic").get_response(
            messages=[{"role": "user", "content": "hi"}],
            chat_params={"model": "claude-sonnet-4-6", "max_tokens": 4096},
        )
        assert mock_comp.call_args.kwargs["model"] == "anthropic/claude-sonnet-4-6"

    @patch("terralingua.utils.llm_client.litellm.completion")
    def test_anthropic_forwards_reasoning_effort(self, mock_comp):
        # litellm (>=1.90.2) maps reasoning_effort to Anthropic adaptive thinking +
        # output_config.effort, so the client must pass it through, not strip it.
        mock_comp.return_value = _mock_resp("hello")
        AgentClient(provider="anthropic").get_response(
            messages=[{"role": "user", "content": "hi"}],
            chat_params={"model": "claude-sonnet-4-6", "reasoning_effort": "low"},
        )
        assert mock_comp.call_args.kwargs["reasoning_effort"] == "low"

    @patch("terralingua.utils.llm_client.litellm.completion")
    def test_anthropic_caches_system_message(self, mock_comp):
        mock_comp.return_value = _mock_resp("hello")
        AgentClient(provider="anthropic").get_response(
            messages=[{"role": "system", "content": "be helpful"}, {"role": "user", "content": "hi"}],
            chat_params={"model": "claude-sonnet-4-6"},
        )
        sent = mock_comp.call_args.kwargs["messages"]
        assert sent[0]["content"] == [
            {"type": "text", "text": "be helpful", "cache_control": {"type": "ephemeral"}}
        ]

    @patch("terralingua.utils.llm_client.litellm.completion")
    def test_local_model_prefixed_with_openai(self, mock_comp):
        mock_comp.return_value = _mock_resp("hello")
        AgentClient(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY").get_response(
            messages=[{"role": "user", "content": "hi"}],
            chat_params={"model": "Qwen/Qwen2.5-32B-Instruct", "temperature": 1},
        )
        kw = mock_comp.call_args.kwargs
        assert kw["model"] == "openai/Qwen/Qwen2.5-32B-Instruct"
        assert kw["base_url"] == "http://127.0.0.1:8000/v1"
        assert kw["api_key"] == "EMPTY"

    @patch("terralingua.utils.llm_client.litellm.completion")
    def test_none_usage_defaults_to_zero(self, mock_comp):
        m = _mock_resp("hello")
        m.usage.prompt_tokens = None
        m.usage.completion_tokens = None
        mock_comp.return_value = m
        resp = AgentClient(provider="openai").get_response(
            [{"role": "user", "content": "hi"}], {"model": "gpt-4o"}
        )
        assert resp.input_tokens == 0
        assert resp.output_tokens == 0


class TestAgentClientGetResponseAsync:
    @patch("terralingua.utils.llm_client.litellm.acompletion", new_callable=AsyncMock)
    def test_async_openai(self, mock_acomp):
        mock_acomp.return_value = _mock_resp("async hello")
        resp = asyncio.run(
            AgentClient(provider="openai").get_response_async(
                [{"role": "user", "content": "hi"}], {"model": "gpt-4o"}
            )
        )
        assert resp.content == "async hello"
        assert mock_acomp.call_args.kwargs["model"] == "gpt-4o"
