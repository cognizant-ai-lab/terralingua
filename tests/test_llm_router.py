"""Model routing keeps provider names and per-run settings consistent."""

from terralingua.experiment.llm_router import LLMRouter


def test_opus_55_uses_canonical_anthropic_model_and_keeps_run_settings():
    router = LLMRouter(
        model_short="claude-opus-5.5", ports=None, instances=2,
        max_tokens=100000, reasoning_effort="medium",
    )
    messages = [
        {"role": "system", "content": "Return JSON."},
        {"role": "user", "content": "Choose an action."},
    ]
    for _ in range(3):
        client, params = router.next()
        request = client._build_kwargs(messages, params)
        assert client.provider == "anthropic"
        assert request["model"] == "anthropic/claude-opus-5-5"
        assert request["max_tokens"] == 100000
        assert request["reasoning_effort"] == "medium"
        assert request["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}

    router.refresh(instances=1)
    client, params = router.next()
    assert client.provider == "anthropic"
    assert params["model"] == "claude-opus-5-5"
    assert params["max_tokens"] == 100000
    assert params["reasoning_effort"] == "medium"
