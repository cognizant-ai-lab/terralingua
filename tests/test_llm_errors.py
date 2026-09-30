"""The LLM error classifier handles every body shape a provider may send."""

from types import SimpleNamespace

from terralingua.utils.llm_errors import llm_error_type


def test_nested_type_wins():
    exc = SimpleNamespace(status_code=429, body={"error": {"type": "rate_limit_error"}})
    assert llm_error_type(exc) == "rate_limit_error"


def test_string_error_falls_back_to_top_level_type_then_status():
    assert llm_error_type(SimpleNamespace(status_code=400, body={"error": "bad request", "type": "invalid_request"})) == "invalid_request"
    assert llm_error_type(SimpleNamespace(status_code=500, body={"error": ["a", "b"]})) == "http_500"
    assert llm_error_type(SimpleNamespace(status_code=502, body="gateway")) == "http_502"


def test_without_status_code_is_not_an_api_error():
    assert llm_error_type(ValueError("x")) is None
