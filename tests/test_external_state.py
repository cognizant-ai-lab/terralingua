"""Tests for the external-server shared-state registry and base handler."""

import pytest

from terralingua.external.external_state import (
    ExternalStateHandler,
    make_state_handler,
    register_state_handler,
    state_handler_kinds,
)


@register_state_handler
class _Counter(ExternalStateHandler):
    kind = "counter_test"
    info_key = "counter"

    def __init__(self):
        self.seen = 0

    def observe(self, messages):
        self.seen += len(messages)

    def summary(self):
        return f"seen={self.seen}" if self.seen else ""


def test_unknown_kind_rejected():
    with pytest.raises(ValueError, match="Unknown external state handler"):
        make_state_handler("nope")


def test_register_requires_kind():
    with pytest.raises(ValueError, match="non-empty 'kind'"):

        @register_state_handler
        class _Bad(ExternalStateHandler):
            pass


def test_registered_kind_is_listed_and_built():
    assert "counter_test" in state_handler_kinds()
    assert isinstance(make_state_handler("counter_test"), _Counter)


def test_inject_writes_summary_under_info_key():
    h = make_state_handler("counter_test")
    infos = {"a0": {}, "a1": {}}
    h.inject(infos)
    assert infos == {"a0": {}, "a1": {}}
    h.observe(["x", "y"])
    h.inject(infos)
    assert infos["a0"]["counter"] == "seen=2"
    assert infos["a1"]["counter"] == "seen=2"
