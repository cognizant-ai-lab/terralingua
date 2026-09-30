"""Correlated tool replies remain distinct from delivery and observed effects."""

import asyncio
import json
import threading
import types
from collections import defaultdict, deque

import pytest

from terralingua.external.request_manager import BATCH_TOOL_NAME, ExternalRequestManager


def result(text="", *, error=False, structured=None):
    return types.SimpleNamespace(
        content=[types.SimpleNamespace(text=text)] if text else [],
        isError=error, structuredContent=structured,
    )


def manager(session):
    mgr = ExternalRequestManager.__new__(ExternalRequestManager)
    mgr._lock = threading.Lock()
    mgr._agent_queues = defaultdict(deque)
    mgr._broadcast_queue = deque()
    mgr._session = session
    mgr.tool_info = types.SimpleNamespace(name="act")
    mgr.tools = {"act": types.SimpleNamespace(inputSchema={}), BATCH_TOOL_NAME: object()}
    return mgr


class ImmediateSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_out_of_order_replies_keep_original_request_id():
    async def exercise():
        started = asyncio.Event()
        release = asyncio.Event()

        class Session:
            async def call_tool(self, name, arguments):
                if arguments["value"] == "old":
                    started.set()
                    await release.wait()
                return result(json.dumps(arguments))

        mgr = manager(Session())
        old = asyncio.create_task(mgr._deliver_message_async("a", {"value": "old"}, "old-request"))
        await started.wait()
        await mgr._deliver_message_async("a", {"value": "new"}, "new-request")
        first = mgr.get_incoming_records("a")
        assert [rec["message_id"] for rec in first] == ["new-request"]
        assert json.loads(first[0]["text"]) == {"value": "new"}
        release.set()
        await old
        late = mgr.get_incoming_records("a")
        assert [rec["message_id"] for rec in late] == ["old-request"]
        assert json.loads(late[0]["text"]) == {"value": "old"}

    asyncio.run(exercise())


def test_mcp_error_is_retained_even_when_body_looks_successful():
    mgr = manager(ImmediateSession(result('{"reward": 50}', error=True)))
    delivered, status = asyncio.run(mgr._deliver_message_async("a", {}, "request-1"))
    assert delivered is False
    assert "tool error" in status
    reply = mgr.get_incoming_records("a")[0]
    assert reply["message_id"] == "request-1"
    assert reply["tool_error"] is True
    assert reply["delivery_failed"] is False
    assert reply["text"] == '{"reward": 50}'


def test_empty_reply_has_correlation_but_no_outcome_text():
    mgr = manager(ImmediateSession(result()))
    asyncio.run(mgr._deliver_message_async("a", {}, "empty-request"))
    assert mgr.get_incoming_records("a") == [{
        "text": "", "message_id": "empty-request", "tool_error": False,
        "delivery_failed": False,
    }]
    assert mgr.get_incoming_messages("a") == []


def test_structured_only_mcp_result_is_preserved():
    mgr = manager(ImmediateSession(result(structured={"counter": 2})))
    asyncio.run(mgr._deliver_message_async("a", {}, "request-1"))
    reply = mgr.get_incoming_records("a")[0]
    assert json.loads(reply["text"]) == {"counter": 2}
    assert reply["structured_content"] == {"counter": 2}


def test_transport_failure_keeps_request_identity_without_effect_claim():
    mgr = manager(ImmediateSession(TimeoutError("late")))
    delivered, status = asyncio.run(mgr._deliver_message_async("a", {}, "request-1"))
    assert delivered is False
    assert "failed" in status
    reply = mgr.get_incoming_records("a")[0]
    assert reply["message_id"] == "request-1"
    assert reply["delivery_failed"] is True
    assert "effect_status" not in reply


def test_empty_error_callback_keeps_request_identity():
    mgr = manager(ImmediateSession(result()))
    mgr._queue_callback(types.SimpleNamespace(
        target=types.SimpleNamespace(agent_tag="a"), payload="  ",
        meta={"request_id": "old-request"}, type="error",
    ))
    assert mgr.get_incoming_records("a") == [{
        "text": "", "message_id": "old-request", "tool_error": True,
        "delivery_failed": False,
    }]


def test_callback_message_id_is_not_assumed_to_be_a_request_id():
    mgr = manager(ImmediateSession(result()))
    mgr._queue_callback(types.SimpleNamespace(
        id="an-envelope-id", target=types.SimpleNamespace(agent_tag="a"),
        payload="An observation", meta={}, type="event",
    ))
    assert mgr.get_incoming_records("a")[0]["message_id"] == ""


def test_batch_replies_keep_batch_id_and_agent_identity():
    mgr = manager(ImmediateSession(result(json.dumps({"results": {
        "a": {"counter": 1}, "b": {"counter": 2}, "unsolicited": "notice",
    }}))))
    entries = [{"agent_tag": tag, "tool": "act", "args": {}} for tag in ("a", "b")]
    asyncio.run(mgr._deliver_batch_async(entries, message_id="batch-1"))
    assert mgr._session.calls == [(BATCH_TOOL_NAME, {"entries": entries})]
    a, b = mgr.get_incoming_records("a")[0], mgr.get_incoming_records("b")[0]
    assert a["message_id"] == b["message_id"] == "batch-1"
    assert json.loads(a["text"]) == {"counter": 1}
    assert json.loads(b["text"]) == {"counter": 2}
    unsolicited = mgr.get_incoming_records("unsolicited")[0]
    assert unsolicited["message_id"] == ""
    assert unsolicited["text"] == "notice"


def test_batch_mcp_error_produces_correlated_error_for_each_request():
    mgr = manager(ImmediateSession(result("batch failed", error=True)))
    entries = [{"agent_tag": tag, "tool": "act", "args": {}} for tag in ("a", "b")]
    with pytest.raises(ValueError, match="MCP tool returned an error"):
        asyncio.run(mgr._deliver_batch_async(entries, message_id="batch-1"))
    for tag in ("a", "b"):
        reply = mgr.get_incoming_records(tag)[0]
        assert reply["message_id"] == "batch-1"
        assert reply["tool_error"] is True
