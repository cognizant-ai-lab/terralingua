"""Tests for batched external dispatch (servers.json ``batch: true``).

Covers the ServerSpec flag, the manager's one-call-per-round delivery, and the
runner's dispatch branch and action-spec filtering.
"""

import asyncio
import json
import threading
import types
from collections import defaultdict, deque

import pytest

from terralingua.experiment.runner import SimulationRunner, _ExternalState
from terralingua.external.request_manager import BATCH_TOOL_NAME, ExternalRequestManager
from terralingua.external.servers_config import ServerSpec
from terralingua.voting import IndependentVotingManager
from terralingua.voting.rewards import DirectRewards

# ---- ServerSpec.batch --------------------------------------------------------

def test_batch_flag_rejects_non_bool():
    with pytest.raises(ValueError, match="batch"):
        ServerSpec(name="s", url="http://a/mcp", callback_port=9000, batch="yes")


# ---- ExternalRequestManager.send_outgoing_batch ------------------------------

class _FakeToolResult:
    def __init__(self, text):
        self.content = [types.SimpleNamespace(text=text)]


class _FakeSession:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return _FakeToolResult(self.reply)


def _bare_manager(reply):
    mgr = ExternalRequestManager.__new__(ExternalRequestManager)
    mgr._lock = threading.Lock()
    mgr._agent_queues = defaultdict(deque)
    mgr._session = _FakeSession(reply)
    mgr.tools = {BATCH_TOOL_NAME: object()}
    return mgr


def test_deliver_batch_rejects_missing_results():
    mgr = _bare_manager(json.dumps({"nope": []}))
    with pytest.raises(ValueError, match="results"):
        asyncio.run(mgr._deliver_batch_async([]))


def test_deliver_batch_requires_batch_tool():
    mgr = _bare_manager("{}")
    mgr.tools = {}
    with pytest.raises(RuntimeError, match=BATCH_TOOL_NAME):
        asyncio.run(mgr._deliver_batch_async([]))


# ---- Runner: batch dispatch branch and action-spec filtering -----------------

class _StubBatchManager:
    def __init__(self):
        self.batch_calls = []
        self.single_calls = []

    def send_outgoing_batch(self, requests, tools):
        self.batch_calls.append((requests, tools))
        return {tag: [{"message_id": "m", "status": "sent and received response"}]
                for tag in requests}

    def send_outgoing_messages(self, msgs, tool_name=None):
        self.single_calls.append((msgs, tool_name))
        return {tag: [{"message_id": "m", "status": "sent and received response"}]
                for tag in msgs}

    def list_tools(self):
        return [
            types.SimpleNamespace(name="act", description="Act.", inputSchema={}),
            types.SimpleNamespace(
                name=BATCH_TOOL_NAME, description="Round.", inputSchema={}
            ),
        ]


def _runner_self(batch: bool):
    manager = _StubBatchManager()
    spec = ServerSpec(
        name="world", url="http://a/mcp", callback_port=9010, batch=batch
    )
    fs = types.SimpleNamespace(
        external_managers={"world": manager},
        external_specs={"world": spec},
        external_actions_spec={
            "world_act": {"server": "world", "tool": "act", "cost": 0,
                         "mode": "independent"},
            "world_observe": {"server": "world", "tool": "observe", "cost": 0,
                             "mode": "independent"},
        },
        voting_managers={"world": IndependentVotingManager(DirectRewards(0.0))},
        _stay_action={"action": "noop", "params": {}},
        env=types.SimpleNamespace(),
    )
    return fs, manager


def test_prestep_batches_all_tools_into_one_call():
    fs, manager = _runner_self(batch=True)
    actions = {
        "a1": {"action": "world_act", "params": {"moves": []}},
        "a2": {"action": "world_observe", "params": {}},
    }
    state = SimulationRunner._process_external_prestep(fs, actions)
    assert isinstance(state, _ExternalState)
    assert len(manager.batch_calls) == 1
    assert manager.single_calls == []
    requests, tools = manager.batch_calls[0]
    assert set(requests) == {"a1", "a2"}
    assert tools == {"a1": "act", "a2": "observe"}
    assert set(state.sent_statuses) == {"a1", "a2"}


def test_prestep_without_batch_keeps_per_tool_calls():
    fs, manager = _runner_self(batch=False)
    actions = {
        "a1": {"action": "world_act", "params": {"moves": []}},
        "a2": {"action": "world_observe", "params": {}},
    }
    SimulationRunner._process_external_prestep(fs, actions)
    assert manager.batch_calls == []
    assert len(manager.single_calls) == 2


def test_actions_spec_skips_batch_tool():
    fs, manager = _runner_self(batch=True)
    SimulationRunner._build_external_actions_spec(fs)
    assert "world_act" in fs.external_actions_spec
    assert f"world_{BATCH_TOOL_NAME}" not in fs.external_actions_spec
