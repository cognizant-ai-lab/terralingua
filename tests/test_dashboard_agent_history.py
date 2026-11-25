"""Dashboard history renders both current and pre-migration message records."""

import asyncio
import json

import pytest

from terralingua.experiment.runner import SimulationRunner


@pytest.mark.parametrize("action,expected", [
    ({"action": "noop", "message": "Hello"}, {"broadcast": "Hello"}),
    ({"action": "send_direct_message", "message": "Everyone",
      "params": {"target": "Ada", "content": "Only you"}},
     {"broadcast": "Everyone", "Ada": "Only you"}),
    ({"action": "send_direct_message", "params": {"target": "Ada", "content": "Only you"}},
     {"Ada": "Only you"}),
    ({"action": "noop", "messages": {"Ada": "Current"}, "message": "Obsolete"},
     {"Ada": "Current"}),
    ({"action": "noop"}, {}),
])
def test_dashboard_normalizes_sent_messages_without_rewriting_logs(tmp_path, action, expected):
    runner = SimulationRunner.__new__(SimulationRunner)
    runner.exp_logdir = tmp_path
    logs = tmp_path / "agent_logs"
    logs.mkdir()
    path = logs / "being0.jsonl"
    record = {
        "timestamp": "3", "action": action,
        "observation": {"energy": 12, "time": 8, "inventory": ["Notes"]},
        "internal_memory": "Remember",
    }
    original = json.dumps(record) + "\n"
    path.write_text(original)

    result = asyncio.run(runner._dispatch_cmd("get_agent_log", {"tag": "being0", "since": 3}))
    assert result["ok"] is True
    entry, = result["data"]["entries"]
    assert entry["messages"] == expected
    assert entry["step"] == 3 and entry["action"] == action["action"]
    assert entry["energy"] == 12 and entry["inventory"] == ["Notes"]
    assert path.read_text() == original
