import json

from terralingua.anthropologist.analysis_utils import (
    load_agent_log,
    received_messages,
    sent_messages,
)


def test_received_messages_joins_broadcasts_and_direct_messages():
    observation = {
        "incoming_broadcasts": {"Ada": "come to the river", "Bo": "hello"},
        "incoming_dms": {"Bo": "just you and me", "Cy": "psst"},
    }
    assert received_messages(observation) == {
        "Ada": "come to the river",
        "Bo": "hello\njust you and me",
        "Cy": "psst",
    }
    assert received_messages({"incoming_broadcasts": {}}) == {}


def test_reduced_agent_log_keeps_received_messages(tmp_path):
    record = {
        "timestamp": "3",
        "agent": "Bo",
        "agent_tag": "b0",
        "action": {"action": "move", "params": {"direction": "north"}, "message": "on my way"},
        "observation": {
            "observation": {"(0, 0)": ["Ada"]},
            "observation_text": "Ada is here",
            "incoming_broadcasts": {"Ada": "come to the river"},
            "energy": 10,
            "time": 20,
            "inventory": [],
        },
        "internal_memory": "",
        "available_actions": {},
        "input_prompt": "",
    }
    path = tmp_path / "b0.jsonl"
    path.write_text(json.dumps(record) + "\n")
    reduced = load_agent_log(path, reduce=True)
    assert reduced[3]["received_messages"] == {"Ada": "come to the river"}
    assert reduced[3]["sent_messages"] == {"broadcast": "on my way"}


def test_sent_messages_keeps_every_receiver_with_a_text():
    action = {"action": "noop", "messages": {"broadcast": "hi all", "Bo": "psst", "Cy": ""}}
    assert sent_messages(action) == {"broadcast": "hi all", "Bo": "psst"}
    assert sent_messages({"action": "noop", "messages": {}}) == {}
    assert sent_messages({"action": "noop"}) == {}
