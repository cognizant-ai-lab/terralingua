"""External-server replies name beings, never show their tags."""

import json

from terralingua.agents.agent_mixin import _visible_response
from terralingua.agents.llm_agent import LLMAgent
from terralingua.genome.no_traits import Genome as NoTraitsGenome

NAMES = {"being1": "Ada", "being17": "Jerry"}


def test_agent_tags_become_names_at_every_depth():
    raw = json.dumps({
        "executed_joint_actions": [
            {"agent_tag": "being17", "action": "connect_road"},
            {"agent_tag": "being1", "action": "act"},
        ],
        "nested": {"inner": {"agent_tag": "being1"}},
    })
    shown = _visible_response(raw, [], NAMES)
    assert "agent_tag" not in shown
    assert json.loads(shown) == {
        "executed_joint_actions": [
            {"agent": "Jerry", "action": "connect_road"},
            {"agent": "Ada", "action": "act"},
        ],
        "nested": {"inner": {"agent": "Ada"}},
    }


def test_tag_without_a_name_keeps_its_value():
    raw = json.dumps({"agent_tag": "being99"})
    assert json.loads(_visible_response(raw, [], NAMES)) == {"agent": "being99"}


def test_reply_without_tags_is_returned_as_is():
    raw = '{"status":   "ok"}'
    assert _visible_response(raw, [], NAMES) is raw
    assert _visible_response("plain text", [], NAMES) == "plain text"


def test_renamed_reply_keeps_its_compact_layout():
    raw = json.dumps({"agent_tag": "being1", "x": 1})
    assert _visible_response(raw, [], NAMES) == json.dumps({"agent": "Ada", "x": 1})


def test_hidden_keys_are_dropped_and_names_applied():
    raw = json.dumps({"secret": 1, "moves": [{"agent_tag": "being17"}]})
    shown = _visible_response(raw, ["secret"], NAMES)
    assert json.loads(shown) == {"moves": [{"agent": "Jerry"}]}


def test_llm_agent_shows_names_in_external_replies(tmp_path):
    agent = LLMAgent(
        agent_name="Ada",
        agent_tag="being1",
        log_dir=tmp_path,
        system_prompt_template="grid.j2",
        genome=NoTraitsGenome(),
    )
    agent.names_by_tag = NAMES
    obs = {
        "observation_text": "",
        "incoming_broadcasts": {},
        "inventory": [],
        "energy": 10.0,
        "time": 10.0,
        "external_response": [json.dumps({"moves": [{"agent_tag": "being17"}]})],
    }
    formatted = agent._format_observation(obs)
    assert json.loads(formatted["external_response"]) == {"moves": [{"agent": "Jerry"}]}
