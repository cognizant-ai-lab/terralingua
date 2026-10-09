"""Checks for readable current-state and history fields."""

import copy
import json

import pytest

from terralingua.agents.agent_mixin import AgentMixin

PUBLIC_KEY = "Artifacts here"
PRIVATE_KEY = "Artifacts in your inventory"


class _PromptAgent(AgentMixin):
    def __init__(self, template="social_graph.j2", *, inventory=True):
        self.system_prompt_template = template
        self.history = []
        self.max_history = 2
        self.internal_memory_size = 500
        self.use_inventory = inventory
        self.max_message_length = 200


def _formatted(label="current", *, lifespan=9, inventory="private current artifact"):
    return {
        "observation": label,
        "message": "<none>",
        "inventory": inventory,
        "energy": 17,
        "time": lifespan,
    }


def _entry(label="past", *, lifespan=10, inventory="private past artifact", info=None):
    return (_formatted(label, lifespan=lifespan, inventory=inventory), "noop", "", {}, info)


def _render(agent, info=None):
    return agent._make_prompt(_formatted(), {"noop": {"params": {}}}, "remember", info)


def test_history_labels_are_relative_and_keep_the_window_order():
    agent = _PromptAgent()
    agent.history = [_entry("discarded"), _entry("earlier"), _entry("latest")]
    prompt = _render(agent)
    history = prompt.split("=== Current State ===", 1)[0]

    assert "t-2 to t-1, oldest first" in history
    assert "History t-1:" in history
    assert "History t-2:" in history
    assert "Step 1:" not in history
    assert "discarded" not in history
    assert history.index("earlier") < history.index("latest")
    assert len(agent.history) == 3


def test_history_and_current_state_use_the_same_energy_visibility():
    agent = _PromptAgent()
    agent.history = [_entry()]
    prompt = _render(agent)
    history, current = prompt.split("=== Current State ===", 1)

    assert "Energy: 17" in history
    assert "Energy: 17" in current
    assert "Remaining lifespan (steps): 10" in history
    assert "Remaining lifespan (steps): 9" in current
    assert "Remaining time:" not in prompt


@pytest.mark.parametrize("template", ["social_graph.j2", "grid.j2"])
@pytest.mark.parametrize("inventory", [False, True])
def test_inventory_visibility_and_privacy_labels(template, inventory):
    agent = _PromptAgent(template, inventory=inventory)
    agent.history = [_entry()]
    prompt = _render(agent)

    assert ("Social observation:\ncurrent" in prompt) == (template == "social_graph.j2")
    if template != "social_graph.j2":
        assert "Observation:\ncurrent" in prompt
    assert ("private past artifact" in prompt) == inventory
    assert ("private current artifact" in prompt) == inventory
    private_label = "Inventory (private artifacts):"
    if inventory and template == "social_graph.j2":
        assert prompt.count(private_label) == 2
    else:
        assert private_label not in prompt
    if inventory and template != "social_graph.j2":
        assert prompt.count("Inventory:") == 2


@pytest.mark.parametrize("template", ["social_graph.j2", "grid.j2"])
def test_artifact_content_labels_are_social_only_and_preserve_raw_data(template):
    info = {
        PUBLIC_KEY: ["Artifact guide\nContent: first line\nSecond line", "Artifact notes\nContent: more"],
        PRIVATE_KEY: ["Artifact draft\nContent: private text"],
        "custom_payload": {"node": "keep-user-data", "list": ["embedded\nline"]},
        "custom_text": "User text mentions a node.\nKeep this line.",
    }
    original_info = copy.deepcopy(info)
    agent = _PromptAgent(template)
    agent.history = [_entry(info=info)]
    original_history = copy.deepcopy(agent.history)
    prompt = _render(agent, info)

    assert info == original_info
    assert agent.history == original_history
    assert str(info["custom_payload"]) in prompt
    assert info["custom_text"] in prompt
    if template == "social_graph.j2":
        assert prompt.count("Your public artifact contents:") == 2
        assert prompt.count("Your private artifact contents:") == 2
        assert "Artifact guide\nContent: first line\nSecond line\n\nArtifact notes" in prompt
        assert PUBLIC_KEY not in prompt
        assert PRIVATE_KEY not in prompt
    else:
        assert PUBLIC_KEY in prompt
        assert PRIVATE_KEY in prompt
        assert str(info[PUBLIC_KEY]) in prompt
        assert "Your public artifact contents:" not in prompt
        assert "Your private artifact contents:" not in prompt


def test_unknown_artifact_value_types_are_not_discarded():
    agent = _PromptAgent()
    info = {PUBLIC_KEY: [{"effect": 3}, "text"], PRIVATE_KEY: {"result": "ok"}}
    prompt = _render(agent, info)

    assert str(info[PUBLIC_KEY]) in prompt
    assert str(info[PRIVATE_KEY]) in prompt


def test_published_context_appears_once_and_other_info_is_preserved():
    agent = _PromptAgent()
    past_info = {
        "external_context": {"notes": "old context"},
        "ordinary_feedback": {"result": "past result"},
    }
    info = {
        "external_context": {"notes": "current context"},
        "ordinary_feedback": {"result": "current result"},
    }
    agent.history = [_entry(info=past_info)]
    original = copy.deepcopy((past_info, info, agent.history))
    prompt = _render(agent, info)

    assert "old context" not in prompt
    assert prompt.count("current context") == 1
    assert "past result" in prompt
    assert "current result" in prompt
    assert (past_info, info, agent.history) == original


def test_action_parameters_and_reply_schema_are_unchanged():
    agent = _PromptAgent()
    actions = {"tool_call": {"params": {"target": {"choices": ["a", "b"]}}}}
    original = copy.deepcopy(actions)
    prompt = agent._make_prompt(_formatted(), actions, "my memory", {})

    assert json.dumps(actions, indent=4) in prompt
    assert actions == original
    assert '"action": "<one of tool_call>"' in prompt
    assert '"message":' in prompt
    assert '"params": {"key": "value"}' in prompt
    assert '"internal_memory":' in prompt
    assert "my memory" in prompt
