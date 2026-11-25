"""Offline checks for bounded history and temporary retry settings."""

import asyncio
import copy
import json
from types import SimpleNamespace

import httpx
import pytest
from openai import BadRequestError

from terralingua.agents.human_agent import HumanAgent
from terralingua.agents.llm_agent import LLMAgent
from terralingua.agents.remote_agent import RemoteAgent
from terralingua.genome.no_traits import Genome
from terralingua.utils.llm_utils import async_select_with_retry

OBS = {"observation_text": "current scene", "incoming_broadcasts": {},
       "inventory": [], "energy": 50, "time": 100}
ACTIONS = {"noop": {"description": "Wait", "params": {}}}
ACTION = {"action": "noop", "params": {}, "messages": {}}


def make_agent(kind, limit):
    cls = HumanAgent if kind == "human" else RemoteAgent if kind == "remote" else LLMAgent
    agent = cls(
        agent_name="Alice", agent_tag="a", genome=Genome(), max_history=limit,
        logger=SimpleNamespace(save_genome=lambda **_: None, log=lambda **_: None),
        internal_memory_size=0, verbose=0,
        **({"motivation_prompt": ""} if kind == "remote" else {}),
    )
    return agent


@pytest.mark.parametrize("kind,limit", [
    ("llm", 0), ("llm", 2), ("fallback", 2), ("remote", 2), ("human", 2),
])
def test_history_storage_is_bounded_for_every_recording_path(kind, limit, monkeypatch):
    agent = make_agent(kind, limit)
    if kind == "human":
        monkeypatch.setattr(agent, "_prompt_for_action", lambda *_: "noop")
        monkeypatch.setattr(agent, "_prompt_for_params", lambda *_: {})
        monkeypatch.setattr(agent, "_prompt_for_messages", lambda: {})

    async def respond(**_):
        content = "invalid response" if kind == "fallback" else json.dumps(ACTION)
        return SimpleNamespace(content=content, input_tokens=0, output_tokens=0, cost=0)

    for step in range(4):
        obs = {**OBS, "energy": step}
        if kind == "remote":
            agent.record_action(obs, ACTION, {})
        elif kind == "human":
            agent._select_action(obs, ACTIONS, 0, {}, step)
        else:
            result = asyncio.run(agent.select_action_async(
                obs, ACTIONS, 0, {}, step, {}, SimpleNamespace(get_response_async=respond),
                max_attempts=1,
            ))
            assert result["source"] == ("default" if kind == "fallback" else "llm")
        assert len(agent.history) == min(step + 1, limit)
    assert [entry[0]["energy"] for entry in agent.history] == ([2, 3] if limit else [])


def test_zero_history_does_not_render_restored_entries():
    agent = make_agent("llm", 0)
    past = agent._format_observation({**OBS, "observation_text": "PRIVATE_OLD_SCENE"})
    agent.history = [(past, "noop", "", {}, {})]
    prompt = agent._make_prompt(agent._format_observation(OBS), ACTIONS, "", {})
    assert "PRIVATE_OLD_SCENE" not in prompt
    assert "History entry" not in prompt
    assert "current scene" in prompt


@pytest.mark.parametrize("template", ["grid.j2", "social_graph.j2"])
def test_history_keeps_inventory_names_and_events_without_passive_contents(template):
    agent = make_agent("llm", 2)
    agent.system_prompt_template = template
    inventory = ["A(text,movable): route notes", "A(text,movable): budget notes"]
    past = agent._format_observation({**OBS, "inventory": inventory, "step": 5})
    past_info = {
        "Artifacts in your inventory": [
            "PAST_ROUTE_CONTENTS",
            'Budget notes before: {"field":"/balance","value":10000}',
        ] * 3,
        "Events": [{"event": "budget warning", "observed": 9000}],
        "Action outcome": "PAST_ACTION_RESULT",
        "Artifacts here": ["PUBLIC_ARTIFACT_CONTENTS"],
    }
    agent.history = [(past, "noop", {}, {}, past_info)]
    saved_history = copy.deepcopy(agent.history)
    current_info = {
        "Artifacts in your inventory": [
            "CURRENT_ROUTE_CONTENTS",
            'Budget notes now: {"field":"/balance","value":10000}',
        ],
        "Events": [{"event": "budget warning", "observed": 8000}],
    }
    saved_current = copy.deepcopy(current_info)

    prompt = agent._make_prompt(
        agent._format_observation({**OBS, "inventory": inventory, "step": 6}),
        ACTIONS, "", current_info,
    )
    history, current = prompt.split("=== Current State ===", 1)

    for name in inventory:
        assert name in history
    for contents in past_info["Artifacts in your inventory"]:
        assert contents not in history
    assert "'observed': 9000" in history
    assert "PAST_ACTION_RESULT" in history
    assert "PUBLIC_ARTIFACT_CONTENTS" in history
    for contents in current_info["Artifacts in your inventory"]:
        assert contents in current
    assert "'observed': 8000" in current
    assert agent.history == saved_history
    assert current_info == saved_current


def bad_request():
    return BadRequestError(
        "context too long", response=httpx.Response(400, request=httpx.Request("POST", "https://invalid.test")),
        body=None,
    )


@pytest.mark.parametrize("initial", [1, 3])
@pytest.mark.parametrize("succeed", [False, True])
def test_retry_history_never_becomes_negative_and_is_restored(initial, succeed):
    seen = []
    agent = SimpleNamespace(max_history=initial, agent_name="Alice", agent_tag="a",
                            record_default_action=lambda *args: args[-1])

    async def select(**_):
        seen.append(agent.max_history)
        if succeed and len(seen) == 3:
            return ACTION
        raise bad_request()

    agent.select_action_async = select
    result = asyncio.run(async_select_with_retry(agent, OBS, ACTIONS, 0, {}, 0, None, {}))
    assert seen == [initial, max(0, initial - 1), max(0, initial - 2)]
    assert agent.max_history == initial
    if succeed:
        assert result == ACTION
    else:
        assert result["default_reason"] == "max_retries"


def test_retry_restores_history_when_timeout_cancels_the_turn():
    async def run():
        waiting = asyncio.Event()
        calls = 0
        agent = SimpleNamespace(max_history=2, agent_name="Alice", agent_tag="a")

        async def select(**_):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise bad_request()
            assert agent.max_history == 1
            waiting.set()
            await asyncio.Future()

        agent.select_action_async = select
        task = asyncio.create_task(async_select_with_retry(agent, OBS, ACTIONS, 0, {}, 0, None, {}))
        await waiting.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert agent.max_history == 2

    asyncio.run(run())


@pytest.mark.parametrize("social", [False, True])
@pytest.mark.parametrize("failure", ["max_retries", "unexpected_error"])
def test_retry_failure_uses_world_pass_action(social, failure):
    agent = SimpleNamespace(max_history=1, agent_name="Alice", agent_tag="a",
                            record_default_action=lambda *args: args[-1])
    if social:
        agent.stay_action = ACTION

    async def select(**_):
        if failure == "max_retries":
            raise bad_request()
        raise ValueError("invalid client response")

    agent.select_action_async = select
    result = asyncio.run(async_select_with_retry(agent, OBS, ACTIONS, 0, {}, 0, None, {}, retries=1))
    assert result["action"] == ("noop" if social else "move")
    assert result["params"] == ({} if social else {"direction": "stay"})
    assert result["default_reason"] == failure
    assert agent.max_history == 1


@pytest.mark.parametrize("failure", ["max_retries", "unexpected_error"])
def test_retry_failure_keeps_a_turn_for_the_agent_log(failure):
    records = []
    agent = make_agent("llm", 2)
    agent.logger = SimpleNamespace(save_genome=lambda **_: None, log=lambda **record: records.append(record))

    async def select(**_):
        if failure == "max_retries":
            raise bad_request()
        raise ValueError("invalid client response")

    agent.select_action_async = select
    result = asyncio.run(async_select_with_retry(agent, OBS, ACTIONS, 0, {}, 7, None, {}, retries=1))
    assert result["default_reason"] == failure
    assert len(agent.history) == 1
    assert records == []
    agent.log_turn(9)
    assert [r["time"] for r in records] == ["9"]
    assert records[0]["action"] is result
    agent.log_turn(10)
    assert len(records) == 1
