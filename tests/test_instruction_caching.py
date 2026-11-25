"""Stable MCP instructions and per-turn availability retain separate meanings."""

import asyncio
import copy
import json
from types import SimpleNamespace

from terralingua.agents.llm_agent import LLMAgent
from terralingua.genome.no_traits import Genome
from terralingua.utils.llm_client import Response


def test_mcp_definitions_stay_stable_when_state_and_availability_change(tmp_path):
    agent = LLMAgent(
        agent_name="Alice",
        agent_tag="a",
        genome=Genome(),
        log_dir=tmp_path,
        external_actions=True,
        max_history=0,
        internal_memory_size=0,
        server_instructions="Stable world objective and rules.",
    )
    tool = {
        "description": "Look up the town.",
        "params": {"town_id": "Town ID"},
        "input_schema": {
            "type": "object",
            "properties": {"town_id": {"type": "integer"}},
        },
    }
    actions = {"noop": {"description": "Wait", "params": {}}, "world_query": tool}
    original = copy.deepcopy(actions)
    seen = []

    async def respond(messages, **_):
        seen.append(copy.deepcopy(messages))
        return Response(
            content=json.dumps({"action": "noop", "params": {}, "messages": {}}),
            input_tokens=1500,
            output_tokens=10,
            cached_input_tokens=1024,
            cache_write_input_tokens=0,
        )

    obs = {
        "observation_text": "state one",
        "incoming_broadcasts": {},
        "inventory": [],
        "energy": 50,
        "time": 100,
    }
    client = SimpleNamespace(get_response_async=respond)
    first = asyncio.run(agent.select_action_async(obs, actions, 0, {}, 0, {}, client))
    asyncio.run(
        agent.select_action_async(
            {**obs, "observation_text": "state two"},
            {"noop": actions["noop"]},
            0,
            {},
            1,
            {},
            client,
        )
    )
    assert seen[0][0] == seen[1][0]
    assert "Stable world objective and rules." in seen[0][0]["content"]
    assert "Look up the town." in seen[0][0]["content"]
    assert "Look up the town." not in seen[0][1]["content"]
    assert "world_query" in seen[0][1]["content"]
    assert "world_query" not in seen[1][1]["content"]
    assert "state two" in seen[1][1]["content"]
    assert actions == original
    assert agent.history == []
    assert first["cached_input_tokens"] == 1024
    assert first["cache_write_input_tokens"] == 0
    assert agent.get_state_ckpt()["mcp_tool_definitions"] == {"world_query": tool}


def test_cache_counts_include_failed_parse_attempts(tmp_path):
    agent = LLMAgent(
        agent_name="Alice",
        agent_tag="a",
        genome=Genome(),
        log_dir=tmp_path,
        internal_memory_size=0,
    )
    replies = iter(
        [
            Response(
                "invalid",
                1500,
                10,
                cached_input_tokens=1000,
                cache_write_input_tokens=200,
            ),
            Response(
                '{"action":"noop","params":{},"messages":{}}',
                1600,
                20,
                cached_input_tokens=1200,
                cache_write_input_tokens=100,
            ),
        ]
    )

    async def respond(**_):
        return next(replies)

    result = asyncio.run(
        agent.select_action_async(
            {
                "observation_text": "state",
                "incoming_broadcasts": {},
                "inventory": [],
                "energy": 50,
                "time": 100,
            },
            {"noop": {"description": "Wait", "params": {}}},
            0,
            {},
            0,
            {},
            SimpleNamespace(get_response_async=respond),
        )
    )
    assert result["cached_input_tokens"] == 2200
    assert result["cache_write_input_tokens"] == 300
