"""Synchronous decisions use each model and record each applied/default turn once."""

import asyncio
import json

import pytest

from terralingua.config.models import ExperimentConfig
from terralingua.experiment import runner as runner_module
from terralingua.experiment.runner import SimulationRunner
from terralingua.utils.llm_client import AgentClient, Response

LUNA = "gpt-6-luna"
TERRA = "gpt-6.1-terra"


def make_runner(tmp_path, monkeypatch, *, heartbeat=-1):
    monkeypatch.setattr(runner_module, "LOGS_DIR", tmp_path)
    return SimulationRunner(
        ExperimentConfig(
            agent={
                "model": LUNA,
                "model_assignment": {TERRA: 1},
                "max_history": 2,
                "internal_memory_size": 0,
                "scenario_specific_instructions": "none",
            },
            env={
                "world_type": "social_graph",
                "init_agents": 2,
                "food_mechanism": False,
                "init_food": 0,
                "food_spawn_rate": 0,
                "energy_upkeep": 0,
                "agent_lifespan": -1,
                "artifact_creation_cost": -1,
                "graph": {"topology": "complete", "n_nodes": 2},
            },
            run={
                "exp_name": "sync_turns",
                "seed": 7,
                "env_heartbeat": heartbeat,
                "save_video": False,
            },
        )
    )


def apply_turn(runner, actions, step):
    runner._capture_action_requests(actions, step)
    external = runner._process_external_prestep(actions)
    runner.obs, runner.rewards, runner.dones, _, runner.infos = runner.env.step(actions)
    runner._process_external_poststep(external, step)
    runner._record_local_action_receipts(actions, external, step)
    for tag in actions:
        runner.agents[tag].log_turn(step)


def records(runner, tag):
    path = runner.exp_logdir / "agent_logs" / f"{tag}.jsonl"
    return (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )


def test_each_model_is_used_and_logs_only_after_action_application(
    tmp_path, monkeypatch
):
    seen = []

    async def reply(client, messages, chat_params):
        seen.append(chat_params["model"])
        return Response(
            json.dumps(
                {
                    "action": "noop",
                    "params": {},
                    "messages": {"broadcast": chat_params["model"]},
                }
            ),
            12,
            3,
        )

    monkeypatch.setattr(AgentClient, "get_response_async", reply)
    runner = make_runner(tmp_path, monkeypatch)
    actions = asyncio.run(runner._collect_actions(0))
    assert sorted(seen) == sorted([LUNA, TERRA])
    assert set(actions) == set(runner.agents)
    for tag, agent in runner.agents.items():
        assert actions[tag]["messages"]["broadcast"] == agent.model
        assert actions[tag]["decision_step"] == 0
        assert not records(runner, tag)

    apply_turn(runner, actions, 0)
    for tag, agent in runner.agents.items():
        (row,) = records(runner, tag)
        assert row["model"] == agent.model
        assert row["timestamp"] == "0"
        assert row["action"]["action"] == "noop"
        assert row["action"]["seconds"] >= 0
        assert len(agent.history) == 1
        agent.log_turn(1)
        assert len(records(runner, tag)) == 1


@pytest.mark.parametrize("failure", ["timeout", "unexpected_error"])
def test_failed_decision_records_world_pass_and_does_not_erase_other_model_turn(
    tmp_path,
    monkeypatch,
    failure,
):
    cancelled = []

    async def reply(client, messages, chat_params):
        if chat_params["model"] == TERRA:
            if failure == "unexpected_error":
                raise ValueError("Malformed provider result")
            try:
                await asyncio.Future()
            finally:
                cancelled.append(True)
        return Response('{"action":"noop","params":{},"messages":{}}', 12, 3)

    monkeypatch.setattr(AgentClient, "get_response_async", reply)
    runner = make_runner(tmp_path, monkeypatch, heartbeat=0.1)
    actions = asyncio.run(runner._collect_actions(0))
    by_model = {agent.model: tag for tag, agent in runner.agents.items()}
    failed_tag = by_model[TERRA]
    failed = actions[failed_tag]
    assert failed["action"] == "noop"
    assert failed["source"] == "default"
    assert failed["default_reason"] == (
        "agent_timeout" if failure == "timeout" else "unexpected_error"
    )
    assert actions[by_model[LUNA]]["source"] == "llm"
    assert len(runner.agents[failed_tag].history) == 1
    assert runner.agents[failed_tag].history[-1][1] == "noop"
    assert not records(runner, failed_tag)
    if failure == "timeout":
        assert cancelled == [True]

    apply_turn(runner, actions, 0)
    (row,) = records(runner, failed_tag)
    assert row["action"]["default_reason"] == failed["default_reason"]
    assert row["model"] == TERRA
    runner.agents[failed_tag].log_turn(1)
    assert len(records(runner, failed_tag)) == 1


def test_disconnected_remote_uses_world_pass_and_records_one_turn(
    tmp_path, monkeypatch
):
    async def reply(client, messages, chat_params):
        return Response('{"action":"noop","params":{},"messages":{}}', 12, 3)

    monkeypatch.setattr(AgentClient, "get_response_async", reply)
    runner = make_runner(tmp_path, monkeypatch)
    tag, original = next(iter(runner.agents.items()))
    remote = runner._make_remote_agent(tag, original.agent_name, "", original.genome)
    runner.agents[tag] = remote
    assert remote.connection_manager is None
    assert remote.stay_action == runner.env.stay_action
    saved = remote.get_state_ckpt()
    assert saved["stay_action"] == runner.env.stay_action
    remote.set_state_ckpt(saved)

    actions = asyncio.run(runner._collect_actions(0))
    assert actions[tag]["action"] == "noop"
    assert actions[tag]["default_reason"] == "no_connection"
    assert len(remote.history) == 1
    assert remote.history[-1][1] == "noop"
    assert not records(runner, tag)
    apply_turn(runner, actions, 0)
    (row,) = records(runner, tag)
    assert row["action"]["default_reason"] == "no_connection"
    remote.log_turn(1)
    assert len(records(runner, tag)) == 1
