"""External side effects are admitted before dispatch and charged once."""

from types import SimpleNamespace

import pytest

from terralingua.environment.grid_env import OpenGridWorld
from terralingua.experiment.runner import SimulationRunner
from terralingua.voting import IndependentVotingManager, UnanimousVotingManager
from terralingua.voting.rewards import DirectRewards


@pytest.fixture
def runner(tmp_path):
    spec = {
        "server": "service", "tool": "query", "cost": 5, "mode": "independent",
        "description": "Query a value.", "params": {"value": "An integer."},
        "input_schema": {
            "type": "object", "properties": {"value": {"type": "integer", "minimum": 0}},
            "required": ["value"], "additionalProperties": False,
        },
    }
    env = OpenGridWorld(
        grid_size=5, init_food=0, food_spawn_rate=0, food_mechanism=False,
        energy_death=False, energy_upkeep=0, init_agent_energy=10,
        reproduction_cost=-1, external_actions_spec={"service_query": spec},
        log_path=tmp_path, headless=True,
    )
    for tag, position in (("a", (1, 1)), ("b", (1, 2))):
        env.add_agent(tag, tag.upper(), "no_traits", position=position)
    env.restart_env(seed=0, agent_poses={"a": (1, 1), "b": (1, 2)})
    calls = []

    def send(requests, tool_name):
        calls.append((requests, tool_name))
        return {tag: [{"message_id": tag, "status": "queued for delivery"}] for tag in requests}

    result = SimulationRunner.__new__(SimulationRunner)
    result.env = env
    result.external_actions_spec = env.external_actions_spec
    result.external_managers = {"service": SimpleNamespace(send_outgoing_messages=send)}
    result.external_specs = {"service": SimpleNamespace(batch=False)}
    result.voting_managers = {"service": IndependentVotingManager(DirectRewards(0))}
    result.external_state_handlers = {}
    result._stay_action = env.stay_action
    result.calls = calls
    result.infos = {tag: {} for tag in ("a", "b")}
    result.agents = {}
    result.exp_logdir = tmp_path
    yield result
    env.close()


def action(params=None):
    return {"action": "service_query", "params": {"value": 1} if params is None else params,
            "messages": {"broadcast": "hello"}}


@pytest.mark.parametrize("params", [{}, {"value": "bad"}, {"value": -1},
                                    {"value": 1, "unknown": True}, [1]])
def test_invalid_arguments_never_reach_server(runner, params):
    actions = {"a": action(params)}
    runner._capture_action_requests(actions, 0)
    state = runner._process_external_prestep(actions)
    assert runner.calls == []
    assert runner.env.agent_energy["a"] == 10
    assert "a" in state.rejected
    assert actions["a"]["messages"] == {"broadcast": "hello"}
    runner.obs, _, _, _, runner.infos = runner.env.step(actions)
    runner._record_external_outcomes(state, {}, 0)
    runner._record_local_action_receipts(actions, state, 0)
    receipts = runner.infos["a"]["action_receipts"]
    assert len(receipts) == 1
    assert receipts[0]["operation_status"] == "rejected"
    assert receipts[0]["effect_status"] == "not_observed"


def test_current_permission_is_checked_without_overwriting_cached_menu(runner):
    cached = runner.env.agent_avail_actions["a"]
    runner.env.agent_excluded_actions["a"].add("service_query")
    state = runner._process_external_prestep({"a": action()})
    assert runner.calls == []
    assert "not currently available" in state.rejected["a"]
    assert runner.env.agent_avail_actions["a"] is cached


def test_insufficient_energy_and_forged_paid_marker_cannot_dispatch(runner):
    runner.env.agent_energy["a"] = 4
    actions = {"a": {**action(), "external_cost_paid": True}}
    state = runner._process_external_prestep(actions)
    assert runner.calls == []
    assert "Not enough energy" in state.rejected["a"]
    assert runner.env.agent_energy["a"] == 4


def test_exact_energy_is_charged_once_across_dispatch_and_world_step(runner):
    runner.env.agent_energy["a"] = 5
    actions = {"a": action()}
    state = runner._process_external_prestep(actions)
    assert state.dispatched == {"a"}
    assert runner.calls == [({"a": {"value": 1}}, "query")]
    assert runner.env.agent_energy["a"] == 0
    runner.env.step(actions)
    assert runner.env.agent_energy["a"] == 0


def test_unselected_proposals_are_refunded(runner):
    runner.voting_managers["service"] = UnanimousVotingManager(DirectRewards(0))
    actions = {"a": action(), "b": action()}
    state = runner._process_external_prestep(actions)
    assert len(state.dispatched) == 1
    representative = next(iter(state.dispatched))
    other = ({"a", "b"} - state.dispatched).pop()
    assert runner.env.agent_energy[representative] == 5
    assert runner.env.agent_energy[other] == 10
    runner.env.step(actions)
    assert runner.env.agent_energy[representative] == 5
    assert runner.env.agent_energy[other] == 10


def test_unavailable_server_rejects_without_charging(runner):
    runner.external_managers.clear()
    state = runner._process_external_prestep({"a": action()})
    assert "unavailable" in state.rejected["a"]
    assert runner.env.agent_energy["a"] == 10
