import asyncio
import json
from collections import Counter

import pytest
from pydantic import ValidationError

from terralingua.config.cli import parse_argv
from terralingua.config.compose import compose
from terralingua.config.models import ExperimentConfig
from terralingua.experiment import runner as runner_module
from terralingua.experiment.runner import SimulationRunner
from terralingua.utils.llm_client import AgentClient, Response

LUNA = "gpt-6-luna"
TERRA = "gpt-6.1-terra"


@pytest.mark.parametrize("assignment", [
    {TERRA: -1}, {TERRA: 1.5}, {TERRA: True}, {TERRA: 4, LUNA: 2}, {"": 1},
])
def test_invalid_counts_fail_before_launch(assignment):
    with pytest.raises(ValidationError, match="model_assignment"):
        ExperimentConfig(agent={"model_assignment": assignment}, env={"init_agents": 5, "init_human_agents": 1})


def test_mixed_models_route_log_and_resume(tmp_path, monkeypatch):
    _, overrides, _ = parse_argv([
        "--init_agents", "5", "--model", LUNA, "--model_assignment", json.dumps({TERRA: 2}),
        "--run.seed", "12", "--exp_name", "mixed", "--env_heartbeat", "-1",
        "--world_type", "social_graph", "--graph.topology", "complete", "--graph.n_nodes", "5",
        "--reasoning_effort", "low", "--max_tokens", "500",
    ])
    config = compose(overrides=overrides)
    monkeypatch.setattr(runner_module, "LOGS_DIR", tmp_path)

    async def echo_model(client, messages, chat_params):
        assert client.provider == "openai"
        assert chat_params["reasoning_effort"] == "low" and chat_params["max_tokens"] == 500
        message = {"action": "noop", "messages": {"broadcast": chat_params["model"]}, "params": {}}
        return Response(content=json.dumps(message), input_tokens=1, output_tokens=1)

    monkeypatch.setattr(AgentClient, "get_response_async", echo_model)
    runner = SimulationRunner(config)
    assignments = {tag: agent.model for tag, agent in runner.agents.items()}
    assert Counter(assignments.values()) == {LUNA: 3, TERRA: 2}

    async def check_turns(runner):
        step = runner.start_ts
        actions = await runner._collect_actions(step)
        assert actions.keys() == assignments.keys()
        runner._capture_action_requests(actions, step)
        external = runner._process_external_prestep(actions)
        runner.obs, runner.rewards, runner.dones, _, runner.infos = runner.env.step(actions)
        runner._process_external_poststep(external, step)
        runner._record_local_action_receipts(actions, external, step)
        for tag in actions:
            runner.agents[tag].log_turn(step)
        for tag, model in assignments.items():
            assert actions[tag]["messages"]["broadcast"] == model
            rows = (runner.exp_logdir / "agent_logs" / f"{tag}.jsonl").read_text().splitlines()
            assert json.loads(rows[-1])["model"] == model

    asyncio.run(asyncio.wait_for(check_turns(runner), timeout=10))
    asyncio.run(runner._save_checkpoint(1))
    resumed = SimulationRunner(config, resume=True)
    assert {tag: agent.model for tag, agent in resumed.agents.items()} == assignments
    asyncio.run(asyncio.wait_for(check_turns(resumed), timeout=10))

    # The same seed reproduces the assignment on a fresh run too.
    config.run.exp_name = "fresh"
    fresh = SimulationRunner(config)
    assert {tag: agent.model for tag, agent in fresh.agents.items()} == assignments


def test_legacy_checkpoint_and_replacement_agents_use_run_default(tmp_path, monkeypatch):
    config = ExperimentConfig(agent={"model": LUNA, "model_assignment": {TERRA: 1}},
                              env={"init_agents": 1, "init_human_agents": 1},
                              run={"exp_name": "legacy"})
    monkeypatch.setattr(runner_module, "LOGS_DIR", tmp_path)
    runner = SimulationRunner(config)
    founder = runner.agents["being0"]
    assert founder.model == TERRA
    assert runner.agents["human0"].model is None
    new_agent = runner._make_llm_agent("new", "New", founder.genome)
    assert new_agent.model == LUNA
    state = founder.get_state_ckpt()
    del state["model"]
    new_agent.set_state_ckpt(state)
    assert new_agent.model == LUNA


@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("chosen", [None, LUNA, TERRA, "gpt-5-mini"])
def test_offspring_model_through_spawn(tmp_path, monkeypatch, paired, chosen):
    # The third model has no founders. The default also appears in the assignment.
    settings = {"model": LUNA, "model_assignment": {TERRA: 1, LUNA: 1, "gpt-5-mini": 0}}
    if chosen is not None:
        settings["inherited_model"] = False
    config = ExperimentConfig(
        agent=settings,
        env={"world_type": "social_graph", "init_agents": 2, "reproduction_cost": 0,
             "two_parent_spawn": paired, "graph": {"topology": "complete", "n_nodes": 2}},
        run={"exp_name": "birth", "seed": 12},
    )
    monkeypatch.setattr(runner_module, "LOGS_DIR", tmp_path)
    runner = SimulationRunner(config)
    parent, partner = sorted(runner.agents.values(), key=lambda agent: agent.model != TERRA)
    tag = parent.agent_tag
    actions = runner.infos[tag]["available_actions"]
    spawn_params = actions["spawn"]["params"]
    params = {"name": "Child"}
    if paired:
        params.update(partner=partner.agent_name, offspring_genome="")
    if chosen is None:
        assert "offspring_model" not in spawn_params
    else:
        assert spawn_params["offspring_model"]["choices"] == [LUNA, TERRA, "gpt-5-mini"]
        params["offspring_model"] = chosen

    decision = {"action": "spawn", "params": params}
    _, _, parsed, _, _ = parent._parse_response(json.dumps(decision), actions)
    runner.obs, runner.rewards, runner.dones, _, runner.infos = runner.env.step(
        {tag: {"action": "spawn", "params": parsed}}
    )
    record = runner.infos[tag]["spawn"]
    assert record["status"] == "successful"
    if paired:
        assert record["parent_b_tag"] == partner.agent_tag
    asyncio.run(runner._handle_spawn())
    child = runner.agents[record["child_tag"]]
    assert child.model == (chosen or TERRA)
    assert child.get_state_ckpt()["model"] == child.model
    assert runner._router_for(child.model).next()[1]["model"] == child.model


@pytest.mark.parametrize("inherit,params", [
    (False, {}), (False, {"offspring_model": "gpt-unlisted"}), (True, {"offspring_model": TERRA}),
])
def test_invalid_model_choice_creates_no_child(tmp_path, monkeypatch, inherit, params):
    config = ExperimentConfig(
        agent={"model": LUNA, "model_assignment": {TERRA: 0}, "inherited_model": inherit},
        env={"init_agents": 1, "reproduction_cost": 0}, run={"exp_name": "invalid"},
    )
    monkeypatch.setattr(runner_module, "LOGS_DIR", tmp_path)
    runner = SimulationRunner(config)
    infos = runner.env.step({"being0": {"action": "spawn", "params": {"name": "Child", **params}}})[4]
    assert set(runner.env.agent_registry) == {"being0"}
    failure = infos["being0"].get("Action error") or infos["being0"]["spawn"]["reason"]
    assert "offspring_model" in failure
