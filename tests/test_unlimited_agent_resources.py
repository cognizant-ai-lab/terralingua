"""Unlimited resources stay unlimited and do not add survival constraints to prompts."""

from types import SimpleNamespace

import numpy as np
import pytest

from terralingua.agents.human_agent import HumanAgent
from terralingua.agents.llm_agent import LLMAgent
from terralingua.agents.remote_agent import RemoteAgent
from terralingua.environment.grid_env import OpenGridWorld
from terralingua.experiment.runner import SimulationRunner
from terralingua.genome.no_traits import Genome


def observation(energy=float("inf"), time=float("inf")):
    return {"energy": energy, "time": time, "observation": {(0, 0): ["Alice"]},
            "observation_text": "Observation at simulation step 12",
            "incoming_broadcasts": {}, "inventory": []}


def make_agent(kind, tmp_path, **kwargs):
    cls = LLMAgent if kind == "llm" else RemoteAgent
    options = {"scenario_specific_instructions": "none"} if kind == "llm" else {"motivation_prompt": ""}
    return cls(agent_name="Alice", agent_tag="a", log_dir=tmp_path, genome=Genome(),
               food_mechanism=False, energy_death=True, external_actions=True,
               **options, **kwargs)


@pytest.mark.parametrize("kind", ["llm", "remote"])
@pytest.mark.parametrize("energy", [20, float("inf")])
@pytest.mark.parametrize("remaining", [30, float("inf")])
def test_prompt_visibility_uses_actual_resources(tmp_path, kind, energy, remaining):
    agent = make_agent(kind, tmp_path, system_prompt_template="grid.j2")
    obs = observation(energy, remaining)
    old_obs = agent._format_observation(obs)
    agent.history = [(old_obs, "noop", "", {}, {})]
    formatted, prompt = agent._build_prompt_sync(obs, {"noop": {"params": {}}}, {})
    assert formatted["energy"] == energy
    assert ("Energy:" in prompt) == np.isfinite(energy)
    assert ("Remaining lifespan" in prompt) == np.isfinite(remaining)
    assert ("When your energy reaches" in agent.system_prompt) == np.isfinite(energy)
    assert ("set life span" in agent.system_prompt) == np.isfinite(remaining)
    if not np.isfinite(energy):
        assert "energy" not in agent.system_prompt.lower()
    if not np.isfinite(remaining):
        assert "Time left" not in agent.system_prompt
        assert "- Time" not in agent.system_prompt
    assert "simulation step 12" in prompt
    assert "History t-1" in prompt
    assert "inf" not in prompt.lower().replace("info", "")
    agent.close()


@pytest.mark.parametrize("kind", ["llm", "remote"])
def test_infinite_parent_does_not_hide_finite_child_budget(tmp_path, kind):
    agent = make_agent(kind, tmp_path, finite_energy=False, finite_lifespan=False)
    agent._build_prompt_sync(observation(10), {}, {})
    assert "When your energy reaches 0" in agent.system_prompt
    assert "set life span" not in agent.system_prompt
    agent.close()


@pytest.mark.parametrize("kind", ["llm", "remote"])
def test_checkpoint_prompt_refreshes_when_actual_resources_change(tmp_path, kind):
    agent = make_agent(kind, tmp_path)
    state = agent.get_state_ckpt()
    state["system_prompt"] = "STALE Energy: inf. Time left in your life: inf."
    agent.set_state_ckpt(state)
    _, prompt = agent._build_prompt_sync(observation(), {}, {})
    assert "STALE" not in agent.system_prompt
    assert "energy" not in agent.system_prompt.lower()
    assert "Remaining lifespan" not in prompt
    # A finite remaining lifetime in another checkpoint is preserved, not reset.
    agent._build_prompt_sync(observation(23, 7), {}, {})
    assert "set life span" in agent.system_prompt
    agent.close()


def test_remote_payload_hides_unlimited_resources_without_changing_log_data(tmp_path):
    agent = make_agent("remote", tmp_path)
    obs = observation()
    payload = agent.build_payload(obs, {}, 0, {}, 42)
    assert payload["step"] == 42
    assert "energy" not in payload["observation_raw"]
    assert "time" not in payload["observation_raw"]
    assert obs["energy"] == obs["time"] == float("inf")
    agent.close()


@pytest.mark.parametrize("energy,remaining", [(float("inf"), float("inf")), (11, 5)])
def test_human_prompt_and_generated_memory(tmp_path, monkeypatch, capsys, energy, remaining):
    agent = HumanAgent(agent_name="Alice", agent_tag="a", log_dir=tmp_path)
    monkeypatch.setattr("builtins.input", lambda _: "noop")
    formatted = agent._format_observation(observation(energy, remaining), {})
    assert agent._prompt_for_action(formatted, {"noop": {}}) == "noop"
    text = capsys.readouterr().out
    memory = agent._update_memory(formatted, "noop", {})
    assert ("Energy:" in text) == np.isfinite(energy)
    assert ("Remaining lifespan" in text) == np.isfinite(remaining)
    assert ("'energy'" in memory) == np.isfinite(energy)
    assert ("time_left" in memory) == np.isfinite(remaining)
    agent.close()


def make_world(tmp_path, lifespan):
    world = OpenGridWorld(grid_size=3, lifespan=lifespan, init_agent_energy=-1,
                          food_mechanism=False, init_food=0, food_spawn_rate=0,
                          food_decay_rate=0, reproduction_cost=10, headless=True,
                          log_path=tmp_path)
    world.add_agent("a", "Alice", "no_traits", position=(0, 0))
    world.restart_env(seed=1, agent_poses={"a": (0, 0)})
    return world


def test_unlimited_life_survives_steps_and_checkpoint(tmp_path):
    world = make_world(tmp_path, -1)
    for _ in range(120):
        world.step({})
    assert "a" in world.agent_registry
    assert np.isinf(world.agent_time["a"])
    state = world.get_state_ckpt()
    world.set_state_ckpt(state)
    world.step({})
    assert np.isinf(world.agent_time["a"])
    world.close()


def test_finite_life_still_counts_down_and_expires(tmp_path):
    world = make_world(tmp_path, 3)
    world.step({})
    state = world.get_state_ckpt()
    world.set_state_ckpt(state)
    assert world.agent_time["a"] == 2
    world.step({})
    assert world.agent_time["a"] == 1
    world.step({})
    assert "a" not in world.agent_registry
    world.close()


def test_runner_assigns_unlimited_life_to_new_agents():
    runner = SimulationRunner.__new__(SimulationRunner)
    runner.params = SimpleNamespace(env=SimpleNamespace(agent_lifespan=-1))
    runner.env = SimpleNamespace(agent_time={"a": 1}, rng=np.random.default_rng(10))
    runner.obs = {"a": {"time": 1}}
    assert np.isinf(runner._assign_sampled_lifespan("a"))
    assert np.isinf(runner.env.agent_time["a"])
    assert np.isinf(runner.obs["a"]["time"])


def test_food_does_not_claim_energy_death_when_disabled(tmp_path):
    agent = make_agent("llm", tmp_path)
    agent.food_mechanism = True
    agent.energy_death = False
    agent.update_system_prompt()
    assert "lose 1 energy" in agent.system_prompt
    assert "When your energy reaches 0, you die" not in agent.system_prompt
    agent.close()
