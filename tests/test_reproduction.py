"""Births transfer parent energy while ordinary arrivals keep founder energy."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from terralingua.config.models import ExperimentConfig, GraphConfig
from terralingua.environment.graph_env import OpenGraphWorld
from terralingua.environment.grid_env import OpenGridWorld
from terralingua.environment.social_graph_env import OpenSocialGraphWorld
from terralingua.experiment.checkpoint import CheckpointManager
from terralingua.experiment.runner import SimulationRunner

KINDS = ("grid", "social")


def make_world(kind, directory, *, cost=20, food=False, paired=False, cap=None):
    options = dict(
        log_path=directory, headless=True, init_agent_energy=80, lifespan=500,
        init_food=1, food_spawn_rate=0, food_mechanism=food, energy_death=False,
        drop_food_on_death=False, reproduction_cost=cost, two_parent_spawn=paired,
        max_agents=cap,
    )
    if kind == "grid":
        world = OpenGridWorld(grid_size=8, vision_radius=2, **options)
        positions = {"a": (3, 3), "b": (3, 4)}
    else:
        cls = OpenSocialGraphWorld if kind == "social" else OpenGraphWorld
        world = cls(graph_cfg=GraphConfig.complete(n_nodes=4), **options)
        positions = {"a": "0", "b": "1"}
    for tag, name in (("a", "Alice"), ("b", "Bob")):
        world.add_agent(tag, name, "no_traits", position=positions[tag])
    world.restart_env(seed=11, agent_poses=positions)
    world.food.clear()
    return world


def birth(params=None):
    return {"action": "spawn", "message": "", "params": {"name": "Child", **(params or {})}}


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("cost", [0, 20])
@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("gift", [None, 7])
def test_birth_energy_matches_mode_and_only_initiating_parent_pays(tmp_path, kind, cost, paired, gift):
    world = make_world(kind, tmp_path, cost=cost, paired=paired)
    params = {"partner": "Bob"} if paired else {}
    if gift is not None:
        params["energy"] = gift
    before = sum(world.agent_energy.values())
    infos = world.step({"a": birth(params)})[4]
    assert infos["a"]["spawn"]["status"] == "successful"
    child = infos["a"]["spawn"]["child_tag"]
    base = 80 if cost == 0 else cost
    assert world.agent_energy[child] == base + (gift or 0)
    assert world.agent_energy["a"] == 80 - cost - (gift or 0)
    assert world.agent_energy["b"] == 80
    assert sum(world.agent_energy.values()) == before + (80 if cost == 0 else 0)


def test_food_drain_is_separate_from_paid_birth_transfer(tmp_path):
    world = make_world("grid", tmp_path, food=True)
    infos = world.step({"a": birth({"energy": 7})})[4]
    child = infos["a"]["spawn"]["child_tag"]
    assert world.agent_energy == {"a": 52, "b": 79, child: 26}


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("reason", ["disabled", "balance", "cap"])
def test_failed_cached_spawn_creates_no_child_and_charges_only_when_enabled(tmp_path, kind, reason):
    world = make_world(kind, tmp_path)
    assert "spawn" in world.agent_avail_actions["a"]
    if reason == "disabled":
        world.reproduction_cost = -1
    elif reason == "balance":
        world.agent_energy["a"] = 19
    else:
        world.max_agents = 2
    before_energy = dict(world.agent_energy)
    before_agents = set(world.agent_registry)
    before_nodes = set(world.world_graph.all_nodes()) if kind != "grid" else None
    infos = world.step({"a": birth()})[4]
    assert infos["a"]["spawn"]["status"] == "failed"
    expected = dict(before_energy)
    if reason != "disabled":
        expected["a"] -= 20
    assert world.agent_energy == expected
    assert world.agent_registry == before_agents
    assert world.agent_spawn["a"] == []
    if before_nodes is not None:
        assert set(world.world_graph.all_nodes()) == before_nodes
    assert "spawn" not in world._get_avail_actions("a")


@pytest.mark.parametrize("gift", [-1, "invalid", 1.5, 61])
def test_invalid_or_unaffordable_gift_does_not_create_child(tmp_path, gift):
    world = make_world("grid", tmp_path)
    before = dict(world.agent_energy)
    infos = world.step({"a": birth({"energy": gift})})[4]
    assert infos["a"]["spawn"]["status"] == "failed"
    assert world.agent_energy == {**before, "a": before["a"] - 20}
    assert len(world.agent_registry) == 2


def test_last_population_slot_is_used_only_once(tmp_path):
    world = make_world("grid", tmp_path, cap=3)
    infos = world.step({"a": birth(), "b": birth({"name": "Other"})})[4]
    assert len(world.agent_registry) == 3
    assert sorted(infos[tag]["spawn"]["status"] for tag in ("a", "b")) == ["failed", "successful"]
    assert sorted(world.agent_energy[tag] for tag in ("a", "b")) == [60, 60]
    assert sum(world.agent_energy.values()) == 140


@pytest.mark.parametrize("kind", ["grid", "graph"])
def test_no_birth_position_consumes_cost_without_transferring_gift(tmp_path, kind):
    world = make_world(kind, tmp_path)
    if kind == "grid":
        for x in range(2, 5):
            for y in range(2, 5):
                if not world.pos_to_agent[(x, y)]:
                    tag = f"block_{x}_{y}"
                    world.add_agent(tag, tag, "no_traits", position=(x, y))
    else:
        for neighbor in list(world.world_graph.neighbors("0")):
            world.world_graph.remove_edge("0", neighbor)
    before = dict(world.agent_energy)
    infos = world.step({"a": birth({"energy": 7})})[4]
    assert infos["a"]["spawn"]["status"] == "failed"
    assert world.agent_energy == {**before, "a": before["a"] - 20}
    assert "Child" not in world.name_to_tag


@pytest.mark.parametrize("cost", [-1, 20])
def test_minimum_population_arrivals_keep_founder_energy(tmp_path, cost):
    world = make_world("social", tmp_path, cost=cost)
    world._kill("a")
    world._kill("b")
    runner = SimulationRunner.__new__(SimulationRunner)
    runner.params = ExperimentConfig(agent={"genome": "no_traits"}, env={
        "world_type": "social_graph", "init_agents": 1, "min_agents": 1,
        "init_agent_energy": 80, "reproduction_cost": cost,
    })
    runner.env = world
    runner.agents = {}
    runner.obs, runner.infos, runner.dones = {}, {}, {}
    runner.last_spawn_idx = 0
    runner._make_llm_agent = lambda **kwargs: SimpleNamespace(genome=kwargs["genome"])
    runner._assign_sampled_lifespan = lambda _tag: None
    runner._respawn_if_needed()
    assert len(runner.agents) == 1
    assert list(world.agent_energy.values()) == [80]


def test_saved_parameters_preserve_reproduction_energy_state(tmp_path):
    (tmp_path / "params.json").write_text(json.dumps({"env": {
        "init_agent_energy": 80, "reproduction_cost": 20,
    }}))
    config = CheckpointManager(tmp_path).update_parameters()
    assert config.env.reproduction_cost == 20
    assert config.env.init_agent_energy == 80
    source = make_world("social", tmp_path / "source", cost=config.env.reproduction_cost)
    source.step({"a": birth({"energy": 7})})
    restored = make_world("social", tmp_path / "restored", cost=config.env.reproduction_cost)
    restored.set_state_ckpt(source.get_state_ckpt())
    assert restored.agent_energy == source.agent_energy
    assert sorted(restored.agent_energy.values()) == [27, 53, 80]


def test_invalid_partner_consumes_cost_but_not_gift(tmp_path):
    world = make_world("grid", tmp_path, paired=True)
    infos = world.step({"a": birth({"partner": "Missing", "energy": 7})})[4]
    assert infos["a"]["spawn"]["status"] == "failed"
    assert world.agent_energy == {"a": 60, "b": 80}
    assert len(world.agent_registry) == 2


def test_runner_resume_refreshes_legacy_spawn_menus_without_changing_balances(tmp_path):
    source = make_world("social", tmp_path / "source")
    source.agent_energy.update(a=55, b=32)
    legacy = {"spawn": {"description": "Legacy birth", "params": {"name": "Name"}}}
    source.agent_avail_actions = {tag: deepcopy(legacy) for tag in ("a", "b")}
    checkpoint = {
        "ts": 7, "last_spawn_idx": 1, "env": source.get_state_ckpt(),
        "agents": {tag: {"type": "LLMAgent", "tag": tag, "name": name}
                   for tag, name in (("a", "Alice"), ("b", "Bob"))},
        "env_outs": {
            "obs": {tag: source._build_obs(tag)[0] for tag in ("a", "b")},
            "infos": {tag: {"available_actions": deepcopy(legacy), "saved_result": "retain"}
                      for tag in ("a", "b")},
            "rewards": {}, "dones": {"a": False, "b": False},
        },
    }
    # A saved runner can still hold an agent already removed by the world.
    checkpoint["agents"]["dead"] = {"type": "LLMAgent", "tag": "dead", "name": "Departed"}
    checkpoint["env_outs"]["dones"]["dead"] = True
    checkpoint["env_outs"]["infos"]["dead"] = {"available_actions": deepcopy(legacy)}
    params = ExperimentConfig(agent={"genome": "no_traits"}, env={
        "world_type": "social_graph",
        "init_agent_energy": 80, "reproduction_cost": 20,
    })
    runner = SimulationRunner.__new__(SimulationRunner)
    runner.agents = {}
    runner.params = params
    runner.checkpointer = SimpleNamespace(
        load_checkpoint=lambda: checkpoint, update_parameters=lambda: params,
    )
    runner._restore_reward_state = lambda _state: None
    runner._make_env = lambda: setattr(runner, "env", make_world("social", tmp_path / "restored"))
    runner._make_llm_agent = lambda **kwargs: SimpleNamespace(
        genome=kwargs["genome"], set_state_ckpt=lambda _state: None,
    )
    runner._load_state()
    assert runner.env.agent_energy == {"a": 55, "b": 32}
    assert "dead" not in runner.env.agent_avail_actions
    for tag in ("a", "b"):
        live = runner.env.agent_avail_actions[tag]
        delivered = runner.infos[tag]["available_actions"]
        assert live == delivered
        assert "energy" in live["spawn"]["params"]
        assert "energy" in live["spawn"]["optional"]
        assert live["spawn"]["description"] != legacy["spawn"]["description"]
        assert runner.infos[tag]["saved_result"] == "retain"
