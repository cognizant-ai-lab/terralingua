"""Per-agent action wording hides unlimited budgets without changing mechanics."""

import copy
import math

import pytest

from terralingua.agents.prompt_templates import resolve_instructions
from terralingua.config.models import GraphConfig
from terralingua.environment.actions import (
    ACTION_TEXT,
    LocationAffordance,
    build_direct_message_action,
    build_spawn_action,
)
from terralingua.environment.social_graph_env import OpenSocialGraphWorld


@pytest.fixture
def mixed_world(tmp_path):
    spec = {
        "description": "Query the world. Costs 11 energy.",
        "description_without_cost": "Query the world.",
        "cost": 11,
        "params": {"query": "Question about the world."},
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
        "optional": ["query"],
    }
    env = OpenSocialGraphWorld(
        graph_cfg=GraphConfig.complete(n_nodes=3, direct_message_cost=7),
        log_path=tmp_path, headless=True, food_mechanism=False,
        init_agent_energy=100, lifespan=200, reproduction_cost=20,
        init_food=0, food_spawn_rate=0, use_inventory=True,
        artifact_creation_cost=9,
        external_actions_spec={"world_query": spec},
    )
    poses = {}
    for tag, name, node in zip(("a", "b"), ("Alice", "Bob"), env.world_graph.all_nodes()):
        env.add_agent(tag, name, "no_traits", position=node)
        poses[tag] = node
    env.restart_env(agent_poses=poses)
    env.agent_energy["a"] = math.inf
    return env


def test_actual_actor_balance_controls_cost_prose_and_preserves_schema(mixed_world):
    env = mixed_world
    template_before = copy.deepcopy(ACTION_TEXT)
    specs_before = copy.deepcopy(env.external_actions_spec)
    unlimited = env._get_avail_actions("a")
    finite = env._get_avail_actions("b")

    assert set(unlimited) == set(finite)
    for action, amount in (("create_artifact", 9), ("send_direct_message", 7), ("world_query", 11)):
        assert f"{amount} energy" not in unlimited[action]["description"]
        assert f"{amount} energy" in finite[action]["description"]
        assert unlimited[action]["params"] == finite[action]["params"] or action == "send_direct_message"
    assert unlimited["world_query"]["description"] == "Query the world."
    assert unlimited["world_query"]["input_schema"] == finite["world_query"]["input_schema"]
    assert unlimited["world_query"]["optional"] == ["query"]
    assert "your current energy" not in unlimited["give"]["params"]["amount"]
    assert "your current energy" in finite["give"]["params"]["amount"]
    assert "target's current energy" in unlimited["take"]["params"]["amount"]
    assert "costs" not in unlimited["spawn"]["description"].lower()
    assert "new being starts with 20 energy" in unlimited["spawn"]["description"]
    assert "your balance" not in unlimited["spawn"]["params"]["energy"]
    assert "your balance" in finite["spawn"]["params"]["energy"]
    assert ACTION_TEXT == template_before
    assert env.external_actions_spec == specs_before

    # A later budget change must update wording without rewriting shared specs.
    env.agent_energy["a"] = 100
    assert "Costs 11 energy." in env._get_avail_actions("a")["world_query"]["description"]


def test_infinite_actor_dm_keeps_parameters_without_cost():
    _, unlimited = build_direct_message_action(["Bob"], 7, finite_energy=False)
    _, finite = build_direct_message_action(["Bob"], 7)
    assert "energy" not in unlimited["description"]
    assert unlimited["params"] == finite["params"]
    assert "uses your action for the turn" in unlimited["description"]


@pytest.mark.parametrize("cost,child_budget", [(20, 20), (0, 100), (0, math.inf)])
def test_infinite_parent_keeps_finite_child_budget_and_gift(cost, child_budget):
    initial = math.inf if math.isinf(child_budget) else 100
    _, entry = build_spawn_action(cost, initial, "sentence_directed", [], finite_energy=False)
    assert "Spawning costs" not in entry["description"]
    assert "Failed attempts still consume" not in entry["description"]
    assert "unlimited energy" not in entry["description"]
    if math.isfinite(child_budget):
        assert f"new being starts with {child_budget} energy" in entry["description"]
    else:
        assert "energy" not in entry["description"]
    assert entry["optional"] == ["energy"]
    assert "nonnegative integer" in entry["params"]["energy"]
    assert "balance" not in entry["params"]["energy"]
    assert "deducted" not in entry["params"]["energy"]
    assert "offspring_genome" in entry["params"]
    assert "partner" not in entry["params"]


def test_infinite_parent_spawning_keeps_finite_child_mechanics(mixed_world):
    env = mixed_world
    env.step({"a": {
        "action": "spawn", "message": "", "params": {"name": "Child", "energy": 5},
    }})
    child = env.name_to_tag["Child"]
    assert env.agent_energy[child] == 25
    assert math.isinf(env.agent_energy["a"])
    assert "Spawning costs 20 energy" in env._get_avail_actions(child)["spawn"]["description"]


@pytest.mark.parametrize("handler", ["_handle_drain_energy", "_handle_add_energy"])
@pytest.mark.parametrize("finite", [False, True])
def test_effect_receipts_hide_undefined_delta_but_preserve_finite_target_feedback(mixed_world, handler, finite):
    env = mixed_world
    env.agent_energy["b"] = 100 if finite else math.inf
    aff = LocationAffordance(action="support", effect={"amount": 5})
    info = {}
    getattr(env, handler)("a", {"target": "Bob"}, info, aff)
    assert info["a"]["support"]["status"] == "successful"
    assert info["a"]["support"]["target"] == "Bob"
    if finite:
        delta = -5 if handler == "_handle_drain_energy" else 5
        assert info["a"]["support"]["energy_change"] == delta
        assert f"Your energy changed by {delta}" in info["b"]["support"]
    else:
        assert "energy_change" not in info["a"]["support"]
        assert info["b"]["support"] == "Alice: support."
        assert math.isinf(env.agent_energy["b"])


def test_file_instructions_render_mortality_flags_and_default_to_finite(tmp_path):
    partials = tmp_path / "_partials"
    partials.mkdir()
    (partials / "memory.md").write_text(
        "{% if finite_lifespan %}Nothing survives your death.{% else %}Findings stay available.{% endif %}\n"
    )
    page = tmp_path / "instructions.md"
    page.write_text(
        '{% include "_partials/memory.md" %}\n'
        "{% if finite_energy %}Your energy runs out.{% else %}Your energy is unlimited.{% endif %}\n"
    )
    default = resolve_instructions(str(page))
    finite = resolve_instructions(str(page), finite_energy=True, finite_lifespan=True)
    unlimited = resolve_instructions(str(page), finite_energy=False, finite_lifespan=False)
    assert default == finite
    assert "your death" in finite
    assert "runs out" in finite
    assert "your death" not in unlimited
    assert "runs out" not in unlimited
    assert "Findings stay available" in unlimited
