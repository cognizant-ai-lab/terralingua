"""energy_upkeep: the per-step energy drain is configurable and independent of natural food."""

import pytest
from pydantic import ValidationError

from terralingua.agents.prompt_templates import render_system_prompt
from terralingua.config.evaluation import evaluate
from terralingua.config.models import EnvConfig, GraphConfig
from terralingua.environment.graph_env import OpenGraphWorld

STAY = {"action": "move", "message": "", "params": {"direction": "stay"}}


def make_env(tmp_path, **kwargs):
    options = dict(
        log_path=tmp_path, headless=True, init_food=0, food_spawn_rate=0, food_decay_rate=0,
        init_agent_energy=5, lifespan=200, reproduction_cost=-1,
    )
    options.update(kwargs)
    env = OpenGraphWorld(graph_cfg=GraphConfig.complete(n_nodes=3), **options)
    env.add_agent("a", "Alice", "no_traits", position="0")
    env.restart_env(seed=1, agent_poses={"a": "0"})
    return env


def test_upkeep_without_food_drains_energy_and_kills_at_zero(tmp_path):
    env = make_env(tmp_path, food_mechanism=False, energy_upkeep=2)
    assert env.energy_upkeep == 2 and env.energy_death is True
    env.step({"a": STAY})
    assert env.agent_energy["a"] == 3
    env.step({"a": STAY})
    assert env.agent_energy["a"] == 1
    env.step({"a": STAY})
    assert "a" not in env.agent_registry


def test_defaults_follow_food_mechanism(tmp_path):
    with_food = make_env(tmp_path / "food", food_mechanism=True)
    assert with_food.energy_upkeep == 1 and with_food.energy_death is True
    without = make_env(tmp_path / "nofood", food_mechanism=False)
    assert without.energy_upkeep == 0 and without.energy_death is False
    without.step({"a": STAY})
    assert without.agent_energy["a"] == 5


def test_zero_upkeep_with_food_keeps_energy(tmp_path):
    env = make_env(tmp_path, food_mechanism=True, energy_upkeep=0)
    env.step({"a": STAY})
    assert env.agent_energy["a"] == 5
    assert env.energy_death is True  # food is on


def test_negative_upkeep_is_rejected(tmp_path):
    with pytest.raises(ValidationError):
        EnvConfig(energy_upkeep=-1)
    with pytest.raises(ValueError):
        make_env(tmp_path, food_mechanism=False, energy_upkeep=-1)


@pytest.mark.parametrize("food,upkeep,death,expected_death,expected_upkeep", [
    (True, None, None, True, 1),
    (False, None, None, False, 0),
    (False, 1, None, True, 1),
    (False, 3, False, False, 3),
    (True, 0, None, True, 0),
])
def test_evaluation_reports_the_resolved_upkeep_and_death(food, upkeep, death, expected_death, expected_upkeep):
    result = evaluate(overrides={"food_mechanism": food, "energy_upkeep": upkeep, "energy_death": death})
    assert result["valid"], result
    assert result["derived"]["energy_death"] is expected_death
    assert result["derived"]["energy_upkeep"] == expected_upkeep
    assert result["resolved"]["env"]["energy_upkeep"] == upkeep


def render(template, **flags):
    base = dict(agent_name="Alice", use_internal_memory=False, use_inventory=False, artifact_creation=False,
                finite_energy=True, finite_lifespan=True, external_actions=False, scenario_specific_instructions="")
    base.update(flags)
    return render_system_prompt(template, **base)


@pytest.mark.parametrize("template", ["graph.j2", "grid.j2", "social_graph.j2"])
def test_prompt_describes_the_drain_without_food(template):
    text = render(template, food_mechanism=False, energy_death=True, energy_upkeep=1)
    assert "lose 1 energy" in text and "you die" in text
    assert "refill your energy" not in text and "containing food" not in text and "receiving food" not in text
    text3 = render(template, food_mechanism=False, energy_death=True, energy_upkeep=3)
    assert "lose 3 energy" in text3


def test_prompt_keeps_the_food_sentence_with_food_and_drops_energy_without_any_drain():
    text = render("graph.j2", food_mechanism=True, energy_death=True, energy_upkeep=1)
    assert "lose 1 energy" in text and "containing food" in text
    none = render("graph.j2", food_mechanism=False, energy_death=False, energy_upkeep=0)
    assert "- Energy" not in none
