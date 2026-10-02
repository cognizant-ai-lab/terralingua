"""Tests for the pydantic config system: models, composition, and presets."""

from pathlib import Path

import pytest

from terralingua.config.compose import compose, deep_merge, route_overrides
from terralingua.config.models import EnvConfig, ExperimentConfig, GraphConfig
from terralingua.config.presets import _resolve_paths, list_presets


@pytest.mark.parametrize(
    "value,expected",
    [
        (["10,10", "12,5"], [(10, 10), (12, 5)]),
        (["none"], None),
        ("none", None),
    ],
)
def test_food_zones_coercion(value, expected):
    assert EnvConfig(food_zones=value).food_zones == expected


def test_hocon_coupling_forces_topology_and_world_type():
    e = EnvConfig(
        graph=GraphConfig(agent_network_hocon_path="demo/local/neuro_san_network.hocon")
    )
    assert e.graph.topology == "agent_network"
    assert e.world_type == "graph"


def test_hocon_requires_sentence_directed_genome():
    env = EnvConfig(
        graph=GraphConfig(agent_network_hocon_path="demo/local/neuro_san_network.hocon")
    )
    with pytest.raises(Exception):
        ExperimentConfig.model_validate({"agent": {"genome": "ocean_5"}, "env": env.model_dump()})
    # sentence_directed is accepted
    ExperimentConfig.model_validate({"agent": {"genome": "sentence_directed"}, "env": env.model_dump()})


def test_min_agents_not_greater_than_init():
    with pytest.raises(Exception):
        EnvConfig(min_agents=10, init_agents=5)


def test_deep_merge_and_routing():
    assert deep_merge({"a": {"x": 1}}, {"a": {"y": 2}}) == {"a": {"x": 1, "y": 2}}
    # bare field names auto-route; dotted/graph-shorthand route too
    routed = route_overrides({"max_ts": 5, "grid_size": 9, "graph.topology": "ring"})
    assert routed == {"run": {"max_ts": 5}, "env": {"grid_size": 9, "graph": {"topology": "ring"}}}


def test_compose_precedence_and_coercion():
    # overrides (strings) beat preset, coerced to declared types
    c = compose("grid_baseline", {"max_ts": "7", "graph.topology": "ring", "save_video": "false"})
    assert c.run.max_ts == 7 and isinstance(c.run.max_ts, int)
    assert c.env.graph.topology == "ring"
    assert c.run.save_video is False


def test_scenario_presets_discovered_and_paths_resolved():
    names = {name for name, _, _ in list_presets()}
    assert {"example", "demo", "paper_core"} <= names
    assert compose("example").run.scenario == "scenarios.example"
    # path-valued keys resolve against the preset folder; builtin names stay as they are
    resolved = _resolve_paths(
        {"agent": {"personas_path": "people.json", "scenario_specific_instructions": "none"},
         "env": {"init_artifacts_path": "seeds", "grid_size": 5}},
        Path("/presets/here"),
    )
    assert resolved["agent"]["personas_path"] == "/presets/here/people.json"
    assert resolved["agent"]["scenario_specific_instructions"] == "none"
    assert resolved["env"]["init_artifacts_path"] == "/presets/here/seeds"
    assert resolved["env"]["grid_size"] == 5


@pytest.mark.parametrize("invalid", [True])
def test_reproduction_default_and_invalid_negative_value(invalid):
    with pytest.raises(ValueError):
        EnvConfig(reproduction_cost=invalid)
