"""Argument parsing needs no runner, secret files, or service initialization."""

import pytest

from terralingua.config.cli import parse_argv
from terralingua.config.compose import compose


@pytest.mark.parametrize("key,prefix", [("max_ts", "--"), ("graph.n_nodes", "--no-")])
def test_nonboolean_fields_require_values(key, prefix):
    with pytest.raises(SystemExit, match="requires a value"):
        parse_argv([prefix + key])


def test_flags_aliases_explicit_booleans_and_negative_values():
    preset, overrides, resume = parse_argv([
        "grid_baseline", "--remote_api_enabled", "--no-env.food_mechanism",
        "--env.energy_death=false", "--max_ts", "-1", "--resume",
    ])
    config = compose(preset, overrides)
    assert resume and preset == "grid_baseline"
    assert config.run.remote_api_enabled is True
    assert config.env.food_mechanism is False
    assert config.env.energy_death is False
    assert config.run.max_ts == -1


@pytest.mark.parametrize("key", ["agent.max_tokens", "max_agents", "energy_death", "graph.seed", "external_server_port"])
def test_nullable_values_clear_existing_settings(key):
    _, overrides, _ = parse_argv([f"--{key}=null"])
    assert overrides[key] is None
    compose(overrides=overrides)


def test_structured_collections_and_scenario_options():
    _, overrides, _ = parse_argv([
        "--ports", "[9010,9011]", "--excluded_actions", '["spawn","set_color"]',
        "--scenario_options", '{"seed": 17, "messages_enabled": false}',
        "--food_zones", "[[2,3],[4,5]]",
    ])
    config = compose(overrides=overrides)
    assert config.run.ports == (9010, 9011)
    assert config.env.excluded_actions == ["spawn", "set_color"]
    assert config.run.scenario_options == {"seed": 17, "messages_enabled": False}
    assert config.env.food_zones == [(2, 3), (4, 5)]


def test_scalar_food_zone_count_keeps_union_support():
    _, overrides, _ = parse_argv(["--food_zones", "3"])
    assert compose(overrides=overrides).env.food_zones == 3


@pytest.mark.parametrize("key", ["ports"])
def test_invalid_collection_text_has_actionable_error(key):
    with pytest.raises(SystemExit, match="valid JSON"):
        parse_argv([f"--{key}", "not JSON"])


def test_freeform_dictionary_subpaths_require_explicit_values():
    _, overrides, _ = parse_argv(["--run.scenario_options.seed", "17"])
    assert overrides == {"run.scenario_options.seed": "17"}
    with pytest.raises(SystemExit, match="requires a value"):
        parse_argv(["--run.scenario_options.messages_enabled"])


@pytest.mark.parametrize("key", ["agent.model.child"])
def test_unknown_paths_are_rejected_before_launch(key):
    with pytest.raises(SystemExit):
        parse_argv([f"--{key}=1"])


def test_strings_keep_literal_null_and_none():
    _, overrides, _ = parse_argv(["--exp_description=null", "--scenario_specific_instructions=none"])
    assert overrides == {"exp_description": "null", "scenario_specific_instructions": "none"}


def test_demo_and_empty_invocations_keep_their_contract():
    assert parse_argv([]) == (None, {}, False)
    assert parse_argv(["--demo"]) == ("demo", {}, False)


@pytest.mark.parametrize("argv", [
    ["--max_ts", "5", "--max_ts", "6"],
    ["grid_baseline", "--demo"],
    ["--scenario_options", '{"x":1,"x":2}'],
    ["--scenario_options", '{"x":NaN}'],
])
def test_cli_rejects_overwritten_or_nonfinite_input(argv):
    with pytest.raises(SystemExit):
        parse_argv(argv)
