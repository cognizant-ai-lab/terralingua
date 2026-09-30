"""Config input keeps explicit settings and reports ambiguous input."""

import pytest

from terralingua.config import presets
from terralingua.config.compose import compose, compose_input, route_overrides
from terralingua.config.models import ExperimentConfig


@pytest.mark.parametrize(
    "overrides",
    [
        {"env.typo": 5},
        {"agent.model.child": "value"},
        {"env..grid_size": 5},
        {"graph": {"typo": 5}},
    ],
)
def test_unknown_or_invalid_override_paths_are_rejected(overrides):
    with pytest.raises(KeyError):
        route_overrides(overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_ts": 10, "run.max_ts": 20},
        {"graph": {"topology": "ring"}, "graph.n_nodes": 4},
        {"graph.n_nodes": 4, "graph": {"topology": "ring"}},
        {"run.scenario_options": {"seed": 1}, "run.scenario_options.seed": 2},
    ],
)
def test_overlapping_aliases_are_rejected_in_either_order(overrides):
    with pytest.raises(ValueError, match="overlap"):
        route_overrides(overrides)


def test_dictionary_settings_keep_arbitrary_keys_and_values():
    values = {
        "run.scenario_options.seed": 7,
        "run.scenario_options.nested.option": True,
    }
    routed = route_overrides(values)
    assert routed["run"]["scenario_options"] == {
        "seed": 7,
        "nested": {"option": True},
    }
    ExperimentConfig.model_validate(routed)


@pytest.mark.parametrize(
    "config",
    [
        {"env": {"graph": {"toplogy": "ring"}}},
    ],
)
def test_unknown_nested_config_keys_are_rejected(config):
    with pytest.raises(KeyError):
        compose_input(config=config)


def test_raw_layers_preserve_requested_values_before_validation():
    requested = compose_input(
        "grid_baseline",
        {"max_ts": "7"},
        {"run": {"max_ts": 8}, "env": {"grid_size": 9}},
    )
    assert requested["run"]["max_ts"] == "7"
    assert requested["env"]["grid_size"] == 9
    assert "agent" not in requested
    effective = compose(
        "grid_baseline",
        {"max_ts": "7"},
        config={"run": {"max_ts": 8}, "env": {"grid_size": 9}},
    )
    assert effective.run.max_ts == 7
    assert effective.env.grid_size == 9


def test_raw_layers_and_builtin_presets_are_independent_copies():
    config = {"env": {"excluded_actions": ["take"]}}
    requested = compose_input(config=config)
    requested["env"]["excluded_actions"].append("give")
    assert config["env"]["excluded_actions"] == ["take"]
    first = presets.get_preset("grid_baseline")
    first["env"]["grid_size"] = 999
    assert presets.get_preset("grid_baseline")["env"]["grid_size"] == 50


@pytest.mark.parametrize("field,value", [("spawn_allowed", False)])
def test_legacy_reproduction_inputs_are_rejected_in_all_layers(field, value):
    with pytest.raises(KeyError, match="Unknown"):
        compose(config={"env": {field: value}})
    for path in (field, f"env.{field}"):
        with pytest.raises(KeyError, match="Unknown"):
            compose(overrides={path: value})


@pytest.mark.parametrize("value", ["base"])
def test_builtin_instruction_names_are_not_resolved_as_paths(tmp_path, value):
    path = tmp_path / "preset.yaml"
    path.write_text(
        f"name: custom\nconfig:\n  agent:\n    scenario_specific_instructions: {value}\n"
    )
    data = presets._discover_presets(tmp_path)["custom"][1]
    assert data["agent"]["scenario_specific_instructions"] == value
    assert ExperimentConfig.model_validate(data).agent.scenario_specific_instructions == value


def test_relative_instruction_file_still_resolves(tmp_path):
    instructions = tmp_path / "instructions.md"
    instructions.write_text("Use the available actions.")
    (tmp_path / "preset.yaml").write_text(
        "config:\n  agent:\n    scenario_specific_instructions: instructions.md\n"
    )
    data = next(iter(presets._discover_presets(tmp_path).values()))[1]
    assert data["agent"]["scenario_specific_instructions"] == str(instructions)


@pytest.mark.parametrize(
    "contents",
    [
        "",
        "name: []\nconfig: {}",
        "description: []\nconfig: {}",
        "config: []",
        "config: {run: []}",
        "config: {env: {graph: []}}",
        "config: {unknown: {}}",
        "config: {}\nextra: ignored",
        "config: {run: {max_ts: 1, max_ts: 2}}",
        "config: [",
    ],
)
def test_invalid_preset_input_is_reported_with_path(tmp_path, contents):
    path = tmp_path / "preset.yaml"
    path.write_text(contents)
    with pytest.raises(ValueError, match="preset.yaml"):
        presets._discover_presets(tmp_path)


def test_duplicate_preset_names_are_rejected(tmp_path):
    (tmp_path / "a.preset.yaml").write_text("name: duplicate\nconfig: {}\n")
    (tmp_path / "b.preset.yaml").write_text("name: duplicate\nconfig: {}\n")
    with pytest.raises(ValueError, match="both.*a.preset.yaml.*b.preset.yaml"):
        presets._discover_presets(tmp_path)


def test_builtin_preset_name_collision_is_rejected(tmp_path):
    (tmp_path / "preset.yaml").write_text("name: grid_baseline\nconfig: {}\n")
    with pytest.raises(ValueError, match="built-in"):
        presets._discover_presets(tmp_path)


def test_yaml_merge_can_be_overridden_explicitly(tmp_path):
    (tmp_path / "preset.yaml").write_text(
        "config:\n"
        "  env: &environment\n"
        "    init_agents: 2\n"
        "  run:\n"
        "    scenario_options:\n"
        "      <<: *environment\n"
        "      init_agents: 3\n"
    )
    data = next(iter(presets._discover_presets(tmp_path).values()))[1]
    assert data["run"]["scenario_options"]["init_agents"] == 3


def test_reference_template_is_not_discovered(tmp_path):
    (tmp_path / "template_preset.yaml").write_text("config: {}")
    assert presets._discover_presets(tmp_path) == {}


@pytest.mark.parametrize("name", [name for name, _, _ in presets.list_presets()])
def test_all_installed_presets_validate(name):
    compose(name)



def test_scenario_dictionary_keys_are_not_config_file_paths(tmp_path):
    (tmp_path / "preset.yaml").write_text(
        "config:\n"
        "  run:\n"
        "    scenario_options:\n"
        "      file_path: local-input\n"
        "      scenario_specific_instructions: custom-marker\n"
    )
    data = next(iter(presets._discover_presets(tmp_path).values()))[1]
    assert data["run"]["scenario_options"] == {
        "file_path": "local-input",
        "scenario_specific_instructions": "custom-marker",
    }
