"""Regression checks for parameter dependencies and the inspection interface."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from terralingua.config.compose import compose
from terralingua.config.evaluation import evaluate
from terralingua.config.inspection import describe, field_definitions
from terralingua.config.models import EnvConfig, GraphConfig

REPO = Path(__file__).resolve().parents[1]


def test_contract_covers_every_field_and_links_both_directions():
    schema = describe()
    assert schema["format_version"] == 1
    assert set(schema["fields"]) == {path for path, *_ in field_definitions()}
    for path, field in schema["fields"].items():
        assert field["description"], path
        Draft202012Validator.check_schema(field["active_when"])
        for dependency in field["depends_on"]:
            assert path in schema["fields"][dependency]["affects"]


def test_contract_choices_and_fingerprint_are_complete_and_stable():
    schema = describe()
    original_hash = schema["schema_hash"]
    schema["fields"]["agent.max_history"]["default"] = 999
    schema["fields"]["env.food_spawn_rate"]["active_when"].clear()
    current = describe()
    assert current["schema_hash"] == original_hash
    assert current["fields"]["agent.max_history"]["default"] == 10
    assert current["fields"]["env.food_spawn_rate"]["active_when"]


@pytest.mark.parametrize("overrides,field,active", [
    ({"food_mechanism": False}, "env.food_spawn_rate", False),
    ({"food_mechanism": False}, "env.food_zones", True),
    ({"world_type": "social_graph"}, "env.food_zones", False),
    ({"reproduction_cost": -1}, "env.two_parent_spawn", False),
    ({"reproduction_cost": 0}, "env.two_parent_spawn", True),
    ({"genome": "no_traits"}, "env.genome_mutation_rate", False),
    ({"genome": "sentence_mutate"}, "env.genome_mutation_rate", True),
    ({"world_type": "graph"}, "env.graph.hop_radius", True),
    ({"world_type": "social_graph"}, "env.graph.hop_radius", False),
    ({"world_type": "graph", "graph.topology": "tree"}, "env.graph.small_world_p", False),
    ({"world_type": "graph", "graph.topology": "small_world"}, "env.graph.small_world_p", True),
    ({"world_type": "social_graph", "artifact_creation_cost": -1, "use_inventory": False}, "env.library_preview", True),
    ({"world_type": "social_graph", "inert_artifacts": True}, "env.library_preview", False),
    ({"world_type": "social_graph", "inert_artifacts": True}, "env.use_library", True),
    ({"artifact_creation_cost": -1}, "env.allow_fixed_artifacts", False),
    ({"artifact_creation_cost": 0}, "env.allow_fixed_artifacts", True),
])
def test_applicability_matches_independent_mechanisms(overrides, field, active):
    result = evaluate(overrides=overrides)
    assert result["valid"], result["diagnostics"]
    assert result["fields"][field]["active"] is active
    assert bool(result["fields"][field]["reason"]) is not active


def test_inactive_values_survive_toggling_and_do_not_change_inputs():
    submitted = {"save_video": False, "video_fps": 37}
    result = evaluate(overrides=submitted)
    assert submitted == {"save_video": False, "video_fps": 37}
    assert result["resolved"]["run"]["video_fps"] == 37
    assert result["inactive_values"]["run.video_fps"] == 37
    restored = evaluate(config=result["resolved"], overrides={"save_video": True})
    assert restored["active_values"]["run.video_fps"] == 37


@pytest.mark.parametrize("cost,initial,expected", [
    (-1, 500, None), (10, 500, 10), (0, -1, "unlimited"),
])
def test_reproduction_summary_matches_funding(cost, initial, expected):
    result = evaluate(overrides={
        "reproduction_cost": cost, "init_agent_energy": initial, "food_mechanism": False,
    })
    assert result["valid"]
    assert result["derived"]["newborn_base_energy"] == expected
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("food,death,expected", [
    (True, None, True), (False, None, False), (False, True, True),
])
def test_energy_death_inheritance_is_reported_without_overwriting_input(food, death, expected):
    result = evaluate(overrides={"food_mechanism": food, "energy_death": death})
    assert result["valid"]
    assert result["derived"]["energy_death"] is expected
    assert result["resolved"]["env"]["energy_death"] is death


def test_hocon_normalization_is_visible_and_requested_values_are_preserved():
    result = evaluate(config={
        "agent": {"genome": "sentence_directed"},
        "env": {"world_type": "grid", "graph": {
            "topology": "ring", "agent_network_hocon_path": "not-opened.hocon",
        }},
    })
    assert result["valid"]
    assert result["requested"]["env"]["world_type"] == "grid"
    assert result["resolved"]["env"]["world_type"] == "graph"
    changes = {change["field"]: change for change in result["normalizations"]}
    assert changes["env.world_type"]["requested"] == "grid"
    assert changes["env.graph.topology"]["effective"] == "agent_network"


@pytest.mark.parametrize("overrides", [
    {"env.typo": 1}, {"max_parallel_workers": 0}, {"max_ts": True},
    {"world_type": "graph", "graph.topology": "file"},
    {"food_zones": []},
    {"agents_name_prefix": "human", "init_agents": 2, "init_human_agents": 1},
])
def test_invalid_configuration_returns_errors_instead_of_partial_success(overrides):
    result = evaluate(overrides=overrides)
    assert not result["valid"]
    assert result["resolved"] is None
    assert result["diagnostics"]
    assert all(item["severity"] == "error" for item in result["diagnostics"])
    assert result["input"]["overrides"] == overrides


@pytest.mark.parametrize("value", [float("nan")])
def test_nonfinite_json_input_is_reported_safely(value):
    result = evaluate(config={"run": {"scenario_options": {"value": value}}})
    assert not result["valid"]
    assert result["input"] is None
    json.dumps(result, allow_nan=False)


def test_inactive_food_zones_and_graph_node_ids_remain_intact():
    assert EnvConfig(food_mechanism=False, food_zones=0).food_zones == 0
    assert EnvConfig(world_type="graph", food_zones=["100"]).food_zones == ["100"]
    assert EnvConfig(world_type="graph", food_zones=["hub"]).food_zones == ["hub"]
    assert EnvConfig(food_zones=["1,2"]).food_zones == [(1, 2)]


def test_topology_parameters_follow_current_settings():
    graph = GraphConfig.small_world(n_nodes=20, k=4, p=0.1)
    graph.small_world_k = 6
    graph.small_world_p = 0.8
    assert graph._topology_params == {"k": 6, "p": 0.8}
    assert "_topology_params" not in graph.model_dump()


def test_free_dictionary_values_are_preserved_as_atomic_parameters():
    result = evaluate(overrides={
        "runner_class": "package:Runner", "scenario_options": {"nested": {"choice": 3}},
    })
    assert result["valid"]
    assert result["active_values"]["run.scenario_options"] == {"nested": {"choice": 3}}
    assert any(item["code"] == "scenario_validation_deferred" for item in result["diagnostics"])


def test_inspection_import_does_not_load_runner_or_dotenv():
    script = (
        "import sys; from terralingua.config.inspection import describe; describe(); "
        "assert 'main' not in sys.modules; "
        "assert 'terralingua.experiment.runner' not in sys.modules; "
        "assert 'terralingua.external.request_manager' not in sys.modules"
    )
    process = subprocess.run([sys.executable, "-c", script], cwd=REPO, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr


@pytest.mark.parametrize("command,payload,code", [
    (["describe"], None, 0),
    (["evaluate", "--overrides", "-"], '{"reproduction_cost":10,"food_mechanism":false}', 0),
    (["evaluate", "--overrides", "-"], '{"max_parallel_workers":0}', 2),
    (["evaluate", "--overrides", "-"], 'not-json', 2),
])
def test_machine_interface_writes_only_json(command, payload, code):
    process = subprocess.run(
        [sys.executable, "-m", "terralingua.config", *command], cwd=REPO,
        input=payload, capture_output=True, text=True,
        env={**os.environ, "PYGAME_HIDE_SUPPORT_PROMPT": "1"},
    )
    assert process.returncode == code, process.stderr
    result = json.loads(process.stdout)
    assert "Traceback" not in process.stderr
    if command[0] == "evaluate":
        assert result["valid"] is (code == 0)


def test_social_food_is_disabled_by_default_and_cannot_be_enabled():
    result = evaluate(overrides={"world_type": "social_graph"})
    assert result["valid"]
    assert result["resolved"]["env"]["food_mechanism"] is False
    assert result["fields"]["env.food_mechanism"]["active"] is False
    assert result["derived"]["energy_death"] is False
    assert not evaluate(overrides={"world_type": "social_graph", "food_mechanism": True})["valid"]
    assert describe()["fields"]["env.food_mechanism"]["default_when"]["value"] is False


def test_nonfinite_values_from_presets_are_reported_safely(monkeypatch):
    monkeypatch.setattr("terralingua.config.compose.get_preset", lambda _name: {"env": {"food_decay_rate": float("nan")}})
    result = evaluate("bad")
    assert not result["valid"]
    assert result["requested"] is None
    assert result["input"]["preset"] == "bad"
    json.dumps(result, allow_nan=False)


def test_normal_launch_composition_reports_inactive_values(caplog):
    config = compose(overrides={"save_video": False, "video_fps": 37})
    assert config.run.video_fps == 37
    assert "run.video_fps is inactive" in caplog.text


@pytest.mark.parametrize("prefix,humans,valid", [
    ("human", 1, False), ("human1", 10, True), ("human1", 11, False),
    ("human01", 1000, True),
])
def test_initial_tag_collision_checks_numeric_prefix_suffixes(prefix, humans, valid):
    result = evaluate(overrides={"agents_name_prefix": prefix, "init_agents": 1, "init_human_agents": humans})
    assert result["valid"] is valid
