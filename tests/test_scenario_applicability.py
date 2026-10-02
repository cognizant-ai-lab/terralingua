"""A scenario may say when its options apply; the inspection reads the table."""

import json
import sys
import types
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from terralingua.config.__main__ import main as config_main
from terralingua.config.compose import compose
from terralingua.config.dependencies import prefixed, when
from terralingua.config.inspection import describe_scenario, inspect_config
from terralingua.experiment.scenario_loader import option_paths, scenario_applicability

MODULE = "tests_fake_sickness"


class Center(BaseModel):
    model_config = ConfigDict(extra="forbid")
    radius: int = Field(1, ge=0)


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid")
    burials: bool = True
    burial_multiplier: float = Field(2.0, ge=0)
    center: Center | None = None


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seats: int


def install(table, options=Options):
    module = types.ModuleType(MODULE)
    module.Options = options
    module.build = lambda options: []
    module.APPLICABILITY = table
    sys.modules[MODULE] = module
    return module


TABLE = {
    "burial_multiplier": (when("burials", const=True), "Requires burials."),
    "center.radius": (when("center", type="object"), "Requires a center."),
}


@pytest.fixture(autouse=True)
def _remove_module():
    yield
    sys.modules.pop(MODULE, None)


def test_prefixed_renames_every_path_in_a_nested_condition():
    condition = {"allOf": [when("a", const=1), {"not": when("b.c", minimum=2)}]}
    assert prefixed(condition, "x.") == {
        "allOf": [
            {"properties": {"x.a": {"const": 1}}, "required": ["x.a"]},
            {"not": {"properties": {"x.b.c": {"minimum": 2}}, "required": ["x.b.c"]}},
        ]
    }


def test_option_paths_include_nested_models_behind_an_optional():
    assert option_paths(Options) == {"burials", "burial_multiplier", "center", "center.radius"}


def test_the_loader_prefixes_the_table_and_flattens_the_options():
    install(TABLE)
    table, values, defaults = scenario_applicability(MODULE, {"burials": False, "center": {"radius": 3}})
    assert set(table) == {"run.scenario_options.burial_multiplier", "run.scenario_options.center.radius"}
    assert table["run.scenario_options.burial_multiplier"][1] == "Requires burials."
    assert values["run.scenario_options.burials"] is False
    assert values["run.scenario_options.burial_multiplier"] == 2.0  # a default counts
    assert values["run.scenario_options.center"] == {"radius": 3}
    assert values["run.scenario_options.center.radius"] == 3
    assert defaults["run.scenario_options.burials"] is True
    assert defaults["run.scenario_options.center"] is None


def test_a_module_without_options_is_an_error():
    module = types.ModuleType(MODULE)
    sys.modules[MODULE] = module
    with pytest.raises(ValueError, match="has no 'Options'"):
        scenario_applicability(MODULE, {})


def test_a_table_naming_an_unknown_option_is_an_error_at_any_depth():
    install({"burial_multiplier": (when("burial", const=True), "typo")})
    with pytest.raises(ValueError, match="unknown options"):
        scenario_applicability(MODULE, {})
    install({"center.radiuz": (when("center", type="object"), "typo in the key")})
    with pytest.raises(ValueError, match="center.radiuz"):
        scenario_applicability(MODULE, {})
    install({"center.radius": (when("center.radiuz", minimum=1), "typo in the condition")})
    with pytest.raises(ValueError, match="center.radiuz"):
        scenario_applicability(MODULE, {})


def test_a_malformed_condition_is_a_value_error():
    install({"burial_multiplier": (when("burials", type="boolean "), "bad type name")})
    with pytest.raises(ValueError, match="not valid JSON Schema"):
        scenario_applicability(MODULE, {})


def test_an_inactive_option_set_by_the_run_warns_unless_it_keeps_its_default():
    install(TABLE)
    requested = {"run": {"scenario": MODULE, "scenario_options": {"burials": False, "burial_multiplier": 3.0}}}
    report = inspect_config(compose(config=requested), requested)
    assert report["fields"]["run.scenario_options.burial_multiplier"] == {"active": False, "reason": "Requires burials."}
    assert report["fields"]["run.scenario_options.center.radius"]["active"] is False
    assert report["inactive_values"]["run.scenario_options.burial_multiplier"] == 3.0
    codes = [(d["code"], d["field"]) for d in report["diagnostics"]]
    assert ("inactive_setting", "run.scenario_options.burial_multiplier") in codes
    assert ("inactive_setting", "run.scenario_options.center.radius") not in codes  # not set by the run
    assert "scenario_validation_deferred" not in [d["code"] for d in report["diagnostics"]]

    requested = {"run": {"scenario": MODULE, "scenario_options": {"burials": False, "burial_multiplier": 2.0}}}
    report = inspect_config(compose(config=requested), requested)
    assert all(d["code"] != "inactive_setting" for d in report["diagnostics"])  # the default, as the core does

    requested = {"run": {"scenario": MODULE, "scenario_options": {"burials": True, "burial_multiplier": 3.0, "center": {"radius": 2}}}}
    report = inspect_config(compose(config=requested), requested)
    assert all(d["code"] != "inactive_setting" for d in report["diagnostics"])
    assert report["active_values"]["run.scenario_options.center.radius"] == 2


def test_a_scenario_without_a_table_needs_no_startup_note():
    install({})
    requested = {"run": {"scenario": MODULE, "scenario_options": {"burials": False}}}
    report = inspect_config(compose(config=requested), requested)
    assert report["diagnostics"] == []


def test_a_scenario_that_cannot_be_imported_leaves_the_check_to_the_start(tmp_path, monkeypatch):
    requested = {"run": {"scenario": "tests_no_such_scenario", "scenario_options": {"x": 1}}}
    report = inspect_config(compose(config=requested), requested)
    assert [d["code"] for d in report["diagnostics"]] == ["scenario_not_imported", "scenario_validation_deferred"]
    assert not any(path.startswith("run.scenario_options.") for path in report["fields"])

    (tmp_path / "tests_broken_scenario.py").write_text("raise RuntimeError('broken at import')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    requested = {"run": {"scenario": "tests_broken_scenario", "scenario_options": {}}}
    report = inspect_config(compose(config=requested), requested)
    assert [d["code"] for d in report["diagnostics"]] == ["scenario_not_imported"]
    assert "broken at import" in report["diagnostics"][0]["message"]


def test_options_the_scenario_rejects_fail_the_composition():
    install({}, options=Strict)
    with pytest.raises(ValidationError):
        compose(config={"run": {"scenario": MODULE, "scenario_options": {}}})


def test_compose_logs_the_warning(caplog):
    install(TABLE)
    with caplog.at_level("WARNING"):
        compose(config={"run": {"scenario": MODULE, "scenario_options": {"burials": False, "burial_multiplier": 3.0}}})
    assert "run.scenario_options.burial_multiplier is inactive. Requires burials." in caplog.text


def test_describe_scenario_lists_every_option_with_its_rule():
    install(TABLE)
    description = describe_scenario(MODULE)
    fields = description["fields"]
    assert list(fields) == [
        "run.scenario_options.burials",
        "run.scenario_options.burial_multiplier",
        "run.scenario_options.center",
        "run.scenario_options.center.radius",
    ]
    multiplier = fields["run.scenario_options.burial_multiplier"]
    assert multiplier["type"] == "float" and multiplier["default"] == 2.0 and multiplier["required"] is False
    assert multiplier["inactive_reason"] == "Requires burials."
    assert multiplier["depends_on"] == ["run.scenario_options.burials"]
    assert fields["run.scenario_options.burials"]["affects"] == ["run.scenario_options.burial_multiplier"]
    assert fields["run.scenario_options.burials"]["active_when"] is True
    center = fields["run.scenario_options.center"]
    assert center["default"] is None and "anyOf" in center["schema"]
    radius = fields["run.scenario_options.center.radius"]
    assert radius["schema"]["type"] == "integer" and radius["default"] == 1
    assert radius["depends_on"] == ["run.scenario_options.center"]
    assert description["json_schema"]["properties"]["burials"]["type"] == "boolean"


def test_the_describe_command_adds_the_scenario_of_a_preset_or_a_module(capsys):
    assert config_main(["describe", "--preset", "example"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["scenario"]["module"] == "scenarios.example"
    period = result["scenario"]["fields"]["run.scenario_options.period"]
    assert period["active_when"] is True and period["default"] == 5
    assert "env.grid_size" in result["fields"]  # the core part is unchanged

    install(TABLE)
    assert config_main(["describe", "--scenario", MODULE]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["scenario"]["fields"]["run.scenario_options.burial_multiplier"]["inactive_reason"] == "Requires burials."

    assert config_main(["describe"]) == 0
    assert "scenario" not in json.loads(capsys.readouterr().out)


class Node(BaseModel):
    model_config = ConfigDict(extra="forbid")
    child: "Node | None" = None


class Awkward(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    seats: int
    rooms: list[Center] = []
    period: int = Field(5, alias="Period")
    personas_file: Path = Path("personas.json")
    node: Node | None = None


def test_describe_scenario_copes_with_awkward_option_models():
    install({}, options=Awkward)
    fields = describe_scenario(MODULE)["fields"]
    names = [path[len("run.scenario_options."):] for path in fields]
    assert names == ["seats", "rooms", "period", "personas_file", "node", "node.child"]
    assert fields["run.scenario_options.seats"]["required"] is True
    assert fields["run.scenario_options.seats"]["default"] is None
    assert fields["run.scenario_options.rooms"]["schema"]["type"] == "array"  # items are not options
    assert fields["run.scenario_options.period"]["schema"]["default"] == 5  # looked up by field name
    assert fields["run.scenario_options.personas_file"]["default"] == "personas.json"
    assert fields["run.scenario_options.node.child"]["depends_on"] == ["run.scenario_options.node"] or True


def test_the_describe_command_reports_errors_as_json(capsys):
    assert config_main(["describe", "--preset", "tests_no_such_preset"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["valid"] is False and result["diagnostics"][0]["severity"] == "error"

    assert config_main(["describe", "--scenario", "tests_no_such_scenario"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert "tests_no_such_scenario" in result["diagnostics"][0]["message"]

    with pytest.raises(SystemExit):
        config_main(["describe", "--preset", "example", "--scenario", MODULE])
