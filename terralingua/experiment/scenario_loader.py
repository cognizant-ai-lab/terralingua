"""Load the scenario named by run.scenario and build its mechanics."""

import importlib
import sys
from pathlib import Path
from typing import get_args

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import BaseModel, ValidationError

from terralingua.config.dependencies import dependencies, prefixed
from terralingua.environment.mechanic import Mechanic

OPTIONS_PREFIX = "run.scenario_options."


def import_scenario(module_path: str):
    """Import the scenario module.

    The working directory is put on the import path first, as `python -m` does,
    so a scenario package next to the presets imports under a console command.
    The runner calls this before it reads servers.json. Registrations that run
    at import time (artifact types, external state handlers) are then in place.
    """
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    return importlib.import_module(module_path)


def load_scenario(module_path: str, raw_options: dict | None) -> list[Mechanic]:
    """Import the scenario module, validate its options, and build its mechanics.

    The module exposes `Options`, a pydantic model for run.scenario_options, and
    `build(options)`, which returns the mechanics to attach to the world.
    """
    module = import_scenario(module_path)
    for attribute in ("Options", "build"):
        if not hasattr(module, attribute):
            raise ValueError(f"Scenario module '{module_path}' has no '{attribute}'")
    options = module.Options.model_validate(raw_options or {})
    mechanics = list(module.build(options))
    for mechanic in mechanics:
        if not isinstance(mechanic, Mechanic):
            raise TypeError(
                f"Scenario '{module_path}' built a {type(mechanic).__name__}, not a Mechanic"
            )
    return mechanics


def flatten_options(options: dict, parent: str = ""):
    """Yield (dotted path, value) for every option, nested ones included."""
    for key, value in options.items():
        path = f"{parent}{key}"
        yield path, value
        if isinstance(value, dict):
            yield from flatten_options(value, f"{path}.")


def option_paths(model: type[BaseModel], parent: str = "") -> set[str]:
    """Every dotted option name a pydantic model accepts, nested models included."""
    paths = set()
    for name, field in model.model_fields.items():
        path = f"{parent}{name}"
        paths.add(path)
        for candidate in (field.annotation, *get_args(field.annotation)):
            if isinstance(candidate, type) and issubclass(candidate, BaseModel):
                paths |= option_paths(candidate, f"{path}.")
    return paths


class ScenarioImportError(ImportError):
    """The scenario module could not be imported from the working directory."""


def scenario_applicability(module_path: str, raw_options: dict | None) -> tuple[dict, dict, dict]:
    """The scenario's applicability table, its option values and its defaults, all
    under run.scenario_options.

    The module may expose `APPLICABILITY`: option name to (condition, reason), in
    the shape of the core's table, with conditions over option names. The values
    come from the validated options, so defaults count. The defaults are empty
    when the model has required options. Raises ScenarioImportError when the
    module cannot be imported, pydantic's ValidationError when the options are
    invalid, and ValueError when the table names an unknown option or holds a
    condition that is not valid JSON Schema.
    """
    try:
        module = import_scenario(module_path)
    except Exception as exc:
        raise ScenarioImportError(f"{module_path}: {exc}") from exc
    if not hasattr(module, "Options"):
        raise ValueError(f"Scenario module '{module_path}' has no 'Options'")
    table = getattr(module, "APPLICABILITY", {})
    options = module.Options.model_validate(raw_options or {}).model_dump(mode="json")
    values = {OPTIONS_PREFIX + path: value for path, value in flatten_options(options)}
    try:
        defaults = module.Options().model_dump(mode="json")
    except ValidationError:
        defaults = {}
    defaults = {OPTIONS_PREFIX + path: value for path, value in flatten_options(defaults)}
    known = option_paths(module.Options)
    applicability = {}
    for path, (condition, reason) in table.items():
        condition = prefixed(condition, OPTIONS_PREFIX)
        try:
            Draft202012Validator.check_schema(condition)
        except SchemaError as exc:
            raise ValueError(
                f"Scenario '{module_path}' applicability for '{path}' is not valid JSON Schema: {exc.message}"
            ) from exc
        names = [path] + [ref[len(OPTIONS_PREFIX):] for ref in dependencies(condition)]
        unknown = sorted(name for name in names if name not in known)
        if unknown:
            raise ValueError(
                f"Scenario '{module_path}' applicability names unknown options: {unknown}"
            )
        applicability[OPTIONS_PREFIX + path] = (condition, reason)
    return applicability, values, defaults
