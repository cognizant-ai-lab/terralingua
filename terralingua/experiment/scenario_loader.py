"""Load the scenario named by run.scenario and build its mechanics."""

import importlib
import sys
from pathlib import Path

from terralingua.environment.mechanic import Mechanic


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
