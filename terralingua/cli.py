#!/usr/bin/env python
# Copyright © 2025 Cognizant Technology Solutions Corp, www.cognizant.com.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# END COPYRIGHT

"""Single entry point: configure and run a simulation.

    terralingua                       # model defaults (grid world), no preset
    terralingua grid_baseline         # a named preset (see --list)
    terralingua example --max_ts 50 --model claude-sonnet-4-6
    terralingua --list                # list available presets
    terralingua <preset> --resume     # resume from the latest checkpoint
    terralingua --demo                # alias for the `demo` preset (scenarios/demo)

Config layers, lowest-to-highest precedence: model defaults -> preset -> CLI
overrides. Any `--field`, `--section.field`, `--graph.field`, or `--no-flag`
token is an override; string values are coerced to the model's declared types.
run_suite.sh starts Redis, the dashboard, a run, and the anthropologist together;
the Docker demos live under demo/.
"""

import asyncio
import importlib
import logging
import os
import sys

from dotenv import find_dotenv, load_dotenv

from terralingua.config.cli import parse_argv
from terralingua.config.compose import compose
from terralingua.config.models import iter_config_fields
from terralingua.config.presets import list_presets
from terralingua.experiment.checkpoint import CheckpointManager
from terralingua.experiment.runner import SimulationRunner
from terralingua.server.redis_bus import init_redis_pool, redis_url
from terralingua.server.redis_connection_manager import RedisConnectionManager
from terralingua.server.redis_publisher import RedisPublisher
from terralingua.server.redis_registry import RedisAgentRegistry
from terralingua.utils.generic import LOGS_DIR
from terralingua.utils.logging_setup import setup_logging

_dotenv_path = find_dotenv(usecwd=True)
load_dotenv(_dotenv_path, override=True)

setup_logging()
log = logging.getLogger(__name__)
_key = os.getenv("ANTHROPIC_API_KEY", "")
log.debug("[DEBUG] .env path : %r", _dotenv_path)
log.debug("[DEBUG] API key set: %s", bool(_key))


def _runner_class(spec: str | None) -> type[SimulationRunner]:
    """Load an optional scenario runner. Existing presets use the base runner."""
    if spec is None:
        return SimulationRunner
    module_name, separator, class_name = spec.partition(":")
    if not separator or not module_name or not class_name:
        raise ValueError("runner_class must use module:Class syntax")
    candidate = getattr(importlib.import_module(module_name), class_name)
    if not isinstance(candidate, type) or not issubclass(candidate, SimulationRunner):
        raise ValueError("runner_class must inherit SimulationRunner")
    return candidate


async def _run(params, resume: bool) -> None:
    if resume:
        if not params.run.exp_name:
            raise ValueError("Resume requires exp_name to locate saved parameters")
        # Saved parameters govern the runner and every attached service.
        params = CheckpointManager(LOGS_DIR / params.run.exp_name).update_parameters()
    runner = _runner_class(params.run.runner_class)(params=params, resume=resume)
    if runner.params.run.remote_api_enabled:
        init_redis_pool()
        runner.registry = await RedisAgentRegistry.create()
        publisher = RedisPublisher()
        await publisher.start()
        cm = RedisConnectionManager()
        await cm.start()
        runner.dashboard_manager = publisher
        runner.connection_manager = cm
        runner._use_redis = True
        if not resume:
            await publisher.reset()
        log.info("[Runner] Redis: %s", redis_url())
        log.info("[Runner] API server must be started separately")
    await runner.run_async()


def _print_presets(root: str | None) -> None:
    presets = list_presets(root)
    if not presets:
        print(f"No presets found under {root}.")
        return
    print(f"Available presets{f' under {root}' if root else ''}:")
    for name, desc, location in presets:
        print(f"  {name:24} {location}")
        if desc:
            print(f"      {desc}")


def _print_help() -> None:
    print(__doc__)
    print("Configurable parameters (override with `--field VALUE`, `--section.field VALUE`,")
    print("or `--no-flag` for booleans). Defaults shown are the model defaults:\n")
    current = None
    for section, name, type_str, default, desc in iter_config_fields():
        if section != current:
            print(f"\n[{section}]")
            current = section
        print(f"  --{name:<32} {type_str:<16} (default: {default!r})")
        if desc:
            print(f"      {desc}")
    print("\nList presets with:  terralingua --list  [PATH]")
    print("A run finds presets under the working directory, or under TL_PRESET_ROOT when it is set.")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--help" in argv or "-h" in argv:
        _print_help()
        return
    if "--list" in argv or any(a.startswith("--list=") for a in argv):
        root: str | None = None
        for i, a in enumerate(argv):
            if a == "--list":
                if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                    root = argv[i + 1]
                break
            if a.startswith("--list="):
                root = a.split("=", 1)[1]
                break
        _print_presets(root)
        return
    preset, overrides, resume = parse_argv(argv)
    params = compose(preset, overrides)
    asyncio.run(_run(params, resume))

