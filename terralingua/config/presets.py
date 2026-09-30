"""Named run presets — the "config groups" picker.

Two sources, merged:

1. Built-in presets defined here (``grid_baseline``, ``core``) — runs that
   are not tied to a scenario folder.
2. File presets discovered from any ``preset.yaml`` anywhere under the repo
   (e.g. ``scenarios/<name>/preset.yaml``) — config lives beside its HOCON /
   servers.json / instructions. Path-valued keys in a file preset
   (``external_servers_config``, ``agent_network_hocon_path``, ...) are resolved
   relative to that ``preset.yaml``'s folder, so the file is self-contained.

A preset is a nested dict of deltas from the model defaults (same shape as
``ExperimentConfig``). ``compose(preset, overrides)`` layers it over the defaults
and under any CLI/programmatic overrides. A preset is pure config; scenarios that
need companion services (dashboard, MCP server) bring them up in their thin
``run.sh`` and then call ``terralingua <preset>``.
"""

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from terralingua.agents.prompt_templates import BUILTIN_INSTRUCTION_NAMES

# Directories skipped while discovering preset.yaml files (huge / irrelevant).
_EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".pytest_cache",
    "logs",
    ".vscode",
}

# Config keys whose values are filesystem paths; in a scenario preset.yaml they
# are written relative to the scenario folder and resolved to absolute here.
_PATH_FIELDS = {
    ("run", "external_servers_config"),
    ("env", "graph", "agent_network_hocon_path"),
    ("env", "roles_hocon_path"),
    ("agent", "scenario_specific_instructions"),
    ("env", "init_artifacts_path"),
    ("env", "affordances_file_path"),
    ("env", "graph", "file_path"),
}

# name -> (description, config-deltas)
_BUILTIN_PRESETS: dict[str, tuple[str, dict[str, Any]]] = {
    "grid_baseline": (
        "Plain grid world, LLM agents, no external services.",
        {
            "run": {"exp_name": "grid_baseline", "max_ts": 300},
            "env": {
                "world_type": "grid",
                "grid_size": 50,
                "init_agents": 10,
                "init_food": 100,
                "food_zones": 1,
            },
        },
    ),
    "core": (
        "Baseline grid run with an API model: 50x50 grid, 20 agents, 3000 steps, colors on.",
        {
            "run": {
                "exp_name": "core_run_1",
                "exp_description": "Core experiment baseline from the TerraLingua paper",
                "max_ts": 3000,
                "ckpt_interval": 100,
                "max_parallel_workers": 30,
                "save_video": True,
                "video_fps": 10,
            },
            "agent": {
                "model": "claude-haiku-4-5",
                "agents_name_prefix": "being",
                "scenario_specific_instructions": "base",
                "genome": "ocean_5",
                "max_history": 1,
                "internal_memory_size": 150,
                "use_inventory": True,
                "use_colors": True,
            },
            "env": {
                "grid_size": 50,
                "init_agents": 20,
                "init_human_agents": 0,
                "min_agents": 10,
                "init_agent_energy": 50,
                "init_food": 200,
                "food_zones": 2,
                "food_spawn_rate": 1,
                "food_decay_rate": 0.05,
                "food_mechanism": True,
                "static_food": False,
                "agent_lifespan": 100,
                "vision_radius": 5,
                "artifact_creation_cost": 0,
                "inert_artifacts": False,
                "reproduction_cost": 50,
            },
        },
    ),
}


def _resolve_paths(cfg: Any, base: Path, prefix: tuple[str, ...] = ()) -> Any:
    """Resolve path-valued keys in a scenario preset relative to its folder."""
    if isinstance(cfg, dict):
        out: dict = {}
        for key, value in cfg.items():
            field_path = (*prefix, key)
            if (
                field_path in _PATH_FIELDS
                and not (
                    key == "scenario_specific_instructions"
                    and value in BUILTIN_INSTRUCTION_NAMES
                )
                and isinstance(value, str)
                and not Path(value).is_absolute()
            ):
                out[key] = str(base / value)
            else:
                out[key] = _resolve_paths(value, base, field_path)
        return out
    return cfg


def _default_root() -> Path:
    """The working directory, or TL_PRESET_ROOT when it is set."""
    return Path(os.environ.get("TL_PRESET_ROOT") or Path.cwd())


def _preset_files(root: str | Path | None = None) -> list[Path]:
    """All preset files under `root` (default: the working directory, or TL_PRESET_ROOT).

    A preset file is named ``preset.yaml`` or ``<name>.preset.yaml`` — so one
    folder can hold several presets (e.g. a scenario family).
    Heavy/irrelevant directories are pruned for speed.
    """
    base = Path(root) if root is not None else _default_root()
    paths: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [
            d
            for d in dirnames
            if d not in _EXCLUDE_DIRS and not d.endswith(".egg-info")
        ]
        for fname in filenames:
            if fname == "preset.yaml" or fname.endswith(".preset.yaml"):
                paths.append(Path(dirpath) / fname)
    return sorted(paths)


def _preset_name(path: Path, data: dict) -> str:
    """Preset name: explicit ``name:``, else the file stem / folder name."""
    if data.get("name"):
        return data["name"]
    if path.name == "preset.yaml":
        return path.parent.name
    return path.name[: -len(".preset.yaml")]


class _PresetLoader(yaml.SafeLoader):
    """Reject duplicate explicit keys while preserving YAML merge behavior."""

    def construct_mapping(self, node, deep=False):
        explicit_keys = set()
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key = self.construct_object(key_node, deep=deep)
            try:
                if key in explicit_keys:
                    raise yaml.constructor.ConstructorError(
                        "while reading a preset",
                        node.start_mark,
                        f"duplicate key: {key!r}",
                        key_node.start_mark,
                    )
                explicit_keys.add(key)
            except TypeError as exc:
                raise yaml.constructor.ConstructorError(
                    "while reading a preset",
                    node.start_mark,
                    "preset keys must be scalar values",
                    key_node.start_mark,
                ) from exc
        return super().construct_mapping(node, deep=deep)


def _read_preset(path: Path) -> dict:
    """Read one preset without silently dropping invalid input."""
    try:
        data = yaml.load(path.read_text(), Loader=_PresetLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid preset YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Preset {path} must contain a dictionary.")
    unknown = set(data) - {"name", "description", "config"}
    if unknown:
        raise ValueError(f"Unknown preset keys in {path}: {sorted(unknown, key=str)}.")
    if "name" in data and (
        not isinstance(data["name"], str) or not data["name"].strip()
    ):
        raise ValueError(f"Preset name in {path} must be a nonempty string.")
    if "description" in data and not isinstance(data["description"], str):
        raise ValueError(f"Preset description in {path} must be a string.")
    if not isinstance(data.get("config"), dict):
        raise ValueError(f"Preset config in {path} must be a dictionary.")
    config = data["config"]
    unknown_sections = set(config) - {"agent", "env", "run"}
    if unknown_sections:
        raise ValueError(
            f"Unknown config sections in {path}: "
            f"{sorted(unknown_sections, key=str)}."
        )
    for section, values in config.items():
        if not isinstance(values, dict):
            raise ValueError(f"Preset config.{section} in {path} must be a dictionary.")
    graph = config.get("env", {}).get("graph", {})
    if not isinstance(graph, dict):
        raise ValueError(f"Preset config.env.graph in {path} must be a dictionary.")
    return data


def _discover_presets(
    root: str | Path | None = None,
) -> dict[str, tuple[str, dict, Path]]:
    """Discover presets and reject ambiguous names."""
    found: dict[str, tuple[str, dict, Path]] = {}
    for path in _preset_files(root):
        data = _read_preset(path)
        name = _preset_name(path, data)
        if name in _BUILTIN_PRESETS:
            raise ValueError(
                f"Preset '{name}' in {path} conflicts with a built-in preset."
            )
        if name in found:
            raise ValueError(
                f"Preset '{name}' is defined in both {found[name][2]} and {path}."
            )
        found[name] = (
            data.get("description", ""),
            _resolve_paths(data["config"], path.parent),
            path,
        )
    return found


def get_preset(name: str) -> dict[str, Any]:
    found = _discover_presets()
    if name in _BUILTIN_PRESETS:
        return deepcopy(_BUILTIN_PRESETS[name][1])
    if name in found:
        return deepcopy(found[name][1])
    available = ", ".join(sorted([*_BUILTIN_PRESETS, *found])) or "(none)"
    raise KeyError(f"Unknown preset '{name}'. Available: {available}")


def list_presets(root: str | Path | None = None) -> list[tuple[str, str, str]]:
    """[(name, description, location), ...] sorted by name — powers `terralingua --list`.

    With no root: built-in presets (location ``(built-in)``) plus every preset
    file discovered under the working directory. With a root: only presets found under it.
    """
    items: list[tuple[str, str, str]] = []
    if root is None:
        items.extend(
            (name, desc, "(built-in)") for name, (desc, _) in _BUILTIN_PRESETS.items()
        )
    base = Path(root) if root is not None else _default_root()
    for name, (desc, _cfg, path) in _discover_presets(root).items():
        try:
            location = str(path.relative_to(base))
        except ValueError:
            location = str(path)
        items.append((name, desc, location))
    return sorted(items)
