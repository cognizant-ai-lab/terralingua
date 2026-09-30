"""Compose configuration layers and reject ambiguous input.

Layers apply in this order: preset, nested config, then flat overrides.
Pydantic applies defaults and validates values after these layers are merged.
"""

import logging
from copy import deepcopy
from typing import Any, get_origin

from pydantic import BaseModel

from terralingua.config.inspection import inspect_config
from terralingua.config.models import (
    AgentConfig,
    EnvConfig,
    ExperimentConfig,
    GraphConfig,
    RunConfig,
)
from terralingua.config.presets import get_preset

logger = logging.getLogger(__name__)

_SECTION_MODELS = {"agent": AgentConfig, "env": EnvConfig, "run": RunConfig}


def deep_merge(*dicts: dict) -> dict:
    """Merge nested dictionaries. Later layers take precedence."""
    out: dict = {}
    for data in dicts:
        for key, value in (data or {}).items():
            if isinstance(value, dict) and isinstance(out.get(key), dict):
                out[key] = deep_merge(out[key], value)
            else:
                out[key] = deepcopy(value)
    return out


def _field_path(name: str) -> tuple[str, ...]:
    """Resolve a bare field name."""
    for section, model in _SECTION_MODELS.items():
        if name in model.model_fields:
            return (section, name)
    if name in GraphConfig.model_fields:
        return ("env", "graph", name)
    raise KeyError(
        f"Unknown config field '{name}'. Run 'terralingua --help' for valid fields."
    )


def _validate_path(path: tuple[str, ...]) -> None:
    """Check model fields while allowing keys inside dictionary settings."""
    model = ExperimentConfig
    for index, part in enumerate(path):
        if not part:
            raise KeyError(f"Empty field in config path '{'.'.join(path)}'.")
        field = model.model_fields.get(part)
        if field is None:
            raise KeyError(f"Unknown config path '{'.'.join(path)}'.")
        if index == len(path) - 1:
            return
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            model = annotation
        elif annotation is dict or get_origin(annotation) is dict:
            if any(not key for key in path[index + 1 :]):
                raise KeyError(f"Empty field in config path '{'.'.join(path)}'.")
            return
        else:
            raise KeyError(f"Config field '{'.'.join(path[: index + 1])}' has no fields.")


def _resolve_path(key: str) -> tuple[str, ...]:
    if not isinstance(key, str):
        raise TypeError("Config field names must be strings.")
    parts = key.split(".")
    if len(parts) == 1:
        path = _field_path(parts[0])
    elif parts[0] in _SECTION_MODELS:
        path = tuple(parts)
    elif parts[0] == "graph":
        path = ("env", "graph", *parts[1:])
    else:
        raise KeyError(f"Unknown config path '{key}'.")
    _validate_path(path)
    return path


def _validate_keys(data: dict, model=ExperimentConfig, prefix: tuple[str, ...] = ()) -> None:
    """Reject unknown model keys before model validation can discard them."""
    if not isinstance(data, dict):
        raise TypeError(f"Config '{'.'.join(prefix) or 'root'}' must be a dictionary.")
    for key, value in data.items():
        if not isinstance(key, str):
            raise TypeError("Config field names must be strings.")
        field = model.model_fields.get(key)
        if field is None:
            raise KeyError(f"Unknown config field '{'.'.join((*prefix, key))}'.")
        annotation = field.annotation
        if (
            isinstance(value, dict)
            and isinstance(annotation, type)
            and issubclass(annotation, BaseModel)
        ):
            _validate_keys(value, annotation, (*prefix, key))


def route_overrides(flat: dict[str, Any]) -> dict:
    """Resolve aliases without silently overwriting another explicit setting."""
    if not isinstance(flat, dict):
        raise TypeError("Config overrides must be a dictionary.")
    nested: dict = {}
    supplied: dict[tuple[str, ...], str] = {}
    for key, value in flat.items():
        path = _resolve_path(key)
        for earlier_path, earlier_key in supplied.items():
            common = min(len(path), len(earlier_path))
            if path[:common] == earlier_path[:common]:
                raise ValueError(
                    f"Config overrides '{earlier_key}' and '{key}' overlap. "
                    "Supply each setting once."
                )
        supplied[path] = key
        cursor = nested
        for part in path[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[path[-1]] = deepcopy(value)
    _validate_keys(nested)
    return nested


def compose_input(
    preset: str | None = None,
    overrides: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict:
    """Return requested layers before defaults, coercion, or normalization."""
    layers: list[dict] = []
    if preset:
        preset_data = get_preset(preset)
        _validate_keys(preset_data)
        layers.append(preset_data)
    if config is not None:
        _validate_keys(config)
        layers.append(config)
    if overrides is not None:
        layers.append(route_overrides(overrides))
    return deep_merge(*layers)


def compose(
    preset: str | None = None,
    overrides: dict[str, Any] | None = None,
    *,
    config: dict[str, Any] | None = None,
) -> ExperimentConfig:
    """Validate the preset, nested config, and flat override layers."""
    requested = compose_input(preset, overrides, config)
    result = ExperimentConfig.model_validate(requested)
    for diagnostic in inspect_config(result, requested)["diagnostics"]:
        if diagnostic["severity"] == "warning":
            logger.warning(diagnostic["message"])
    return result
