"""Describe configuration fields and inspect their effective behavior."""

import copy
import hashlib
import json
from functools import lru_cache

from jsonschema import Draft202012Validator
from pydantic import TypeAdapter

from terralingua.config.dependencies import (
    APPLICABILITY,
    CONSTRAINTS,
    DERIVED_DEPENDENCIES,
    GROUPS,
    NOTES,
    dependencies,
    is_active,
)
from terralingua.config.models import (
    AgentConfig,
    EnvConfig,
    ExperimentConfig,
    GraphConfig,
    RunConfig,
    _field_default,
    _type_str,
)
from terralingua.experiment.scenario_loader import (
    OPTIONS_PREFIX,
    ScenarioImportError,
    flatten_options,
    scenario_applicability,
)

FORMAT_VERSION = 1
MODELS = {
    "agent": AgentConfig,
    "env": EnvConfig,
    "env.graph": GraphConfig,
    "run": RunConfig,
}


def field_definitions():
    """Yield canonical paths and the existing model fields."""
    for section, model in MODELS.items():
        for name, field in model.model_fields.items():
            if section == "env" and name == "graph":
                continue
            yield f"{section}.{name}", section, model, name, field


def flatten_values(config: dict) -> dict:
    """Keep dictionary-valued parameters intact."""
    values = {}
    for path, section, _model, name, _field in field_definitions():
        container = config
        for part in section.split("."):
            container = container.get(part, {}) if isinstance(container, dict) else {}
        if isinstance(container, dict) and name in container:
            values[path] = container[name]
    return values


def _json_value(value):
    return json.loads(json.dumps(value, allow_nan=False))


@lru_cache(maxsize=1)
def _description() -> dict:
    definitions = list(field_definitions())
    paths = {path for path, *_ in definitions}
    groups = {}
    for group, members in GROUPS.items():
        for path in members:
            if path in groups:
                raise ValueError(f"Duplicate configuration group for {path}.")
            groups[path] = group
    if set(groups) != paths:
        missing = sorted(paths - groups.keys())
        extra = sorted(groups.keys() - paths)
        raise ValueError(f"Configuration metadata mismatch. Missing: {missing}; extra: {extra}.")
    if not set(APPLICABILITY) <= paths or not set(NOTES) <= paths:
        raise ValueError("Configuration metadata names an unknown field.")
    schema = ExperimentConfig.model_json_schema()
    fields = {}
    for path, section, model, name, field in definitions:
        condition, reason = APPLICABILITY.get(path, (True, ""))
        Draft202012Validator.check_schema(condition)
        refs = dependencies(condition)
        if not set(refs) <= paths:
            raise ValueError(f"Unknown applicability dependency for {path}.")
        aliases = [name]
        if section == "env.graph":
            aliases.append(f"graph.{name}")
        fields[path] = {
            "path": path,
            "aliases": aliases,
            "group": groups[path],
            "type": _type_str(field.annotation),
            "schema": copy.deepcopy(schema["$defs"][model.__name__]["properties"][name]),
            "default": _json_value(_field_default(field)),
            "description": field.description or "",
            "active_when": condition,
            "inactive_reason": reason,
            "depends_on": refs,
            "affects": [],
            "derived_effects": [],
            "constraints": [],
            "notes": NOTES.get(path, ""),
        }
    fields["env.food_mechanism"]["default_when"] = {
        "condition": {"properties": {"env.world_type": {"const": "social_graph"}}, "required": ["env.world_type"]},
        "value": False,
    }
    for path, field in fields.items():
        for dependency in field["depends_on"]:
            fields[dependency]["affects"].append(path)
    for derived, refs in DERIVED_DEPENDENCIES.items():
        for dependency in refs:
            if dependency not in fields:
                raise ValueError(f"Unknown dependency for derived value {derived}.")
            fields[dependency]["derived_effects"].append(derived)
    for constraint in CONSTRAINTS:
        for dependency in constraint["fields"]:
            if dependency not in fields:
                raise ValueError(f"Unknown constraint dependency: {dependency}.")
            fields[dependency]["constraints"].append(constraint["id"])
    result = {
        "format_version": FORMAT_VERSION,
        "condition_format": "json-schema-2020-12",
        "condition_context": "canonical_flat_values",
        "validation": "Use evaluate for authoritative model and cross-field validation.",
        "json_schema": schema,
        "groups": [{"name": group, "fields": members} for group, members in GROUPS.items()],
        "fields": fields,
        "constraints": CONSTRAINTS,
        "scope": "Standard run configuration. Runtime permissions, injected events, and custom runners add conditions.",
        "derived_dependencies": DERIVED_DEPENDENCIES,
    }
    content = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
    result["schema_hash"] = hashlib.sha256(content.encode()).hexdigest()
    return result


def describe() -> dict:
    """Return a versioned contract without importing a runner or opening services."""
    return copy.deepcopy(_description())


def _normalized_requested(requested: dict, defaults: dict) -> dict:
    """Coerce individual values without applying cross-field normalization."""
    values = {**flatten_values(defaults), **flatten_values(requested)}
    for path, _section, _model, _name, field in field_definitions():
        try:
            values[path] = _json_value(TypeAdapter(field.annotation).validate_python(values[path]))
        except (ValueError, TypeError):
            # A model validator may intentionally accept a saved representation.
            pass
    return values


def inspect_config(config: ExperimentConfig, requested: dict | None = None) -> dict:
    """Explain a validated configuration without changing its saved values."""
    requested = copy.deepcopy(requested or {})
    description = _description()
    resolved = config.to_json()
    values = flatten_values(resolved)
    defaults = ExperimentConfig().to_json()
    default_values = flatten_values(defaults)
    before = _normalized_requested(requested, defaults)
    explicit = flatten_values(requested)
    states = {}
    active_values = {}
    inactive_values = {}
    diagnostics = []
    normalizations = []
    for path, field in description["fields"].items():
        active = is_active(field["active_when"], values)
        reason = "" if active else field["inactive_reason"]
        states[path] = {"active": active, "reason": reason}
        (active_values if active else inactive_values)[path] = values[path]
        if not active and path in explicit and values[path] != default_values[path]:
            diagnostics.append({
                "severity": "warning",
                "code": "inactive_setting",
                "field": path,
                "message": f"{path} is inactive. {reason}",
            })
        if before[path] != values[path]:
            change = {"field": path, "source": "explicit" if path in explicit else "default", "requested": before[path], "effective": values[path]}
            normalizations.append(change)
            diagnostics.append({
                "severity": "warning",
                "code": "normalized_setting",
                "field": path,
                "message": f"{path} resolves from {before[path]!r} to {values[path]!r}.",
            })
    cost = config.env.reproduction_cost
    energy_death = config.env.energy_death
    newborn = None if cost < 0 else (config.env.init_agent_energy if cost == 0 else cost)
    if newborn is not None and newborn < 0:
        newborn = "unlimited"
    derived = {
        "artifact_creation_enabled": config.env.artifact_creation_cost >= 0,
        "internal_memory_enabled": config.agent.internal_memory_size > 0,
        "energy_death": config.env.food_mechanism if energy_death is None else energy_death,
        "reproduction_enabled": cost >= 0,
        "newborn_base_energy": newborn,
        "failed_birth_cost": max(cost, 0),
        "hop_radius": 1 if config.env.world_type == "social_graph" else (
            config.env.graph.hop_radius if config.env.world_type == "graph" else None
        ),
    }
    deferred = bool(config.run.scenario_options)
    if config.run.scenario:
        try:
            applicability, option_values, option_defaults = scenario_applicability(
                config.run.scenario, config.run.scenario_options
            )
        except ScenarioImportError as exc:
            applicability, option_values, option_defaults = {}, {}, {}
            diagnostics.append({
                "severity": "info",
                "code": "scenario_not_imported",
                "field": "run.scenario",
                "message": f"The scenario module is not importable from here: {exc}",
            })
        else:
            deferred = False  # the options were validated here
        if applicability:
            explicit_options = {
                OPTIONS_PREFIX + path
                for path, _value in flatten_options(
                    requested.get("run", {}).get("scenario_options") or {}
                )
            }
            for path, (condition, reason) in applicability.items():
                active = is_active(condition, option_values)
                states[path] = {"active": active, "reason": "" if active else reason}
                (active_values if active else inactive_values)[path] = option_values.get(path)
                if (
                    not active
                    and path in explicit_options
                    and option_values.get(path) != option_defaults.get(path)
                ):
                    diagnostics.append({
                        "severity": "warning",
                        "code": "inactive_setting",
                        "field": path,
                        "message": f"{path} is inactive. {reason}",
                    })
    if deferred:
        diagnostics.append({
            "severity": "info",
            "code": "scenario_validation_deferred",
            "field": "run.scenario_options",
            "message": "The selected scenario runner validates these settings at startup.",
        })
    if config.run.external_servers_config:
        diagnostics.append({
            "severity": "info",
            "code": "external_validation_deferred",
            "field": "run.external_servers_config",
            "message": "Server contents and connectivity are checked when external services start.",
        })
    return {
        "format_version": FORMAT_VERSION,
        "schema_hash": description["schema_hash"],
        "valid": True,
        "requested": _json_value(requested),
        "resolved": resolved,
        "fields": states,
        "active_values": active_values,
        "inactive_values": inactive_values,
        "normalizations": normalizations,
        "derived": derived,
        "diagnostics": diagnostics,
    }
