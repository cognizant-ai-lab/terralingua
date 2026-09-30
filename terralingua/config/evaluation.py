"""Evaluate proposed settings through the normal composition path."""

import json

from pydantic import ValidationError

from terralingua.config.compose import compose_input
from terralingua.config.inspection import FORMAT_VERSION, describe, inspect_config
from terralingua.config.models import ExperimentConfig


def evaluate(preset=None, overrides=None, *, config=None) -> dict:
    """Return diagnostics instead of starting a simulation.

    Inputs must be JSON-compatible values. Raw layers remain available in input.
    Requested holds merged layers before defaults and model normalization.
    """
    submitted = None
    requested = None
    try:
        submitted = json.loads(json.dumps({
            "preset": preset, "config": config, "overrides": overrides,
        }, allow_nan=False))
        requested = json.loads(json.dumps(compose_input(preset, overrides, config), allow_nan=False))
        validated = ExperimentConfig.model_validate(requested)
        result = inspect_config(validated, requested)
    except (ValidationError, ValueError, TypeError, KeyError) as exc:
        if isinstance(exc, ValidationError):
            diagnostics = [{
                "severity": "error",
                "code": error["type"],
                "field": ".".join(str(part) for part in error["loc"]),
                "message": error["msg"],
            } for error in exc.errors(include_url=False, include_input=False)]
        else:
            diagnostics = [{
                "severity": "error",
                "code": "configuration_input",
                "field": None,
                "message": str(exc),
            }]
        result = {
            "format_version": FORMAT_VERSION,
            "schema_hash": describe()["schema_hash"],
            "valid": False,
            "requested": requested,
            "resolved": None,
            "fields": {},
            "active_values": {},
            "inactive_values": {},
            "normalizations": [],
            "derived": {},
            "diagnostics": diagnostics,
        }
    result["input"] = submitted
    return result
