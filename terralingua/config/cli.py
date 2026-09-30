"""Parse command-line values using the experiment field definitions."""

from types import UnionType
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel

from terralingua.config.compose import _resolve_path
from terralingua.config.json_input import read_json
from terralingua.config.models import ExperimentConfig


def _annotation(key: str):
    """Resolve aliases and dictionary paths without loading a scenario runner."""
    try:
        path = _resolve_path(key)
    except (KeyError, TypeError) as error:
        raise SystemExit(str(error)) from error
    annotation = ExperimentConfig
    for part in path:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            annotation = annotation.model_fields[part].annotation
        elif get_origin(annotation) is dict or annotation is dict:
            args = get_args(annotation)
            annotation = args[1] if args else Any
        else:
            return Any
    return annotation


def _variants(annotation) -> tuple:
    return get_args(annotation) if get_origin(annotation) in (Union, UnionType) else (annotation,)


def _value(key: str, raw: str):
    variants = _variants(_annotation(key))
    if raw == "null" and type(None) in variants:
        return None
    structured = any(get_origin(item) in (list, tuple, dict) or item in (list, tuple, dict)
                     for item in variants)
    if structured:
        try:
            return read_json(raw)
        except ValueError as error:
            raise SystemExit(f"--{key} requires valid JSON. {error}") from error
    return raw


def _flag(key: str, value: bool) -> bool:
    if bool not in _variants(_annotation(key)):
        raise SystemExit(f"--{key} requires a value; bare flags are only for booleans.")
    return value


def _set_override(overrides: dict, key: str, value) -> None:
    if key in overrides:
        raise SystemExit(f"Duplicate override --{key}. Supply each setting once.")
    overrides[key] = value


def parse_argv(argv: list[str]) -> tuple[str | None, dict, bool]:
    """Parse preset, overrides, and resume without starting application services.

    Collections use JSON. Optional fields accept ``null``. Only boolean fields
    accept bare ``--flag`` or ``--no-flag`` forms.
    """
    tokens = list(argv)
    preset: str | None = None
    if tokens and not tokens[0].startswith("-"):
        preset = tokens.pop(0)

    overrides: dict[str, object] = {}
    resume = False
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("--"):
            raise SystemExit(f"Unexpected argument: {token!r}")
        body = token[2:]
        if body == "resume":
            resume = True
            i += 1
        elif body == "demo":
            if preset is not None and preset != "demo":
                raise SystemExit("Choose either a named preset or --demo.")
            preset = "demo"
            i += 1
        elif "=" in body:
            key, raw = body.split("=", 1)
            _set_override(overrides, key, _value(key, raw))
            i += 1
        elif body.startswith("no-"):
            key = body[3:]
            _set_override(overrides, key, _flag(key, False))
            i += 1
        elif i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            _set_override(overrides, body, _value(body, tokens[i + 1]))
            i += 2
        else:
            _set_override(overrides, body, _flag(body, True))
            i += 1
    return preset, overrides, resume
