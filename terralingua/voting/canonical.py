"""Deterministic parsing that makes differently spelled ballot args comparable."""

import json
import re
from typing import Any, Dict, List, Optional

CALL_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*\((.*)\)\s*$", re.DOTALL)


def canonical_value(value: Any) -> Any:
    """Numeric-normalized form of one param value ("03" == 3, " 1.5" == 1.5,
    "2.0" == 2.0 == 2)."""
    if isinstance(value, str):
        text = value.strip()
        try:
            return int(text)
        except ValueError:
            pass
        try:
            value = float(text)
        except ValueError:
            return text
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def split_top_level(text: str) -> List[str]:
    """Split on commas outside quotes/brackets."""
    parts, depth, quote, start = [], 0, "", 0
    for i, ch in enumerate(text):
        if quote:
            if ch == quote:
                quote = ""
            continue
        if ch in "'\"":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return [p.strip() for p in parts if p.strip()]


def parse_scalar(text: str) -> Any:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        return text[1:-1]
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    if text.startswith("[") and text.endswith("]"):
        return [parse_scalar(p) for p in split_top_level(text[1:-1])]
    return canonical_value(text)


def structural_parse(value: str) -> Optional[Dict[str, Any]]:
    """Deterministic parse of call-syntax and JSON-object strings; None when
    the string has no machine shape this parser knows."""
    text = (value or "").strip()
    if text.startswith("{"):
        try:
            obj = json.loads(text)
        except (ValueError, TypeError):
            return None
        return obj if isinstance(obj, dict) else None
    match = CALL_RE.match(text)
    if not match:
        return None
    name, body = match.groups()
    fields: Dict[str, Any] = {"action_name": name}
    for i, part in enumerate(split_top_level(body)):
        key, eq, raw = part.partition("=")
        key = key.strip()
        if eq and re.fullmatch(r"[A-Za-z_]\w*", key):
            value = parse_scalar(raw)
        else:
            key, value = f"arg{i}", parse_scalar(part)
        if key in fields:
            return None
        fields[key] = value
    return fields
