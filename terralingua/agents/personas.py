"""Personas for the first beings of a run, read from a JSON file."""

import json
from pathlib import Path


def load_personas(path: str | Path) -> list[dict]:
    """The entries of the file, expanded by count, in file order.

    An entry is the persona text, or an object with "persona", an optional
    "name" and an optional "count" (default 1). A name applies only to an entry
    with count 1, and no two entries may carry the same name, so every being
    keeps a distinct name. Any other key of an entry, such as a role, stays
    with each persona for the scenario to read.
    """
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: the personas file must hold a JSON list")
    personas = []
    names = set()
    for index, entry in enumerate(data):
        if isinstance(entry, str):
            entry = {"persona": entry}
        if not isinstance(entry, dict) or not isinstance(entry.get("persona"), str) or not entry["persona"].strip():
            raise ValueError(f"{path}: entry {index} needs a 'persona' text")
        count = entry.get("count", 1)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"{path}: entry {index} has an invalid 'count' {count!r}")
        name = entry.get("name")
        if name is not None and not isinstance(name, str):
            raise ValueError(f"{path}: entry {index} has a 'name' that is not text")
        persona = {key: value for key, value in entry.items() if key not in ("name", "count")}
        persona["persona"] = entry["persona"].strip()
        if count == 1 and name:
            if name in names:
                raise ValueError(f"{path}: the name {name!r} appears twice")
            names.add(name)
            persona["name"] = name
        personas.extend([dict(persona) for _ in range(count)])
    return personas
