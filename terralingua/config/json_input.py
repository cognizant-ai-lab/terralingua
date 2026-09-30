"""Read JSON without discarding duplicate keys or accepting nonfinite numbers."""

import json


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}.")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"JSON numbers must be finite: {value}.")


def read_json(text: str):
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
