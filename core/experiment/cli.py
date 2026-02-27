# cli.py
import argparse
from dataclasses import fields
from typing import Any, Union, get_args, get_origin

from core.experiment.config import AgentConfig, EnvConfig, RunConfig


def _coerce_food_zones(values):
    """Accepts:
    --food_zones 3
    --food_zones 10,10 12,5 20,0
    --food_zones None
    """
    if values is None:
        return None

    # case: explicit None string
    if len(values) == 1 and values[0].lower() in ("none", "null"):
        return None

    # case: single integer
    if len(values) == 1 and values[0].isdigit():
        return int(values[0])

    # case: list of coordinate pairs
    coords = []
    for item in values:
        if "," not in item:
            raise ValueError(f"Invalid food zone format: '{item}' (expected 'x,y')")
        x, y = item.split(",", 1)
        coords.append((int(x), int(y)))
    return coords


AUTO_COERCERS = {
    "food_zones": _coerce_food_zones,
}


def _add_dataclass_group(parser: argparse.ArgumentParser, group_name: str, cls: Any):
    group = parser.add_argument_group(group_name)
    for f in fields(cls):
        meta = f.metadata or {}
        if meta.get("excluded", False):
            continue

        arg_name = f"--{f.name}"

        kwargs: dict[str, Any] = {
            "help": meta.get("help"),
            "default": argparse.SUPPRESS,
        }

        if "choices" in meta:
            kwargs["choices"] = meta["choices"]

        # booleans
        if f.type is bool:
            kwargs["action"] = argparse.BooleanOptionalAction

        # auto-coerced types (e.g., food_zones)
        elif meta.get("autocoerce"):
            kwargs["type"] = str
            kwargs["nargs"] = "+"

        # explicit type override
        elif "arg_type" in meta:
            kwargs["type"] = meta["arg_type"]
            if "nargs" in meta:
                kwargs["nargs"] = meta["nargs"]

        # SIMPLE fallback for everything else
        else:
            if f.type in (int, float, str):
                kwargs["type"] = f.type
            else:
                # Optional[str], tuple, list, etc → parse as string
                kwargs["type"] = str

        group.add_argument(arg_name, **kwargs)

    return group


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulation parameters for multi-agent environment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    _add_dataclass_group(parser, "Agent Parameters", AgentConfig)
    _add_dataclass_group(parser, "Environment Parameters", EnvConfig)
    _add_dataclass_group(parser, "Run Parameters", RunConfig)

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint",
    )

    args = parser.parse_args()

    # Auto-coercion pass
    args_dict = vars(args)
    for name, value in args_dict.items():
        if value is argparse.SUPPRESS:
            continue
        if name in AUTO_COERCERS:
            args_dict[name] = AUTO_COERCERS[name](value)

    return args
