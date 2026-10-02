"""Inspect configuration without launching agents or services.

python -m terralingua.config describe
python -m terralingua.config describe --preset example
python -m terralingua.config describe --scenario scenarios.example
python -m terralingua.config evaluate --preset example
python -m terralingua.config evaluate --config config.json --overrides '{"reproduction_cost": 10}'
Use "-" as a file path to read JSON from standard input.
"""

import argparse
import json
import sys
from pathlib import Path

from terralingua.config.compose import compose
from terralingua.config.evaluation import evaluate
from terralingua.config.inspection import describe, describe_scenario
from terralingua.config.json_input import read_json


def _read_json(value: str, *, file: bool = False):
    if value == "-":
        return read_json(sys.stdin.read())
    if file:
        return read_json(Path(value).read_text())
    return read_json(value)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    describe_parser = subparsers.add_parser("describe", help="Export fields, defaults, and dependencies.")
    scenario_source = describe_parser.add_mutually_exclusive_group()
    scenario_source.add_argument("--preset", help="Also describe the options of this preset's scenario.")
    scenario_source.add_argument("--scenario", help="Also describe the options of this scenario module.")
    evaluate_parser = subparsers.add_parser("evaluate", help="Validate and explain proposed settings.")
    evaluate_parser.add_argument("--preset")
    evaluate_parser.add_argument("--config", help="Nested configuration JSON file, or - for stdin.")
    evaluate_parser.add_argument("--overrides", help="Flat JSON overrides, or - for stdin.")
    args = parser.parse_args(argv)
    try:
        if args.command == "describe":
            result = describe()
            if args.preset:
                config = compose(args.preset)
                result["scenario"] = (
                    describe_scenario(config.run.scenario) if config.run.scenario else None
                )
            elif args.scenario:
                result["scenario"] = describe_scenario(args.scenario)
        else:
            if args.config == "-" and args.overrides == "-":
                parser.error("Only one input can read standard input.")
            result = evaluate(
                args.preset,
                _read_json(args.overrides) if args.overrides else None,
                config=_read_json(args.config, file=True) if args.config else None,
            )
    except (OSError, ValueError, TypeError, KeyError, ImportError, RecursionError) as exc:
        result = {"valid": False, "diagnostics": [{
            "severity": "error", "code": "input", "field": None, "message": str(exc),
        }]}
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result.get("valid", True) else 2


if __name__ == "__main__":
    sys.exit(main())
