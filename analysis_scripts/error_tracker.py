import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional


class ErrorTracker:
    """
    Centralized error tracking for analysis scripts.

    Stores errors with context (agent/community/experiment names) and provides
    methods to print summaries and save to JSON.
    """

    def __init__(self, show_stacktraces: bool = False):
        self.show_stacktraces = show_stacktraces
        self.errors: Dict[str, Any] = {}
        self.failed_experiments: List[Dict[str, Any]] = []

    def add_error(
        self,
        context: str,
        error: Exception,
        error_type: str = "error",
        additional_info: Optional[Dict[str, Any]] = None,
    ):
        """
        Add an error to the tracker.

        Args:
            context: Identifier for what failed (e.g., agent name, community idx, experiment name)
            error: The exception that occurred
            error_type: Type of error (currently only 'error' is used)
            additional_info: Any additional context to store (e.g., experiment name, community members)
        """
        error_data = {
            "error": str(error),
            "traceback": traceback.format_exc(),
            "error_type": error_type,
        }

        if additional_info:
            error_data.update(additional_info)

        self.errors[context] = error_data

    def add_experiment_failure(self, experiment: str, error: Exception):
        """Add a failed experiment with full error details."""
        self.failed_experiments.append(
            {
                "exp": experiment,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )

    def has_errors(self) -> bool:
        """Check if any errors were recorded."""
        return bool(self.errors or self.failed_experiments)

    def print_summary(self):
        """Print a formatted summary of all errors and issues."""
        if not self.has_errors():
            print("✓ No errors occurred.")
            return

        print("\n" + "=" * 80)
        print("ERROR SUMMARY")
        print("=" * 80)

        # Print failed experiments
        if self.failed_experiments:
            print(f"\n{len(self.failed_experiments)} EXPERIMENT(S) FAILED:")
            print("-" * 80)
            for failure in self.failed_experiments:
                print(f"\nExperiment: {failure['exp']}")
                print(f"Error: {failure['error']}")
                if self.show_stacktraces:
                    print(f"\nStack trace:\n{failure['traceback']}")
                print("-" * 80)

        # Print errors grouped by experiment
        if self.errors:
            # Group errors by experiment
            errors_by_exp: Dict[str, List[tuple]] = {}
            standalone_errors: List[tuple] = []

            for context, error_data in self.errors.items():
                exp = error_data.get("experiment")
                if exp:
                    if exp not in errors_by_exp:
                        errors_by_exp[exp] = []
                    errors_by_exp[exp].append((context, error_data))
                else:
                    standalone_errors.append((context, error_data))

            # Print errors by experiment
            if errors_by_exp:
                print(f"\n{len(errors_by_exp)} EXPERIMENT(S) WITH PARTIAL FAILURES:")
                print("-" * 80)
                for exp, exp_errors in errors_by_exp.items():
                    print(f"\nExperiment: {exp}")
                    print(f"Number of items with errors: {len(exp_errors)}")
                    for context, error_data in exp_errors:
                        print(f"\n  - {context}:")
                        print(f"    Error: {error_data['error']}")
                        if self.show_stacktraces:
                            print(f"    Stack trace:\n{error_data['traceback']}")
                    print("-" * 80)

            # Print standalone errors
            if standalone_errors:
                print(f"\n{len(standalone_errors)} STANDALONE ERROR(S):")
                print("-" * 80)
                for context, error_data in standalone_errors:
                    print(f"\nContext: {context}")
                    print(f"Error: {error_data['error']}")
                    if self.show_stacktraces:
                        print(f"\nStack trace:\n{error_data['traceback']}")
                    print("-" * 80)

    def save_to_file(self, filepath: Path):
        """Save error summary to JSON file."""
        # Group errors by experiment for the JSON output
        errors_by_exp = {}
        standalone_errors = {}

        for context, error_data in self.errors.items():
            exp = error_data.get("experiment")
            if exp:
                if exp not in errors_by_exp:
                    errors_by_exp[exp] = {}
                errors_by_exp[exp][context] = error_data
            else:
                standalone_errors[context] = error_data

        experiments_with_issues = [
            {"exp": exp, "errors": errors} for exp, errors in errors_by_exp.items()
        ]

        summary = {
            "failed_experiments": self.failed_experiments,
            "experiments_with_issues": experiments_with_issues,
            "standalone_errors": standalone_errors,
        }

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(summary, f, indent=4)

        print(f"\n✓ Error summary saved to: {filepath}")
