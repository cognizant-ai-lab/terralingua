from pathlib import Path

from terralingua.anthropologist import Anthropologist
from terralingua.anthropologist.error_tracker import ErrorTracker
from terralingua.utils import LOGS_DIR, ROOT

"""
Runs the full anthropologist pipeline via the Anthropologist orchestrator.
Calls all steps in order: agent annotation, graph, group annotation,
artifact novelty + classification, artifact phylogeny.
Logic lives in anthropologist/.
"""

EXPERIMENTS_NAMES = []  # e.g. ['core_run', 'scarcity_run', ...]

# Setup
# ---------------------------
AUDIT = True
SHOW_STACKTRACES = False
LLM_PROVIDER = "anthropic"
LLM_MODEL = "claude-sonnet-4-5-20250929"
FORCE_LONG_CONTEXT = False

STEPS = [1, 2, 3, 4, 5]  # subset of steps to run, or None for all
# ---------------------------


def main(exp_path: Path | str, error_tracker: ErrorTracker):
    anthropologist = Anthropologist(
        model=LLM_MODEL,
        provider=LLM_PROVIDER,
        audit=AUDIT,
        parallel=True,
        force_long_context=FORCE_LONG_CONTEXT,
    )
    anthropologist.analyze(
        exp_path=exp_path,
        steps=STEPS,
        save=True,
        error_tracker=error_tracker,
    )


if __name__ == "__main__":
    main_tracker = ErrorTracker(show_stacktraces=SHOW_STACKTRACES)

    for exp in EXPERIMENTS_NAMES:
        print(f"Running anthropologist on experiment: {exp}")
        try:
            main(exp_path=LOGS_DIR / exp, error_tracker=main_tracker)
        except Exception as e:
            main_tracker.add_experiment_failure(exp, e)
        print()

    main_tracker.print_summary()
    main_tracker.save_to_file(ROOT / "analysis_summary" / "007_summary.json")
