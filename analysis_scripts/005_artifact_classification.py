from pathlib import Path

from terralingua.anthropologist.analysts.artifact_analyst import ArtifactAnalyst
from terralingua.anthropologist.error_tracker import ErrorTracker
from terralingua.utils import LOGS_DIR, ROOT

"""
Classifies artifacts into categories 1-4 (or -1) using an LLM.
Logic lives in analysis_scripts/analysts/artifact_analyst.py.
"""

EXPERIMENTS_NAMES = []  # e.g. ['core_run', 'scarcity_run', ...]

# Setup
# ---------------------------
LLM_PROVIDER = "anthropic"
LLM_MODEL = "claude-haiku-4-5"
PARALLEL_WORKERS = 8
PARALLEL = True
# ---------------------------


def main(
    exp_path: Path,
    error_tracker: ErrorTracker,
):
    analyst = ArtifactAnalyst(model=LLM_MODEL, provider=LLM_PROVIDER, parallel=PARALLEL)
    analyst.classify_artifacts(
        exp_path=exp_path,
        parallel_workers=PARALLEL_WORKERS,
        error_tracker=error_tracker,
    )


if __name__ == "__main__":
    main_tracker = ErrorTracker(show_stacktraces=False)

    for exp in EXPERIMENTS_NAMES:
        print(f"Running analysis on experiment: {exp}")
        try:
            main(
                exp_path=LOGS_DIR / exp,
                error_tracker=main_tracker,
            )
        except Exception as e:
            main_tracker.add_experiment_failure(exp, e)
        print()

    main_tracker.print_summary()
    main_tracker.save_to_file(ROOT / "analysis_summary" / "005_summary.json")
