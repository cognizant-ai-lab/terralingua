from pathlib import Path

from terralingua.anthropologist.analysts.artifact_analyst import ArtifactAnalyst
from terralingua.anthropologist.error_tracker import ErrorTracker
from terralingua.utils import LOGS_DIR, ROOT

"""
Traces artifact ancestry via hand-detection and LLM inference.
Logic lives in analysis_scripts/analysts/artifact_analyst.py.
"""

EXPERIMENTS_NAMES = []  # e.g. ['core_run', 'scarcity_run', ...]

# Setup
# ---------------------------
SHOW_STACKTRACES = False
BYNARY = True  # Whether to use the binary ancestor detection or the finer one
LLM_PROVIDER = "anthropic"
LLM_MODEL = "claude-haiku-4-5"

PARALLEL = True
MAX_PARALLEL_WORKERS = 8
HAND_PHYLOGENY = True
LLM_PHYLOGENY = True
# ---------------------------


def main(
    exp_path: Path,
    error_tracker: ErrorTracker,
):

    analyst = ArtifactAnalyst(model=LLM_MODEL, provider=LLM_PROVIDER, parallel=PARALLEL)
    analyst.trace_phylogeny(
        exp_path=exp_path,
        hand_phylogeny=HAND_PHYLOGENY,
        llm_phylogeny=LLM_PHYLOGENY,
        binary=BYNARY,
        parallel=PARALLEL,
        max_workers=MAX_PARALLEL_WORKERS,
        error_tracker=error_tracker,
    )


if __name__ == "__main__":
    main_tracker = ErrorTracker(show_stacktraces=SHOW_STACKTRACES)

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
    main_tracker.save_to_file(ROOT / "analysis_summary" / "006_summary.json")
