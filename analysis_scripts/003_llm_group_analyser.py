from pathlib import Path

from terralingua.anthropologist.analysts.group_analyst import GroupAnalyst
from terralingua.anthropologist.error_tracker import ErrorTracker
from terralingua.utils import LOGS_DIR, ROOT

"""
Annotates community/group dynamics using LLMs.
Logic lives in analysis_scripts/analysts/group_analyst.py.
"""

EXPERIMENTS_NAMES = []  # e.g. ['core_run', 'scarcity_run', ...]

# Setup
# ---------------------------
PERFORM_AUDIT = True
LLM_PROVIDER = "anthropic"
LLM_MODEL = "claude-sonnet-4-5-20250929"
FORCE_LONG_CONTEXT = False
# ---------------------------


def main(exp_path: Path | str, error_tracker: ErrorTracker):
    exp_path = Path(exp_path)

    save_path = exp_path / "community_annotations" / LLM_MODEL
    analyst = GroupAnalyst(
        model=LLM_MODEL,
        provider=LLM_PROVIDER,
        audit=PERFORM_AUDIT,
        parallel=True,
        force_long_context=FORCE_LONG_CONTEXT,
    )
    analyst.annotate_groups(
        exp_path=exp_path,
        save_path=save_path,
        error_tracker=error_tracker,
    )


if __name__ == "__main__":
    main_tracker = ErrorTracker(show_stacktraces=False)

    for exp in EXPERIMENTS_NAMES:
        print(f"Running analysis on experiment: {exp}")
        try:
            main(exp_path=LOGS_DIR / exp, error_tracker=main_tracker)
        except Exception as e:
            main_tracker.add_experiment_failure(exp, e)
        print()

    main_tracker.print_summary()
    main_tracker.save_to_file(ROOT / "analysis_summary" / "003_summary.json")
