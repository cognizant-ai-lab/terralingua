from pathlib import Path

from terralingua.anthropologist.analysts.artifact_analyst import ArtifactAnalyst
from terralingua.anthropologist.error_tracker import ErrorTracker
from terralingua.utils import LOGS_DIR, ROOT

"""
Computes artifact novelty scores and complexity metrics.
Logic lives in analysis_scripts/analysts/artifact_analyst.py.
"""

EXPERIMENTS_NAMES = []  # e.g. ['core_run', 'scarcity_run', ...]

# Setup
# ---------------------------
NOVELTY = True
METRICS = True
EXPANSION = False
EMBED = True
NOVELTY_SAMPLES = 5

PARALLEL = True
LLM_PROVIDER = "anthropic"
LLM_MODEL = "claude-sonnet-4-5-20250929"

EXPANSION_METRIC = "cosine"  # 'cosine' or 'euclidean'
# ---------------------------


def main(
    exp_path: Path,
    metrics: list,
    error_tracker: ErrorTracker,
):

    analyst = ArtifactAnalyst(model=LLM_MODEL, provider=LLM_PROVIDER, parallel=PARALLEL)
    analyst.analyze_novelty(
        exp_path=exp_path,
        metrics=metrics,
        embed=EMBED,
        expansion=EXPANSION,
        novelty_samples=NOVELTY_SAMPLES,
        error_tracker=error_tracker,
    )


if __name__ == "__main__":
    if METRICS:
        import sys

        sys.path.insert(0, str(ROOT / "analysis_scripts"))
        from terralingua.anthropologist.artifact_complexity import (
            CompressedSize,
            InverseCompressionRate,
            LexicalSophistication,
            LMSurprisal,
            SyntacticDepth,
        )

        metrics = [
            LMSurprisal(),
            CompressedSize(),
            InverseCompressionRate(),
            SyntacticDepth(),
            LexicalSophistication(),
        ]
    else:
        metrics = []

    main_tracker = ErrorTracker(show_stacktraces=False)

    for exp in EXPERIMENTS_NAMES:
        print(f"Running analysis on experiment: {exp}")
        try:
            main(
                exp_path=LOGS_DIR / exp,
                metrics=metrics,
                error_tracker=main_tracker,
            )
        except Exception as e:
            main_tracker.add_experiment_failure(exp, e)
        print()

    main_tracker.print_summary()
    main_tracker.save_to_file(ROOT / "analysis_summary" / "004_summary.json")
