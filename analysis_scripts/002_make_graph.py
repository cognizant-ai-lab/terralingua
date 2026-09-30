import traceback

from terralingua.anthropologist.analysts.graph_analyst import GraphAnalyst
from terralingua.utils import LOGS_DIR, ROOT

"""
Builds interaction graphs and computes network metrics.
Logic lives in analysis_scripts/analysts/graph_analyst.py.
"""

EXPERIMENTS_NAMES = []  # e.g. ['core_run', 'scarcity_run', ...]


def main(exp_name):
    analyst = GraphAnalyst()
    analyst.build_graph(exp_path=LOGS_DIR / exp_name, save=True)


if __name__ == "__main__":
    failed = []
    for exp_name in EXPERIMENTS_NAMES:
        try:
            print("Processing", exp_name)
            main(exp_name)
            print(f"Done with {exp_name}")
            print()
        except Exception as e:
            failed.append(exp_name)
            print(f"Experiment {exp_name} failed: {e}")
            traceback.print_exc()
            print()

    if failed:
        print("The following experiments failed:")
        for exp in failed:
            print(f"- {exp}")
