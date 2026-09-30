"""Every test runs from the repository root.

Preset discovery and the logs folder follow the working directory, so the
suite must not depend on where pytest was started.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent


@pytest.fixture(autouse=True)
def _run_from_repo_root(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
