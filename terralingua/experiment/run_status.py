"""Persist simulation progress separately from exports and process failures."""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStatus:
    """A small atomic status file. Export failure cannot erase simulation completion."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data = self.read()

    def read(self) -> dict:
        if not self.path.exists():
            return {}
        value = json.loads(self.path.read_text())
        if not isinstance(value, dict):
            raise ValueError("Run status must contain an object.")
        return value

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["updated_at"] = _now()
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=self.path.parent, prefix=self.path.name + ".", delete=False
            ) as stream:
                temporary = Path(stream.name)
                json.dump(self.data, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def start(self, start_step: int, step_budget: int | None = None) -> None:
        self.data = {
            **self.data,
            "simulation": {
                "status": "running", "start_step": start_step,
                "step_budget": step_budget, "completed_steps": 0, "started_at": _now(),
            },
            "exports": {},
            "process": {"status": "running", "started_at": _now()},
        }
        self._save()

    def simulation_finished(
        self, *, completed_steps: int, reason: str,
        error: BaseException | str | None = None, completed: bool = False,
    ) -> None:
        self.data["simulation"] = {
            **self.data.get("simulation", {}),
            "status": "failed" if error is not None else "complete" if completed else "stopped_early",
            "completed_steps": completed_steps, "reason": reason,
            "error": str(error) if error is not None else None, "finished_at": _now(),
        }
        self._save()

    def export_started(self, name: str) -> None:
        self.data.setdefault("exports", {})[name] = {"status": "running", "started_at": _now()}
        self._save()

    def export_finished(
        self, name: str, *, error: BaseException | str | None = None, skipped: bool = False,
    ) -> None:
        exports = self.data.setdefault("exports", {})
        exports[name] = {
            **exports.get(name, {}),
            "status": "failed" if error is not None else "skipped" if skipped else "complete",
            "error": str(error) if error is not None else None, "finished_at": _now(),
        }
        self._save()

    def process_finished(self, error: BaseException | str | None = None) -> None:
        self.data["process"] = {
            **self.data.get("process", {}),
            "status": "failed" if error is not None else "complete",
            "error": str(error) if error is not None else None, "finished_at": _now(),
        }
        self._save()
