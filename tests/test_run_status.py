"""Completed simulation records survive optional export and process failures."""

import json

from terralingua.experiment.run_status import RunStatus


def test_export_failure_keeps_completed_simulation(tmp_path):
    path = tmp_path / "run_status.json"
    status = RunStatus(path)
    status.start(start_step=1000, step_budget=500)
    status.simulation_finished(completed_steps=500, reason="step_budget", completed=True)
    # This record must exist before export code can fail.
    assert json.loads(path.read_text())["simulation"]["status"] == "complete"
    status.export_started("video")
    error = RuntimeError("No frames")
    status.export_finished("video", error=error)
    status.process_finished(error=error)
    reopened = RunStatus(path).data
    assert reopened["simulation"]["status"] == "complete"
    assert reopened["simulation"]["completed_steps"] == 500
    assert reopened["exports"]["video"]["status"] == "failed"
    assert reopened["process"]["status"] == "failed"
    assert not list(tmp_path.glob("run_status.json.*"))


def test_early_stop_and_export_success_remain_distinct(tmp_path):
    status = RunStatus(tmp_path / "status.json")
    status.start(start_step=0, step_budget=500)
    status.simulation_finished(completed_steps=13, reason="user_stop")
    status.export_finished("video", skipped=True)
    status.export_finished("checkpoint")
    status.process_finished()
    assert status.read()["simulation"]["status"] == "stopped_early"
    assert status.read()["exports"]["video"]["status"] == "skipped"
    assert status.read()["exports"]["checkpoint"]["status"] == "complete"
    assert status.read()["process"]["status"] == "complete"


def test_simulation_error_is_not_erased_by_successful_cleanup(tmp_path):
    status = RunStatus(tmp_path / "status.json")
    status.start(start_step=0)
    status.simulation_finished(completed_steps=2, reason="error", error="World unavailable")
    status.export_finished("checkpoint")
    status.process_finished(error="World unavailable")
    assert status.read()["simulation"]["status"] == "failed"
    assert status.read()["simulation"]["error"] == "World unavailable"
    assert status.read()["exports"]["checkpoint"]["status"] == "complete"


def test_existing_metadata_survives_status_updates(tmp_path):
    path = tmp_path / "status.json"
    path.write_text(json.dumps({"experiment_id": "comparison-1"}))
    status = RunStatus(path)
    status.start(start_step=0)
    status.process_finished()
    assert status.read()["experiment_id"] == "comparison-1"
