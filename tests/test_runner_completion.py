"""Exercise completion accounting through the real asynchronous runner loop."""

import asyncio
from types import SimpleNamespace

import pytest

from terralingua.experiment.run_status import RunStatus
from terralingua.experiment.runner import SimulationRunner, _ExternalState


class WorldFailure(RuntimeError):
    pass


class StubEnvironment:
    def __init__(self, fail_on=None):
        self.artifacts = {}
        self.attempted = 0
        self.committed = 0
        self.fail_on = fail_on

    def step(self, actions):
        self.attempted += 1
        if self.attempted == self.fail_on:
            raise WorldFailure("World step failed before committing.")
        self.committed += 1
        return {"a": {"time": self.committed}}, {"a": 0}, {"a": False}, {}, {"a": {}}


def make_runner(tmp_path, *, fail_on=None, max_steps=2):
    runner = SimulationRunner.__new__(SimulationRunner)
    runner.params = SimpleNamespace(run=SimpleNamespace(
        max_parallel_workers=1, empty_countdown=-1, max_ts=40 + max_steps,
        stop_on_external_solved=False,
    ))
    runner.start_ts = 40
    runner._use_redis = False
    runner.dashboard_manager = None
    runner.terminate = False
    runner.env = StubEnvironment(fail_on)
    runner.agents = {"a": SimpleNamespace()}
    runner.obs = {"a": {"time": 0}}
    runner.infos = {"a": {}}
    runner.rewards = {"a": 0}
    runner.dones = {"a": False}
    runner.run_status = RunStatus(tmp_path / "run_status.json")
    runner.teardown_states = []

    async def no_async(*args, **kwargs):
        return None

    async def collect(ts):
        return {"a": {"action": "noop", "params": {}, "message": "", "source": "scripted"}}

    async def teardown(ts):
        runner.teardown_states.append({"next_step": ts, "status": runner.run_status.read()})

    for name in (
        "_setup_signal_handlers", "_render", "_maybe_refresh_llm_router",
        "_record_external_outcomes", "_inject_external_state", "_capture_action_requests",
        "_log_step_usage", "_process_external_poststep", "_record_local_action_receipts",
    ):
        setattr(runner, name, lambda *args, **kwargs: None)
    for name in (
        "_maybe_checkpoint", "_advance_population", "_push_dashboard_frame",
    ):
        setattr(runner, name, no_async)
    runner._handle_incoming_external_messages = lambda *args: {}
    runner._process_external_prestep = lambda actions: _ExternalState()
    runner._collect_actions = collect
    runner._teardown = teardown
    return runner


def test_completed_simulation_is_persisted_before_teardown_raises(tmp_path):
    runner = make_runner(tmp_path)
    failure = RuntimeError("Export failed after simulation completion.")

    async def failed_teardown(ts):
        runner.teardown_states.append({"next_step": ts, "status": runner.run_status.read()})
        raise failure

    runner._teardown = failed_teardown
    with pytest.raises(RuntimeError) as caught:
        asyncio.run(runner.run_async())
    assert caught.value is failure
    before_export = runner.teardown_states[0]
    assert before_export["next_step"] == 42
    assert before_export["status"]["simulation"]["status"] == "complete"
    assert before_export["status"]["simulation"]["completed_steps"] == runner.env.committed == 2
    status = runner.run_status.read()
    assert status["simulation"]["status"] == "complete"
    assert status["process"]["status"] == "failed"


@pytest.mark.parametrize("fail_on", [1, 2])
def test_world_step_failure_propagates_and_counts_only_committed_steps(tmp_path, fail_on):
    runner = make_runner(tmp_path, fail_on=fail_on)
    with pytest.raises(WorldFailure, match="before committing"):
        asyncio.run(runner.run_async())
    status = runner.run_status.read()
    assert status["simulation"]["status"] == "failed"
    assert status["simulation"]["reason"] == "error"
    assert status["simulation"]["completed_steps"] == runner.env.committed == fail_on - 1
    assert runner.env.attempted == fail_on
    assert runner.teardown_states[0]["next_step"] == 40 + runner.env.committed
    assert runner.teardown_states[0]["status"]["simulation"]["status"] == "failed"
    assert status["process"]["status"] == "failed"


def test_maintenance_failure_does_not_rewind_the_committed_world_step(tmp_path):
    runner = make_runner(tmp_path)

    def broken_maintenance(*args):
        raise RuntimeError("Receipt maintenance failed after the world step.")

    runner._record_local_action_receipts = broken_maintenance
    with pytest.raises(RuntimeError, match="maintenance failed"):
        asyncio.run(runner.run_async())
    status = runner.run_status.read()
    assert status["simulation"]["status"] == "failed"
    assert status["simulation"]["completed_steps"] == runner.env.committed == 1
    assert runner.teardown_states[0]["next_step"] == 41
    assert runner.env.attempted == 1
