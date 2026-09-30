"""Saved configuration must govern services before checkpoint state is restored."""

import ast
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from terralingua.config.models import ExperimentConfig
from terralingua.experiment import runner as runner_module
from terralingua.experiment.checkpoint import CheckpointManager
from terralingua.experiment.runner import SimulationRunner


def save_params(tmp_path, config):
    folder = tmp_path / config.run.exp_name
    folder.mkdir()
    (folder / "params.json").write_text(json.dumps(config.to_json()))
    return folder


def test_saved_settings_govern_external_services_before_restore(tmp_path, monkeypatch):
    servers = tmp_path / "servers.json"
    servers.write_text(json.dumps([{
        "name": "saved", "url": "http://localhost:9000/mcp", "callback_port": 9001,
        "mode": "independent", "hidden_keys": ["private"],
    }]))
    saved = ExperimentConfig(run={
        "exp_name": "resume", "external_servers_config": str(servers),
        "external_sync_send": False, "reward_to_energy_coefficient": 2.5,
        "remote_api_enabled": True,
    })
    folder = save_params(tmp_path, saved)
    requested = ExperimentConfig(run={
        "exp_name": "resume",
        "external_sync_send": True, "reward_to_energy_coefficient": 99,
    })
    created = []
    restored = []
    monkeypatch.setattr(runner_module, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(runner_module, "ExternalRequestManager", lambda **kw: created.append(kw) or SimpleNamespace(instructions="", **kw))
    monkeypatch.setattr(SimulationRunner, "_build_external_actions_spec", lambda self: None)
    monkeypatch.setattr(SimulationRunner, "_load_state", lambda self: restored.append(self.params))
    monkeypatch.setattr(runner_module, "LLMRouter", lambda **kw: SimpleNamespace(**kw))

    runner = SimulationRunner(requested, resume=True)
    assert runner.params == saved
    assert restored == [saved]
    assert created == [{"server_url": "http://localhost:9000/mcp", "listen_port": 9001,
                        "request_timeout_s": None, "sync_send": False}]
    assert runner.voting_managers["saved"].rewards.coefficient == 2.5
    assert runner.hidden_external_keys == ["private"]
    assert json.loads((folder / "params.json").read_text()) == saved.to_json()


def test_resume_requires_an_existing_run_name(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_module, "LOGS_DIR", tmp_path)
    with pytest.raises(ValueError, match="exp_name"):
        SimulationRunner(ExperimentConfig(), resume=True)
    assert not list(tmp_path.iterdir())


def entrypoint_run(namespace):
    # Execute only this function. Importing the cli module would initialize dotenv/logging.
    path = Path(__file__).parents[1] / "terralingua" / "cli.py"
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "_run")
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_run"]


@pytest.mark.parametrize("saved_enabled", [False, True])
def test_entrypoint_uses_saved_dashboard_choice_for_base_runner(tmp_path, saved_enabled):
    saved = ExperimentConfig(run={"exp_name": "resume", "remote_api_enabled": saved_enabled})
    save_params(tmp_path, saved)
    requested = ExperimentConfig(run={"exp_name": "resume", "remote_api_enabled": not saved_enabled})
    calls = []

    async def record_start():
        calls.append("start")

    async def registry():
        calls.append("registry")
        return object()

    class OfflineRunner:
        def __init__(self, params, resume):
            self.params = params
            assert params == saved and resume
            calls.append("runner")

        async def run_async(self):
            calls.append("run")

    namespace = {
        "CheckpointManager": CheckpointManager, "LOGS_DIR": tmp_path,
        "_runner_class": lambda _: OfflineRunner,
        "init_redis_pool": lambda: calls.append("redis"),
        "RedisAgentRegistry": SimpleNamespace(create=registry),
        "RedisPublisher": lambda: SimpleNamespace(start=record_start),
        "RedisConnectionManager": lambda: SimpleNamespace(start=record_start),
        "redis_url": lambda: "offline", "log": SimpleNamespace(info=lambda *_: None),
    }
    asyncio.run(entrypoint_run(namespace)(requested, True))
    assert calls == (["runner", "redis", "registry", "start", "start", "run"]
                     if saved_enabled else ["runner", "run"])
