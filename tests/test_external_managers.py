"""Tests for the runner's multi-server external-manager wiring (M2).

Covers spec resolution (servers.json or none) and the manager-building loop
(one manager per server, each on its declared callback_port, dict keyed by name)
without constructing a full SimulationRunner.
"""

import json
import types

from terralingua.experiment.runner import SimulationRunner
from terralingua.external.servers_config import ServerSpec
from terralingua.voting import IndependentVotingManager, UnanimousVotingManager


def _fake_self(*, servers_config=None, sync_send=True,
               server_port=None, server_callback_port=None):
    run = types.SimpleNamespace(
        external_servers_config=servers_config,
        external_sync_send=sync_send,
        reward_to_energy_coefficient=0.0,
        external_server_port=server_port,
        external_server_callback_port=server_callback_port,
    )
    return types.SimpleNamespace(params=types.SimpleNamespace(run=run))


# ---- _resolve_server_specs -------------------------------------------------

def test_resolve_applies_port_overrides(tmp_path):
    # run.external_server_port / external_server_callback_port supersede the sole
    # server's URL port / callback_port (so parallel runs need not edit servers.json).
    path = tmp_path / "servers.json"
    path.write_text(json.dumps([
        {"name": "world", "url": "http://localhost:8011/mcp", "callback_port": 9001,
         "mode": "unanimous", "vote_key": "slot"},
    ], indent=2))
    fs = _fake_self(servers_config=str(path), server_port=8022, server_callback_port=9099)
    specs = SimulationRunner._resolve_server_specs(fs)
    assert specs[0].url == "http://localhost:8022/mcp"
    assert specs[0].callback_port == 9099
    assert specs[0].mode == "unanimous"  # non-port fields preserved


# ---- _build_external_managers ----------------------------------------------

class _StubManager:
    instances = []

    def __init__(self, server_url, listen_port, request_timeout_s, sync_send):
        self.server_url = server_url
        self.listen_port = listen_port
        self.sync_send = sync_send
        self.tool_info = types.SimpleNamespace(name="step_world")
        _StubManager.instances.append(self)


def _build_with_specs(monkeypatch, specs):
    monkeypatch.setattr(
        "terralingua.experiment.runner.ExternalRequestManager", _StubManager
    )
    _StubManager.instances = []
    fs = _fake_self()
    fs._resolve_server_specs = lambda: specs
    fs.external_specs = {}
    fs.external_managers = {}
    SimulationRunner._build_external_managers(fs)
    return fs


def test_build_one_manager_per_server(monkeypatch):
    specs = [
        ServerSpec("market", "http://a/mcp", callback_port=9000),
        ServerSpec("judge", "http://b/mcp", callback_port=9001),
        ServerSpec("world", "http://c/mcp", callback_port=9002, mode="unanimous"),
    ]
    fs = _build_with_specs(monkeypatch, specs)
    assert list(fs.external_managers) == ["market", "judge", "world"]
    assert [m.listen_port for m in _StubManager.instances] == [9000, 9001, 9002]
    assert fs.external_managers["judge"].server_url == "http://b/mcp"
    assert fs.external_specs["world"].mode == "unanimous"
    assert set(fs.voting_managers) == {"market", "judge", "world"}
    assert isinstance(fs.voting_managers["market"], IndependentVotingManager)
    assert isinstance(fs.voting_managers["world"], UnanimousVotingManager)
