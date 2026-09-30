"""Tests for the servers.json loader (terralingua/external/servers_config.py).

A scenario declares its external MCP servers in a JSON array; each entry becomes a
ServerSpec with per-server cost/mode/callback_port. These tests cover parsing,
defaults, and the validation error cases.
"""

import json

import pytest

from terralingua.external.external_state import (
    ExternalStateHandler,
    register_state_handler,
)
from terralingua.external.servers_config import (
    ServerSpec,
    apply_port_overrides,
    load_servers_config,
)


@register_state_handler
class _SharedState(ExternalStateHandler):
    kind = "shared_test"


def _write(tmp_path, data):
    path = tmp_path / "servers.json"
    path.write_text(json.dumps(data, indent=2))
    return str(path)


def test_loads_full_entry(tmp_path):
    path = _write(
        tmp_path,
        [{"name": "world", "url": "http://localhost:8080/mcp", "callback_port": 9000,
          "cost": 2, "mode": "unanimous", "state": "shared_test", "vote_key": "slot",
          "hidden_keys": ["difficulty", "layout"], "batch": True}],
    )
    specs = load_servers_config(path)
    assert specs == [
        ServerSpec("world", "http://localhost:8080/mcp", callback_port=9000,
                   cost=2, mode="unanimous", state="shared_test", vote_key="slot",
                   hidden_keys=["difficulty", "layout"], batch=True)
    ]


def test_applies_defaults(tmp_path):
    path = _write(tmp_path, [{"name": "judge", "url": "http://localhost:8081/mcp",
                              "callback_port": 9000}])
    spec = load_servers_config(path)[0]
    assert spec.callback_port == 9000
    assert spec.cost == 0
    assert spec.mode == "independent"
    assert spec.state == ""


def test_invalid_state_rejected(tmp_path):
    path = _write(tmp_path, [{"name": "world", "url": "http://a/mcp",
                              "callback_port": 9000, "state": "bogus"}])
    with pytest.raises(ValueError, match="invalid state"):
        load_servers_config(path)


def test_multiple_servers_preserve_order(tmp_path):
    path = _write(
        tmp_path,
        [
            {"name": "market", "url": "http://a/mcp", "callback_port": 9000},
            {"name": "judge", "url": "http://b/mcp", "callback_port": 9001},
            {"name": "world", "url": "http://c/mcp", "callback_port": 9002,
             "mode": "unanimous"},
        ],
    )
    specs = load_servers_config(path)
    assert [s.name for s in specs] == ["market", "judge", "world"]
    assert [s.callback_port for s in specs] == [9000, 9001, 9002]
    assert specs[2].mode == "unanimous"


def test_duplicate_names_rejected(tmp_path):
    path = _write(
        tmp_path,
        [{"name": "market", "url": "http://a/mcp", "callback_port": 9000},
         {"name": "market", "url": "http://b/mcp", "callback_port": 9001}],
    )
    with pytest.raises(ValueError, match="Duplicate server name"):
        load_servers_config(path)


def test_duplicate_callback_port_rejected(tmp_path):
    path = _write(
        tmp_path,
        [{"name": "market", "url": "http://a/mcp", "callback_port": 9000},
         {"name": "judge", "url": "http://b/mcp", "callback_port": 9000}],
    )
    with pytest.raises(ValueError, match="Duplicate callback_port"):
        load_servers_config(path)


def test_invalid_mode_rejected(tmp_path):
    path = _write(tmp_path, [{"name": "market", "url": "http://a/mcp",
                              "callback_port": 9000, "mode": "bogus"}])
    with pytest.raises(ValueError, match="invalid mode"):
        load_servers_config(path)


@pytest.mark.parametrize("name", ["Market", "ma-rket"])
def test_invalid_names_rejected(tmp_path, name):
    path = _write(tmp_path, [{"name": name, "url": "http://a/mcp", "callback_port": 9000}])
    with pytest.raises(ValueError, match="invalid"):
        load_servers_config(path)


def test_missing_required_keys_rejected(tmp_path):
    path = _write(tmp_path, [{"name": "market"}])
    with pytest.raises(ValueError, match="must include 'name', 'url', and 'callback_port'"):
        load_servers_config(path)


@pytest.mark.parametrize("port", [0, 70000])
def test_invalid_callback_port_rejected(tmp_path, port):
    path = _write(tmp_path, [{"name": "market", "url": "http://a/mcp",
                              "callback_port": port}])
    with pytest.raises(ValueError, match="invalid callback_port"):
        load_servers_config(path)


def test_negative_cost_rejected(tmp_path):
    path = _write(tmp_path, [{"name": "market", "url": "http://a/mcp",
                              "callback_port": 9000, "cost": -1}])
    with pytest.raises(ValueError, match="invalid cost"):
        load_servers_config(path)


def test_non_array_rejected(tmp_path):
    path = _write(tmp_path, {"name": "market", "url": "http://a/mcp"})
    with pytest.raises(ValueError, match="must be a JSON array"):
        load_servers_config(path)


def test_empty_array_rejected(tmp_path):
    path = _write(tmp_path, [])
    with pytest.raises(ValueError, match="no servers"):
        load_servers_config(path)


def test_non_object_entry_rejected(tmp_path):
    path = _write(tmp_path, ["market"])
    with pytest.raises(ValueError, match="must be an object"):
        load_servers_config(path)


# ── apply_port_overrides ─────────────────────────────────────────────────────


def test_override_url_port_only_preserves_host_and_path():
    specs = [ServerSpec("world", "http://localhost:8011/mcp", callback_port=9001)]
    out = apply_port_overrides(specs, url_port=8022)
    assert out[0].url == "http://localhost:8022/mcp"
    assert out[0].callback_port == 9001  # untouched


def test_override_callback_port_only():
    specs = [ServerSpec("world", "http://localhost:8011/mcp", callback_port=9001)]
    out = apply_port_overrides(specs, callback_port=9002)
    assert out[0].callback_port == 9002
    assert out[0].url == "http://localhost:8011/mcp"  # untouched


def test_override_invalid_callback_port_rejected():
    specs = [ServerSpec("world", "http://localhost:8011/mcp", callback_port=9001)]
    with pytest.raises(ValueError, match="invalid callback_port"):
        apply_port_overrides(specs, callback_port=70000)


def test_override_requires_single_server():
    specs = [
        ServerSpec("market", "http://a/mcp", callback_port=9000),
        ServerSpec("judge", "http://b/mcp", callback_port=9001),
    ]
    with pytest.raises(ValueError, match="exactly one declared server"):
        apply_port_overrides(specs, url_port=8022)
