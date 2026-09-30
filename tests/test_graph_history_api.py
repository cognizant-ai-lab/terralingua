"""The dashboard reads only the selected run's recorded social graphs."""

import asyncio
import importlib
import time

import dotenv
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from terralingua.experiment.social_graph_history import SocialGraphHistoryRecorder
from terralingua.server.dashboard_manager import DashboardManager
from terralingua.server.graph_history import get_social_graph_history
from terralingua.server.redis_bus import KEY_META
from tests.test_social_graph_env import make_social_env, seed_n_agents


class MetadataRedis:
    """Expose stored metadata without a runner or a Redis service."""

    def __init__(self, logdir=None):
        self.logdir = logdir
        self.metadata_reads = []

    async def hget(self, key, field):
        self.metadata_reads.append((key, field))
        assert (key, field) == (KEY_META, "exp_logdir")
        return self.logdir

    async def get(self, key):
        return None


class SessionRegistry:
    def __init__(self):
        self.released = []

    async def validate_for_connect(self, token):
        return token == "valid-test-session"

    async def get_user_id(self, token):
        return None

    async def release(self, token):
        self.released.append(token)


def record_run(directory, count=5):
    directory.mkdir(parents=True, exist_ok=True)
    env = make_social_env(directory, n_nodes=3)
    seed_n_agents(env, ["Alice", "Bob", "Cara"])
    recorder = SocialGraphHistoryRecorder(
        directory / "social_graph.jsonl", run_id=directory.name
    )
    for step in range(count):
        env.step_count = step
        recorder.record(
            env,
            segment_start=step == 0,
            reason="start" if step == 0 else None,
        )
    return recorder


def read_page(redis, **kwargs):
    return asyncio.run(get_social_graph_history(redis, **kwargs))


@pytest.fixture
def dashboard_client(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost/test")
    monkeypatch.setenv("SESSION_SECRET", "graph-history-test-session-secret")
    monkeypatch.setenv("JWT_SECRET", "graph-history-test-jwt-secret")
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    api = importlib.import_module("terralingua.server.api")
    clients = []

    def make_client(redis):
        app = api.create_app(dashboard_manager=DashboardManager())
        app.state.redis = redis
        app.state.session_registry = SessionRegistry()
        app.state.config_cache = (
            time.monotonic(),
            {"ok": True, "data": {"allow_human_agents": True}},
        )
        # No lifespan context: these tests never start Redis, DB, or LLM services.
        client = TestClient(app)
        clients.append(client)
        return client, app

    yield make_client
    for client in clients:
        client.close()


def test_pages_cover_all_snapshots_without_duplicates(tmp_path):
    record_run(tmp_path, count=7)
    redis = MetadataRedis(str(tmp_path))
    pages = []
    offset = 0
    identity = {}
    while True:
        page = read_page(redis, offset=offset, limit=2, **identity)
        assert page["available"] is True
        assert page["offset"] == offset
        assert page["total"] == 7
        assert page["experiment"] == tmp_path.name
        assert len(page["snapshots"]) <= 2
        identity = {key: page[key] for key in ("run_id", "segment_id")}
        pages.extend(page["snapshots"])
        if page["next_offset"] is None:
            break
        assert page["next_offset"] > offset
        offset = page["next_offset"]
    assert [snapshot["step"] for snapshot in pages] == list(range(7))
    assert redis.metadata_reads == [(KEY_META, "exp_logdir")] * 4


@pytest.mark.parametrize("offset", [-1, True, "1"])
def test_invalid_offsets_are_rejected_before_reading(offset):
    redis = MetadataRedis("/unused")
    with pytest.raises(ValueError, match="offset"):
        read_page(redis, offset=offset)
    assert redis.metadata_reads == []


@pytest.mark.parametrize("limit", [0, 201, True, "2"])
def test_invalid_page_sizes_are_rejected_before_reading(limit):
    redis = MetadataRedis("/unused")
    with pytest.raises(ValueError, match="page size"):
        read_page(redis, limit=limit)
    assert redis.metadata_reads == []


@pytest.mark.parametrize("limit", [1, 200])
def test_page_size_boundaries_are_allowed(tmp_path, limit):
    record_run(tmp_path, count=3)
    page = read_page(MetadataRedis(str(tmp_path)), limit=limit)
    assert len(page["snapshots"]) == min(limit, 3)


def test_offset_at_end_returns_a_complete_empty_page(tmp_path):
    record_run(tmp_path, count=3)
    page = read_page(MetadataRedis(str(tmp_path)), offset=3)
    assert page["available"] is True
    assert page["total"] == 3
    assert page["offset"] == 3
    assert page["snapshots"] == []
    assert page["next_offset"] is None


@pytest.mark.parametrize("source", ["no_redis", "no_metadata", "missing_file"])
def test_unavailable_recordings_return_a_friendly_empty_result(tmp_path, source):
    redis = (
        None
        if source == "no_redis"
        else MetadataRedis(None if source == "no_metadata" else str(tmp_path))
    )
    page = read_page(redis)
    assert page["available"] is False
    assert page["snapshots"] == []
    assert page["total"] == 0
    assert page["run_id"] is None
    assert page["segment_id"] is None
    assert page["next_offset"] is None
    assert page["message"]


@pytest.mark.parametrize("as_bytes", [False, True])
def test_saved_history_loads_without_a_runner(tmp_path, as_bytes):
    record_run(tmp_path, count=2)
    logdir = str(tmp_path)
    redis = MetadataRedis(logdir.encode() if as_bytes else logdir)
    page = read_page(redis)
    assert [snapshot["step"] for snapshot in page["snapshots"]] == [0, 1]
    assert page["available"] is True
    assert redis.metadata_reads == [(KEY_META, "exp_logdir")]


@pytest.mark.parametrize("identity_key", ["run_id", "segment_id"])
def test_identity_mismatch_rejects_a_later_page(tmp_path, identity_key):
    record_run(tmp_path, count=3)
    redis = MetadataRedis(str(tmp_path))
    first = read_page(redis, limit=1)
    identity = {key: first[key] for key in ("run_id", "segment_id")}
    identity[identity_key] = "another-recording"
    with pytest.raises(ValueError, match="recording changed"):
        read_page(redis, offset=first["next_offset"], limit=1, **identity)


def test_dashboard_rejects_history_without_a_valid_session(dashboard_client):
    redis = MetadataRedis("/unused")
    client, app = dashboard_client(redis)
    with client.websocket_connect("/ws/dashboard?token=invalid-test-session") as websocket:
        with pytest.raises(WebSocketDisconnect) as closed:
            websocket.receive_json()
    assert closed.value.code == 4001
    assert redis.metadata_reads == []


def test_dashboard_history_uses_only_trusted_metadata_path(tmp_path, dashboard_client):
    selected = tmp_path / "selected"
    other = tmp_path / "other"
    record_run(selected, count=3)
    record_run(other, count=8)
    redis = MetadataRedis(str(selected))
    client, app = dashboard_client(redis)
    with client.websocket_connect(
        "/ws/dashboard?token=valid-test-session"
    ) as websocket:
        assert websocket.receive_json()["type"] == "config"
        websocket.send_json(
            {
                "cmd": "get_social_graph_history",
                "req": "history-1",
                "limit": 2,
                "path": str(other / "social_graph.jsonl"),
                "exp_logdir": str(other),
                "experiment": "other",
            }
        )
        response = websocket.receive_json()
        assert response["res"] == "history-1"
        assert response["ok"] is True
        assert response["data"]["experiment"] == "selected"
        assert response["data"]["total"] == 3
        assert len(response["data"]["snapshots"]) == 2
        websocket.send_json(
            {
                "cmd": "get_social_graph_history",
                "req": "history-invalid",
                "limit": True,
            }
        )
        failure = websocket.receive_json()
        assert failure["res"] == "history-invalid"
        assert failure["ok"] is False
        assert "page size" in failure["error"]
    assert app.state.session_registry.released == ["valid-test-session"]
    assert redis.metadata_reads == [(KEY_META, "exp_logdir")]
