"""Shared Redis channel/key constants, client factory, and async utilities."""
import asyncio
import logging
import os

from redis.asyncio import ConnectionPool, Redis

logger = logging.getLogger(__name__)

# Pub/sub channels
CHANNEL_FRAMES = "ogw:frames"
CHANNEL_PUSH = "ogw:push"
CHANNEL_CMD = "ogw:cmd"
CHANNEL_AGENT_EVENT = "ogw:agent_event"

# Global keys (not experiment-specific)
KEY_LAST_FRAME = "ogw:last_frame"
KEY_META = "ogw:meta"
KEY_CONNECTED_AGENTS = "ogw:connected_agents"
KEY_RUNNER_STATUS = "ogw:runner_status"
KEY_EXP_CONFIG = "ogw:exp_config"  # JSON {exp_name, exp_path} — global pointer to current experiment
KEY_ANTHRO_ADMIN_API_KEY = "ogw:anthro:admin_api_key"  # Fernet-encrypted fallback key for ownerless-agent annotation


# Experiment-scoped key builders
def key_field_notes_all(exp_name: str) -> str:
    return f"ogw:{exp_name}:field_notes_all"


def key_field_notes_latest(exp_name: str) -> str:
    return f"ogw:{exp_name}:field_notes_latest"


def key_last_detect_step(exp_name: str) -> str:
    return f"ogw:{exp_name}:last_detect_step"


def key_postmortem_queue(exp_name: str) -> str:
    return f"ogw:{exp_name}:postmortem_queue"


def key_live_annotations(exp_name: str) -> str:
    return f"ogw:{exp_name}:live_annotations"


def key_artifact_classifications(exp_name: str) -> str:
    return f"ogw:{exp_name}:artifact_classifications"


def key_phylogeny_queue(exp_name: str) -> str:
    return f"ogw:{exp_name}:phylogeny_queue"


def rsp_channel(req_id: str) -> str:
    return f"ogw:rsp:{req_id}"


def obs_channel(tag: str) -> str:
    return f"ogw:obs:{tag}"


def action_channel(tag: str) -> str:
    return f"ogw:action:{tag}"


def key_bg_owner(tag: str) -> str:
    return f"ogw:bg_owner:{tag}"


def key_anon_alive(anon_id: str) -> str:
    """Heartbeat key for an anonymous browser session.

    Refreshed via dashboard-WS heartbeat messages. server_llm_loop instances
    spawned for this anon_id exit when the key expires (browser closed all tabs).
    """
    return f"ogw:anon_alive:{anon_id}"


def key_anon_apikey(anon_id: str) -> str:
    """Fernet-encrypted API key for an anonymous browser session.

    Lives only in Redis (no disk persistence by default), encrypted with the
    same FERNET_SECRET_KEY used for logged-in users' DB-stored keys. TTL is
    refreshed by the dashboard-WS heartbeat alongside key_anon_alive; the key
    expires within ANON_HEARTBEAT_TTL of the last heartbeat. Used by
    _watch_child_agents to spawn loops for children of anon parents.
    """
    return f"ogw:anon_apikey:{anon_id}"


def redis_url() -> str:
    return os.environ.get("REDIS_URL", "redis://localhost:6379")


_pool = None


def init_redis_pool() -> None:
    global _pool
    # Bounded pool: caps how many connections each worker can open to Redis, so
    # a burst of activity back-pressures instead of exhausting Redis's client
    # limit. Default 1000/worker (4 workers => ~4000, well under maxclients).
    # NOTE: the plain ConnectionPool raises ConnectionError when exhausted
    # rather than blocking; switch to BlockingConnectionPool if you'd prefer
    # callers to queue for a free connection instead.
    max_conn = int(os.environ.get("REDIS_MAX_CONNECTIONS", "1000"))
    _pool = ConnectionPool.from_url(
        redis_url(), decode_responses=True, max_connections=max_conn
    )


async def close_redis_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.aclose()
        _pool = None


async def get_redis():
    if _pool is None:
        raise RuntimeError("Redis pool not initialized — call init_redis_pool() first")
    return Redis(connection_pool=_pool)


async def retry_loop(coro_fn, label: str, max_backoff: float = 60.0):
    """Run coro_fn() in a loop, retrying with exponential backoff on failure.

    Exits cleanly on CancelledError (normal shutdown).
    """
    backoff = 1.0
    while True:
        try:
            await coro_fn()
            backoff = 1.0
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.error(f"{label}: {e}. Retrying in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)
