"""
FastAPI server for the OpenGridWorld web UI (API process).

All simulation commands are routed to the runner process via Redis pub/sub.
Only the `llm` command is handled locally (no runner state needed).

Endpoints:
  WS   /ws/dashboard Authenticated dashboard channel
  WS   /ws/{tag}     Remote agent WebSocket (bridges obs/actions via Redis)
  GET  /dashboard    Dashboard UI; issues a one-time session token
"""

import asyncio
import hashlib
import hmac
import html as _html
import json
import logging
import os
import re
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from fastapi import (
    Depends,
    FastAPI,
    Form,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi_users.authentication.strategy import DatabaseStrategy
from fastapi_users.exceptions import (
    InvalidPasswordException,
    InvalidResetPasswordToken,
    UserAlreadyExists,
    UserNotExists,
)
from fastapi_users_db_sqlalchemy.access_token import SQLAlchemyAccessTokenDatabase
from pydantic import BaseModel
from redis.asyncio import Redis as AsyncRedis
from redis.exceptions import ConnectionError as RedisConnectionError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.middleware.sessions import SessionMiddleware
from wonderwords import RandomWord

from terralingua.server.auth import decrypt_api_key, encrypt_api_key
from terralingua.server.graph_history import get_social_graph_history
from terralingua.server.models import (
    AccessToken,
    User,
    UserCreate,
    UserRead,
    UserUpdate,
)
from terralingua.server.redis_bus import (
    CHANNEL_AGENT_EVENT,
    CHANNEL_CMD,
    CHANNEL_PUSH,
    KEY_ANTHRO_ADMIN_API_KEY,
    KEY_CONNECTED_AGENTS,
    KEY_EXP_CONFIG,
    KEY_LAST_FRAME,
    KEY_RUNNER_STATUS,
    action_channel,
    close_redis_pool,
    get_redis,
    init_redis_pool,
    key_anon_alive,
    key_anon_apikey,
    key_artifact_classifications,
    key_bg_owner,
    key_field_notes_all,
    key_live_annotations,
    key_phylogeny_queue,
    key_postmortem_queue,
    obs_channel,
    rsp_channel,
)
from terralingua.server.redis_registry import (
    RedisAgentRegistry,
    RedisDashboardSessionRegistry,
)
from terralingua.server.redis_subscriber import RedisSubscriber
from terralingua.server.user_db import (
    async_session_maker,
    create_db_and_tables,
    get_async_session,
)
from terralingua.server.user_manager import (
    auth_backend,
    cookie_secure_default,
    current_user,
    fastapi_users,
    get_user_manager,
    optional_current_user,
    superuser,
)
from terralingua.utils.llm_client import AgentClient
from terralingua.utils.llm_errors import llm_error_type as _llm_error_type
from terralingua.utils.models import models_for_dashboard

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"
_AGENT_TAG_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _static_assets_hash() -> str:
    """Short fingerprint of static files for cache-busting CSS/JS links.

    Uses path + mtime + size (not content) so it's cheap enough to call on every
    request — guarantees browsers fetch fresh assets the moment files change on
    disk, with no server restart needed.
    """
    h = hashlib.md5()
    for p in sorted(_STATIC_DIR.rglob("*")):
        if p.is_file():
            st = p.stat()
            h.update(str(p).encode())
            h.update(str(st.st_mtime_ns).encode())
            h.update(str(st.st_size).encode())
    return h.hexdigest()[:8]


class NoCacheStaticFiles(StaticFiles):
    """StaticFiles that forces browser revalidation on every request.

    Starlette already sets Last-Modified/ETag, so revalidation is cheap (304
    when unchanged). This closes the cache-busting gap for ES module imports
    inside JS files, which the HTML-rewrite in /dashboard cannot reach.
    """

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response


def _valid_agent_tag(tag: str) -> bool:
    return bool(tag and _AGENT_TAG_RE.match(tag))


def _agent_vitals(agent_tag: str, exp_path: Path) -> dict:
    """Read the first and last lines of an agent JSONL log to get
    birth_step, steps_lived, and energy_at_death."""
    if not _valid_agent_tag(agent_tag):
        return {}
    try:
        log_path = exp_path / "agent_logs" / f"{agent_tag}.jsonl"
        first_line = last_line = None
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if first_line is None:
                    first_line = line
                last_line = line
        if not first_line:
            return {}
        first = json.loads(first_line)
        last = json.loads(last_line)  # type: ignore
        ts_first = first.get("timestamp", "")
        birth_step = int(ts_first) if str(ts_first).isdigit() else None
        first_obs = first.get("observation", {})
        last_obs = last.get("observation", {})
        time_at_birth = first_obs.get("time")
        time_at_death = last_obs.get("time")
        steps_lived = (
            time_at_birth - time_at_death + 1
            if time_at_birth is not None and time_at_death is not None
            else None
        )
        return {
            "birth_step": birth_step,
            "steps_lived": steps_lived,
            "energy_at_death": last_obs.get("energy"),
        }
    except Exception:
        return {}


load_dotenv(find_dotenv(usecwd=True), override=True)
DASHBOARD_PASSWORD: str | None = os.environ.get("DASHBOARD_PASSWORD") or None
# Randomised per dashboard server invocation so that restarting invalidates existing
# gate cookies. In multi-worker mode the parent seeds OGW_GATE_NONCE before
# forking so every worker validates the same cookie.
_GATE_NONCE: str = os.environ.get("OGW_GATE_NONCE") or secrets.token_hex(32)


def _make_auth_cookie() -> str:
    key = (os.environ.get("SESSION_SECRET", "") + _GATE_NONCE).encode()
    return hmac.new(key, b"dashboard_auth", hashlib.sha256).hexdigest()


def _check_auth(request: Request) -> bool:
    if not DASHBOARD_PASSWORD:
        return False
    token = request.cookies.get("dashboard_auth", "")
    if not token:
        return False
    return hmac.compare_digest(token, _make_auth_cookie())


def _check_guest(request: Request) -> bool:
    return request.cookies.get("guest_ok") == "1"


class _TestingGate:
    """Pure ASGI middleware that gates all HTTP routes behind DASHBOARD_PASSWORD."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            if scope.get("path", "") != "/gate":
                cookies = {}
                for name, value in scope.get("headers", []):
                    if name == b"cookie":
                        for part in value.decode().split(";"):
                            part = part.strip()
                            if "=" in part:
                                k, v = part.split("=", 1)
                                cookies[k.strip()] = v.strip()
                token = cookies.get("dashboard_auth", "")
                if not (
                    DASHBOARD_PASSWORD
                    and token
                    and hmac.compare_digest(token, _make_auth_cookie())
                ):
                    await send(
                        {
                            "type": "http.response.start",
                            "status": 302,
                            "headers": [
                                (b"location", b"/gate"),
                                (b"content-length", b"0"),
                            ],
                        }
                    )
                    await send({"type": "http.response.body", "body": b""})
                    return
        await self.app(scope, receive, send)


# ------------------------------------------------------------------
# Pydantic schemas
# ------------------------------------------------------------------


class DeleteAccountRequest(BaseModel):
    password: str


class ForgotPasswordRequest(BaseModel):
    email: str


class ResetPasswordRequest(BaseModel):
    token: str
    password: str


class AdminUpdateUserRequest(BaseModel):
    is_active: bool | None = None
    is_superuser: bool | None = None


# ------------------------------------------------------------------
# Redis command helper
# ------------------------------------------------------------------


async def _redis_cmd(
    redis: AsyncRedis, cmd: str, params: dict, timeout: float = 15.0
) -> dict:
    """Send a command to the runner via Redis and wait for the response."""
    req_id = str(uuid.uuid4())
    rsp_ch = rsp_channel(req_id)
    pubsub = redis.pubsub()
    try:
        await pubsub.subscribe(rsp_ch)
        # Drain subscribe confirmation before publishing to avoid race
        async for msg in pubsub.listen():
            if msg["type"] == "subscribe":
                break
        await redis.publish(
            CHANNEL_CMD, json.dumps({"req_id": req_id, "cmd": cmd, "params": params})
        )

        async def _wait() -> dict:
            async for msg in pubsub.listen():
                if msg["type"] == "message":
                    return json.loads(msg["data"])
            return {"ok": False, "error": "pubsub connection closed"}

        return await asyncio.wait_for(_wait(), timeout=timeout)
    except asyncio.TimeoutError:
        return {"ok": False, "error": f"Runner timeout on command '{cmd}'"}
    except RedisConnectionError:
        return {"ok": False, "error": "Redis connection lost"}
    finally:
        await pubsub.unsubscribe(rsp_ch)
        await pubsub.aclose()


# Runner config is static for the lifetime of a run, so re-fetching it on every
# dashboard connect is pure waste — and under a connect storm the per-connect
# pub/sub round-trip to the single runner piles up held Redis connections and
# exhausts the client pool. Caching collapses thousands of round-trips into
# ~one per worker per TTL window.
CONFIG_CACHE_TTL = 10.0  # seconds


async def _get_cached_config(state) -> dict:
    """Return the runner's get_config response, cached per-worker for
    CONFIG_CACHE_TTL seconds. Serves the last good value if the runner blips."""
    now = time.monotonic()
    cached = state.config_cache
    if cached and now - cached[0] < CONFIG_CACHE_TTL:
        return cached[1]
    # Single-flight: only one refresh runs at a time; concurrent callers await
    # it and then read the freshly-populated cache instead of each hitting the
    # runner (prevents a stampede when the cache is cold under a connect burst).
    async with state.config_lock:
        cached = state.config_cache
        if cached and time.monotonic() - cached[0] < CONFIG_CACHE_TTL:
            return cached[1]
        cfg = await _redis_cmd(state.redis, "get_config", {})
        if cfg.get("ok"):
            state.config_cache = (time.monotonic(), cfg)
        elif cached:
            return cached[1]  # serve stale rather than fail the connect
        return cfg


# Dashboard-WS commands the API forwards to the runner without API-side
# authorization. Two categories qualify:
#   - read-only (no shared state mutated);
#   - already runner-authorized (kill_agent / update_agent require a per-agent
#     token that only the original spawner holds — see runner.py).
# All other commands (notably the artifact mutations) must have an explicit
# handler in _handle_dashboard_message.
_WS_RUNNER_PASSTHROUGH = frozenset(
    {
        "get_config",
        "get_status",
        "get_messages",
        "get_genome_info",
        "get_agent_log",
        "get_actions",
        "get_agent_names",
        "download_messages",
        "update_agent",
        "kill_agent",
    }
)


def _artifact_owner_key(name: str) -> str:
    """Redis key recording who created a dashboard-spawned artifact. Used to
    gate modify_artifact / destroy_artifact (the runner has no per-user authz
    on those commands)."""
    return f"ogw:artifact_owner:{name}"


def _ws_caller_id(user_id: str | None, msg: dict) -> str | None:
    """Stable identifier for the dashboard WS caller. ``user:<uuid>`` for
    logged-in users, ``anon:<token>`` for guests with an anon_id. None for an
    unidentifiable caller (no user_id and no anon_id in the message)."""
    if user_id:
        return f"user:{user_id}"
    anon = msg.get("anon_id")
    if anon:
        return f"anon:{anon}"
    return None


STAY_ACTION = {"action": "move", "message": "", "params": {"direction": "stay"}}

# Cross-worker bg-loop ownership lock TTL (seconds). Refreshed at TTL/2 by the loop
# while alive; on worker SIGKILL the key naturally expires within this window so a
# different worker (or a fresh boot) can re-elect ownership.
BG_OWNER_TTL = 60

# Anonymous browser-session heartbeat TTL (seconds). The browser refreshes the
# anon_alive key over the dashboard WS every HEARTBEAT_INTERVAL_MS (5s); if no
# heartbeat arrives within ANON_HEARTBEAT_TTL the key expires and any bg loops
# spawned for that anon_id exit on their next poll. Set generously so a brief
# network blip / paused tab doesn't kill a guest's agents — abandoned tabs
# still clean up within a minute.
ANON_HEARTBEAT_TTL = 60


async def _publish_anthro_error(
    redis,
    *,
    target_user_id: str | None,
    target_session_token: str | None,
    operation: str,
    reason: str = "missing_api_key",
) -> None:
    """Publish a precondition-fail anthropologist event so the dashboard can
    render a targeted toast for the requester. Mirrors the same payload shape
    the anthropologist server publishes for its own missing-key paths."""
    if not redis:
        return
    if not target_user_id and not target_session_token:
        return
    payload = {
        "type": "anthro_error",
        "target_user_id": target_user_id,
        "target_session_token": target_session_token,
        "operation": operation,
        "reason": reason,
    }
    try:
        await redis.publish(CHANNEL_PUSH, json.dumps(payload))
    except Exception as e:
        logger.warning("anthro_error publish failed: %s", e)


async def _log_api_error(redis, agent_tag: str, error_type: str, message: str) -> None:
    """Append an API error entry to {exp_path}/api_errors.jsonl if an experiment is running."""
    if not redis:
        return
    try:
        cfg_raw = await redis.get(KEY_EXP_CONFIG)
        if not cfg_raw:
            return
        exp_path = Path(json.loads(cfg_raw).get("exp_path", ""))
        if not exp_path or not exp_path.exists():
            return
        entry = json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent_tag": agent_tag,
                "error_type": error_type,
                "message": message,
            }
        )
        with open(exp_path / "api_errors.jsonl", "a") as f:
            f.write(entry + "\n")
    except Exception as e:
        logger.warning("[API] failed to write api_errors.jsonl: %s", e)


def parse_agent_action(text: str, available_actions: dict) -> dict:
    """Extract a JSON action dict from an LLM response text. Returns stay on failure."""
    try:
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
        json_str = m.group(1) if m else None
        if not json_str:
            m = re.search(r"\{.*\}", text, re.S)
            json_str = m.group(0) if m else None
        if not json_str:
            return {**STAY_ACTION, "source": "default", "default_reason": "no_json"}
        try:
            obj = json.loads(json_str)
        except json.JSONDecodeError:
            json_str = re.sub(
                r"([{\s,])([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', json_str
            )
            json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
            obj = json.loads(json_str)
        data = {k.lower(): v for k, v in obj.items()}
        action = data.get("action", "move")
        if available_actions and action not in available_actions:
            return {
                **STAY_ACTION,
                "source": "default",
                "default_reason": "invalid_action",
            }
        parsed = {
            "action": action,
            "message": data.get("message", ""),
            "params": data.get("params", {}),
            "internal_memory": data.get("internal_memory", ""),
            "source": "llm",
        }
        return parsed
    except Exception:
        return {**STAY_ACTION, "source": "default", "default_reason": "parse_error"}


# ------------------------------------------------------------------
# App factory
# ------------------------------------------------------------------


def create_app(
    dashboard_manager=None,
    host: str = "localhost",
    port: int = 8765,
    ssl: bool = False,
    testing: bool = False,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        init_redis_pool()
        redis = await get_redis()
        app.state.redis = redis
        app.state.registry = RedisAgentRegistry(redis)
        app.state.session_registry = RedisDashboardSessionRegistry(redis)
        app.state.bg_llm_tasks = {}
        app.state.shutting_down = False
        # Per-worker cache of the runner's get_config response (see
        # _get_cached_config) so dashboard connects don't each round-trip to the
        # runner — that per-connect round-trip is what exhausts Redis's client
        # pool under a connect storm.
        app.state.config_cache = None
        app.state.config_lock = asyncio.Lock()
        # Distinguish this worker among siblings — used to claim ownership of bg
        # loops via SETNX on key_bg_owner(tag).
        app.state.worker_id = f"{os.getpid()}-{secrets.token_hex(4)}"
        if dashboard_manager is not None:
            dashboard_manager.set_redis(redis)
        await create_db_and_tables()
        # Restart background LLM loops that were persisted before last shutdown.
        # Under multi-worker each persisted ogw:bg:{tag} key will be claimed by
        # exactly one worker via SETNX on key_bg_owner(tag).
        cursor = 0
        while True:
            cursor, keys = await redis.scan(cursor, match="ogw:bg:*", count=100)
            for key in keys:
                raw = await redis.get(key)
                if not raw:
                    continue
                try:
                    info = json.loads(raw)
                    agent_tag = key.split(":", 2)[2]
                    if "user_id" not in info:
                        # Anonymous record — its API key is in Redis with TTL
                        # tied to the heartbeat, so by the time we restart
                        # there's nothing to resume from. Drop the (likely
                        # stale) record entirely.
                        await redis.delete(key)
                        continue
                    won = await redis.set(
                        key_bg_owner(agent_tag),
                        app.state.worker_id,
                        nx=True,
                        ex=BG_OWNER_TTL,
                    )
                    if not won:
                        continue
                    user = await get_user_by_id(info["user_id"])
                    if user and user.encrypted_api_key:
                        api_key = decrypt_api_key(user.encrypted_api_key)
                        t = asyncio.create_task(
                            server_llm_loop(agent_tag, api_key, info["model"])
                        )
                        app.state.bg_llm_tasks[agent_tag] = t
                        logger.info(
                            "[API] Resumed background LLM loop for %s (worker %s)",
                            agent_tag,
                            app.state.worker_id,
                        )
                    else:
                        # User gone or no key — drop ownership and the persisted record.
                        await redis.delete(key_bg_owner(agent_tag), key)
                except Exception as e:
                    logger.warning("[API] Failed to resume bg loop for %s: %s", key, e)
            if cursor == 0:
                break
        subscriber = RedisSubscriber(dashboard_manager)
        task = asyncio.create_task(subscriber.run())
        logger.info("[API] Redis subscriber started")
        cleanup_task = asyncio.create_task(cleanup_unverified_accounts())
        logger.info("[API] Unverified-account cleanup task started (TTL: 48 h)")
        child_agent_task = asyncio.create_task(_watch_child_agents())
        logger.info("[API] Child-agent watcher started")
        yield
        app.state.shutting_down = True
        if dashboard_manager is not None:
            await dashboard_manager.release_observer_count()
        bg = list(app.state.bg_llm_tasks.values())
        for t in bg:
            t.cancel()
        if bg:
            await asyncio.gather(*bg, return_exceptions=True)
        for t in (task, cleanup_task, child_agent_task):
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
        await close_redis_pool()

    session_secret = os.environ.get("SESSION_SECRET", "")
    if not session_secret:
        raise ValueError(
            "SESSION_SECRET is not set. "
            'Generate one with: python -c "import secrets; print(secrets.token_hex(32))" '
            "and add it to your .env file."
        )

    app = FastAPI(
        title="TerraLingua Remote Agent API",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.mount("/static", NoCacheStaticFiles(directory=_STATIC_DIR), name="static")

    async def get_user_by_id(user_id_str: str) -> User | None:
        try:
            uid = uuid.UUID(user_id_str)
            async with async_session_maker() as session:
                result = await session.execute(select(User).where(User.id == uid))
                return result.scalar_one_or_none()
        except Exception:
            return None

    async def cleanup_unverified_accounts() -> None:
        """Delete accounts that were never verified within UNVERIFIED_TTL_HOURS. Runs hourly."""
        UNVERIFIED_TTL_HOURS = 48
        while True:
            await asyncio.sleep(3600)
            try:
                cutoff = datetime.now(timezone.utc) - timedelta(
                    hours=UNVERIFIED_TTL_HOURS
                )
                async with async_session_maker() as session:
                    result = await session.execute(
                        select(User).where(
                            User.is_verified == False,  # noqa: E712
                            User.created_at.isnot(None),
                            User.created_at < cutoff,
                        )
                    )
                    expired = result.scalars().all()
                    for user in expired:
                        logger.info(
                            "[API] Removing expired unverified account: %s", user.email
                        )
                        await session.delete(user)
                    if expired:
                        await session.commit()
                        logger.info(
                            "[API] Purged %d expired unverified account(s)",
                            len(expired),
                        )
            except Exception as e:
                logger.warning("[API] cleanup_unverified_accounts error: %s", e)

    async def server_llm_loop(
        agent_tag: str,
        api_key: str,
        model: str,
        anon_id: str | None = None,
    ) -> None:
        """Drive an agent's obs→LLM→action cycle from the dashboard server process.

        anon_id (optional): if set, the loop polls key_anon_alive(anon_id) every
        ANON_HEARTBEAT_TTL/3 seconds and exits when it expires (browser closed).
        Logged-in agents pass anon_id=None and run until explicitly killed.
        """
        provider = "anthropic" if model.startswith("claude") else "openai"
        client = AgentClient(provider=provider, api_key=api_key)
        r = await get_redis()
        pubsub = r.pubsub()
        await pubsub.subscribe(obs_channel(agent_tag))
        await pubsub.subscribe(CHANNEL_AGENT_EVENT)
        # Keep the agent in the connected set so /auth/my-agents returns it.
        await r.sadd(KEY_CONNECTED_AGENTS, agent_tag)
        await r.publish(
            CHANNEL_AGENT_EVENT, json.dumps({"event": "connect", "tag": agent_tag})
        )

        # exit_event is set by any background-task watcher (owner lock lost,
        # anon heartbeat expired) to break us out of the main listen loop.
        exit_event = asyncio.Event()

        async def _refresh_owner() -> None:
            try:
                while True:
                    await asyncio.sleep(BG_OWNER_TTL // 2)
                    try:
                        await r.expire(key_bg_owner(agent_tag), BG_OWNER_TTL)
                        current = await r.get(key_bg_owner(agent_tag))
                        if current != app.state.worker_id:
                            logger.warning(
                                "[server_llm] owner lock for %s lost (now %s); exiting",
                                agent_tag,
                                current,
                            )
                            exit_event.set()
                            return
                    except Exception as e:
                        logger.warning(
                            "[server_llm] owner refresh error for %s: %s", agent_tag, e
                        )
            except asyncio.CancelledError:
                return

        async def _check_anon_heartbeat() -> None:
            if anon_id is None:
                return  # never spawned in this case; narrows type for the checker
            heartbeat_key = key_anon_alive(anon_id)
            try:
                while True:
                    await asyncio.sleep(max(2, ANON_HEARTBEAT_TTL // 3))
                    try:
                        if not await r.exists(heartbeat_key):
                            logger.info(
                                "[server_llm] anon heartbeat for %s expired; exiting %s",
                                anon_id,
                                agent_tag,
                            )
                            exit_event.set()
                            return
                    except Exception as e:
                        logger.warning("[server_llm] anon heartbeat check error: %s", e)
            except asyncio.CancelledError:
                return

        owner_refresh = asyncio.create_task(_refresh_owner())
        anon_check = asyncio.create_task(_check_anon_heartbeat()) if anon_id else None
        try:
            async for msg in pubsub.listen():
                if exit_event.is_set():
                    break
                if msg["type"] != "message":
                    continue
                if msg["channel"] == CHANNEL_AGENT_EVENT:
                    try:
                        event = json.loads(msg["data"])
                        if event.get("tag") == agent_tag and event.get("event") in (
                            "died",
                            "kill_bg",
                        ):
                            break
                    except Exception:
                        pass
                    continue
                try:
                    payload = json.loads(msg["data"])
                    if payload.get("step") is None:
                        continue
                    messages = [
                        {"role": "system", "content": payload["system_prompt"]},
                        {"role": "user", "content": payload["user_prompt"]},
                    ]
                    resp = await client.get_response_async(
                        messages, {"model": model, "max_tokens": 1024}
                    )
                    action = parse_agent_action(
                        resp.content, payload.get("available_actions", {})
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    error_type = _llm_error_type(e)
                    if error_type:
                        logger.error(
                            "[server_llm] %s for %s: %s", error_type, agent_tag, e
                        )
                        await _log_api_error(r, agent_tag, error_type, str(e))
                        await r.publish(
                            CHANNEL_PUSH,
                            json.dumps(
                                {
                                    "type": "agent_api_error",
                                    "agent_tag": agent_tag,
                                    "error_type": error_type,
                                    "error_message": str(e),
                                }
                            ),
                        )
                    else:
                        logger.warning("[server_llm] %s: %s", agent_tag, e)
                    action = {
                        **STAY_ACTION,
                        "source": "default",
                        "default_reason": "api_error",
                    }
                await r.publish(action_channel(agent_tag), json.dumps(action))
        except asyncio.CancelledError:
            pass
        finally:
            owner_refresh.cancel()
            try:
                await owner_refresh
            except (asyncio.CancelledError, Exception):
                pass
            if anon_check is not None:
                anon_check.cancel()
                try:
                    await anon_check
                except (asyncio.CancelledError, Exception):
                    pass
            await pubsub.aclose()
            # Always release our ownership claim so a future _activate_server_llm
            # (whether on this worker or another) can re-elect cleanly. Only
            # delete the key if it still belongs to us — avoids stomping on a
            # newer owner that already took over after lock expiry.
            try:
                current = await r.get(key_bg_owner(agent_tag))
                if current == app.state.worker_id:
                    await r.delete(key_bg_owner(agent_tag))
            except Exception:
                pass
            await r.aclose()
            app.state.bg_llm_tasks.pop(agent_tag, None)
            # On graceful shutdown: leave ogw:bg and KEY_CONNECTED_AGENTS intact so
            # the loop restarts on next boot.
            # On any other exit (explicit kill, connection drop): clean up fully.
            if not app.state.shutting_down:
                redis = app.state.redis
                if redis is not None:
                    await redis.delete(f"ogw:bg:{agent_tag}")
                    await redis.srem(KEY_CONNECTED_AGENTS, agent_tag)
                    await redis.publish(
                        CHANNEL_AGENT_EVENT,
                        json.dumps({"event": "disconnect", "tag": agent_tag}),
                    )

    async def _activate_server_llm(
        state,
        agent_tag: str,
        user_id: str | None,
        model: str,
        api_key: str,
        anon_id: str | None = None,
    ) -> bool:
        """Persist (if applicable) and start a server-side LLM loop for an agent.

        Under multi-worker the spawn is gated by SETNX on key_bg_owner(tag): every
        worker may call this for the same agent (e.g. via _watch_child_agents),
        but only the worker that wins the lock spawns the loop locally.

        For logged-in users we persist {user_id, model} under ogw:bg:{tag} so the
        lifespan resume scan can re-spawn the loop after a restart (the API key
        is decryptable from the user's DB row).

        For anonymous users we skip the persistence step — the API key only
        lives in this worker's RAM, so a restart cannot resume the loop. The
        loop is also gated on a heartbeat key that the browser refreshes over
        the dashboard WS; the loop exits when the user closes their tabs.
        """
        if state.redis is None:
            return False
        if user_id is not None:
            # Logged-in: persist forever so the lifespan resume scan can
            # re-spawn after a dashboard server restart (key decryptable from DB).
            await state.redis.set(
                f"ogw:bg:{agent_tag}",
                json.dumps({"user_id": user_id, "model": model}),
            )
        elif anon_id is not None:
            # Anonymous: persist transiently so _watch_child_agents can map
            # parent_tag → anon_id and look up the encrypted API key. TTL
            # tracks the heartbeat — refreshed on each cmd:"heartbeat" arrival.
            await state.redis.set(
                f"ogw:bg:{agent_tag}",
                json.dumps({"anon_id": anon_id, "model": model}),
                ex=ANON_HEARTBEAT_TTL,
            )
        won = await state.redis.set(
            key_bg_owner(agent_tag),
            state.worker_id,
            nx=True,
            ex=BG_OWNER_TTL,
        )
        if not won:
            return False
        if agent_tag in state.bg_llm_tasks:
            return False
        t = asyncio.create_task(
            server_llm_loop(agent_tag, api_key, model, anon_id=anon_id)
        )
        state.bg_llm_tasks[agent_tag] = t
        logger.info(
            "[API] Started server LLM loop for %s (worker %s, %s)",
            agent_tag,
            state.worker_id,
            "anon" if anon_id else f"user={user_id}",
        )
        return True

    async def _watch_child_agents() -> None:
        """Start server-side LLM loops for offspring of server-llm agents."""
        while True:
            try:
                r = await get_redis()
                pubsub = r.pubsub()
                await pubsub.subscribe(CHANNEL_PUSH)
                try:
                    async for msg in pubsub.listen():
                        if msg["type"] != "message":
                            continue
                        try:
                            data = json.loads(msg["data"])
                        except Exception:
                            continue
                        if data.get("type") != "child_agent":
                            continue
                        child_tag = data.get("agent_tag")
                        parent_tag = data.get("parent_tag")
                        if not child_tag or not parent_tag:
                            continue
                        bg_raw = await r.get(f"ogw:bg:{parent_tag}")
                        if not bg_raw:
                            continue
                        try:
                            bg_info = json.loads(bg_raw)
                            model = bg_info.get("model") or "claude-sonnet-4-5"
                        except Exception:
                            continue
                        if "user_id" in bg_info:
                            # Logged-in parent — child inherits the user's
                            # encrypted DB key.
                            user_id = bg_info["user_id"]
                            user = await get_user_by_id(user_id)
                            if not user or not user.encrypted_api_key:
                                continue
                            api_key = decrypt_api_key(user.encrypted_api_key)
                            await app.state.registry.set_owner(child_tag, user_id)
                            # _activate_server_llm logs "Started …" only on the
                            # SETNX-winning worker; the others no-op silently.
                            await _activate_server_llm(
                                app.state, child_tag, user_id, model, api_key
                            )
                        elif "anon_id" in bg_info:
                            # Anonymous parent — child inherits the anon_id and
                            # the Fernet-encrypted API key stashed in Redis.
                            # Lifetime is gated by the same heartbeat as the
                            # parent's, so when the user closes their tab the
                            # whole family expires together.
                            anon_id = bg_info["anon_id"]
                            encrypted = await r.get(key_anon_apikey(anon_id))
                            if not encrypted:
                                continue
                            try:
                                api_key = decrypt_api_key(encrypted)
                            except Exception as e:
                                logger.warning(
                                    "[API] Could not decrypt anon api key for %s: %s",
                                    anon_id,
                                    e,
                                )
                                continue
                            # Track the child under the parent's anon_id so
                            # /auth/my-agents and the next heartbeat refresh
                            # cover it too.
                            await r.sadd(f"ogw:anon:{anon_id}:agents", child_tag)
                            await r.expire(
                                f"ogw:anon:{anon_id}:agents", ANON_HEARTBEAT_TTL
                            )
                            # _activate_server_llm logs "Started …" only on the
                            # SETNX-winning worker; the others no-op silently.
                            await _activate_server_llm(
                                app.state,
                                child_tag,
                                None,
                                model,
                                api_key,
                                anon_id=anon_id,
                            )
                        else:
                            continue
                finally:
                    await pubsub.aclose()
                    await r.aclose()
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.warning("[API] _watch_child_agents error: %s", e)
                await asyncio.sleep(5)

    app.state.dashboard_manager = dashboard_manager
    app.state.base_url = f"{'wss' if ssl else 'ws'}://{host}:{port}"

    # Anonymous session tracking (stable signed cookie per browser)
    app.add_middleware(SessionMiddleware, secret_key=session_secret)
    if testing:
        app.add_middleware(_TestingGate)

    @app.middleware("http")
    async def _security_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "connect-src 'self' wss: https://api.anthropic.com https://api.openai.com; "
            "img-src 'self' data: blob:; "
            "font-src 'self' https://fonts.gstatic.com;"
        )
        if request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    # Rate limiting (brute-force protection on login)
    limiter = Limiter(
        key_func=get_remote_address,
        storage_uri=os.environ.get("REDIS_URL", "redis://localhost:6379"),
    )
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

    # fastapi-users routers (users/me only — register is custom for rate limiting)
    app.include_router(
        fastapi_users.get_users_router(UserRead, UserUpdate),
        prefix="/users",
        tags=["users"],
    )
    app.include_router(
        fastapi_users.get_verify_router(UserRead),
        prefix="/auth",
        tags=["auth"],
    )

    @app.post("/auth/forgot-password", status_code=202, tags=["auth"])
    @limiter.limit("3/minute")
    async def forgot_password(
        request: Request,
        body: ForgotPasswordRequest,
        user_manager=Depends(get_user_manager),
    ):
        try:
            user = await user_manager.get_by_email(body.email)
            await user_manager.forgot_password(user, request)
        except Exception:
            pass  # always 202 — no email enumeration
        return Response(status_code=202)

    @app.post("/auth/reset-password", tags=["auth"])
    @limiter.limit("5/minute")
    async def reset_password(
        request: Request,
        body: ResetPasswordRequest,
        user_manager=Depends(get_user_manager),
    ):

        try:
            await user_manager.reset_password(body.token, body.password, request)
        except (InvalidResetPasswordToken, UserNotExists):
            raise HTTPException(status_code=400, detail="RESET_PASSWORD_BAD_TOKEN")
        except InvalidPasswordException as e:
            raise HTTPException(status_code=422, detail=str(e.reason))

    # Login / logout — custom so we can apply the rate-limit decorator
    @app.post("/auth/jwt/login", tags=["auth"])
    @limiter.limit("10/minute")
    async def auth_login(
        request: Request,
        credentials: OAuth2PasswordRequestForm = Depends(),
        user_manager=Depends(get_user_manager),
        strategy=Depends(auth_backend.get_strategy),
    ):
        user = await user_manager.authenticate(credentials)
        if user is None or not user.is_active:
            raise HTTPException(status_code=400, detail="LOGIN_BAD_CREDENTIALS")
        return await auth_backend.login(strategy, user)

    @app.post("/auth/jwt/logout", tags=["auth"])
    async def auth_logout(
        request: Request,
        user: User = Depends(current_user),
        strategy=Depends(auth_backend.get_strategy),
    ):
        token = request.cookies.get("ogw_user_session", "")
        return await auth_backend.logout(strategy, user, token)

    @app.get("/auto-login", include_in_schema=False)
    async def auto_login():
        """Demo shortcut: sets the session cookie for the pre-seeded demo user and redirects to /dashboard.
        Only active when DEMO_MODE=true. Returns 404 otherwise."""
        if os.environ.get("DEMO_MODE", "").lower() != "true":
            raise HTTPException(status_code=404)

        demo_email = os.environ.get("DEMO_USER_EMAIL", "demo@demo.com")

        async with async_session_maker() as session:
            user = await session.scalar(select(User).where(User.email == demo_email))

        if user is None or not user.is_active or not user.is_verified:
            raise HTTPException(
                status_code=503,
                detail="Demo user not found — run demo/create_demo_db.py first",
            )

        async with async_session_maker() as token_session:
            access_token_db = SQLAlchemyAccessTokenDatabase(token_session, AccessToken)
            strategy = DatabaseStrategy(access_token_db, lifetime_seconds=86400)
            token = await strategy.write_token(user)

        response = RedirectResponse("/dashboard", status_code=302)
        response.set_cookie(
            key="ogw_user_session",
            value=token,
            max_age=86400,
            path="/",
            httponly=True,
            samesite="lax",
            secure=cookie_secure_default(),
        )
        return response

    @app.post("/auth/register", response_model=UserRead, status_code=201, tags=["auth"])
    @limiter.limit("5/minute")
    async def auth_register(
        request: Request,
        user_create: UserCreate,
        user_manager=Depends(get_user_manager),
    ):

        try:
            user = await user_manager.create(user_create, safe=True, request=request)
            return UserRead.model_validate(user)
        except UserAlreadyExists:
            raise HTTPException(status_code=400, detail="REGISTER_USER_ALREADY_EXISTS")
        except InvalidPasswordException as e:
            raise HTTPException(status_code=422, detail=str(e.reason))

    @app.get("/verify-email", include_in_schema=False)
    async def verify_email_page(token: str = ""):
        dest = f"/login?verify_token={token}" if token else "/login"
        return RedirectResponse(dest, status_code=302)

    # ------------------------------------------------------------------
    # Dashboard WebSocket  (must be before /ws/{agent_tag})
    # ------------------------------------------------------------------

    @app.websocket("/ws/dashboard")
    async def dashboard_ws(websocket: WebSocket, token: str = ""):
        sr = app.state.session_registry
        if sr is None or not await sr.validate_for_connect(token):
            await websocket.accept()
            await websocket.close(code=4001, reason="Invalid or expired token")
            return
        user_id = await sr.get_user_id(token) if sr else None

        # Mirror the logged-in user's encrypted API key into a session-scoped
        # Redis entry so postmortem/phylogeny handlers can read it without
        # touching the DB. Anon users populate this via the set_api_key WS
        # command.
        if user_id:
            _u = await get_user_by_id(user_id)
            if _u and _u.encrypted_api_key:
                await sr.set_session_api_key(token, _u.encrypted_api_key)

        dm = app.state.dashboard_manager
        await dm.on_connect(websocket)

        pending_tasks: set[asyncio.Task] = set()
        try:
            redis = app.state.redis
            allow_human = True
            if redis is not None:
                cfg = await _get_cached_config(app.state)
                if cfg.get("ok"):
                    allow_human = cfg.get("data", {}).get("allow_human_agents", True)
                else:
                    logger.warning(
                        "[API] dashboard init: runner unavailable (%s)",
                        cfg.get("error"),
                    )
            await websocket.send_json(
                {"type": "config", "allow_human_agents": allow_human}
            )

            # Send snapshot of latest frame stored in Redis
            initial_frame = None
            if redis is not None:
                raw = await redis.get(KEY_LAST_FRAME)
                initial_frame = json.loads(raw) if raw else None
            await dm.send_initial_frame(websocket, initial_frame)

            # Send field notes history, obituaries, and live annotations snapshots if available
            if redis is not None:
                exp_name = None
                raw = await redis.get(KEY_EXP_CONFIG)
                if raw:
                    cfg = json.loads(raw)
                    exp_name = cfg.get("exp_name")
                    raw_fn = (
                        await redis.get(key_field_notes_all(exp_name))
                        if exp_name
                        else None
                    )
                    if raw_fn:
                        await websocket.send_json(
                            {
                                "type": "field_notes_snapshot",
                                "events": json.loads(raw_fn),
                            }
                        )
                    ann_dir = Path(cfg["exp_path"]) / "annotations"
                    if ann_dir.is_dir():
                        skip = {"anthropologist_notes.json"}
                        tags = [
                            f.stem for f in ann_dir.glob("*.json") if f.name not in skip
                        ]
                        if tags:
                            await websocket.send_json(
                                {"type": "obituaries_index", "tags": tags}
                            )
                if exp_name:
                    raw = await redis.get(key_live_annotations(exp_name))
                    if raw:
                        await websocket.send_json(
                            {"type": "live_annotations", "annotations": json.loads(raw)}
                        )
                raw = await redis.get(KEY_RUNNER_STATUS)
                if raw:
                    await websocket.send_json({"type": "runner_status", "status": raw})
                if exp_name:
                    raw = await redis.get(key_artifact_classifications(exp_name))
                    if raw:
                        await websocket.send_json(json.loads(raw))

            while True:
                raw = await websocket.receive_text()
                task = asyncio.create_task(
                    _handle_dashboard_message(websocket, raw, app.state, user_id, token)
                )
                pending_tasks.add(task)
                task.add_done_callback(pending_tasks.discard)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            for task in pending_tasks:
                task.cancel()
            await dm.on_disconnect(websocket)
            await sr.release(token)

    async def _handle_dashboard_message(
        websocket: WebSocket,
        raw: str,
        state,
        user_id: str | None = None,
        session_token: str | None = None,
    ) -> None:
        """Dispatch a dashboard WS message.

        Most commands are request/response with a 'req' id; a few are
        fire-and-forget notifications (heartbeat) that have no req field.
        """
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return
        cmd = msg.get("cmd")
        if not cmd:
            return

        # Fire-and-forget notifications — handled before the req_id gate so they
        # don't need to fake a request id.
        if cmd == "heartbeat":
            anon_id = msg.get("anon_id")
            if anon_id and not user_id and state.redis is not None:
                # alive: re-create the key so it survives even if it briefly
                # expired (the loop uses EXISTS, so a missed beat exits it).
                await state.redis.set(
                    key_anon_alive(anon_id), "1", ex=ANON_HEARTBEAT_TTL
                )
                # apikey: just bump TTL — only valid if it hasn't been gone too
                # long. If it has, the user has effectively been disconnected
                # and the loops have already exited.
                await state.redis.expire(key_anon_apikey(anon_id), ANON_HEARTBEAT_TTL)
                # Refresh the family-tracking SET first so a slightly delayed
                # heartbeat doesn't read an empty set, then refresh each
                # member's bg record (used by _watch_child_agents to resolve
                # parent_tag → anon_id for spawn).
                await state.redis.expire(
                    f"ogw:anon:{anon_id}:agents", ANON_HEARTBEAT_TTL
                )
                tags = await state.redis.smembers(f"ogw:anon:{anon_id}:agents")
                for tag in tags:
                    await state.redis.expire(f"ogw:bg:{tag}", ANON_HEARTBEAT_TTL)
            return

        req_id = msg.get("req")
        if not req_id:
            return

        async def respond(data=None, error=None):
            await websocket.send_json(
                {
                    "res": req_id,
                    "ok": error is None,
                    **({"data": data} if data is not None else {}),
                    **({"error": error} if error is not None else {}),
                }
            )

        try:
            if cmd == "request_artifact_phylogeny":
                artifact_name = msg.get("artifact_name", "")
                if not artifact_name:
                    await respond(error="artifact_name is required")
                    return
                redis = state.redis
                if redis is None:
                    await respond(
                        error="Phylogeny queue unavailable — is the anthropologist server running?"
                    )
                    return
                # Phylogeny is requester-pays. Pull the requester's
                # session-scoped encrypted key; reject + notify if none.
                requester_key = None
                if session_token and state.session_registry is not None:
                    requester_key = await state.session_registry.get_session_api_key(
                        session_token
                    )
                if not requester_key:
                    await _publish_anthro_error(
                        redis,
                        target_user_id=user_id,
                        target_session_token=session_token,
                        operation="artifact_phylogeny",
                    )
                    await respond(error="API key not set. Open settings to add one.")
                    return
                cfg_raw = await redis.get(KEY_EXP_CONFIG)
                exp_name = json.loads(cfg_raw).get("exp_name") if cfg_raw else None
                if exp_name:
                    await redis.rpush(
                        key_phylogeny_queue(exp_name),
                        json.dumps(
                            {
                                "name": artifact_name,
                                "encrypted_api_key": requester_key,
                                "user_id": user_id,
                                "session_token": session_token,
                            }
                        ),
                    )
                    logger.info(
                        f"Queued on-demand phylogeny for artifact '{artifact_name}'"
                    )
                await respond({"queued": True})
            elif cmd == "request_postmortem":
                agent_tag = msg.get("agent_tag", "")
                if not agent_tag:
                    await respond(error="agent_tag is required")
                    return
                agent_name = msg.get("agent_name") or None
                label = f"{agent_name}({agent_tag})" if agent_name else agent_tag
                redis = state.redis
                logger.info(
                    f"request_postmortem received for {label} (redis={'yes' if redis else 'no'})"
                )
                if redis is None:
                    await respond(
                        error="Postmortem queue unavailable — is the anthropologist server running?"
                    )
                    return
                # Postmortem is requester-pays. Pull the requester's
                # session-scoped encrypted key; reject + notify if none.
                requester_key = None
                if session_token and state.session_registry is not None:
                    requester_key = await state.session_registry.get_session_api_key(
                        session_token
                    )
                if not requester_key:
                    await _publish_anthro_error(
                        redis,
                        target_user_id=user_id,
                        target_session_token=session_token,
                        operation="postmortem",
                    )
                    await respond(error="API key not set. Open settings to add one.")
                    return
                cfg_raw = await redis.get(KEY_EXP_CONFIG)
                exp_name = json.loads(cfg_raw).get("exp_name") if cfg_raw else None
                queued = False
                if exp_name:
                    await redis.rpush(
                        key_postmortem_queue(exp_name),
                        json.dumps(
                            {
                                "tag": agent_tag,
                                "name": agent_name,
                                "encrypted_api_key": requester_key,
                                "user_id": user_id,
                                "session_token": session_token,
                            }
                        ),
                    )
                    logger.info(f"Pushed {label} to postmortem queue")
                    queued = True
                else:
                    logger.warning(
                        "request_postmortem: no exp_name in config — not queued"
                    )
                if queued:
                    await respond({"queued": True})
                else:
                    await respond(
                        error="Postmortem queue unavailable — is the anthropologist server running?"
                    )
            elif cmd == "fetch_obituary":
                agent_tag = msg.get("agent_tag", "")
                if not agent_tag:
                    await respond(error="agent_tag is required")
                    return
                if not _valid_agent_tag(agent_tag):
                    await respond(error="invalid agent_tag")
                    return
                redis = state.redis
                ann_path = None
                if redis is not None:
                    cfg_raw = await redis.get(KEY_EXP_CONFIG)
                    if cfg_raw:
                        cfg = json.loads(cfg_raw)
                        ann_path = (
                            Path(cfg["exp_path"]) / "annotations" / f"{agent_tag}.json"
                        )
                        exp_path_obj = Path(cfg["exp_path"])
                if ann_path is None or not ann_path.exists():
                    await respond(error="obituary not found")
                    return
                try:
                    annotation = json.loads(ann_path.read_text())
                    vitals = _agent_vitals(agent_tag, exp_path_obj)
                    await websocket.send_json(
                        {
                            "type": "agent_postmortem",
                            "agent_tag": agent_tag,
                            "annotation": annotation,
                            **vitals,
                        }
                    )
                    await respond({"ok": True})
                except Exception as e:
                    logger.exception("[API] fetch_obituary failed: %s", e)
                    await respond(error="Failed to fetch obituary")
            elif cmd == "get_api_key_provider":
                provider = None
                has_key = False
                if user_id:
                    u = await get_user_by_id(user_id)
                    if u and u.encrypted_api_key:
                        has_key = True
                        try:
                            key = decrypt_api_key(u.encrypted_api_key)
                            if key.startswith("sk-ant-"):
                                provider = "anthropic"
                            elif key.startswith("sk-"):
                                provider = "openai"
                        except Exception:
                            pass
                await respond({"provider": provider, "has_key": has_key})
            elif cmd == "set_api_key":
                # Save / rotate the user's API key. Single command path for
                # both registered and anonymous users; the server branches on
                # whether user_id is set.
                #   Registered: writes to DB, mirrors to every AgentRecord
                #               this user owns, mirrors to session entry.
                #               Superuser: also writes the admin fallback.
                #   Anonymous:  writes to the session entry only.
                # Plaintext key is received over TLS, encrypted server-side
                # immediately, and never persisted unencrypted.
                plaintext = (msg.get("api_key") or "").strip()
                if not plaintext:
                    await respond(error="API key is required.")
                    return
                try:
                    encrypted = encrypt_api_key(plaintext)
                except Exception as e:
                    logger.exception("[API] set_api_key encrypt failed: %s", e)
                    await respond(error="Failed to encrypt API key")
                    return

                if user_id:
                    # Persist to DB.
                    target_user = None
                    async with async_session_maker() as session:
                        target_user = await session.get(User, uuid.UUID(user_id))
                        if target_user is None:
                            await respond(error="User not found")
                            return
                        target_user.encrypted_api_key = encrypted
                        session.add(target_user)
                        await session.commit()
                    # Mirror to all of this user's AgentRecords.
                    try:
                        owned = await state.registry.get_agents_for_user(user_id)
                        for rec in owned:
                            await state.registry.set_api_key(rec.agent_tag, encrypted)
                    except Exception as e:
                        logger.warning(
                            "[API] set_api_key: mirror to AgentRecords failed: %s", e
                        )
                    # Admin fallback (superuser only).
                    if target_user is not None and getattr(
                        target_user, "is_superuser", False
                    ):
                        try:
                            await state.redis.set(KEY_ANTHRO_ADMIN_API_KEY, encrypted)
                        except Exception as e:
                            logger.warning(
                                "[API] set_api_key: admin key write failed: %s", e
                            )

                # Mirror to session entry for anyone (registered + anon alike).
                if session_token and state.session_registry is not None:
                    try:
                        await state.session_registry.set_session_api_key(
                            session_token, encrypted
                        )
                    except Exception as e:
                        logger.warning(
                            "[API] set_api_key: session mirror failed: %s", e
                        )

                await respond({"ok": True})
            elif cmd == "register":
                redis = state.redis
                if redis is None:
                    await respond(error="Not connected to runner")
                    return
                is_human = msg.get("agent_type") == "remote_human"
                anon_id = msg.get("anon_id")

                # Resolve the API key we'll use to drive this agent's bg loop.
                # Logged-in users: read encrypted key from their account.
                # Anonymous: take the key from the registration message; it
                # lives only in this worker's RAM for the lifetime of the loop.
                # Also keep the Fernet-encrypted form around so we can stash it
                # on the AgentRecord (anthropologist reads from there for live
                # annotation).
                api_key: str | None = None
                encrypted_for_record: str | None = None
                if not is_human:
                    if user_id:
                        _reg_user = await get_user_by_id(user_id)
                        if not _reg_user or not _reg_user.encrypted_api_key:
                            await respond(
                                error="API key not set. Please save your API key in the settings."
                            )
                            return
                        encrypted_for_record = _reg_user.encrypted_api_key
                        api_key = decrypt_api_key(_reg_user.encrypted_api_key)
                    else:
                        api_key = (msg.get("api_key") or "").strip() or None
                        if not api_key:
                            await respond(error="API key is required.")
                            return
                        encrypted_for_record = encrypt_api_key(api_key)

                result = await _redis_cmd(redis, cmd, msg)
                if not result.get("ok", False):
                    await respond(error=result.get("error", "Runner error"))
                    return
                data = result.get("data", {})
                agent_tag = data.get("agent_tag")

                if not user_id and anon_id and agent_tag:
                    await redis.sadd(f"ogw:anon:{anon_id}:agents", agent_tag)
                    await redis.expire(f"ogw:anon:{anon_id}:agents", ANON_HEARTBEAT_TTL)
                    # Seed the heartbeat so the bg loop's first poll succeeds;
                    # the browser refreshes it over the dashboard WS.
                    await redis.set(key_anon_alive(anon_id), "1", ex=ANON_HEARTBEAT_TTL)
                    # Stash the encrypted API key in Redis so children of this
                    # anon parent can be spawned on any worker — encrypted with
                    # the same Fernet key used for DB-stored logged-in keys, and
                    # with the same TTL as the heartbeat (gone within
                    # ANON_HEARTBEAT_TTL of tab close).
                    if api_key:
                        await redis.set(
                            key_anon_apikey(anon_id),
                            encrypt_api_key(api_key),
                            ex=ANON_HEARTBEAT_TTL,
                        )
                if user_id and agent_tag:
                    await state.registry.set_owner(agent_tag, user_id)
                # Stash the encrypted API key on the AgentRecord so the
                # anthropologist can find it by agent_tag for live annotation
                # without needing DB access. None for human-controlled agents.
                if agent_tag and encrypted_for_record:
                    await state.registry.set_api_key(agent_tag, encrypted_for_record)

                if not is_human and agent_tag and api_key:
                    model = msg.get("model_info") or "claude-sonnet-4-5"
                    await _activate_server_llm(
                        state,
                        agent_tag,
                        user_id,
                        model,
                        api_key,
                        anon_id=anon_id if not user_id else None,
                    )
                await respond(data)
            elif cmd == "create_artifact":
                redis = state.redis
                if redis is None:
                    await respond(error="Not connected to runner")
                    return
                caller = _ws_caller_id(user_id, msg)
                if caller is None:
                    await respond(
                        error="Login or guest session required to create artifacts"
                    )
                    return
                if not (msg.get("name") or "").strip():
                    await respond(error="'name' is required")
                    return
                result = await _redis_cmd(redis, cmd, msg)
                if not result.get("ok", False):
                    await respond(error=result.get("error", "Runner error"))
                    return
                data = result.get("data") or {}
                # Key ownership on the runner-assigned final name, not the
                # requested one. The runner suffixes duplicates ("X" → "X_1")
                # and artifact names are unique forever, so the final name is
                # guaranteed unique and no hijack race is possible.
                final_name = data.get("name")
                if final_name:
                    await redis.set(_artifact_owner_key(final_name), caller)
                await respond(data)
            elif cmd in ("modify_artifact", "destroy_artifact"):
                redis = state.redis
                if redis is None:
                    await respond(error="Not connected to runner")
                    return
                caller = _ws_caller_id(user_id, msg)
                if caller is None:
                    await respond(error="Login or guest session required")
                    return
                name = (msg.get("name") or "").strip()
                if not name:
                    await respond(error="'name' is required")
                    return
                # Superusers can edit/destroy anything (matches the dashboard's
                # client-side check at artifacts.js:12).
                is_super = False
                if user_id:
                    _u = await get_user_by_id(user_id)
                    is_super = bool(_u and _u.is_superuser)
                if not is_super:
                    owner = await redis.get(_artifact_owner_key(name))
                    if owner is None:
                        await respond(error="Artifact not owned by you")
                        return
                    if isinstance(owner, bytes):
                        owner = owner.decode()
                    if owner != caller:
                        await respond(error="Artifact not owned by you")
                        return
                result = await _redis_cmd(redis, cmd, msg)
                if not result.get("ok", False):
                    await respond(error=result.get("error", "Runner error"))
                    return
                if cmd == "destroy_artifact":
                    await redis.delete(_artifact_owner_key(name))
                await respond(result.get("data"))
            elif cmd == "get_social_graph_history":
                try:
                    data = await get_social_graph_history(
                        state.redis,
                        offset=msg.get("offset", 0),
                        limit=msg.get("limit", 100),
                        run_id=msg.get("run_id"),
                        segment_id=msg.get("segment_id"),
                    )
                except ValueError as exc:
                    await respond(error=str(exc))
                else:
                    await respond(data)
            elif cmd in _WS_RUNNER_PASSTHROUGH:
                redis = state.redis
                if redis is None:
                    await respond(error="Not connected to runner")
                    return
                result = await _redis_cmd(redis, cmd, msg)
                if result.get("ok", False):
                    await respond(result.get("data"))
                else:
                    await respond(error=result.get("error", "Runner error"))
            else:
                logger.warning(
                    "[API] rejected unknown dashboard cmd '%s' from %s",
                    cmd,
                    _ws_caller_id(user_id, msg) or "<unidentified>",
                )
                await respond(error=f"Unknown command: {cmd}")

        except WebSocketDisconnect:
            raise
        except Exception as e:
            logger.exception("[API] dashboard command '%s' raised: %s", cmd, e)
            try:
                await respond(error="Internal server error")
            except Exception:
                pass  # best-effort: WebSocket may already be closed

    # ------------------------------------------------------------------
    # Agent WebSocket  (wildcard — must be after /ws/dashboard)
    # ------------------------------------------------------------------

    @app.websocket("/ws/{agent_tag}")
    async def websocket_endpoint(websocket: WebSocket, agent_tag: str, token: str = ""):
        record = await app.state.registry.get_by_token(token)
        if record is None or record.agent_tag != agent_tag:
            await websocket.accept()
            await websocket.close(code=4001, reason="Invalid token")
            return

        redis = app.state.redis
        if redis is None:
            await websocket.accept()
            await websocket.close(code=4001, reason="Not connected to runner")
            return

        await websocket.accept()
        await redis.sadd(KEY_CONNECTED_AGENTS, agent_tag)
        await redis.publish(
            CHANNEL_AGENT_EVENT, json.dumps({"event": "connect", "tag": agent_tag})
        )

        # In the unified design only human-controlled agents open this WS — LLM
        # agents are driven entirely by server_llm_loop subscribed to Redis. So
        # no bg-loop restart logic is needed here; the lifespan resume scan
        # already re-spawns persisted (logged-in) loops on dashboard server boot.

        async def obs_forwarder():
            backoff = 1.0
            while True:
                try:
                    r2 = await get_redis()
                    pubsub = r2.pubsub()
                    await pubsub.subscribe(obs_channel(agent_tag))
                    try:
                        async for msg in pubsub.listen():
                            if msg["type"] == "message":
                                await websocket.send_text(msg["data"])
                    finally:
                        await pubsub.aclose()
                        await r2.aclose()
                    backoff = 1.0
                except asyncio.CancelledError:
                    return
                except Exception as e:
                    logger.warning(
                        f"[API] obs_forwarder for {agent_tag} reconnecting: {e}"
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30.0)

        fwd_task = asyncio.create_task(obs_forwarder())
        try:
            while True:
                raw = await websocket.receive_text()
                await redis.publish(action_channel(agent_tag), raw)
        except WebSocketDisconnect:
            pass
        finally:
            fwd_task.cancel()
            # Publish "disconnect" so the UI connected-badge updates immediately.
            # Do NOT remove from KEY_CONNECTED_AGENTS here — the agent is still
            # alive in the sim and the browser may reconnect shortly (page
            # reload).  The runner removes dead agents from KEY_CONNECTED_AGENTS
            # when they actually die, keeping the set as "alive agents".
            # Background LLM loops handle their own cleanup when they stop. The
            # bg loop may be hosted on a different worker, so ask Redis instead
            # of checking the local bg_llm_tasks dict.
            if not await redis.exists(key_bg_owner(agent_tag)):
                await redis.publish(
                    CHANNEL_AGENT_EVENT,
                    json.dumps({"event": "disconnect", "tag": agent_tag}),
                )

    # ------------------------------------------------------------------
    # Login
    # ------------------------------------------------------------------

    _LOGIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <title>TerraLingua — Login</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{display:flex;align-items:center;justify-content:center;min-height:100vh;
         background:#0f1117;font-family:Inter,system-ui,sans-serif;color:#e2e8f0}}
    form{{background:#1a1d27;border:1px solid #2d3148;border-radius:10px;
          padding:2rem;width:100%;max-width:340px;display:flex;flex-direction:column;gap:1rem}}
    h1{{font-size:1.2rem;font-weight:600;text-align:center}}
    input[type=password]{{background:#0f1117;border:1px solid #2d3148;border-radius:6px;
                          color:#e2e8f0;font-size:0.9rem;padding:0.55rem 0.75rem;width:100%}}
    input[type=password]:focus{{outline:none;border-color:#6366f1}}
    button{{background:#6366f1;border:none;border-radius:6px;color:#fff;cursor:pointer;
            font-size:0.9rem;font-weight:500;padding:0.6rem;width:100%}}
    button:hover{{background:#4f51d8}}
    .err{{color:#f87171;font-size:0.8rem;text-align:center}}
  </style>
</head>
<body>
  <form method="post" action="/gate">
    <h1>🌿 TerraLingua</h1>
    {error}
    <input type="password" name="password" placeholder="Password" autofocus autocomplete="current-password"/>
    <button type="submit">Enter</button>
  </form>
</body>
</html>"""

    @app.get("/")
    async def root():
        return RedirectResponse("/login", status_code=302)

    @app.get("/gate", response_class=HTMLResponse)
    async def gate_page(error: str = ""):
        err_html = '<p class="err">Incorrect password.</p>' if error else ""
        return HTMLResponse(_LOGIN_HTML.format(error=err_html))

    @app.post("/gate")
    async def gate_submit(password: str = Form(...)):
        if DASHBOARD_PASSWORD and hmac.compare_digest(password, DASHBOARD_PASSWORD):
            resp = RedirectResponse("/login", status_code=302)
            resp.set_cookie(
                "dashboard_auth",
                _make_auth_cookie(),
                max_age=86400,
                httponly=True,
                samesite="lax",
                secure=cookie_secure_default(),
            )
            return resp
        return RedirectResponse("/gate?error=1", status_code=302)

    # ------------------------------------------------------------------
    # Utility endpoints
    # ------------------------------------------------------------------

    @app.get("/suggest_agent_name")
    async def suggest_agent_name(
        request: Request,
        existing: str = "",
        user: User | None = Depends(optional_current_user),
    ):
        if user is None and not _check_auth(request) and not _check_guest(request):
            raise HTTPException(status_code=401)
        taken = set(existing.split(",")) if existing else set()
        rw = RandomWord()
        for _ in range(500):
            name = (
                rw.word(include_parts_of_speech=["adjectives"]).capitalize()
                + rw.word(include_parts_of_speech=["nouns"]).capitalize()
            )
            if name not in taken:
                return {"name": name}
        return {"name": f"Agent{secrets.token_hex(3)}"}

    # ------------------------------------------------------------------
    # Dashboard UI
    # ------------------------------------------------------------------

    @app.get("/login", response_class=HTMLResponse)
    async def login_page():
        path = _STATIC_DIR / "account.html"
        if not path.exists():
            return HTMLResponse("<h1>Not found</h1>", status_code=404)
        return HTMLResponse(path.read_text(), headers={"Cache-Control": "no-cache"})

    @app.get("/api/models")
    async def api_models():
        """Public endpoint: canonical list of selectable models, grouped by provider."""
        return models_for_dashboard()

    @app.get("/api/stats")
    async def api_stats():
        """Public endpoint: lightweight world-state snapshot for the landing page."""
        agents, artifacts = None, None
        observers = 0
        try:
            redis = app.state.redis
            if redis is not None:
                raw = await redis.get(KEY_LAST_FRAME)
                if raw:
                    frame = json.loads(raw)
                    agents = len(frame.get("agents") or [])
                    artifacts = len(frame.get("artifacts") or [])
                # Cross-worker counter — sums dashboard connections across all
                # uvicorn worker processes. Best-effort under SIGKILL.
                raw_obs = await redis.get("ogw:dashboard_observers")
                observers = max(0, int(raw_obs)) if raw_obs else 0
        except Exception:
            pass
        return {"agents": agents, "artifacts": artifacts, "observers": observers}

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard(
        request: Request, user: User | None = Depends(optional_current_user)
    ):
        if user is None and not _check_auth(request) and not _check_guest(request):
            return RedirectResponse("/login", status_code=302)
        path = _STATIC_DIR / "dashboard.html"
        if not path.exists():
            return HTMLResponse("<h1>Dashboard not found</h1>", status_code=404)
        html = path.read_text()
        v = _static_assets_hash()
        html = re.sub(
            r'(/static/[^"\']+\.(?:css|js))(["\'])',
            lambda m: f"{m.group(1)}?v={v}{m.group(2)}",
            html,
        )
        if app.state.session_registry is not None:
            token = await app.state.session_registry.issue_token(
                user_id=str(user.id) if user else None
            )
            # Inject the session token as a meta tag (not an inline <script>)
            # so CSP can forbid 'unsafe-inline' on script-src.
            html = html.replace(
                "</head>",
                f'<meta name="dashboard-token" content="{_html.escape(token, quote=True)}">\n</head>',
            )
        return HTMLResponse(
            html,
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    # ------------------------------------------------------------------
    # Auth (custom endpoints — login/register/logout via fastapi-users routers above)
    # ------------------------------------------------------------------

    @app.get("/auth/my-agents")
    async def auth_my_agents(
        request: Request, user: User | None = Depends(optional_current_user)
    ):
        redis = app.state.redis
        registry = app.state.registry
        connected = await redis.smembers(KEY_CONNECTED_AGENTS) if redis else set()

        if user is not None:
            if user.is_superuser:
                records = await registry.all_agents()
            else:
                records = await registry.get_agents_for_user(str(user.id))
            alive = [r for r in records if r.agent_tag in connected]
            return {
                "agents": [
                    {
                        "agent_tag": r.agent_tag,
                        "token": r.token,
                        "model_info": r.model_info,
                    }
                    for r in alive
                ]
            }

        # Anonymous: anon_id lives in localStorage and is sent as a request header.
        anon_id = request.headers.get("X-Anon-Id") or request.session.get("anon_id")
        if not anon_id:
            return {"agents": []}
        tags = await redis.smembers(f"ogw:anon:{anon_id}:agents") if redis else set()
        alive_tags = tags & connected
        agents = []
        for tag in alive_tags:
            record = await registry.get_by_tag(tag)
            if record:
                agents.append(
                    {
                        "agent_tag": record.agent_tag,
                        "token": record.token,
                        "model_info": record.model_info,
                    }
                )
        return {"agents": agents}

    @app.delete("/users/me", tags=["users"])
    async def delete_account(
        request: Request,
        body: DeleteAccountRequest,
        user: User = Depends(current_user),
        user_manager=Depends(get_user_manager),
        strategy=Depends(auth_backend.get_strategy),
    ):
        verified, _ = user_manager.password_helper.verify_and_update(
            body.password, user.hashed_password
        )
        if not verified:
            raise HTTPException(status_code=400, detail="WRONG_PASSWORD")
        registry = app.state.registry
        redis = app.state.redis
        if registry is not None and redis is not None:
            records = await registry.get_agents_for_user(str(user.id))
            for record in records:
                # The bg loop may run on any worker; tell the owning worker to
                # exit by publishing kill_bg, and remove the persisted bg-loop
                # record so the loop doesn't get resumed on next boot.
                await redis.delete(f"ogw:bg:{record.agent_tag}")
                await redis.publish(
                    CHANNEL_AGENT_EVENT,
                    json.dumps({"event": "kill_bg", "tag": record.agent_tag}),
                )
        token = request.cookies.get("ogw_user_session", "")
        logout_response = await auth_backend.logout(strategy, user, token)
        await user_manager.delete(user)
        return logout_response

    # ------------------------------------------------------------------
    # Admin interface (superuser only)
    # ------------------------------------------------------------------

    @app.get("/admin/users", tags=["admin"])
    async def admin_list_users(
        _user: User = Depends(superuser),
        session: AsyncSession = Depends(get_async_session),
    ):
        result = await session.execute(select(User))
        return [UserRead.model_validate(u) for u in result.scalars().all()]

    @app.patch("/admin/users/{user_id}", tags=["admin"])
    async def admin_update_user(
        user_id: uuid.UUID,
        body: AdminUpdateUserRequest,
        _user: User = Depends(superuser),
        session: AsyncSession = Depends(get_async_session),
    ):
        result = await session.execute(select(User).where(User.id == user_id))
        target = result.scalar_one_or_none()
        if not target:
            raise HTTPException(status_code=404, detail="USER_NOT_FOUND")
        if body.is_active is not None:
            target.is_active = body.is_active
        if body.is_superuser is not None:
            target.is_superuser = body.is_superuser
        session.add(target)
        await session.commit()
        return UserRead.model_validate(target)

    @app.delete("/admin/users/{user_id}", tags=["admin"])
    async def admin_delete_user(
        user_id: uuid.UUID,
        _user: User = Depends(superuser),
        user_manager=Depends(get_user_manager),
    ):
        try:
            target = await user_manager.get(user_id)
        except Exception:
            raise HTTPException(status_code=404, detail="USER_NOT_FOUND")
        registry = app.state.registry
        redis = app.state.redis
        if registry is not None and redis is not None:
            records = await registry.get_agents_for_user(str(user_id))
            for record in records:
                await redis.delete(f"ogw:bg:{record.agent_tag}")
                await redis.publish(
                    CHANNEL_AGENT_EVENT,
                    json.dumps({"event": "kill_bg", "tag": record.agent_tag}),
                )
        await user_manager.delete(target)
        return {"ok": True}

    return app
