"""Redis-backed registries for multi-process deployments."""
import json
import secrets
from dataclasses import dataclass

from terralingua.server.redis_bus import redis_url


@dataclass
class AgentRecord:
    agent_tag: str
    agent_name: str
    motivation_prompt: str
    model_info: str
    token: str
    owner_user_id: int | None = None
    # Fernet-encrypted API key for the LLM provider this agent (and the
    # anthropologist when annotating this agent) should use. None for agents
    # spawned outside the WS register flow (e.g. CLI).
    encrypted_api_key: str | None = None


class RedisAgentRegistry:
    """AgentRegistry backed by Redis. Safe to use from multiple processes."""

    # Redis key layout:
    #   ogw:reg:tag:{tag}    -> JSON hash of AgentRecord fields
    #   ogw:reg:tok:{token}  -> agent_tag string
    #   ogw:reg:counter      -> integer counter for tag generation

    def __init__(self, redis):
        self._r = redis

    @classmethod
    async def create(cls):
        from redis.asyncio import Redis
        r = Redis.from_url(redis_url(), decode_responses=True)
        return cls(r)

    async def register(
        self,
        agent_name: str,
        motivation_prompt: str,
        model_info: str = "",
        encrypted_api_key: str | None = None,
    ) -> AgentRecord:
        token = secrets.token_urlsafe(32)
        counter = await self._r.incr("ogw:reg:counter")
        agent_tag = f"remote_{counter}"
        record = AgentRecord(
            agent_tag=agent_tag,
            agent_name=agent_name,
            motivation_prompt=motivation_prompt,
            model_info=model_info,
            token=token,
            encrypted_api_key=encrypted_api_key,
        )
        await self._store(record)
        return record

    async def register_with_tag(
        self,
        agent_tag: str,
        agent_name: str,
        motivation_prompt: str,
        model_info: str = "",
        encrypted_api_key: str | None = None,
    ) -> AgentRecord:
        token = secrets.token_urlsafe(32)
        record = AgentRecord(
            agent_tag=agent_tag,
            agent_name=agent_name,
            motivation_prompt=motivation_prompt,
            model_info=model_info,
            token=token,
            encrypted_api_key=encrypted_api_key,
        )
        await self._store(record)
        return record

    async def _store(self, record: AgentRecord):
        data = json.dumps({
            "agent_tag": record.agent_tag,
            "agent_name": record.agent_name,
            "motivation_prompt": record.motivation_prompt,
            "model_info": record.model_info,
            "token": record.token,
            "owner_user_id": record.owner_user_id,
            "encrypted_api_key": record.encrypted_api_key,
        })
        await self._r.set(f"ogw:reg:tag:{record.agent_tag}", data)
        await self._r.set(f"ogw:reg:tok:{record.token}", record.agent_tag)

    async def get_by_token(self, token: str):
        tag = await self._r.get(f"ogw:reg:tok:{token}")
        if tag is None:
            return None
        return await self.get_by_tag(tag)

    @staticmethod
    def _record_from_dict(d: dict) -> AgentRecord:
        return AgentRecord(
            agent_tag=d["agent_tag"],
            agent_name=d["agent_name"],
            motivation_prompt=d["motivation_prompt"],
            model_info=d["model_info"],
            token=d["token"],
            owner_user_id=d.get("owner_user_id"),
            encrypted_api_key=d.get("encrypted_api_key"),
        )

    async def get_by_tag(self, agent_tag: str):
        raw = await self._r.get(f"ogw:reg:tag:{agent_tag}")
        if raw is None:
            return None
        return self._record_from_dict(json.loads(raw))

    async def set_owner(self, agent_tag: str, user_id: str) -> None:
        raw = await self._r.get(f"ogw:reg:tag:{agent_tag}")
        if not raw:
            return
        d = json.loads(raw)
        d["owner_user_id"] = user_id
        await self._r.set(f"ogw:reg:tag:{agent_tag}", json.dumps(d))

    async def set_api_key(self, agent_tag: str, encrypted_api_key: str | None) -> None:
        """In-place update of the agent's encrypted API key (Fernet-encrypted).

        Called when a registered user rotates their API key — the main server
        walks ``get_agents_for_user`` and calls this for each owned agent so
        live annotation sees the new key on the next cycle without restart.
        """
        raw = await self._r.get(f"ogw:reg:tag:{agent_tag}")
        if not raw:
            return
        d = json.loads(raw)
        d["encrypted_api_key"] = encrypted_api_key
        await self._r.set(f"ogw:reg:tag:{agent_tag}", json.dumps(d))

    async def get_agents_for_user(self, user_id: str) -> list[AgentRecord]:
        records = []
        async for key in self._r.scan_iter("ogw:reg:tag:*"):
            raw = await self._r.get(key)
            if not raw:
                continue
            d = json.loads(raw)
            if d.get("owner_user_id") == user_id:
                records.append(self._record_from_dict(d))
        return records

    async def all_agents(self) -> list[AgentRecord]:
        records = []
        async for key in self._r.scan_iter("ogw:reg:tag:*"):
            raw = await self._r.get(key)
            if not raw:
                continue
            records.append(self._record_from_dict(json.loads(raw)))
        return records

    async def unregister(self, agent_tag: str):
        raw = await self._r.get(f"ogw:reg:tag:{agent_tag}")
        if raw:
            d = json.loads(raw)
            await self._r.delete(f"ogw:reg:tok:{d['token']}")
        await self._r.delete(f"ogw:reg:tag:{agent_tag}")


class RedisDashboardSessionRegistry:
    """DashboardSessionRegistry backed by Redis.

    Key layout (individual keys with TTL, no global sets):
      ogw:sess:p:{token}  pending — issued at GET /dashboard, expires if WS never connects
      ogw:sess:a:{token}  active  — set when WS connects, deleted on disconnect
    """

    _PENDING_TTL = 300   # seconds to open the WS after page load
    _RECONNECT_TTL = 60  # seconds to reconnect after WS disconnect

    def __init__(self, redis):
        self._r = redis

    async def issue_token(self, user_id: str | None = None) -> str:
        t = secrets.token_urlsafe(32)
        payload = json.dumps({"user_id": user_id})
        await self._r.set(f"ogw:sess:p:{t}", payload, ex=self._PENDING_TTL)
        return t

    async def validate_for_connect(self, token: str) -> bool:
        payload = await self._r.getdel(f"ogw:sess:p:{token}")
        if payload is not None:
            await self._r.set(f"ogw:sess:a:{token}", payload)
            return True
        return bool(await self._r.exists(f"ogw:sess:a:{token}"))

    async def get_user_id(self, token: str) -> str | None:
        raw = await self._r.get(f"ogw:sess:a:{token}")
        if not raw:
            return None
        try:
            return json.loads(raw).get("user_id")
        except Exception:
            return None

    async def release(self, token: str):
        payload = await self._r.get(f"ogw:sess:a:{token}")
        await self._r.delete(f"ogw:sess:a:{token}")
        await self._r.set(
            f"ogw:sess:p:{token}",
            payload or json.dumps({"user_id": None}),
            ex=self._RECONNECT_TTL,
        )

    # ── Per-session API key (Fernet-encrypted) ───────────────────────────
    # Stored INSIDE the active-session JSON payload (and migrated to the
    # pending entry on release()) so the key's lifetime is exactly the
    # session's lifetime — no separate TTL that could drift longer than the
    # session and widen the window for a stolen token.

    async def set_session_api_key(self, token: str, encrypted_api_key: str) -> None:
        # Try the active entry first; fall back to the pending entry. We
        # never create a new entry here — the session must already exist
        # (validate_for_connect has run).
        for prefix in ("a", "p"):
            key = f"ogw:sess:{prefix}:{token}"
            raw = await self._r.get(key)
            if raw is None:
                continue
            try:
                payload = json.loads(raw)
            except Exception:
                payload = {"user_id": None}
            payload["encrypted_api_key"] = encrypted_api_key
            # Preserve existing TTL on the entry (pending entries have a
            # bounded TTL we don't want to extend implicitly).
            ttl = await self._r.ttl(key)
            if ttl and ttl > 0:
                await self._r.set(key, json.dumps(payload), ex=ttl)
            else:
                await self._r.set(key, json.dumps(payload))
            return

    async def get_session_api_key(self, token: str) -> str | None:
        for prefix in ("a", "p"):
            raw = await self._r.get(f"ogw:sess:{prefix}:{token}")
            if not raw:
                continue
            try:
                return json.loads(raw).get("encrypted_api_key")
            except Exception:
                return None
        return None
