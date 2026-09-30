"""Redis-backed ConnectionManager for the runner process.

The API process holds the actual WebSocket connections to remote agents.
The runner uses this class to exchange observations/actions over Redis.

Step protocol:
  1. runner calls send_prompt(tag, payload)   -> publishes to ogw:obs:{tag}
  2. API receives it, forwards to agent WS
  3. Agent replies, API publishes to ogw:action:{tag}
  4. Background listener resolves the Future
  5. runner calls wait_for_response(tag, timeout) -> awaits Future
"""

import asyncio
import json
import logging

from redis.asyncio import Redis as AsyncRedis

from terralingua.server.redis_bus import (
    CHANNEL_AGENT_EVENT,
    KEY_CONNECTED_AGENTS,
    action_channel,
    get_redis,
    obs_channel,
    retry_loop,
)

logger = logging.getLogger(__name__)

_STAY_ACTION = {"action": "move", "message": "", "params": {"direction": "stay"}}


class RedisConnectionManager:
    def __init__(self):
        self._pending: dict[str, asyncio.Future] = {}
        self._redis: AsyncRedis | None = None
        self._listener_task: asyncio.Task | None = None
        self._agent_event_task: asyncio.Task | None = None
        self._connected_cache: set = set()

    async def start(self):
        self._redis = await get_redis()
        self._connected_cache = set()
        self._listener_task = asyncio.create_task(self._listen_actions())
        self._agent_event_task = asyncio.create_task(self._listen_agent_events())
        logger.info("[RedisConnectionManager] started")

    async def _listen_actions(self):
        """Pattern-subscribe to all ogw:action:* channels."""
        await retry_loop(self._run_action_listener, "[RedisConnectionManager] action listener")

    async def _run_action_listener(self):
        r = await get_redis()
        pubsub = r.pubsub()
        await pubsub.psubscribe("ogw:action:*")
        try:
            async for msg in pubsub.listen():
                if msg["type"] != "pmessage":
                    continue
                channel = msg["channel"]
                tag = channel.split(":", 2)[2]
                future = self._pending.get(tag)
                if future is not None and not future.done():
                    try:
                        action = json.loads(msg["data"])
                        future.set_result(action)
                    except Exception:
                        future.set_result({**_STAY_ACTION, "source": "default", "default_reason": "malformed_action"})
        finally:
            await pubsub.aclose()

    async def _listen_agent_events(self):
        await retry_loop(self._run_agent_event_listener, "[RedisConnectionManager] agent event listener")

    async def _run_agent_event_listener(self):
        r = await get_redis()
        pubsub = r.pubsub()
        await pubsub.subscribe(CHANNEL_AGENT_EVENT)
        # Reload the full connected set on each (re)connect — events may have
        # been missed during a Redis outage.
        try:
            current = await r.smembers(KEY_CONNECTED_AGENTS)  # type: ignore
            self._connected_cache = set(current)
        except Exception as e:
            logger.warning(f"[RedisConnectionManager] failed to load connected agents: {e}")
        try:
            async for msg in pubsub.listen():
                if msg["type"] != "message":
                    continue
                try:
                    event = json.loads(msg["data"])
                    tag = event.get("tag", "")
                    if event.get("event") == "connect":
                        self._connected_cache.add(tag)
                    elif event.get("event") == "disconnect":
                        self._connected_cache.discard(tag)
                except Exception as e:
                    logger.debug(f"[RedisConnectionManager] malformed agent event: {e}")
        finally:
            await pubsub.aclose()

    async def send_prompt(self, agent_tag: str, payload: dict):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending[agent_tag] = future
        if self._redis is not None:
            await self._redis.publish(obs_channel(agent_tag), json.dumps(payload))

    async def wait_for_response(self, agent_tag: str, timeout_s: float) -> dict:
        future = self._pending.get(agent_tag)
        if future is None:
            return {**_STAY_ACTION, "source": "default", "default_reason": "no_prompt_sent"}
        try:
            if timeout_s > 0:
                return await asyncio.wait_for(asyncio.shield(future), timeout=timeout_s)
            return await future
        except asyncio.TimeoutError:
            return {**_STAY_ACTION, "source": "default", "default_reason": "timeout"}
        finally:
            self._pending.pop(agent_tag, None)

    def connected_agents(self) -> list:
        """Read from the locally cached set of connected agents."""
        return list(self._connected_cache)

    def unregister_agent(self, agent_tag: str):
        future = self._pending.pop(agent_tag, None)
        if future is not None and not future.done():
            future.cancel()

