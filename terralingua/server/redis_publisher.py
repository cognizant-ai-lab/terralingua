"""RedisPublisher: runner-side publisher that replaces DashboardManager."""
import json
import logging

from terralingua.server.redis_bus import (
    CHANNEL_FRAMES,
    CHANNEL_PUSH,
    KEY_CONNECTED_AGENTS,
    KEY_LAST_FRAME,
    KEY_RUNNER_STATUS,
    get_redis,
)

logger = logging.getLogger(__name__)


class RedisPublisher:
    """Publishes simulation frames and push messages to Redis.

    Drop-in replacement for DashboardManager in the runner process.
    API process subscribes and feeds its own DashboardManager.
    """

    def __init__(self):
        self._redis = None

    async def start(self):
        self._redis = await get_redis()
        logger.info("[RedisPublisher] connected to Redis")

    async def _reconnect(self):
        try:
            self._redis = await get_redis()
            logger.info("[RedisPublisher] reconnected to Redis")
        except Exception as e:
            logger.error(f"[RedisPublisher] reconnect failed: {e}")

    async def broadcast_frame(self, frame: dict):
        if self._redis is None:
            return
        data = json.dumps(frame)
        try:
            await self._redis.set(KEY_LAST_FRAME, data)
            await self._redis.publish(CHANNEL_FRAMES, data)
        except Exception as e:
            logger.warning(f"[RedisPublisher] broadcast_frame error, reconnecting: {e}")
            await self._reconnect()

    async def broadcast_status(self, status: str):
        if self._redis is None:
            return
        try:
            await self._redis.set(KEY_RUNNER_STATUS, status)
            await self._redis.publish(CHANNEL_PUSH, json.dumps({"type": "runner_status", "status": status}))
        except Exception as e:
            logger.warning(f"[RedisPublisher] broadcast_status error: {e}")

    async def reset(self):
        """Clear stale Redis state from the previous run."""
        if self._redis is None:
            return
        await self._redis.delete(KEY_LAST_FRAME, KEY_CONNECTED_AGENTS, KEY_RUNNER_STATUS)
        # Purge stale agent registry entries left over from the previous run so
        # that fresh simulations don't collide with old tags.  Only the runner
        # creates ogw:reg:* keys, so it is safe to delete them all here (one
        # runner per Redis instance).  ogw:bg:* keys are owned by the API
        # server and are cleaned up via "died" events in _teardown.
        stale = []
        for pattern in ("ogw:reg:tag:*", "ogw:reg:tok:*"):
            async for key in self._redis.scan_iter(pattern):
                stale.append(key)
        if stale:
            await self._redis.delete(*stale)

    async def broadcast_push(self, msg: dict):
        if self._redis is None:
            return
        try:
            await self._redis.publish(CHANNEL_PUSH, json.dumps(msg))
        except Exception as e:
            logger.warning(f"[RedisPublisher] broadcast_push error, reconnecting: {e}")
            await self._reconnect()
