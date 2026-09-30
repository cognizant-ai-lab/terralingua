"""RedisSubscriber: API-side subscriber that feeds DashboardManager."""
import json
import logging

from terralingua.server.redis_bus import (
    CHANNEL_FRAMES,
    CHANNEL_PUSH,
    get_redis,
    retry_loop,
)

logger = logging.getLogger(__name__)


class RedisSubscriber:
    """Subscribes to ogw:frames and ogw:push, forwards to DashboardManager."""

    def __init__(self, dashboard_manager):
        self._dm = dashboard_manager
        self._redis = None

    async def run(self):
        await retry_loop(self._connect_and_listen, "[RedisSubscriber]")

    async def _connect_and_listen(self):
        self._redis = await get_redis()
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(CHANNEL_FRAMES, CHANNEL_PUSH)
        logger.info("[RedisSubscriber] subscribed to frames and push channels")
        try:
            async for msg in pubsub.listen():
                if msg["type"] != "message":
                    continue
                try:
                    data = json.loads(msg["data"])
                except (json.JSONDecodeError, TypeError):
                    continue
                if msg["channel"] == CHANNEL_FRAMES:
                    await self._dm.broadcast_frame(data)
                elif msg["channel"] == CHANNEL_PUSH:
                    await self._dm.broadcast_push(data)
        finally:
            await pubsub.aclose()
