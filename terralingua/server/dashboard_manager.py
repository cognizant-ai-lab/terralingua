"""
DashboardManager: broadcast simulation frames to connected dashboard WebSocket clients.
"""

import asyncio
import json
import logging
from typing import Optional

from fastapi import WebSocket

logger = logging.getLogger(__name__)

KEY_OBSERVERS = "ogw:dashboard_observers"


class DashboardManager:
    """Tracks active dashboard WebSocket connections and broadcasts simulation frames."""

    def __init__(self):
        self._connections: list = []
        self._redis = None

    def set_redis(self, redis) -> None:
        """Wire a Redis client for the cross-worker observer counter."""
        self._redis = redis

    async def on_connect(self, ws: WebSocket):
        await ws.accept()
        self._connections.append(ws)
        if self._redis is not None:
            try:
                await self._redis.incr(KEY_OBSERVERS)
            except Exception as e:
                logger.warning(f"[DM] failed to incr observer counter: {e}")
        logger.info(f"[DM] dashboard client connected ({len(self._connections)} total)")

    async def on_disconnect(self, ws: WebSocket):
        was_tracked = ws in self._connections
        if was_tracked:
            self._connections.remove(ws)
            if self._redis is not None:
                try:
                    await self._redis.decr(KEY_OBSERVERS)
                except Exception as e:
                    logger.warning(f"[DM] failed to decr observer counter: {e}")
        logger.info(
            f"[DM] dashboard client disconnected ({len(self._connections)} remaining)"
        )

    async def release_observer_count(self) -> None:
        """Best-effort: deduct still-tracked local connections from the global
        counter on graceful shutdown so stale workers don't inflate the count."""
        if self._redis is None or not self._connections:
            return
        try:
            await self._redis.decrby(KEY_OBSERVERS, len(self._connections))
        except Exception as e:
            logger.warning(f"[DM] failed to release observer count: {e}")

    async def _send_to_all(self, payload: dict):
        """Send payload to all connections concurrently; drop any that fail.

        Serializes once per broadcast instead of once per connection — wire
        format is identical to send_json (since that just calls json.dumps
        internally), but the JSON cost goes from O(N) to O(1) per worker per
        frame. Matters once N runs into the hundreds.
        """
        connections = list(self._connections)
        if not connections:
            return
        text = json.dumps(payload)
        results = await asyncio.gather(
            *[ws.send_text(text) for ws in connections],
            return_exceptions=True,
        )
        dead = [
            (ws, r) for ws, r in zip(connections, results) if isinstance(r, Exception)
        ]
        for ws, r in dead:
            logger.warning(f"[DM] failed to send to client, dropping: {r}")
            await self.on_disconnect(ws)

    async def broadcast_frame(self, frame: dict):
        """Push a full simulation state frame to all connected dashboard clients."""
        await self._send_to_all(
            {**frame, "type": "frame", "connected_clients": len(self._connections)}
        )

    async def broadcast_push(self, msg: dict):
        """Push a non-frame message (config, child_agent, etc.) to all clients."""
        await self._send_to_all(msg)

    async def send_initial_frame(self, ws: WebSocket, frame: Optional[dict]) -> None:
        """Send the current frame to a newly connected client."""
        if frame is None:
            return
        try:
            await ws.send_json(
                {**frame, "type": "frame", "connected_clients": len(self._connections)}
            )
        except Exception as e:
            logger.warning(f"[DM] failed to send initial frame: {e}")
