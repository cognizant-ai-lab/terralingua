"""Serve recorded social graphs through the dashboard's existing session."""

import asyncio
from pathlib import Path

from terralingua.experiment.social_graph_history import read_social_graph_history
from terralingua.server.redis_bus import KEY_META


def _empty_history(message: str) -> dict:
    return {
        "schema_version": 1,
        "available": False,
        "run_id": None,
        "segment_id": None,
        "snapshots": [],
        "total": 0,
        "offset": 0,
        "next_offset": None,
        "message": message,
    }


def _read_page(path: Path, offset: int, limit: int, run_id, segment_id) -> dict:
    history = read_social_graph_history(path)
    if not history["snapshots"]:
        return _empty_history(
            "No graph recording is available. New social-graph runs record each step."
        )
    if (
        (run_id is not None and run_id != history["run_id"])
        or (segment_id is not None and segment_id != history["segment_id"])
    ):
        raise ValueError("The recording changed while loading. Load it again.")
    total = len(history["snapshots"])
    end = min(offset + limit, total)
    return {
        "schema_version": history["schema_version"],
        "available": True,
        "run_id": history["run_id"],
        "segment_id": history["segment_id"],
        "snapshots": history["snapshots"][offset:end],
        "total": total,
        "offset": offset,
        "next_offset": end if end < total else None,
        "experiment": path.parent.name,
    }


async def get_social_graph_history(
    redis, *, offset=0, limit=100, run_id=None, segment_id=None
) -> dict:
    """Read the current run's recording without contacting the runner."""
    if type(offset) is not int or offset < 0:
        raise ValueError("History offset must be a nonnegative integer.")
    if type(limit) is not int or not 1 <= limit <= 200:
        raise ValueError("History page size must be between 1 and 200.")
    if redis is None:
        return _empty_history("No simulation is connected. Open a saved recording instead.")
    logdir = await redis.hget(KEY_META, "exp_logdir")
    if not logdir:
        return _empty_history("No recorded run is selected. Open a saved recording instead.")
    if isinstance(logdir, bytes):
        logdir = logdir.decode()
    # The path comes from runner metadata, never from a dashboard request.
    path = Path(logdir) / "social_graph.jsonl"
    return await asyncio.to_thread(_read_page, path, offset, limit, run_id, segment_id)
