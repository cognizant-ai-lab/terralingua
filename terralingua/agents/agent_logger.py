import asyncio
import json
import os
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from terralingua.utils import LOGS_DIR

# ---------------------------------------------------------------------------
# Module-level async write queue.  One background task drains all per-agent
# JSONL log writes so that agent coroutines never block the event loop on
# file I/O.
# ---------------------------------------------------------------------------
_write_queue: Optional[asyncio.Queue] = None
_writer_task: Optional[asyncio.Task] = None

_MAX_LOG_HANDLES = 256  # max simultaneously open file handles in the writer


async def _async_writer(queue: asyncio.Queue) -> None:
    """Background task: drain (filepath, line) pairs from the queue."""
    handles: OrderedDict = OrderedDict()  # path -> file handle (LRU)

    def _get_handle(path: Path):
        if path in handles:
            handles.move_to_end(path)
            return handles[path]
        if len(handles) >= _MAX_LOG_HANDLES:
            _, h = handles.popitem(last=False)
            h.close()
        h = open(path, "a")
        handles[path] = h
        return h

    try:
        while True:
            item = await queue.get()
            if item is None:  # shutdown sentinel
                break
            path, line = item
            _get_handle(path).write(line)
            # Drain any immediately available records before flushing.
            while not queue.empty():
                item = queue.get_nowait()
                if item is None:
                    return
                path, line = item
                _get_handle(path).write(line)
            for h in handles.values():
                h.flush()
    finally:
        for h in handles.values():
            h.close()


def start_async_logging() -> None:
    """Create the write queue and start the background writer task.

    Must be called from inside a running event loop (e.g. at the top of
    ``run_async``).  Safe to call multiple times; subsequent calls are no-ops.
    """
    global _write_queue, _writer_task
    if _write_queue is not None:
        return
    _write_queue = asyncio.Queue()
    _writer_task = asyncio.create_task(_async_writer(_write_queue))


async def stop_async_logging() -> None:
    """Flush and shut down the background writer."""
    global _write_queue, _writer_task
    if _write_queue is not None:
        await _write_queue.put(None)
    if _writer_task is not None:
        await _writer_task
    _write_queue = None
    _writer_task = None


class AgentLogger:
    def __init__(self, log_dir: Path | str | None, agent_tag: str = "agent"):
        if log_dir is None:
            self.log_dir = (
                LOGS_DIR / datetime.now().strftime("%Y%m%d_%H%M") / "agent_logs"
            )
        else:
            self.log_dir = Path(log_dir)

        os.makedirs(self.log_dir, exist_ok=True)
        self.filename = self.log_dir / f"{agent_tag}.jsonl"
        self.data_dict: Dict[str, Dict[str, Any]] = {}

    def save_genome(self, agent_tag: str, genome: dict):
        genome_filename = self.log_dir / f"{agent_tag}_genome.json"
        with open(genome_filename, "w") as f:
            json.dump(genome, f, indent=4)

    def log(
        self,
        agent_name: str,
        agent_tag: str,
        observation: dict,
        available_actions: dict,
        action: dict,
        time: str,
        internal_memory: str,
        input_prompt: str,
    ):
        """
        Record the agent’s observation and action, both in a live file and in-memory dictionary.
        """
        observation["observation"] = {
            str(k): v for k, v in observation["observation"].items()
        }

        record = {
            "timestamp": time,
            "agent": agent_name,
            "agent_tag": agent_tag,
            "action": action,
            "observation": observation,
            "internal_memory": internal_memory,
            "available_actions": available_actions,
            "input_prompt": input_prompt,
        }

        # Save line to file — via async queue when available, else direct write.
        line = json.dumps(record) + "\n"
        if _write_queue is not None:
            _write_queue.put_nowait((self.filename, line))
        else:
            with open(self.filename, "a") as f:
                f.write(line)

        # Save to internal dict
        self.data_dict[time] = {
            "agent": agent_name,
            "agent_tag": agent_tag,
            "action": action,
            "internal_memory": internal_memory,
            "available_actions": available_actions,
            "observation": observation,
            "input_prompt": input_prompt,
        }

    def close(self):
        return
