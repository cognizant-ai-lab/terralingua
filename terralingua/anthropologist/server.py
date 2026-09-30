"""
Live anthropologist during an infinite run.

Runs live detection every --detect-interval simulation steps, publishing
field notes and postmortems to Redis as they happen.

Usage:
    terralingua-anthropologist --exp_name my_run
    terralingua-anthropologist --exp_name my_run --detect-interval 20
    terralingua-anthropologist --exp_name my_run --dispatch-severity 2

Redis channels / keys used:
    ogw:push                pub/sub: field notes and postmortems pushed here
    ogw:field_notes_all     key: JSON array of all field note events, newest first (capped at 2000)
    ogw:last_detect_step    key: int step at which detection last ran (for restart continuity)
    ogw:obituaries          key: JSON list of agent_postmortem payloads (newest first)

Disk:
    logs/{exp}/field_notes.jsonl  append-only JSONL of all events, oldest first
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import os
import signal
import sys
import threading
from pathlib import Path

from redis.asyncio import Redis as AsyncRedis

from terralingua.anthropologist import Anthropologist
from terralingua.anthropologist.analysis_utils import get_current_step, load_agent_log
from terralingua.server.auth import decrypt_api_key
from terralingua.server.redis_bus import (
    CHANNEL_PUSH,
    KEY_ANTHRO_ADMIN_API_KEY,
    KEY_EXP_CONFIG,
    KEY_LAST_FRAME,
    key_artifact_classifications,
    key_field_notes_all,
    key_last_detect_step,
    key_live_annotations,
    key_phylogeny_queue,
    key_postmortem_queue,
    redis_url,
)
from terralingua.server.redis_registry import RedisAgentRegistry
from terralingua.utils import LOGS_DIR
from terralingua.utils.llm_errors import llm_error_type
from terralingua.utils.logging_setup import setup_logging

_FIELD_NOTES_REDIS_CAP = 2000  # max events kept in Redis (oldest are dropped)

setup_logging()
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live anthropologist server")
    p.add_argument(
        "--exp_name", required=True, help="Experiment name (subdirectory of logs/)"
    )
    p.add_argument(
        "--detect-interval",
        type=int,
        default=10,
        help="Run live detection every N simulation steps (default: 10)",
    )
    p.add_argument(
        "--window",
        type=int,
        default=50,
        help="Log scanner window in steps (default: 50)",
    )
    p.add_argument(
        "--dispatch-severity",
        type=int,
        default=1,
        help="Min severity to generate LLM narrative (0=low, 1=medium, 2=high; default: 1)",
    )
    p.add_argument(
        "--model", default="claude-haiku-4-5", help="LLM model name"
    )
    p.add_argument(
        "--fallback-model",
        default=None,
        help=(
            "Optional smaller model to retry with when the primary refuses "
            "(content-policy block) or its retry budget is exhausted. "
            "Postmortems (Sonnet) automatically fall back to --model (Haiku) "
            "even if this flag is unset."
        ),
    )
    p.add_argument(
        "--no-audit", action="store_true", help="Disable audit pass on annotations"
    )
    p.add_argument(
        "--no-redis", action="store_true", help="Run without Redis (print to stdout)"
    )
    p.add_argument(
        "--catchup",
        action="store_true",
        help=(
            "On startup, treat all historical data as new requests: every dead agent "
            "is queued for postmortem, every existing artifact is queued for phylogeny. "
            "Without this flag, only agents that die / artifacts requested via the "
            "dashboard after server start get processed."
        ),
    )
    p.add_argument(
        "--auto-postmortems",
        action="store_true",
        help=(
            "Automatically run a postmortem for every agent that dies during the simulation. "
            "Without this flag, postmortems are only run when explicitly requested via the queue."
        ),
    )
    return p.parse_args()


async def _get_step_from_redis(redis_client: AsyncRedis) -> int:
    """Read current simulation step from KEY_LAST_FRAME in Redis."""
    raw = await redis_client.get(KEY_LAST_FRAME)
    if not raw:
        return 0
    return json.loads(raw).get("step", 0)


class AnthroKeyResolver:
    """Resolves the Fernet-encrypted API key to use for each LLM call.

    Pre-fetches the agent registry once per detect cycle so the per-agent
    lookups inside the orchestrator (called synchronously inside
    asyncio.gather expressions) become plain dict reads. The decrypted
    plaintext keys are kept in memory for the duration of one cycle and
    refreshed at the start of the next.

    Attribution rules implemented here:
      - live annotation: ``get_agent_key(tag)`` returns the agent owner's
        key, falling back to the admin key only for ownerless agents.
      - cross-agent calls: ``get_key_via_chain(tags)`` returns the first
        agent in the list that has a key, falling back to admin.
    """

    def __init__(self, redis_client: AsyncRedis | None):
        self._r = redis_client
        self._registry = RedisAgentRegistry(redis_client) if redis_client else None
        self._agent_keys: dict[str, str | None] = {}
        self._admin_key: str | None = None

    @staticmethod
    def _safe_decrypt(blob: str | None) -> str | None:
        if not blob:
            return None
        try:
            return decrypt_api_key(blob)
        except Exception as e:
            log.warning("Failed to decrypt API key: %s", type(e).__name__)
            return None

    async def refresh(self) -> None:
        """Refresh the in-memory cache from Redis. Call at the start of each cycle."""
        self._agent_keys = {}
        self._admin_key = None
        if self._r is None or self._registry is None:
            return
        try:
            self._admin_key = self._safe_decrypt(
                await self._r.get(KEY_ANTHRO_ADMIN_API_KEY)
            )
        except Exception as e:
            log.warning("admin key fetch failed: %s", e)
        try:
            for rec in await self._registry.all_agents():
                self._agent_keys[rec.agent_tag] = self._safe_decrypt(rec.encrypted_api_key)
        except Exception as e:
            log.warning("agent key prefetch failed: %s", e)

    def get_agent_key(self, agent_tag: str | None) -> str | None:
        """Return the key for an agent's live annotation, or the admin
        fallback if the agent has none on its record. Returns None only
        when no admin fallback is configured either — in that case the
        LLM call falls through to litellm's env-var lookup."""
        if agent_tag is not None:
            key = self._agent_keys.get(agent_tag)
            if key:
                return key
        return self._admin_key

    def get_key_via_chain(self, agent_tags) -> str | None:
        """First agent with a key wins; admin as last-resort fallback."""
        for tag in agent_tags or []:
            key = self._agent_keys.get(tag)
            if key:
                return key
        return self._admin_key

def _is_substantive_str(value) -> bool:
    return isinstance(value, str) and bool(value)


async def _notify_anthro_error(
    redis_client: AsyncRedis | None,
    *,
    target_user_id: str | None,
    target_session_token: str | None,
    operation: str,
    reason: str = "missing_api_key",
) -> None:
    """Publish a precondition-fail event that the dashboard renders as a toast.

    The frontend filters on ``target_user_id`` / ``target_session_token`` so
    the message is private to the right user (not broadcast).
    """
    if redis_client is None:
        return
    if not _is_substantive_str(target_user_id) and not _is_substantive_str(target_session_token):
        log.debug("anthro_error not published — no target user/session for %s/%s", operation, reason)
        return
    payload = {
        "type": "anthro_error",
        "target_user_id": target_user_id,
        "target_session_token": target_session_token,
        "operation": operation,
        "reason": reason,
    }
    try:
        await redis_client.publish(CHANNEL_PUSH, json.dumps(payload))
    except Exception as e:
        log.warning("anthro_error publish failed: %s", e)


async def _notify_anthro_api_error(
    redis_client: AsyncRedis | None,
    exc: BaseException,
    *,
    target_user_id: str | None,
    target_session_token: str | None,
    operation: str,
) -> None:
    """Publish a runtime LLM-error event using the same shape as agent_api_error
    so addStepError() coalesces matching error_types across agent + anthro sources."""
    if redis_client is None:
        return
    if not _is_substantive_str(target_user_id) and not _is_substantive_str(target_session_token):
        log.debug("anthro_api_error not published — no target for %s", operation)
        return
    payload = {
        "type": "anthro_api_error",
        "target_user_id": target_user_id,
        "target_session_token": target_session_token,
        "operation": operation,
        "error_type": llm_error_type(exc) or "api_error",
        "error_message": str(exc)[:500],
    }
    try:
        await redis_client.publish(CHANNEL_PUSH, json.dumps(payload))
    except Exception as e:
        log.warning("anthro_api_error publish failed: %s", e)


def _fmt(tag: str, name: str | None) -> str:
    """Format an agent identity for developer-facing log messages."""
    return f"{name}({tag})" if name else tag


def _get_agent_vitals(exp_path, agent_tag: str, death_step: int) -> dict:
    """Return birth_step and energy_at_death from the agent log, best-effort."""
    try:
        log_path = Path(exp_path) / "agent_logs" / f"{agent_tag}.jsonl"
        agent_log = load_agent_log(log_path, reduce=True)
        if not agent_log:
            return {}
        birth_step = min(agent_log.keys())
        steps = [s for s in agent_log.keys() if s <= death_step]
        energy_at_death = agent_log[max(steps)]["energy"] if steps else None
        return {"birth_step": birth_step, "energy_at_death": energy_at_death}
    except Exception:
        return {}


async def _run_postmortem(
    anthropologist,
    agent_tag: str,
    exp_path: Path,
    death_step: int,
    redis_client: AsyncRedis | None,
    obituary_tags: set[str],
    agent_name: str | None = None,
    api_key: str | None = None,
    requester_user_id: str | None = None,
    requester_session_token: str | None = None,
) -> None:
    """Run (or load) a postmortem for a dead agent and publish the result.

    The LLM call is billed to ``api_key`` (the requester's key, plumbed via
    the queue payload). Adds the tag to ``obituary_tags`` on success and
    republishes the index so the dashboard updates without the detector loop
    having to scan disk."""
    label = _fmt(agent_tag, agent_name)
    log_file = exp_path / "agent_logs" / f"{agent_tag}.jsonl"
    if not log_file.exists():
        log.info(f"Postmortem skipped for {label} (no agent log file)")
        if redis_client is not None:
            await redis_client.publish(
                CHANNEL_PUSH,
                json.dumps({"type": "agent_postmortem_unavailable", "agent_tag": agent_tag}),
            )
        return

    log.info(f"Postmortem starting for {label} (death_step={death_step})")
    try:
        annotation = await anthropologist.annotate_postmortem(
            agent_tag, exp_path, death_step, api_key=api_key,
        )
    except Exception as e:
        log.warning(f"Postmortem LLM call failed for {label}: {e}")
        await _notify_anthro_api_error(
            redis_client, e,
            target_user_id=requester_user_id,
            target_session_token=requester_session_token,
            operation="postmortem",
        )
        return
    if annotation is None:
        log.info(f"Postmortem produced nothing for {label}")
        return

    payload = {
        "type": "agent_postmortem",
        "agent_tag": agent_tag,
        "death_step": death_step,
        "annotation": dataclasses.asdict(annotation),
        **_get_agent_vitals(exp_path, agent_tag, death_step),
    }
    if redis_client is None:
        log.info("Postmortem payload (no-redis): %s", json.dumps(payload, indent=2))
        return
    try:
        await redis_client.publish(CHANNEL_PUSH, json.dumps(payload))
        log.info(f"Postmortem published for {label} (death_step={death_step})")
        if agent_tag not in obituary_tags:
            obituary_tags.add(agent_tag)
            await _publish_obituary_index(redis_client, obituary_tags)
    except Exception as e:
        log.error(f"Failed to publish postmortem for {label}: {e}")


async def _reset_anthro_redis_keys(redis_client: AsyncRedis, exp_name: str) -> None:
    """Wipe anthropologist-owned Redis state on startup.

    The orchestrator's on-disk state file is the source of truth for resuming;
    Redis just mirrors the latest published snapshots. Clearing it guarantees a
    new launch doesn't pick up stale data from a previous session.

    last_detect_step is preserved — it's used to filter already-published
    events when log_scanner offsets aren't restored from a state file.
    """
    keys = [
        key_artifact_classifications(exp_name),
        key_field_notes_all(exp_name),
        key_live_annotations(exp_name),
        key_postmortem_queue(exp_name),
        key_phylogeny_queue(exp_name),
    ]
    try:
        deleted = await redis_client.delete(*keys)
        log.info(f"Reset {deleted} anthropologist Redis key(s) on startup")
    except Exception as e:
        log.warning(f"Failed to reset anthropologist Redis keys: {e}")


def _scan_obituary_tags(exp_path: Path) -> set[str]:
    """One-shot scan of existing postmortem files at startup.

    Flat layout: every JSON file directly under ``annotations/`` is a
    postmortem (model-indexed subfolders are no longer used)."""
    ann_dir = exp_path / "annotations"
    if not ann_dir.is_dir():
        return set()
    return {
        f.stem
        for f in ann_dir.glob("*.json")
        if f.name != "anthropologist_notes.json"
    }


async def _publish_obituary_index(redis_client: AsyncRedis, tags: set[str]) -> None:
    if not tags:
        return
    try:
        await redis_client.publish(
            CHANNEL_PUSH,
            json.dumps({"type": "obituaries_index", "tags": sorted(tags)}),
        )
    except Exception as e:
        log.warning(f"Failed to publish obituary index: {e}")


def _load_field_notes_from_file(path: Path) -> list[dict]:
    """Return all events from field_notes.jsonl, oldest first."""
    if not path.exists():
        return []
    events: list[dict] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        log.warning(f"Failed to load field notes from {path}: {e}")
    return events


def _append_field_notes_to_file(path: Path, events: list[dict]) -> None:
    """Append events to field_notes.jsonl (oldest-first, one JSON object per line)."""
    if not events:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            for event in events:
                f.write(json.dumps(event) + "\n")
    except Exception as e:
        log.warning(f"Failed to append field notes to {path}: {e}")


async def _push_field_notes_to_redis(
    redis_client: AsyncRedis, new_event_dicts: list[dict], exp_name: str
) -> None:
    """Prepend new events to the experiment's field_notes key (newest first)."""
    k = key_field_notes_all(exp_name)
    try:
        existing = json.loads(await redis_client.get(k) or "[]")
        combined = (new_event_dicts + existing)[:_FIELD_NOTES_REDIS_CAP]
        await redis_client.set(k, json.dumps(combined))
    except Exception as e:
        log.warning(f"Failed to update field notes in Redis: {e}")


async def _sync_field_notes_file_to_redis(
    redis_client: AsyncRedis, path: Path, exp_name: str
) -> None:
    """On startup: reset Redis field notes from the file on disk.

    File is the source of truth. If it exists the run is being resumed and its
    events are loaded into Redis. If not, this is a fresh experiment and Redis
    is cleared so the dashboard starts clean.
    """
    k = key_field_notes_all(exp_name)
    events = _load_field_notes_from_file(path)
    if not events:
        try:
            await redis_client.delete(k)
        except Exception as e:
            log.warning(f"Failed to clear field notes from Redis: {e}")
        return
    capped = list(reversed(events[-_FIELD_NOTES_REDIS_CAP:]))
    try:
        await redis_client.set(k, json.dumps(capped))
        log.info(f"Loaded {len(capped)} field note event(s) from file into Redis")
    except Exception as e:
        log.warning(f"Failed to load field notes into Redis: {e}")


async def _publish_classifications(
    redis_client: AsyncRedis | None,
    anthropologist,
    exp_path: Path,
    phylo_requested: set,
    lock: threading.Lock,
    exp_name: str,
) -> None:
    """Build the artifact_classifications payload and push it to Redis.

    Removes names from phylo_requested once they appear in the done set
    (the background worker has finished them, even if with no ancestors).
    """
    done = list(anthropologist.get_phylogeny_processed_names(exp_path))
    with lock:
        for name in list(phylo_requested):
            if name in done:
                phylo_requested.discard(name)
        pending = list(phylo_requested)
    clf_json = json.dumps({
        "type": "artifact_classifications",
        "categories": anthropologist.get_artifact_categories(),
        "phylogeny": anthropologist.get_phylogeny_by_name(exp_path),
        "done": done,
        "pending": pending,
    })
    if redis_client is None:
        log.info("Classification payload (no-redis): %s", clf_json)
        return
    await redis_client.set(key_artifact_classifications(exp_name), clf_json)
    await redis_client.publish(CHANNEL_PUSH, clf_json)


async def _postmortem_drain_loop(args, redis_client: AsyncRedis, submit_postmortem) -> None:
    """Block on the postmortem queue, dispatch each entry to submit_postmortem.

    Queue payload fields used:
      - ``tag`` (required)
      - ``name`` (optional)
      - ``encrypted_api_key`` (required for the LLM call; missing = skip)
      - ``user_id`` / ``session_token`` (for targeting any error toasts back
        to the requester only).
    """
    k = key_postmortem_queue(args.exp_name)
    while True:
        try:
            res = await redis_client.blpop(k, timeout=30)
            if res is None:
                continue
            raw = res[1]
            try:
                current_step = await _get_step_from_redis(redis_client)
            except Exception:
                current_step = 0
            encrypted_key = None
            requester_user_id = None
            requester_session_token = None
            try:
                entry = json.loads(raw)
                agent_tag = entry["tag"]
                agent_name = entry.get("name")
                encrypted_key = entry.get("encrypted_api_key")
                requester_user_id = entry.get("user_id")
                requester_session_token = entry.get("session_token")
            except (json.JSONDecodeError, KeyError):
                agent_tag = raw
                agent_name = None

            api_key: str | None = None
            if encrypted_key:
                try:
                    api_key = decrypt_api_key(encrypted_key)
                except Exception as e:
                    log.warning("Could not decrypt postmortem key for %s: %s", agent_tag, e)

            if api_key is None:
                # The main server should already have notified the user when
                # the request came in without a key, so we only log here as a
                # belt-and-braces measure (covers legacy entries from before
                # the migration).
                log.info("Postmortem dropped — no API key in payload for %s", agent_tag)
                await _notify_anthro_error(
                    redis_client,
                    target_user_id=requester_user_id,
                    target_session_token=requester_session_token,
                    operation="postmortem",
                )
                continue

            submit_postmortem(
                agent_tag, agent_name, current_step,
                api_key, requester_user_id, requester_session_token,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning(f"Postmortem drain error: {e}")
            await asyncio.sleep(1.0)


async def _phylogeny_worker_loop(
    args,
    anthropologist,
    redis_client: AsyncRedis,
    exp_path: Path,
    phylo_requested: set,
    phylo_requested_lock: threading.Lock,
) -> None:
    """Block on the phylogeny queue, dispatch each entry with its own key.

    Each entry is a JSON object:
        {"name": <artifact_name>,
         "encrypted_api_key": <fernet>,
         "user_id": <str|None>,
         "session_token": <str|None>}

    Phylogeny is requester-pays end-to-end (the user who clicked the
    button), so we don't batch across entries that may have different
    requesters — process one request at a time. The intra-request batching
    inside ``process_phylogeny_batch`` still parallelises by creation_time.

    Failed-but-retryable names are pushed back to the queue with the same
    request payload so the retry attempt still bills the original
    requester.
    """
    k = key_phylogeny_queue(args.exp_name)

    async def _on_save():
        await _publish_classifications(
            redis_client, anthropologist, exp_path,
            phylo_requested, phylo_requested_lock, args.exp_name,
        )

    while True:
        try:
            res = await redis_client.blpop(k, timeout=30)
            if res is None:
                continue
            raw = res[1]
            try:
                entry = json.loads(raw)
                if not isinstance(entry, dict):
                    raise ValueError("not a dict")
                name = entry.get("name")
                encrypted_key = entry.get("encrypted_api_key")
                requester_user_id = entry.get("user_id")
                requester_session_token = entry.get("session_token")
            except Exception:
                # Legacy/bare-string entries: nothing to bill, skip.
                log.info("Phylogeny: skipping legacy bare entry %r", raw)
                continue
            if not name:
                continue
            api_key: str | None = None
            if encrypted_key:
                try:
                    api_key = decrypt_api_key(encrypted_key)
                except Exception as e:
                    log.warning("Could not decrypt phylogeny key for %s: %s", name, e)
            if api_key is None:
                log.info("Phylogeny dropped — no API key for %s", name)
                await _notify_anthro_error(
                    redis_client,
                    target_user_id=requester_user_id,
                    target_session_token=requester_session_token,
                    operation="artifact_phylogeny",
                )
                continue
            log.info("Phylogeny request: %s (user=%s)", name, requester_user_id)
            with phylo_requested_lock:
                phylo_requested.add(name)
            try:
                retry_names = await anthropologist.process_phylogeny_batch(
                    [name], exp_path, on_save=_on_save, api_key=api_key,
                )
            except Exception as e:
                log.warning("Phylogeny call failed for %s: %s", name, e)
                await _notify_anthro_api_error(
                    redis_client, e,
                    target_user_id=requester_user_id,
                    target_session_token=requester_session_token,
                    operation="artifact_phylogeny",
                )
                continue
            if retry_names:
                log.info(f"Re-queueing {len(retry_names)} phylogeny retry(ies)")
                for retry_name in retry_names:
                    retry_entry = json.dumps({
                        "name": retry_name,
                        "encrypted_api_key": encrypted_key,
                        "user_id": requester_user_id,
                        "session_token": requester_session_token,
                    })
                    await redis_client.rpush(k, retry_entry)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning(f"Phylogeny worker error: {e}", exc_info=True)
            await asyncio.sleep(1.0)


async def _read_current_step(
    redis_client: AsyncRedis | None, log_path: Path
) -> int:
    if redis_client is not None:
        return await _get_step_from_redis(redis_client)
    return await asyncio.to_thread(get_current_step, log_path)


async def _detector_loop(
    args,
    anthropologist,
    redis_client: AsyncRedis | None,
    exp_path: Path,
    submit_postmortem,
    phylo_requested: set,
    phylo_requested_lock: threading.Lock,
    key_resolver: "AnthroKeyResolver",
) -> None:
    """Async detection cycle.

    Polls for the current sim step every 2s; runs ``run_live_detection`` each
    time it advances by ``detect-interval``. Narrative LLM calls fan out via
    asyncio.gather so sibling drain tasks make progress concurrently.
    """
    field_notes_file = exp_path / "field_notes.jsonl"
    log_path = exp_path / "open_gridworld.log"
    last_detect_step_key = key_last_detect_step(args.exp_name)

    # Restart-continuation state.
    restart_filter_step = -1
    if redis_client is not None:
        if field_notes_file.exists():
            try:
                raw_lds = await redis_client.get(last_detect_step_key)
                if raw_lds is not None:
                    restart_filter_step = int(raw_lds)
            except Exception as e:
                log.warning(f"Failed to read last_detect_step: {e}")
        else:
            try:
                await redis_client.delete(last_detect_step_key)
            except Exception as e:
                log.warning(f"Failed to clear last_detect_step from Redis: {e}")
        await _sync_field_notes_file_to_redis(redis_client, field_notes_file, args.exp_name)

    try:
        startup_step = await _read_current_step(redis_client, log_path)
    except Exception:
        startup_step = 0

    if restart_filter_step >= 0:
        last_detect_step = restart_filter_step
        log.info(f"Resuming: last_detect_step restored to {restart_filter_step}")
    else:
        last_detect_step = startup_step - args.detect_interval

    is_first_cycle = True
    last_logged_step = -1
    checkpoint_every = int(os.environ.get("ANTHRO_CHECKPOINT_EVERY", "10"))
    cycles_since_checkpoint = 0
    phylogeny_queue_key = key_phylogeny_queue(args.exp_name)

    while True:
        try:
            current_step = await _read_current_step(redis_client, log_path)
        except Exception:
            await asyncio.sleep(2.0)
            continue

        if current_step - last_detect_step < args.detect_interval:
            if current_step != last_logged_step:
                steps_to_next = args.detect_interval - (current_step - last_detect_step)
                log.info(
                    f"Last log step {current_step} — next detection in "
                    f"{steps_to_next} step(s) (last detection at {last_detect_step})"
                )
                last_logged_step = current_step
            await asyncio.sleep(2.0)
            continue

        log.info(
            f"Running live detection at step {current_step} "
            f"(+{current_step - last_detect_step} steps since last detection at {last_detect_step})"
        )
        steps_elapsed = current_step - last_detect_step
        # Refresh the per-user key cache once per cycle so the synchronous
        # resolve_key_fn callbacks inside the orchestrator's asyncio.gather
        # expressions are cheap dict lookups.
        await key_resolver.refresh()
        try:
            scan_result, events, graph_snapshot, phylogeny_retry_names = (
                await anthropologist.run_live_detection(
                    exp_path=exp_path,
                    window=args.window,
                    steps_elapsed=steps_elapsed,
                    detect_interval=args.detect_interval,
                    resolve_key_fn=key_resolver.get_agent_key,
                    cross_agent_key_fn=key_resolver.get_key_via_chain,
                )
            )
        except Exception as e:
            log.error(f"Live detection failed: {e}", exc_info=True)
            last_detect_step = current_step
            await asyncio.sleep(2.0)
            continue

        last_detect_step = current_step

        if is_first_cycle and restart_filter_step >= 0:
            before = len(events)
            events = [e for e in events if e.step > restart_filter_step]
            if before != len(events):
                log.info(
                    f"Filtered {before - len(events)} already-published event(s) "
                    f"(step <= {restart_filter_step})"
                )

        died_agents = scan_result.died_agents
        if is_first_cycle and not args.catchup and died_agents:
            log.info(
                f"Skipping {len(died_agents)} historical death(s) "
                f"(use --catchup to process them)"
            )
            died_agents = []

        # Phylogeny is on-demand only: dashboard requests are the normal source.
        # On the first cycle with --catchup, push every existing artifact name so
        # historical artifacts get processed once (the worker dedups).
        # Catchup + deferred-retry have no per-request requester — they're
        # billed to the admin fallback key. If no admin key is set, we log and
        # skip rather than silently use the operator's env key.
        if redis_client is not None:
            admin_encrypted = await redis_client.get(KEY_ANTHRO_ADMIN_API_KEY)
            if is_first_cycle and args.catchup:
                historical_names = anthropologist.list_artifact_names()
                if historical_names:
                    if not admin_encrypted:
                        log.warning(
                            "Catchup phylogeny skipped: %d historical artifact(s) need "
                            "the admin fallback key but none is set",
                            len(historical_names),
                        )
                    else:
                        try:
                            entries = [
                                json.dumps({
                                    "name": n,
                                    "encrypted_api_key": admin_encrypted,
                                    "user_id": None,
                                    "session_token": None,
                                })
                                for n in historical_names
                            ]
                            await redis_client.rpush(phylogeny_queue_key, *entries)
                            log.info(
                                f"Catchup: enqueued {len(historical_names)} historical "
                                f"artifact(s) for phylogeny (admin key)"
                            )
                        except Exception as e:
                            log.warning(f"Failed to catchup-enqueue phylogeny names: {e}")
            if phylogeny_retry_names:
                if not admin_encrypted:
                    log.warning(
                        "Deferred phylogeny skipped: %d name(s) need the admin fallback "
                        "key but none is set",
                        len(phylogeny_retry_names),
                    )
                else:
                    try:
                        entries = [
                            json.dumps({
                                "name": n,
                                "encrypted_api_key": admin_encrypted,
                                "user_id": None,
                                "session_token": None,
                            })
                            for n in phylogeny_retry_names
                        ]
                        await redis_client.rpush(phylogeny_queue_key, *entries)
                    except Exception as e:
                        log.warning(f"Failed to enqueue deferred phylogeny names: {e}")
        is_first_cycle = False

        narratable = [e for e in events if e.severity >= args.dispatch_severity]
        if narratable:
            def _narrative_key(ev) -> str | None:
                involved: list[str] = []
                if "agent_tag" in ev.data and ev.data["agent_tag"]:
                    involved.append(ev.data["agent_tag"])
                if "agent_tags" in ev.data and ev.data["agent_tags"]:
                    involved.extend(ev.data["agent_tags"])
                return key_resolver.get_key_via_chain(involved)

            results = await asyncio.gather(
                *[
                    anthropologist.get_llm_narrative(
                        e, graph_snapshot, api_key=_narrative_key(e),
                    )
                    for e in narratable
                ],
                return_exceptions=True,
            )
            for ev, res in zip(narratable, results):
                if isinstance(res, BaseException):
                    log.warning(f"Narrative generation failed for {ev.kind}: {res}")

        if events:
            event_dicts = [dataclasses.asdict(e) for e in events]
            payload = {"type": "field_note", "step": current_step, "events": event_dicts}
            _append_field_notes_to_file(field_notes_file, event_dicts)
            if redis_client is not None:
                try:
                    await redis_client.publish(CHANNEL_PUSH, json.dumps(payload))
                    await _push_field_notes_to_redis(redis_client, event_dicts, args.exp_name)
                    log.info(f"Published {len(events)} event(s) at step {current_step}")
                except Exception as e:
                    log.error(f"Failed to publish field notes: {e}")
            else:
                log.info("Field notes payload (no-redis): %s", json.dumps(payload, indent=2))

        if redis_client is not None:
            try:
                await redis_client.set(last_detect_step_key, current_step)
            except Exception as e:
                log.warning(f"Failed to persist last_detect_step: {e}")

            live_annotations = anthropologist.get_live_annotations()
            if live_annotations:
                try:
                    await redis_client.set(
                        key_live_annotations(args.exp_name),
                        json.dumps(live_annotations),
                    )
                    await redis_client.publish(
                        CHANNEL_PUSH,
                        json.dumps({"type": "live_annotations", "annotations": live_annotations}),
                    )
                except Exception as e:
                    log.error(f"Failed to publish live annotations: {e}")

        try:
            await _publish_classifications(
                redis_client, anthropologist, exp_path,
                phylo_requested, phylo_requested_lock, args.exp_name,
            )
        except Exception as e:
            log.error(f"Failed to publish artifact classifications: {e}")

        if args.auto_postmortems:
            for dead_agent in died_agents:
                dead_name = scan_result.died_agent_names.get(dead_agent)
                # Auto-postmortems have no requester: use the dead agent's
                # owner key (ownerless → admin fallback). Skip with a log if
                # neither is available.
                auto_key = key_resolver.get_agent_key(dead_agent)
                if auto_key is None:
                    log.info(
                        "Auto-postmortem skipped for %s — no API key (owner has none, no admin fallback)",
                        _fmt(dead_agent, dead_name),
                    )
                    continue
                submit_postmortem(
                    dead_agent, dead_name, current_step,
                    auto_key, None, None,
                )

        cycles_since_checkpoint += 1
        if cycles_since_checkpoint >= checkpoint_every:
            await asyncio.to_thread(anthropologist.save_state, exp_path)
            cycles_since_checkpoint = 0

        await asyncio.sleep(2.0)


async def _async_main(args) -> None:
    """Async entry: builds dependencies and supervises three concurrent tasks."""
    exp_path = LOGS_DIR / args.exp_name
    if not exp_path.exists():
        log.error(f"Experiment path does not exist: {exp_path}")
        sys.exit(1)

    anthropologist = Anthropologist(
        model=args.model,
        audit=not args.no_audit,
        parallel=True,
        agent_annotation_ttl=20,
        fallback_model=args.fallback_model,
    )
    anthropologist.load_state(exp_path)
    # Seed the artifact index from the full world log so dashboard phylogeny
    # requests for pre-restart artifacts can resolve. The live LogScanner only
    # emits new events after a resume, which is otherwise too narrow.
    await asyncio.to_thread(anthropologist.hydrate_artifact_index, exp_path)

    redis_client: AsyncRedis | None = None
    if not args.no_redis:
        try:
            redis_client = AsyncRedis.from_url(redis_url(), decode_responses=True)
            await redis_client.ping()
            log.info("Connected to Redis.")
            await redis_client.set(KEY_EXP_CONFIG, json.dumps({
                "exp_name": args.exp_name,
                "exp_path": str(exp_path),
            }))
            await _reset_anthro_redis_keys(redis_client, args.exp_name)
        except Exception as e:
            log.warning(f"Redis unavailable ({e}). Running without Redis.")
            redis_client = None

    # Bounded async fan-out for postmortems.
    postmortem_sem = asyncio.Semaphore(int(os.environ.get("ANTHRO_MAX_POSTMORTEMS", "8")))
    submitted: set[str] = set()
    in_flight: set[asyncio.Task] = set()
    phylo_requested: set[str] = set()
    phylo_requested_lock = threading.Lock()
    obituary_tags = _scan_obituary_tags(exp_path)
    if redis_client is not None and obituary_tags:
        await _publish_obituary_index(redis_client, obituary_tags)

    # Per-user API key resolver. Refreshed at the start of each detect cycle.
    key_resolver = AnthroKeyResolver(redis_client)

    async def _run_postmortem_bounded(
        agent_tag: str,
        agent_name: str | None,
        current_step: int,
        api_key: str | None = None,
        requester_user_id: str | None = None,
        requester_session_token: str | None = None,
    ) -> None:
        try:
            async with postmortem_sem:
                await _run_postmortem(
                    anthropologist, agent_tag, exp_path, current_step,
                    redis_client, obituary_tags, agent_name,
                    api_key=api_key,
                    requester_user_id=requester_user_id,
                    requester_session_token=requester_session_token,
                )
        except Exception as e:
            log.warning(f"Postmortem failed for {_fmt(agent_tag, agent_name)}: {e}")
        finally:
            submitted.discard(agent_tag)

    def _submit_postmortem(
        agent_tag: str,
        agent_name: str | None,
        current_step: int,
        api_key: str | None = None,
        requester_user_id: str | None = None,
        requester_session_token: str | None = None,
    ) -> None:
        """Submit a postmortem, skipping if it's in flight or already on disk."""
        if agent_tag in submitted:
            log.info(f"Postmortem already submitted for {_fmt(agent_tag, agent_name)}, skipping")
            return
        if agent_tag in anthropologist._postmortem_done:
            log.info(f"Postmortem already on disk for {_fmt(agent_tag, agent_name)}, skipping")
            return
        submitted.add(agent_tag)
        task = asyncio.create_task(
            _run_postmortem_bounded(
                agent_tag, agent_name, current_step,
                api_key, requester_user_id, requester_session_token,
            ),
            name=f"postmortem-{agent_tag}",
        )
        in_flight.add(task)
        task.add_done_callback(in_flight.discard)
        log.info(f"Postmortem queued for {_fmt(agent_tag, agent_name)}")

    if sys.stdin.isatty():
        def _watch_stdin():
            try:
                while True:
                    if sys.stdin.read(1) == "":
                        log.warning("Force kill (Ctrl+D). Terminating immediately...")
                        os._exit(1)
            except Exception:
                pass

        threading.Thread(target=_watch_stdin, daemon=True, name="stdin-watcher").start()
        log.info("Press Ctrl+D to force kill immediately, Ctrl+C to stop after current task.")

    log.info(
        f"Starting live anthropologist: exp={args.exp_name}, "
        f"detect_interval={args.detect_interval} steps, "
        f"window={args.window}, dispatch_severity>={args.dispatch_severity}, "
        f"audit={not args.no_audit}, catchup={args.catchup}"
    )

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _request_stop():
        log.info("Shutting down...")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:
            signal.signal(sig, lambda *_: _request_stop())

    tasks: list[asyncio.Task] = [
        asyncio.create_task(
            _detector_loop(
                args, anthropologist, redis_client, exp_path,
                _submit_postmortem, phylo_requested, phylo_requested_lock,
                key_resolver,
            ),
            name="anthro-detector",
        )
    ]
    if redis_client is not None:
        tasks.append(asyncio.create_task(
            _postmortem_drain_loop(args, redis_client, _submit_postmortem),
            name="anthro-postmortem",
        ))
        tasks.append(asyncio.create_task(
            _phylogeny_worker_loop(
                args, anthropologist, redis_client, exp_path,
                phylo_requested, phylo_requested_lock,
            ),
            name="anthro-phylogeny",
        ))

    try:
        await stop_event.wait()
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if in_flight:
            log.info("Waiting for %d in-flight postmortem(s) to finish...", len(in_flight))
            await asyncio.gather(*in_flight, return_exceptions=True)
        try:
            await asyncio.to_thread(anthropologist.save_state, exp_path)
        except Exception as e:
            log.warning(f"Final state save failed: {e}")
        if redis_client is not None:
            await redis_client.aclose()
        log.info("Anthropologist server stopped.")


def main():
    args = parse_args()
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
