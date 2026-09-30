"""
LogScanner: incremental log reader for the live anthropologist.

Tracks a byte offset per file so each scan() call reads only new lines
since the previous call.

Overlapping windows: when scan(window=N) is used, new events are added to
an in-memory rolling buffer that covers the last N steps.  This means
consecutive calls with the same window return overlapping data — e.g.
scan every 10 steps with window=50 gives 80% overlap, so graph/community
metrics are computed on rich data while the feed updates frequently.
died_agents is never buffered: it always contains only newly-seen deaths.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Edge weights — mirror the defaults in graph_utils.py
_ALPHA_GIVE = 1.0
_ALPHA_STEAL = -1.0
_ALPHA_PARENT = 5.0
_ALPHA_GIVE_ARTIFACT = 5.0

# Exact-match action types → fingerprint bucket
_FINGERPRINT_BUCKET_EXACT = {
    "move": "passive",
    "give": "cooperative",
    "take": "aggressive",
    "spawn": "spawning",
    "set_color": "social",
    "external_action": "tool_use",
    "create_artifact": "artifact_positive",
    "pickup_artifact": "artifact_positive",
    "give_artifact": "artifact_positive",
    "drop_artifact": "artifact_neutral",
    "modify_artifact": "artifact_positive",
    "destroy_artifact": "artifact_destructive",
}
# Prefix-match action types → fingerprint bucket (checked when exact match
# fails). Kept for logs written before modify/destroy became generic actions.
_FINGERPRINT_BUCKET_PREFIX = {
    "modify_artifact_": "artifact_positive",
    "destroy_artifact_": "artifact_destructive",
}
# Location affordance actions and anything unrecognised → "other"

# Files in agent_logs/ that are not per-agent logs. token_counts.jsonl is no
# longer written (usage now lives in the run's costs.csv), but older runs on
# disk still have it, so keep excluding it.
_EXCLUDED_AGENT_LOG_FILES = {"token_counts.jsonl"}


@dataclass
class ScanResult:
    interactions: list[dict]             # {step, source, target, weight, sign}
    artifact_events: list[dict]          # {step, agent_tag, artifact_id, artifact_name, action}
    agent_actions: list[dict]            # {step, agent_tag, action_type, energy, time}
    died_agents: list[str]               # agent_tags from AGENT_DIED — always fresh, never buffered
    died_agent_names: dict[str, str]     # tag → name for agents in died_agents (best-effort)
    expired_artifact_ids: list[str]      # artifact names removed by lifespan expiry — always fresh, never buffered
    raw_artifact_world_events: list[dict] # full raw obj for ARTIFACT_ADDED, ARTIFACT_INTERACTION(modify), ARTIFACT_REMOVED — always fresh, for ExperimentArtifacts.update_from_events()
    steps_covered: tuple[int, int] | None  # (first_step, last_step); None if no events


class LogScanner:
    """
    Incremental reader for the world log and all per-agent logs.

    Each call to scan() reads only bytes appended since the previous call
    (O(new data)).  When a window is specified, an in-memory buffer keeps
    the last `window` steps so that consecutive scans overlap correctly.
    """

    def __init__(self, exp_path: Path):
        self._exp_path = Path(exp_path)
        self._offsets: dict[str, int] = {}  # filepath → byte offset
        self._known_agents: set[str] = set()  # agent log stems already tracked

        # Seed known agents from files that already exist (agents born before
        # the scanner started — their AGENT_ADDED events are already past offset 0).
        agent_logs_dir = self._exp_path / "agent_logs"
        if agent_logs_dir.exists():
            for p in agent_logs_dir.glob("*.jsonl"):
                if p.name not in _EXCLUDED_AGENT_LOG_FILES:
                    self._known_agents.add(p.stem)

        # Rolling buffers for overlapping-window support.
        # Only populated when scan(window=N) is called.
        self._buf_interactions: list[dict] = []
        self._buf_artifact_events: list[dict] = []
        self._buf_agent_actions: list[dict] = []

        # Artifact content registry — persists across scans, never reset.
        # name → {name, payload, step}; updated on ARTIFACT_ADDED and modifications.
        self._latest_artifact_registry: dict[str, dict] = {}

        # Agent name registry — persists across scans, never reset.
        # tag → human-readable name; populated from AGENT_ADDED and AGENT_DIED events.
        self._agent_name_registry: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(self, window: int | None = None) -> ScanResult:
        """
        Read new log lines and return events.

        Args:
            window: Rolling window in simulation steps.  When set, the
                    returned ScanResult covers the last `window` steps
                    (not just the newly-read lines), enabling overlapping
                    windows.  Call every K steps with window >> K for a
                    smooth, data-rich feed.  When None, only newly-read
                    events are returned.

        Returns:
            ScanResult.  died_agents always contains only deaths seen in
            this specific call, regardless of window.
        """
        new_interactions: list[dict] = []
        new_artifact_events: list[dict] = []
        new_agent_actions: list[dict] = []
        died_agents: list[str] = []
        died_agent_names: dict[str, str] = {}
        expired_artifact_ids: list[str] = []
        raw_artifact_world_events: list[dict] = []

        # ── World log ─────────────────────────────────────────────────
        worldlog_path = self._exp_path / "open_gridworld.log"
        if worldlog_path.exists():
            for line in self._read_new_lines(worldlog_path):
                obj = _parse_json_line(line)
                if obj is None:
                    continue
                step = int(obj.get("timestamp", 0))
                event = obj.get("event", "")

                if event == "GIFT_ENERGY":
                    src = obj.get("agent_tag")
                    tgt = obj.get("target_tag")
                    amt = obj.get("amount", 0)
                    if src and tgt and amt:
                        new_interactions.append(
                            {
                                "step": step,
                                "source": str(src),
                                "target": str(tgt),
                                "weight": _ALPHA_GIVE * float(amt),
                                "sign": 1,
                            }
                        )

                elif event == "TAKE_ENERGY":
                    src = obj.get("agent_tag")
                    tgt = obj.get("target_tag")
                    amt = obj.get("amount", 0)
                    if src and tgt and amt:
                        new_interactions.append(
                            {
                                "step": step,
                                "source": str(src),
                                "target": str(tgt),
                                "weight": _ALPHA_STEAL * float(amt),
                                "sign": -1,
                            }
                        )

                elif event.startswith("AGENT_SPAWNED"):
                    src = obj.get("agent_tag")
                    child = obj.get("child_tag")
                    if src and child:
                        new_interactions.append(
                            {
                                "step": step,
                                "source": str(src),
                                "target": str(child).lower(),
                                "weight": _ALPHA_PARENT,
                                "sign": 1,
                            }
                        )

                elif event == "GIVE_ARTIFACT":
                    src = obj.get("agent_tag")
                    tgt = obj.get("target_tag")
                    if src and tgt and obj.get("status") == "Success":
                        new_interactions.append(
                            {
                                "step": step,
                                "source": str(src),
                                "target": str(tgt),
                                "weight": _ALPHA_GIVE_ARTIFACT,
                                "sign": 1,
                            }
                        )

                elif event == "ARTIFACT_PASSIVE_INTERACTION":
                    # Agent is near an artifact and passively reads/observes it.
                    # Not recorded in the agent log as an action, only in world log.
                    tag = obj.get("agent_tag")
                    art_name = obj.get("artifact_name")
                    if tag and art_name:
                        new_artifact_events.append(
                            {
                                "step": step,
                                "agent_tag": str(tag),
                                "artifact_id": str(art_name),
                                "artifact_name": str(art_name),
                                "action": "read",
                            }
                        )

                elif event == "ARTIFACT_ADDED":
                    art = obj.get("artifact", {})
                    name = art.get("name")
                    if name:
                        self._latest_artifact_registry[str(name)] = {
                            "name": str(name),
                            "payload": art.get("payload", ""),
                            "step": step,
                        }
                    raw_artifact_world_events.append(obj)

                elif event == "ARTIFACT_INTERACTION":
                    action = obj.get("action", "")
                    if action == "modify_artifact" or action.startswith(
                        "modify_artifact_"
                    ):
                        art = obj.get("artifact", {})
                        name = art.get("name")
                        if name:
                            self._latest_artifact_registry[str(name)] = {
                                "name": str(name),
                                "payload": art.get("payload", ""),
                                "step": step,
                            }
                        raw_artifact_world_events.append(obj)

                elif event == "ARTIFACT_REMOVED":
                    art = obj.get("artifact", {})
                    name = art.get("name")
                    if name:
                        expired_artifact_ids.append(str(name))
                    raw_artifact_world_events.append(obj)

                elif event == "AGENT_ADDED":
                    tag = obj.get("agent_tag")
                    name = obj.get("agent_name")
                    if tag:
                        self._known_agents.add(str(tag))
                    if tag and name:
                        self._agent_name_registry[str(tag)] = str(name)

                elif event == "AGENT_DIED":
                    tag = obj.get("agent_tag")
                    if tag:
                        died_agents.append(str(tag))
                        name = obj.get("agent_name")
                        if name:
                            died_agent_names[str(tag)] = str(name)
                            self._agent_name_registry[str(tag)] = str(name)

        # ── Agent logs ────────────────────────────────────────────────
        agent_logs_dir = self._exp_path / "agent_logs"
        if agent_logs_dir.exists():
            for tag in sorted(self._known_agents):
                agent_file = agent_logs_dir / f"{tag}.jsonl"
                if not agent_file.exists():
                    continue
                for line in self._read_new_lines(agent_file):
                    obj = _parse_json_line(line)
                    if obj is None:
                        continue
                    step = int(obj.get("timestamp", 0))

                    action_data = obj.get("action", {})
                    if isinstance(action_data, str):
                        action_type = action_data
                        action_params: dict = {}
                    else:
                        action_type = action_data.get("action", "")
                        action_params = action_data.get("params") or {}

                    obs = obj.get("observation") or {}
                    energy = obs.get("energy", 0)
                    time_left = obs.get("time", 0)

                    new_agent_actions.append(
                        {
                            "step": step,
                            "agent_tag": tag,
                            "action_type": str(action_type),
                            "energy": energy,
                            "time": time_left,
                        }
                    )

                    art_event = _parse_artifact_action(action_type, action_params)
                    if art_event is not None:
                        new_artifact_events.append(
                            {
                                "step": step,
                                "agent_tag": tag,
                                "artifact_id": art_event[1],
                                "artifact_name": art_event[1],
                                "action": art_event[0],
                            }
                        )

        # ── Merge into buffers and apply window ───────────────────────
        if window is not None:
            self._buf_interactions.extend(new_interactions)
            self._buf_artifact_events.extend(new_artifact_events)
            self._buf_agent_actions.extend(new_agent_actions)

            # Determine current max step across all buffered events
            all_buffered = (
                self._buf_interactions
                + self._buf_artifact_events
                + self._buf_agent_actions
            )
            if all_buffered:
                max_step = max(e["step"] for e in all_buffered)
                cutoff = max_step - window
                self._buf_interactions = [
                    e for e in self._buf_interactions if e["step"] >= cutoff
                ]
                self._buf_artifact_events = [
                    e for e in self._buf_artifact_events if e["step"] >= cutoff
                ]
                self._buf_agent_actions = [
                    e for e in self._buf_agent_actions if e["step"] >= cutoff
                ]

            interactions = self._buf_interactions
            artifact_events = self._buf_artifact_events
            agent_actions = self._buf_agent_actions
        else:
            interactions = new_interactions
            artifact_events = new_artifact_events
            agent_actions = new_agent_actions

        # ── steps_covered ─────────────────────────────────────────────
        all_steps = (
            [e["step"] for e in interactions]
            + [e["step"] for e in artifact_events]
            + [e["step"] for e in agent_actions]
        )
        steps_covered = (min(all_steps), max(all_steps)) if all_steps else None

        return ScanResult(
            interactions=interactions,
            artifact_events=artifact_events,
            agent_actions=agent_actions,
            died_agents=died_agents,
            died_agent_names=died_agent_names,
            expired_artifact_ids=expired_artifact_ids,
            raw_artifact_world_events=raw_artifact_world_events,
            steps_covered=steps_covered,
        )

    def get_agent_fingerprints(self) -> dict[str, dict[str, float]]:
        """
        Return {agent_tag: {bucket: fraction}} over the current agent action buffer
        (or last scan if no window was used).

        Buckets: artifact_positive, artifact_destructive, artifact_neutral,
                 cooperative, aggressive, spawning, social, tool_use,
                 passive, other (location affordances / unrecognised).
        Used by run_detection() to identify agents whose behaviour shifted.
        """
        source = self._buf_agent_actions if self._buf_agent_actions else []
        counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for action in source:
            tag = action["agent_tag"]
            bucket = _get_fingerprint_bucket(action["action_type"])
            counts[tag][bucket] += 1

        fingerprints: dict[str, dict[str, float]] = {}
        for tag, buckets in counts.items():
            total = sum(buckets.values())
            fingerprints[tag] = (
                {b: c / total for b, c in buckets.items()} if total > 0 else {}
            )
        return fingerprints

    def get_latest_artifact_registry(self) -> dict[str, dict]:
        """
        Return the accumulated artifact content registry.

        Keys are artifact name strings (same identifiers used in artifact_events).
        Values are {name, payload, step} dicts reflecting the latest known version
        of each artifact (updated on creation and modification).
        """
        return self._latest_artifact_registry

    def get_agent_name_registry(self) -> dict[str, str]:
        """
        Return the accumulated agent name registry (tag → human-readable name).

        Populated from AGENT_ADDED and AGENT_DIED events; covers all agents
        ever seen by the scanner, including dead ones.
        """
        return self._agent_name_registry

    # ------------------------------------------------------------------
    # State persistence (snapshot/restore)
    # ------------------------------------------------------------------

    def export_state(self) -> dict:
        """Serialize the scanner's incremental state to a JSON-able dict.

        The file byte offsets are the critical piece — restoring them lets
        the next scan() resume reading exactly where the previous one
        stopped, instead of replaying every log line from byte 0.
        """
        return {
            "offsets": dict(self._offsets),
            "known_agents": sorted(self._known_agents),
            "buf_interactions": list(self._buf_interactions),
            "buf_artifact_events": list(self._buf_artifact_events),
            "buf_agent_actions": list(self._buf_agent_actions),
            "latest_artifact_registry": dict(self._latest_artifact_registry),
            "agent_name_registry": dict(self._agent_name_registry),
        }

    def restore_state(self, state: dict) -> None:
        """Restore scanner state from a previously-exported dict.

        Tolerates missing keys (older snapshots) by leaving the
        corresponding field at its default value.
        """
        self._offsets = dict(state.get("offsets", {}))
        if state.get("known_agents"):
            self._known_agents = set(state["known_agents"])
        if state.get("buf_interactions") is not None:
            self._buf_interactions = list(state["buf_interactions"])
        if state.get("buf_artifact_events") is not None:
            self._buf_artifact_events = list(state["buf_artifact_events"])
        if state.get("buf_agent_actions") is not None:
            self._buf_agent_actions = list(state["buf_agent_actions"])
        if state.get("latest_artifact_registry") is not None:
            self._latest_artifact_registry = dict(state["latest_artifact_registry"])
        if state.get("agent_name_registry") is not None:
            self._agent_name_registry = dict(state["agent_name_registry"])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_new_lines(self, filepath: Path) -> list[str]:
        """
        Read bytes from `filepath` starting at the stored offset.
        Only complete lines (terminated by \\n) are returned; the offset
        advances to just after the last complete line.
        """
        key = str(filepath)
        offset = self._offsets.get(key, 0)
        lines: list[str] = []
        try:
            with open(filepath, "rb") as fh:
                fh.seek(offset)
                chunk = fh.read()
            if not chunk:
                return lines
            last_nl = chunk.rfind(b"\n")
            if last_nl == -1:
                return lines  # no complete line yet
            complete = chunk[: last_nl + 1]
            self._offsets[key] = offset + len(complete)
            lines = complete.decode("utf-8", errors="replace").splitlines()
        except FileNotFoundError:
            pass
        return lines


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _parse_json_line(line: str) -> dict | None:
    """Parse a single JSONL line; return None on error."""
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def _parse_artifact_action(action_type: str, params: dict) -> tuple[str, str] | None:
    """
    Return (normalized_action, artifact_name) for artifact-related agent actions,
    or None if the action is not artifact-related.

    Handles exact matches (create_artifact, pickup_artifact, give_artifact,
    modify_artifact, destroy_artifact) and the legacy prefix forms
    (modify_artifact_<name>, destroy_artifact_<name>).
    artifact_name is used as the artifact identifier in the live system.
    """
    if action_type == "create_artifact":
        name = str(params.get("name", ""))
        return ("create", name) if name else None
    if action_type == "pickup_artifact":
        name = str(params.get("name", ""))
        return ("collect", name) if name else None
    if action_type == "give_artifact":
        name = str(params.get("artifact_name", ""))
        return ("exchange", name) if name else None
    if action_type == "modify_artifact":
        name = str(params.get("name", ""))
        return ("modify", name) if name else None
    if action_type == "destroy_artifact":
        name = str(params.get("name", ""))
        return ("destroy", name) if name else None
    if action_type.startswith("modify_artifact_"):
        name = action_type[len("modify_artifact_") :]
        return ("modify", name) if name else None
    if action_type.startswith("destroy_artifact_"):
        name = action_type[len("destroy_artifact_") :]
        return ("destroy", name) if name else None
    return None


def _get_fingerprint_bucket(action_type: str) -> str:
    """Map a raw action type string to a fingerprint bucket name."""
    bucket = _FINGERPRINT_BUCKET_EXACT.get(action_type)
    if bucket is not None:
        return bucket
    for prefix, b in _FINGERPRINT_BUCKET_PREFIX.items():
        if action_type.startswith(prefix):
            return b
    return "other"


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m terralingua.anthropologist.log_scanner <exp_path> [window]")
        sys.exit(1)

    exp_path = Path(sys.argv[1])
    window = int(sys.argv[2]) if len(sys.argv) > 2 else None

    scanner = LogScanner(exp_path)

    print(f"=== Scan 1 (window={window}) ===")
    r = scanner.scan(window=window)
    print(f"  interactions    : {len(r.interactions)}")
    print(f"  artifact_events : {len(r.artifact_events)}")
    print(f"  agent_actions   : {len(r.agent_actions)}")
    print(f"  died_agents     : {r.died_agents}")
    print(f"  steps_covered   : {r.steps_covered}")

    fps = scanner.get_agent_fingerprints()
    print(f"  fingerprints ({len(fps)} agents):")
    for tag, buckets in sorted(fps.items()):
        parts = ", ".join(f"{b}={v:.2f}" for b, v in sorted(buckets.items()))
        print(f"    {tag}: {parts}")

    print()
    print("=== Scan 2 (incremental check — only new events, same window) ===")
    r2 = scanner.scan(window=window)
    print(f"  interactions    : {len(r2.interactions)}")
    print(f"  artifact_events : {len(r2.artifact_events)}")
    print(f"  agent_actions   : {len(r2.agent_actions)}")
    print(f"  steps_covered   : {r2.steps_covered}")
