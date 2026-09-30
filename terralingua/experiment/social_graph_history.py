"""Durable end-of-step social topology, independent of dashboard/video output.

Rows are full snapshots, not intra-step mutation events. A new segment records the
restored state and supersedes the old branch from that state boundary onward.
"""

import json
import math
import uuid
from pathlib import Path

SCHEMA_VERSION = 1


def _number(value):
    """Keep JSON strict, representing unlimited/missing counters as null."""
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return int(value) if value.is_integer() else value


def _graph_snapshot(env):
    nodes = sorted(env.world_graph.all_nodes())
    node_set = set(nodes)
    agents = []
    for tag in sorted(env.agent_registry):
        agents.append({
            "tag": tag,
            "name": env.agent_names.get(tag, tag),
            "node": env.agent_pos[tag],
            "energy": _number(env.agent_energy.get(tag)),
            "time_left": _number(env.agent_time.get(tag)),
            "role": getattr(env, "_role_assignment", {}).get(tag)
                or getattr(env, "_agent_directives", {}).get(tag, {}).get("role"),
        })
    # Never collapse reciprocal edges: this is the actual directed topology.
    edges = [
        {"source": source, "target": target, "type": "follow"}
        for source in nodes
        for target in sorted(env.world_graph.neighbors(source))
    ]
    for sender, targets in sorted(getattr(env, "_pending_outgoing", {}).items()):
        source = env.agent_pos.get(sender)
        if sender not in env.agent_registry or source not in node_set:
            continue
        for receiver in sorted(targets):
            target = env.agent_pos.get(receiver)
            if receiver in env.agent_registry and target in node_set:
                edges.append({"source": source, "target": target, "type": "pending"})
    layout = {}
    for node, xy in (getattr(env, "_node_layout", None) or {}).items():
        if node in node_set and all(math.isfinite(float(v)) for v in xy):
            layout[node] = [float(v) for v in xy]
    return {
        "env_type": "graph",
        "directed": True,
        "topology": env.graph_cfg.topology,
        "hop_radius": int(env.hop_radius),
        "step": int(env.step_count),
        "nodes": nodes,
        "edges": edges,
        "agents": agents,
        "node_layout": layout,
        "hop_nodes": nodes,
        "food": [],
        "artifacts": [],
        "recent_messages": [],
    }


def _repair_incomplete_tail(path):
    """Discard only the unterminated last record before appending a new segment."""
    if not path.exists():
        return
    with path.open("r+b") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        if not size:
            return
        stream.seek(size - 1)
        if stream.read(1) == b"\n":
            return
        position = size
        while position:
            start = max(0, position - 65536)
            stream.seek(start)
            block = stream.read(position - start)
            newline = block.rfind(b"\n")
            if newline != -1:
                stream.truncate(start + newline + 1)
                return
            position = start
        stream.truncate(0)


class SocialGraphHistoryRecorder:
    def __init__(self, path, run_id, segment_id=None):
        self.path = Path(path)
        self.run_id = run_id
        self.segment_id = segment_id or uuid.uuid4().hex
        self._last_step = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _repair_incomplete_tail(self.path)
        # Refuse to extend a file whose completed records cannot be replayed.
        read_social_graph_history(self.path)

    def record(self, env, *, segment_start=False, reason=None):
        first = self._last_step is None
        if segment_start and not first:
            raise ValueError("Start a new recorder for a new history segment.")
        step = int(env.step_count)
        if not first and step <= self._last_step:
            raise ValueError("Social graph history steps must increase within a segment.")
        snapshot = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "segment_id": self.segment_id,
            "step": step,
            "segment_start": first,
            "graph": _graph_snapshot(env),
        }
        if first:
            snapshot["reason"] = reason or "start"
        elif reason is not None:
            snapshot["reason"] = reason
        _validate_snapshot(snapshot)
        line = json.dumps(snapshot, ensure_ascii=False, allow_nan=False)
        # Opening per row closes and flushes the record before the next phase. Errors
        # propagate: a run must not silently claim to have recorded missing data.
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        self._last_step = step
        return snapshot


def _validate_snapshot(row):
    if not isinstance(row, dict) or row.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported social graph history schema.")
    if any(not isinstance(row.get(key), str) or not row[key]
           for key in ("run_id", "segment_id")):
        raise ValueError("Missing run or segment identity.")
    if type(row.get("step")) is not int or row["step"] < 0:
        raise ValueError("Invalid social graph history step.")
    if type(row.get("segment_start")) is not bool:
        raise ValueError("Missing segment boundary.")
    graph = row.get("graph")
    if not isinstance(graph, dict) or graph.get("env_type") != "graph":
        raise ValueError("Missing graph snapshot.")
    if graph.get("step") != row["step"]:
        raise ValueError("Graph step does not match the snapshot.")
    nodes = graph.get("nodes")
    if not isinstance(nodes, list) or any(not isinstance(n, str) for n in nodes):
        raise ValueError("Invalid graph nodes.")
    if len(nodes) != len(set(nodes)):
        raise ValueError("Duplicate graph node.")
    node_set = set(nodes)
    if not isinstance(graph.get("edges"), list) or not isinstance(graph.get("agents"), list):
        raise ValueError("Missing graph edges or agents.")
    for edge in graph["edges"]:
        if (not isinstance(edge, dict)
                or edge.get("source") not in node_set
                or edge.get("target") not in node_set
                or edge.get("type") not in {"follow", "pending"}):
            raise ValueError("Invalid directed graph edge.")
    for agent in graph["agents"]:
        if (not isinstance(agent, dict) or not isinstance(agent.get("tag"), str)
                or agent.get("node") not in node_set):
            raise ValueError("Invalid graph agent.")


def read_social_graph_history(path):
    """Read the last run's canonical branch; never conceal internal corruption.

    Only an unterminated final line is ignored, as a writer may still be writing
    it or may have crashed. Each resume starts a segment whose initial snapshot
    replaces all prior snapshots at or after its restored environment step.
    """
    path = Path(path)
    history = {
        "schema_version": SCHEMA_VERSION,
        "run_id": None,
        "segment_id": None,
        "snapshots": [],
    }
    try:
        stream = path.open("rb")
    except FileNotFoundError:
        return history
    with stream:
        for line_number, line in enumerate(stream, 1):
            if not line.endswith(b"\n"):
                break
            try:
                row = json.loads(line)
                _validate_snapshot(row)
                snapshots = history["snapshots"]
                if row["segment_start"]:
                    if row["run_id"] != history["run_id"]:
                        snapshots = []
                    else:
                        snapshots = [s for s in snapshots if s["step"] < row["step"]]
                    history.update(run_id=row["run_id"], segment_id=row["segment_id"],
                                   snapshots=snapshots)
                elif (row["run_id"] != history["run_id"]
                      or row["segment_id"] != history["segment_id"]):
                    raise ValueError("History row has no matching segment start.")
                if snapshots and row["step"] <= snapshots[-1]["step"]:
                    raise ValueError("History steps must increase within a segment.")
                snapshots.append(row)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(
                    f"Invalid social graph history at line {line_number}: {exc}"
                ) from exc
    return history
