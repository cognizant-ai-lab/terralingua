#!/usr/bin/env python3
"""
ogw_log.py — shared, scenario-agnostic parsers for an OGW run's world/agent logs.

These loaders reconstruct generic simulation series (population, energy, lifecycle)
from `open_graph_world.log` / `open_social_graph_world.log` + `agent_logs/`, with no
knowledge of any particular scenario. Per-scenario analysis scripts live under
`scenarios/<name>/analysis/` and import from here, so they do not depend on each
other's folders.
"""

import json
from collections import defaultdict
from pathlib import Path


def _resolve_log(exp_dir: Path) -> Path | None:
    """Return the world-log path, whichever naming this run uses."""
    for name in ("open_graph_world.log", "open_social_graph_world.log"):
        path = exp_dir / name
        if path.exists():
            return path
    return None


def _parse_world_log(log_path: Path) -> dict:
    """Single pass over open_graph_world.log, collecting all event data needed by plot()."""
    data: dict = {
        "last_ts": 0,
        "initial_food": 0.0,
        "food_injections": defaultdict(float),        # RESOURCE_SIGNAL where target=None
        "all_resource_injections": defaultdict(float),  # all RESOURCE_SIGNAL
        "ext_req_counts": defaultdict(int),
        "added_by_ts": defaultdict(list),
        "died_by_ts": defaultdict(list),
    }
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            ev = d.get("event")
            ts = int(d.get("timestamp", 0))
            data["last_ts"] = max(data["last_ts"], ts)
            if ev == "ENV_RESET":
                data["initial_food"] = float(d.get("food_count", 0))
            elif ev == "RESOURCE_SIGNAL":
                eff = float(d.get("effective", 0.0))
                data["all_resource_injections"][ts] += eff
                if d.get("target") is None:
                    data["food_injections"][ts] += eff
            elif ev == "EXTERNAL_REQUEST":
                data["ext_req_counts"][ts] += 1
            elif ev == "AGENT_ADDED":
                tag = d.get("agent_tag")
                if tag:
                    data["added_by_ts"][ts].append(tag)
            elif ev == "AGENT_DIED":
                tag = d.get("agent_tag")
                if tag:
                    data["died_by_ts"][ts].append(tag)
    return data


def _load_init_agent_energy(exp_dir: Path) -> float:
    params_path = exp_dir / "params.json"
    if not params_path.exists():
        return 0.0
    try:
        with open(params_path) as f:
            params = json.load(f)
        return float(params.get("env", {}).get("init_agent_energy", 0.0))
    except (json.JSONDecodeError, TypeError, ValueError):
        return 0.0


def load_agent_energy_series(
    exp_dir: Path, mature_age: int = 25, _log_data: dict | None = None
) -> dict[str, list[float] | list[int]]:
    """
    Compute multiple energy series from agent logs and lifecycle events.

    Returns:
      {
        "timesteps": [...],
        "total_energy": [...],            # sum across living agents only
        "avg_live_energy": [...],         # total / live count
        "respawn_adjusted_energy": [...], # excludes agents younger than mature_age
        "live_counts": [...],
      }
    """
    _empty: dict[str, list] = {
        "timesteps": [],
        "total_energy": [],
        "avg_live_energy": [],
        "respawn_adjusted_energy": [],
        "live_counts": [],
    }
    agent_log_dir = exp_dir / "agent_logs"
    if not agent_log_dir.exists():
        return _empty
    if _log_data is None and not (exp_dir / "open_graph_world.log").exists():
        return _empty

    init_agent_energy = _load_init_agent_energy(exp_dir)

    energy_updates_by_ts: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for path in agent_log_dir.glob("*.jsonl"):
        tag = path.stem
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    ts = d.get("timestamp")
                    obs = d.get("observation", {})
                    energy = obs.get("energy")
                    if energy is None and isinstance(obs, dict):
                        energy = obs.get("observation", {}).get("energy")
                    if ts is not None and energy is not None:
                        energy_updates_by_ts[int(ts)].append((tag, float(energy)))
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue

    if _log_data is not None:
        added_by_ts = _log_data["added_by_ts"]
        died_by_ts = _log_data["died_by_ts"]
    else:
        added_by_ts = defaultdict(list)
        died_by_ts = defaultdict(list)
        with open(exp_dir / "open_graph_world.log") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ev = d.get("event")
                ts = int(d.get("timestamp", 0))
                if ev == "AGENT_ADDED":
                    tag = d.get("agent_tag")
                    if tag:
                        added_by_ts[ts].append(tag)
                elif ev == "AGENT_DIED":
                    tag = d.get("agent_tag")
                    if tag:
                        died_by_ts[ts].append(tag)

    all_ts = sorted(set(energy_updates_by_ts) | set(added_by_ts) | set(died_by_ts))
    if not all_ts:
        return _empty

    live_energy: dict[str, float] = {}
    spawn_ts: dict[str, int] = {}
    result_total: list[float] = []
    result_avg: list[float] = []
    result_adjusted: list[float] = []
    result_live_counts: list[int] = []

    for ts in all_ts:
        for tag in added_by_ts.get(ts, []):
            spawn_ts[tag] = ts
            if tag not in live_energy:
                live_energy[tag] = init_agent_energy

        for tag, energy in energy_updates_by_ts.get(ts, []):
            live_energy[tag] = energy
            spawn_ts.setdefault(tag, ts)

        for tag in died_by_ts.get(ts, []):
            live_energy.pop(tag, None)

        total = sum(live_energy.values())
        live_n = len(live_energy)
        adjusted = sum(
            energy
            for tag, energy in live_energy.items()
            if ts - spawn_ts.get(tag, ts) >= mature_age
        )
        result_total.append(total)
        result_avg.append(total / live_n if live_n else 0.0)
        result_adjusted.append(adjusted)
        result_live_counts.append(live_n)

    return {
        "timesteps": all_ts,
        "total_energy": result_total,
        "avg_live_energy": result_avg,
        "respawn_adjusted_energy": result_adjusted,
        "live_counts": result_live_counts,
    }


def load_agent_energy(exp_dir: Path) -> tuple[list[int], list[float]]:
    """Backward-compatible wrapper around the richer energy loader."""
    series = load_agent_energy_series(exp_dir)
    return series["timesteps"], series["total_energy"]


def load_agent_counts(exp_dir: Path, _log_data: dict | None = None) -> tuple[list[int], list[int]]:
    """
    Reconstruct agent population over time from the OGW log.
    Returns (timesteps, counts) lists derived from AGENT_ADDED / AGENT_DIED events.
    """
    if _log_data is None:
        log_path = exp_dir / "open_graph_world.log"
        if not log_path.exists():
            return [], []
        _log_data = _parse_world_log(log_path)

    added_by_ts = _log_data["added_by_ts"]
    died_by_ts = _log_data["died_by_ts"]
    last_ts = _log_data["last_ts"]

    if not added_by_ts and not died_by_ts:
        return [], []

    # AGENT_ADDED events include the initial cohort at t=0, so start from 0.
    delta: dict[int, int] = {}
    for ts, tags in added_by_ts.items():
        delta[ts] = delta.get(ts, 0) + len(tags)
    for ts, tags in died_by_ts.items():
        delta[ts] = delta.get(ts, 0) - len(tags)

    max_ts = max(last_ts, max(delta.keys(), default=0))
    timesteps = list(range(max_ts + 1))
    counts = []
    current = 0
    for t in timesteps:
        current += delta.get(t, 0)
        counts.append(max(current, 0))

    return timesteps, counts


def load_lifecycle_totals(exp_dir: Path, _log_data: dict | None = None) -> tuple[list[int], list[int], list[int]]:
    """Return cumulative deaths and post-start respawns."""
    if _log_data is None:
        log_path = exp_dir / "open_graph_world.log"
        if not log_path.exists():
            return [], [], []
        _log_data = _parse_world_log(log_path)

    added_by_ts = _log_data["added_by_ts"]
    died_by_ts = _log_data["died_by_ts"]
    last_ts = _log_data["last_ts"]

    death_delta: dict[int, int] = defaultdict(int)
    respawn_delta: dict[int, int] = defaultdict(int)
    for ts, tags in died_by_ts.items():
        death_delta[ts] += len(tags)
    for ts, tags in added_by_ts.items():
        if ts > 0:
            respawn_delta[ts] += len(tags)

    if last_ts == 0 and not death_delta and not respawn_delta:
        return [], [], []

    timesteps = list(range(last_ts + 1))
    deaths = []
    respawns = []
    d_running = 0
    r_running = 0
    for ts in timesteps:
        d_running += death_delta.get(ts, 0)
        r_running += respawn_delta.get(ts, 0)
        deaths.append(d_running)
        respawns.append(r_running)
    return timesteps, deaths, respawns
