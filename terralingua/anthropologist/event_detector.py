"""
EventDetector: pure Python event detection for the live anthropologist.

No LLM calls. Receives pre-computed graph data, artifact categories, and
agent tags from run_detection() and diffs against the previous snapshot to
emit structured Event objects.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

from terralingua.anthropologist.log_scanner import LogScanner


@dataclass
class Event:
    kind: str
    severity: int  # 0=low, 1=medium, 2=high
    step: int
    data: dict
    narrative: str = ""  # filled in by Anthropologist.get_llm_narrative()


EVENT_SEVERITIES = {
    "community_split": 1,
    "community_merge": 1,
    "coalition_shift": 0,
    "bridge_agent_lost": 2,
    "polarization_spike": 2,
    "interaction_rate_surge": 1,
    "interaction_rate_collapse": 1,
    "reciprocity_collapse": 1,
    "agent_isolated": 1,
    "agent_emerged_hub": 0,
    "new_institutional_artifact": 2,
    "artifact_category_changed": 1,
    "artifact_burst": 1,
    "artifact_abandoned": 0,
    "artifact_death": 1,
    "artifact_destroyed": 2,
    "contested_artifact": 2,
    "artifact_diffusion": 1,
    "polarization_creep": 2,
    "interaction_rate_erosion": 1,
    "interaction_rate_growth": 0,
    "community_contraction": 0,
    "isolation_wave": 2,
    "artifact_rise": 0,
    "behavioral_drift": 0,
}


def _community_member_sets(
    communities: list, community_membership: dict[str, int]
) -> dict[str, frozenset]:
    """Return agent_tag → frozenset of all agents in that agent's community."""
    com_sets = [frozenset(c) for c in communities]
    return {
        tag: com_sets[idx]
        for tag, idx in community_membership.items()
        if 0 <= idx < len(com_sets)
    }


class EventDetector:
    """
    Stateful event detector.  Call detect() on each analysis cycle; the
    internal state is updated in-place so subsequent calls diff correctly.
    All thresholds are class-level constants and can be overridden per-instance.
    """

    # ── Sharp-event thresholds ─────────────────────────────────────────
    # Tuned for 10-step detect interval with 50-step overlapping window.
    # Consecutive calls share ~80% of data, so per-cycle deltas are small
    # but stable (computed over 50 steps, not 10).
    POLARIZATION_SPIKE_DELTA = 0.10  # single-cycle polarization jump
    INTERACTION_RATE_SURGE_DELTA = 0.05  # single-cycle interaction_rate increase
    INTERACTION_RATE_COLLAPSE_DELTA = -0.05  # single-cycle interaction_rate drop
    RECIPROCITY_COLLAPSE_DELTA = -0.08  # single-cycle reciprocity drop
    ARTIFACT_BURST_COUNT = 3  # new artifacts in one window
    ARTIFACT_DEATH_IDLE_STEPS = 200  # steps of zero access before flagging
    HUB_PERCENTILE = 75
    BRIDGE_CENTRALITY_THRESHOLD = 0.1
    COALITION_SHIFT_JACCARD_THRESHOLD = 0.5  # Jaccard overlap below this → real shift

    # ── Gradual-event thresholds ───────────────────────────────────────
    # N=4 covers ~40 steps of history (4 snapshots × 10-step detect_interval).
    SNAPSHOT_HISTORY_N = 4
    MONOTONICITY_K = 3  # consecutive snapshots in same direction to fire
    POLARIZATION_CREEP_DELTA = (
        0.12  # fallback: polarization rise over window (used before baseline warms)
    )
    ISOLATION_WAVE_MIN = 3  # distinct isolated agents to fire isolation_wave

    # ── Z-score adaptive baseline ──────────────────────────────────────
    # After BASELINE_MIN_SAMPLES cycles, per-metric z-scores replace fixed
    # thresholds for continuous metrics.  Fixed thresholds stay as fallback.
    BASELINE_WINDOW = 30  # rolling history length (cycles) per metric
    ZSCORE_THRESHOLD = 2.0  # standard deviations above/below mean to fire
    BASELINE_MIN_SAMPLES = 8  # cycles before z-score replaces fixed thresholds

    def __init__(self, detect_interval: int = 10):
        self._state: dict = {}
        self._detect_interval = detect_interval
        # Per-metric rolling baselines for z-score detection.
        # Key → deque of recent per-cycle delta (or window-delta) values.
        self._baselines: dict[str, deque] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        graph_data: dict,
        artifact_categories: dict,
        artifact_access_agents: dict,  # artifact_id → set[agent_tag]
        artifact_last_access: dict,  # artifact_id → last step accessed (current window)
        agent_tags: dict,  # agent_tag → list[str]
        current_step: int,
        graph: nx.DiGraph | None = None,
        destroyed_artifacts: list[dict]
        | None = None,  # {artifact_id, agent_tag} — explicitly destroyed this cycle
        expired_artifact_ids: list[str] | None = None,  # lifespan-expired this cycle
    ) -> list[Event]:
        """
        Diff the current observation against the stored snapshot.

        Args:
            graph_data: .graph_data dict from GraphData (see GraphAnalyst).
            artifact_categories: artifact_id → category string.
            artifact_access_agents: artifact_id → set of agent_tags that
                accessed it in this scan window.
            artifact_last_access: artifact_id → last step accessed (window).
            agent_tags: agent_tag → list of LLM-assigned behaviour tag strings.
            current_step: current simulation step.
            graph: optional nx.DiGraph for degree/centrality checks.

        Returns:
            List of Event objects (may be empty).
        """
        # ── Derive current metrics from graph_data ─────────────────────
        communities: list = graph_data.get("communities", [])
        node2com: dict = graph_data.get("node2com", {})
        interaction_rate = float(graph_data.get("interaction_rate", 0.0))
        reciprocity = float(graph_data.get("reciprocity", 0.0))
        polarization = float(graph_data.get("neg_polarization", 0.0))
        nodes: list = graph_data.get("nodes", [])

        community_membership = _primary_community(nodes, node2com)
        community_sizes = {i: len(c) for i, c in enumerate(communities)}
        agent_degrees = _compute_degrees(graph, nodes)
        bridge_agents = _compute_bridges(graph, self.BRIDGE_CENTRALITY_THRESHOLD)
        isolated_agents = {n for n, d in agent_degrees.items() if d < 1e-9}

        # ── Merge artifact_last_access with accumulated state ──────────
        # State carries the most-recent known access step for every artifact;
        # the current window only contains artifacts accessed THIS cycle.
        merged_last_access = dict(self._state.get("artifact_last_access", {}))
        merged_last_access.update(artifact_last_access)

        # ── First call: bootstrap state, emit nothing ──────────────────
        if not self._state:
            self._state = self._make_state(
                current_step,
                communities,
                community_membership,
                community_sizes,
                interaction_rate,
                polarization,
                reciprocity,
                bridge_agents,
                agent_degrees,
                isolated_agents,
                artifact_categories,
                artifact_access_agents,
                merged_last_access,
                agent_tags,
            )
            return []

        prev = self._state
        events: list[Event] = []

        # ── Sharp event detection ──────────────────────────────────────

        # Community split / merge
        prev_count = prev["community_count"]
        new_count = len(communities)
        if new_count > prev_count:
            events.append(
                Event(
                    "community_split",
                    2,
                    current_step,
                    {"old_count": prev_count, "new_count": new_count},
                )
            )
        elif 0 < new_count < prev_count:
            events.append(
                Event(
                    "community_merge",
                    1,
                    current_step,
                    {"old_count": prev_count, "new_count": new_count},
                )
            )

        # Coalition shifts — compare community member sets via Jaccard overlap so
        # that index relabeling between cycles does not produce false positives.
        prev_members = prev.get("community_members", {})
        cur_members = _community_member_sets(communities, community_membership)
        for tag, new_set in cur_members.items():
            old_set = prev_members.get(tag)
            if old_set is None:
                continue
            union = old_set | new_set
            if not union:
                continue
            overlap = len(old_set & new_set) / len(union)
            if overlap < self.COALITION_SHIFT_JACCARD_THRESHOLD:
                prev_com_id = prev["community_membership"].get(tag, -1)
                new_com_id = community_membership.get(tag, -1)
                events.append(
                    Event(
                        "coalition_shift",
                        0,
                        current_step,
                        {
                            "agent_tag": tag,
                            "old_community": prev_com_id,
                            "new_community": new_com_id,
                            "overlap": round(overlap, 2),
                        },
                    )
                )

        # Bridge agent lost
        prev_degrees = prev["agent_degrees"]
        for tag in prev["bridge_agents"]:
            if prev_degrees.get(tag, 0.0) > 0 and agent_degrees.get(tag, 0.0) < 1e-9:
                events.append(
                    Event(
                        "bridge_agent_lost",
                        2,
                        current_step,
                        {"agent_tag": tag, "prev_degree": prev_degrees.get(tag, 0.0)},
                    )
                )

        # Polarization spike
        pol_delta = polarization - prev["polarization"]
        pol_z = self._zscore("pol_delta", pol_delta)
        self._update_baseline("pol_delta", pol_delta)
        if (pol_z is not None and pol_z > self.ZSCORE_THRESHOLD) or (
            pol_z is None and pol_delta > self.POLARIZATION_SPIKE_DELTA
        ):
            events.append(
                Event(
                    "polarization_spike",
                    2,
                    current_step,
                    {
                        "old_val": prev["polarization"],
                        "new_val": polarization,
                        "delta": pol_delta,
                        "z_score": round(pol_z, 2) if pol_z is not None else None,
                    },
                )
            )

        # Interaction rate surge / collapse
        ir_delta = interaction_rate - prev["interaction_rate"]
        ir_z = self._zscore("ir_delta", ir_delta)
        self._update_baseline("ir_delta", ir_delta)
        if (ir_z is not None and ir_z > self.ZSCORE_THRESHOLD) or (
            ir_z is None and ir_delta > self.INTERACTION_RATE_SURGE_DELTA
        ):
            events.append(
                Event(
                    "interaction_rate_surge",
                    1,
                    current_step,
                    {
                        "old": prev["interaction_rate"],
                        "new": interaction_rate,
                        "delta": ir_delta,
                        "z_score": round(ir_z, 2) if ir_z is not None else None,
                    },
                )
            )
        elif (ir_z is not None and ir_z < -self.ZSCORE_THRESHOLD) or (
            ir_z is None and ir_delta < self.INTERACTION_RATE_COLLAPSE_DELTA
        ):
            events.append(
                Event(
                    "interaction_rate_collapse",
                    1,
                    current_step,
                    {
                        "old": prev["interaction_rate"],
                        "new": interaction_rate,
                        "delta": ir_delta,
                        "z_score": round(ir_z, 2) if ir_z is not None else None,
                    },
                )
            )

        # Reciprocity collapse
        rec_delta = reciprocity - prev["reciprocity"]
        rec_z = self._zscore("rec_delta", rec_delta)
        self._update_baseline("rec_delta", rec_delta)
        if (rec_z is not None and rec_z < -self.ZSCORE_THRESHOLD) or (
            rec_z is None and rec_delta < self.RECIPROCITY_COLLAPSE_DELTA
        ):
            events.append(
                Event(
                    "reciprocity_collapse",
                    1,
                    current_step,
                    {
                        "old_val": prev["reciprocity"],
                        "new_val": reciprocity,
                        "z_score": round(rec_z, 2) if rec_z is not None else None,
                    },
                )
            )

        # Agent isolated / emerged hub
        all_degrees = [d for d in agent_degrees.values() if d > 0]
        hub_percentile = float(self.HUB_PERCENTILE)
        hub_threshold = (
            float(np.percentile(all_degrees, hub_percentile)) if all_degrees else 0.0
        )
        for tag in nodes:
            new_deg = agent_degrees.get(tag, 0.0)
            old_deg = prev_degrees.get(tag, 0.0)
            if old_deg > 0 and new_deg < 1e-9:
                events.append(
                    Event(
                        "agent_isolated",
                        1,
                        current_step,
                        {"agent_tag": tag, "prev_degree": old_deg},
                    )
                )
            if hub_threshold > 0 and old_deg <= hub_threshold < new_deg:
                events.append(
                    Event(
                        "agent_emerged_hub",
                        0,
                        current_step,
                        {"agent_tag": tag, "degree": new_deg},
                    )
                )

        # New institutional artifacts (category 3 or 4)
        prev_classified = prev["classified_artifact_ids"]
        new_artifact_ids = set(artifact_categories.keys()) - prev_classified
        new_institutional = [
            aid
            for aid in new_artifact_ids
            if str(artifact_categories[aid]) in ("3", "4")
        ]
        for aid in new_institutional:
            events.append(
                Event(
                    "new_institutional_artifact",
                    2,
                    current_step,
                    {
                        "artifact_id": aid,
                        "name": str(aid),
                        "category": artifact_categories[aid],
                    },
                )
            )

        # Artifact category changed (already-classified artifact reclassified after modification)
        prev_categories = prev["artifact_categories"]
        for aid in artifact_categories.keys() & prev_classified:
            old_cat = str(prev_categories.get(aid, ""))
            new_cat = str(artifact_categories[aid])
            if old_cat and new_cat != old_cat:
                if new_cat in ("3", "4"):
                    sev = 2
                elif old_cat in ("3", "4"):
                    sev = 1
                else:
                    sev = 0
                events.append(
                    Event(
                        "artifact_category_changed",
                        sev,
                        current_step,
                        {
                            "artifact_id": aid,
                            "name": str(aid),
                            "old_category": old_cat,
                            "new_category": new_cat,
                        },
                    )
                )

        # Artifact burst
        n_new = len(new_artifact_ids)
        new_artifact_z = self._zscore("new_artifact_count", float(n_new))
        self._update_baseline("new_artifact_count", float(n_new))
        if (new_artifact_z is not None and new_artifact_z > self.ZSCORE_THRESHOLD) or (
            new_artifact_z is None and n_new >= self.ARTIFACT_BURST_COUNT
        ):
            events.append(
                Event(
                    "artifact_burst",
                    1,
                    current_step,
                    {
                        "count": n_new,
                        "window_steps": current_step - prev["last_step"],
                        "z_score": round(new_artifact_z, 2)
                        if new_artifact_z is not None
                        else None,
                    },
                )
            )

        # Artifact abandoned (idle too long — no access for ARTIFACT_DEATH_IDLE_STEPS)
        for aid, last_step in prev["artifact_last_access"].items():
            if aid not in artifact_last_access:  # not touched in this window
                if current_step - last_step > self.ARTIFACT_DEATH_IDLE_STEPS:
                    events.append(
                        Event(
                            "artifact_abandoned",
                            0,
                            current_step,
                            {
                                "artifact_id": aid,
                                "name": str(aid),
                                "last_access_step": last_step,
                            },
                        )
                    )

        # Artifact death (lifespan expired — removed by the environment)
        for name in expired_artifact_ids or []:
            cat = str(artifact_categories.get(name, "unknown"))
            sev = 1 if cat in ("3", "4") else 0
            events.append(
                Event(
                    "artifact_death",
                    sev,
                    current_step,
                    {"artifact_id": name, "name": name, "category": cat},
                )
            )

        # Artifact destroyed (explicit agent action, or category degraded from 3/4 to 1/2)
        for entry in destroyed_artifacts or []:
            name = entry["artifact_id"]
            cat = str(artifact_categories.get(name, "unknown"))
            sev = 2 if cat in ("3", "4") else 1
            events.append(
                Event(
                    "artifact_destroyed",
                    sev,
                    current_step,
                    {
                        "artifact_id": name,
                        "name": name,
                        "category": cat,
                        "agent_tag": entry["agent_tag"],
                        "reason": "destroyed",
                    },
                )
            )
        for aid in artifact_categories.keys() & prev["classified_artifact_ids"]:
            old_cat = str(prev_categories.get(aid, ""))
            new_cat = str(artifact_categories[aid])
            if old_cat in ("3", "4") and new_cat in ("1", "2"):
                events.append(
                    Event(
                        "artifact_destroyed",
                        2,
                        current_step,
                        {
                            "artifact_id": aid,
                            "name": str(aid),
                            "old_category": old_cat,
                            "new_category": new_cat,
                            "reason": "category_degraded",
                        },
                    )
                )

        # Contested artifact
        if graph is not None:
            for aid, accessing in artifact_access_agents.items():
                if len(accessing) < 2:
                    continue
                agent_list = list(accessing)
                sub = graph.subgraph(agent_list)
                neg = sum(
                    1 for _, _, d in sub.edges(data=True) if d.get("weight", 0) < 0
                )
                total = sub.number_of_edges()
                if total > 0 and neg / total > 0.3:
                    events.append(
                        Event(
                            "contested_artifact",
                            1,
                            current_step,
                            {
                                "artifact_id": aid,
                                "name": str(aid),
                                "agent_tags": agent_list,
                                "neg_edge_count": neg,
                            },
                        )
                    )

        # Artifact diffusion (BFS depth ≥ 2 through graph)
        if graph is not None:
            ug = graph.to_undirected()
            for aid, accessing in artifact_access_agents.items():
                if len(accessing) < 2:
                    continue
                depth = _bfs_depth(ug, set(accessing))
                if depth >= 2:
                    events.append(
                        Event(
                            "artifact_diffusion",
                            0,
                            current_step,
                            {
                                "artifact_id": aid,
                                "name": str(aid),
                                "spread_depth": depth,
                                "unique_agents": len(accessing),
                            },
                        )
                    )

        # ── Gradual event detection ────────────────────────────────────
        snapshot_history: deque = prev["snapshot_history"]
        last_trend_emitted: dict = prev["last_trend_emitted"]

        if len(snapshot_history) >= 2:
            events.extend(
                self._detect_gradual(
                    snapshot_history,
                    last_trend_emitted,
                    current_step,
                    interaction_rate,
                    polarization,
                    community_sizes,
                    isolated_agents,
                    artifact_access_agents,
                    agent_tags,
                )
            )

        # ── Push current snapshot into history before updating state ───
        snapshot_history.append(
            {
                "step": current_step,
                "interaction_rate": interaction_rate,
                "polarization": polarization,
                "community_sizes": dict(community_sizes),
                "isolated_agents": set(isolated_agents),
                "artifact_accessors": {
                    aid: len(ags) for aid, ags in artifact_access_agents.items()
                },
                "agent_tag_sets": {t: frozenset(tl) for t, tl in agent_tags.items()},
            }
        )

        # ── Update persistent state ────────────────────────────────────
        self._state = self._make_state(
            current_step,
            communities,
            community_membership,
            community_sizes,
            interaction_rate,
            polarization,
            reciprocity,
            bridge_agents,
            agent_degrees,
            isolated_agents,
            artifact_categories,
            artifact_access_agents,
            merged_last_access,
            agent_tags,
        )
        self._state["snapshot_history"] = snapshot_history
        self._state["last_trend_emitted"] = last_trend_emitted

        return events

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _zscore(self, key: str, value: float) -> float | None:
        """
        Return the z-score of `value` vs the rolling baseline for `key`.
        Returns None if the baseline has fewer than BASELINE_MIN_SAMPLES entries
        (caller should fall back to fixed threshold in that case).
        Does NOT update the baseline — call _update_baseline separately so the
        same delta can be checked in both directions without double-counting.
        """
        baseline = self._baselines.get(key)
        if baseline is None or len(baseline) < self.BASELINE_MIN_SAMPLES:
            return None
        arr = np.array(baseline, dtype=float)
        std = float(arr.std())
        if std < 1e-9:
            return None
        return float((value - float(arr.mean())) / std)

    def _update_baseline(self, key: str, value: float) -> None:
        """Append value to the rolling baseline for key."""
        if key not in self._baselines:
            self._baselines[key] = deque(maxlen=self.BASELINE_WINDOW)
        self._baselines[key].append(value)

    def _make_state(
        self,
        current_step,
        communities,
        community_membership,
        community_sizes,
        interaction_rate,
        polarization,
        reciprocity,
        bridge_agents,
        agent_degrees,
        isolated_agents,
        artifact_categories,
        artifact_access_agents,
        artifact_last_access,
        agent_tags,
    ) -> dict:
        return {
            "last_step": current_step,
            "community_count": len(communities),
            "community_membership": dict(community_membership),
            "community_members": _community_member_sets(
                communities, community_membership
            ),
            "community_sizes": dict(community_sizes),
            "interaction_rate": interaction_rate,
            "polarization": polarization,
            "reciprocity": reciprocity,
            "bridge_agents": set(bridge_agents),
            "agent_degrees": dict(agent_degrees),
            "isolated_agents": set(isolated_agents),
            "artifact_count": len(artifact_categories),
            "classified_artifact_ids": set(artifact_categories.keys()),
            "artifact_categories": dict(artifact_categories),
            "artifact_counts_by_cat": _count_by_cat(artifact_categories),
            "artifact_last_access": dict(artifact_last_access),
            "artifact_access_agents": {
                k: set(v) for k, v in artifact_access_agents.items()
            },
            "agent_tags": dict(agent_tags),
            # snapshot_history and last_trend_emitted are carried over externally
            "snapshot_history": deque(maxlen=self.SNAPSHOT_HISTORY_N),
            "last_trend_emitted": {},
        }

    def _detect_gradual(
        self,
        history: deque,
        last_trend_emitted: dict,
        current_step: int,
        interaction_rate: float,
        polarization: float,
        community_sizes: dict,
        isolated_agents: set,
        artifact_access_agents: dict,
        agent_tags: dict,
    ) -> list[Event]:
        events: list[Event] = []
        N = len(history)
        oldest = history[0]
        debounce = self._detect_interval
        k = self.MONOTONICITY_K

        def can_emit(kind: str) -> bool:
            return (
                current_step - last_trend_emitted.get(kind, -(debounce + 1))
            ) >= debounce

        def mark(kind: str):
            last_trend_emitted[kind] = current_step

        # Polarization creep (window delta)
        pol_start = oldest.get("polarization", 0.0)
        pol_delta = polarization - pol_start
        pol_wz = self._zscore("pol_window_delta", pol_delta)
        self._update_baseline("pol_window_delta", pol_delta)
        if (
            (pol_wz is not None and pol_wz > self.ZSCORE_THRESHOLD)
            or (pol_wz is None and pol_delta > self.POLARIZATION_CREEP_DELTA)
        ) and can_emit("polarization_creep"):
            events.append(
                Event(
                    "polarization_creep",
                    1,
                    current_step,
                    {
                        "start_val": pol_start,
                        "end_val": polarization,
                        "delta": pol_delta,
                        "n_snapshots": N,
                        "z_score": round(pol_wz, 2) if pol_wz is not None else None,
                    },
                )
            )
            mark("polarization_creep")

        # Interaction rate erosion / growth (monotonicity + z-score on window delta)
        ir_series = [s.get("interaction_rate", 0.0) for s in history] + [
            interaction_rate
        ]
        ir_start = oldest.get("interaction_rate", interaction_rate)
        ir_window_delta = interaction_rate - ir_start
        ir_wz = self._zscore("ir_window_delta", ir_window_delta)
        self._update_baseline("ir_window_delta", ir_window_delta)
        run_down = _monotonic_run(ir_series, -1)
        run_up = _monotonic_run(ir_series, 1)
        if (
            run_down >= k or (ir_wz is not None and ir_wz < -self.ZSCORE_THRESHOLD)
        ) and can_emit("interaction_rate_erosion"):
            events.append(
                Event(
                    "interaction_rate_erosion",
                    1,
                    current_step,
                    {
                        "start_val": ir_series[-run_down - 1]
                        if run_down >= k
                        else ir_start,
                        "end_val": interaction_rate,
                        "run_length": run_down,
                        "z_score": round(ir_wz, 2) if ir_wz is not None else None,
                    },
                )
            )
            mark("interaction_rate_erosion")
        if (
            run_up >= k or (ir_wz is not None and ir_wz > self.ZSCORE_THRESHOLD)
        ) and can_emit("interaction_rate_growth"):
            events.append(
                Event(
                    "interaction_rate_growth",
                    0,
                    current_step,
                    {
                        "start_val": ir_series[-run_up - 1]
                        if run_up >= k
                        else ir_start,
                        "end_val": interaction_rate,
                        "run_length": run_up,
                        "z_score": round(ir_wz, 2) if ir_wz is not None else None,
                    },
                )
            )
            mark("interaction_rate_growth")

        # Community contraction (monotonicity, per-community debounced by id)
        for com_id, cur_size in community_sizes.items():
            kind = f"community_contraction_{com_id}"
            sizes = [s.get("community_sizes", {}).get(com_id, 0) for s in history] + [
                cur_size
            ]
            run = _monotonic_run(sizes, -1)
            if run >= k and can_emit(kind):
                events.append(
                    Event(
                        "community_contraction",
                        0,
                        current_step,
                        {
                            "community_id": com_id,
                            "start_size": sizes[-run - 1],
                            "end_size": cur_size,
                        },
                    )
                )
                mark(kind)

        # Isolation wave (window delta: distinct agents isolated across history)
        all_isolated: set[str] = set(isolated_agents)
        for snap in history:
            all_isolated |= snap.get("isolated_agents", set())
        n_isolated = len(all_isolated)
        iso_z = self._zscore("n_isolated_window", float(n_isolated))
        self._update_baseline("n_isolated_window", float(n_isolated))
        if (
            (iso_z is not None and iso_z > self.ZSCORE_THRESHOLD)
            or (iso_z is None and n_isolated >= self.ISOLATION_WAVE_MIN)
        ) and can_emit("isolation_wave"):
            events.append(
                Event(
                    "isolation_wave",
                    2,
                    current_step,
                    {
                        "agent_tags": sorted(all_isolated),
                        "n_snapshots": N,
                        "z_score": round(iso_z, 2) if iso_z is not None else None,
                    },
                )
            )
            mark("isolation_wave")

        # Artifact rise (monotonicity, per-artifact)
        cur_accessors = {aid: len(ags) for aid, ags in artifact_access_agents.items()}
        for aid, cur_count in cur_accessors.items():
            kind = f"artifact_rise_{aid}"
            counts = [s.get("artifact_accessors", {}).get(aid, 0) for s in history] + [
                cur_count
            ]
            run = _monotonic_run(counts, 1)
            if run >= k and can_emit(kind):
                events.append(
                    Event(
                        "artifact_rise",
                        0,
                        current_step,
                        {
                            "artifact_id": aid,
                            "name": str(aid),
                            "start_count": counts[-run - 1],
                            "end_count": cur_count,
                        },
                    )
                )
                mark(kind)

        # Behavioral drift (window delta: tag changes in ≥ 2 history snapshots)
        for tag, new_tag_list in agent_tags.items():
            cur_set = frozenset(new_tag_list)
            changes = sum(
                1
                for s in history
                if (prev_set := s.get("agent_tag_sets", {}).get(tag)) is not None
                and prev_set != cur_set
            )
            if changes >= 2:
                kind = f"behavioral_drift_{tag}"
                set_key = f"behavioral_drift_set_{tag}"
                # Skip if cur_set is unchanged since the last emission — old
                # snapshots in history keep changes>=2 true for multiple cycles
                # after a single transition, causing duplicate notes.
                if cur_set == last_trend_emitted.get(set_key):
                    continue
                if can_emit(kind):
                    # Only emit transition points within the agent's lifetime
                    tag_changes = []
                    _prev_set = None
                    for s in history:
                        s_tag_sets = s.get("agent_tag_sets", {})
                        if tag not in s_tag_sets:
                            continue  # agent didn't exist yet in this snapshot
                        s_set = s_tag_sets[tag]
                        if s_set != _prev_set:
                            tag_changes.append(
                                {
                                    "step": s.get("step"),
                                    "tags": list(s_set),
                                }
                            )
                            _prev_set = s_set
                    if _prev_set != cur_set:
                        tag_changes.append(
                            {"step": current_step, "tags": list(cur_set)}
                        )
                    events.append(
                        Event(
                            "behavioral_drift",
                            1,
                            current_step,
                            {"agent_tag": tag, "tag_changes": tag_changes},
                        )
                    )
                    mark(kind)
                    last_trend_emitted[set_key] = cur_set

        return events


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _primary_community(nodes: list, node2com: dict) -> dict[str, int]:
    """agent_tag → first (primary) community_id, or -1 if unassigned."""
    result = {}
    for tag in nodes:
        coms = node2com.get(tag, [-1])
        result[tag] = coms[0] if coms else -1
    return result


def _compute_degrees(graph: nx.DiGraph | None, nodes: list) -> dict[str, float]:
    """Sum of absolute edge weights for each node (in + out)."""
    degrees: dict[str, float] = {n: 0.0 for n in nodes}
    if graph is None:
        return degrees
    for n in graph.nodes():
        d = sum(
            abs(data.get("weight", 0)) for _, _, data in graph.out_edges(n, data=True)
        )
        d += sum(
            abs(data.get("weight", 0)) for _, _, data in graph.in_edges(n, data=True)
        )
        degrees[n] = d
    return degrees


def _compute_bridges(graph: nx.DiGraph | None, threshold: float) -> set[str]:
    """Agents whose betweenness centrality ≥ threshold."""
    if graph is None or graph.number_of_nodes() < 2:
        return set()
    try:
        bc = nx.betweenness_centrality(graph, normalized=True, weight="weight")
        return {n for n, v in bc.items() if v >= threshold}
    except Exception:
        return set()


def _bfs_depth(undirected: nx.Graph, agents: set[str]) -> int:
    """
    BFS from an arbitrary seed restricted to `agents` as nodes.
    Returns maximum depth reached (0 = seed only, 1 = one hop, etc.).
    """
    agents = agents & set(undirected.nodes)
    if len(agents) < 2:
        return 0
    sub = undirected.subgraph(agents)
    seed = next(iter(agents))
    visited = {seed}
    queue = deque([(seed, 0)])
    max_depth = 0
    while queue:
        node, depth = queue.popleft()
        max_depth = max(max_depth, depth)
        for nb in sub.neighbors(node):
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, depth + 1))
    return max_depth


def _count_by_cat(categories: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for cat in categories.values():
        k = str(cat)
        counts[k] = counts.get(k, 0) + 1
    return counts


def _monotonic_run(values: list[float], direction: int) -> int:
    """
    Length of the longest strictly monotonic run at the END of `values`.
    direction=1 → increasing, direction=-1 → decreasing.
    """
    if len(values) < 2:
        return 0
    run = 0
    for i in range(len(values) - 1, 0, -1):
        if direction == 1 and values[i] > values[i - 1]:
            run += 1
        elif direction == -1 and values[i] < values[i - 1]:
            run += 1
        else:
            break
    return run
