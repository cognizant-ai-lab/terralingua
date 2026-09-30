"""
Anthropologist: orchestrates all specialist analysts.

Usage:
    from terralingua.anthropologist import Anthropologist
    from pathlib import Path

    a = Anthropologist(model="claude-haiku-4-5", audit=False)
    report = a.analyze(Path("logs/my_exp"), steps=[1, 2, 3])
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import json
import logging
import os
import pickle as pkl
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

from terralingua.anthropologist.analysis_utils import get_last_ts, load_agent_log
from terralingua.anthropologist.analysts.agent_analyst import (
    AgentAnalyst,
    AgentAnnotation,
)
from terralingua.anthropologist.analysts.artifact_analyst import (
    ArtifactAnalyst,
    _process_artifact,
)
from terralingua.anthropologist.analysts.graph_analyst import GraphAnalyst, GraphData
from terralingua.anthropologist.analysts.group_analyst import (
    GroupAnalyst,
    GroupAnnotation,
)
from terralingua.anthropologist.artifact_complexity import ExperimentArtifacts
from terralingua.anthropologist.event_detector import Event, EventDetector
from terralingua.anthropologist.log_scanner import LogScanner, ScanResult
from terralingua.utils.llm_client import LLMClient

# Step map:
#   1 → AgentAnalyst.annotate_agents
#   2 → GraphAnalyst.build_graph
#   3 → GroupAnalyst.annotate_groups  (requires step 2 for community detection)
#   4 → ArtifactAnalyst.analyze_novelty + classify_artifacts
#   5 → ArtifactAnalyst.trace_phylogeny

# Minimum distinct detection cycles before a single-agent emergence event fires.
# Persisting for 2× this threshold upgrades severity from 0 → 1.
_SINGLE_EMERGENCE_MIN_CYCLES = 3
# Emergence signals older than this (in sim steps) are pruned per agent.
_EMERGENCE_SIGNAL_RETENTION_STEPS = 150
# Cap on signals sent to the multi-agent synthesis LLM prompt.
_MULTI_EMERGENCE_PROMPT_MAX_SIGNALS = 20

# Event-kind categories — used to select context in get_llm_narrative
_AGENT_EVENT_KINDS = frozenset(
    {
        "agent_isolated",
        "agent_emerged_hub",
        "behavioral_drift",
    }
)
_ARTIFACT_EVENT_KINDS = frozenset(
    {
        "new_institutional_artifact",
        "artifact_category_changed",
        "artifact_burst",
        "artifact_rise",
        "artifact_abandoned",
        "artifact_death",
        "artifact_destroyed",
        "contested_artifact",
        "artifact_diffusion",
    }
)
# Artifact category legend shared by prompts
_CAT_LEGEND = (
    "1=basic/informational, 2=procedural/coordination, "
    "3=institutional structure, 4=norms/rules/governance"
)

_DEFAULT_STEPS = [1, 2, 3, 4, 5]


@dataclass
class AnthropologistReport:
    experiment: str
    timestamp: str
    step: int  # last completed simulation step
    agent_annotations: list[AgentAnnotation] = field(default_factory=list)
    graph_data: GraphData | None = None
    group_annotations: list[GroupAnnotation] = field(default_factory=list)
    artifact_categories: dict = field(default_factory=dict)
    artifact_novelties: dict = field(default_factory=dict)
    artifact_phylogeny: dict = field(default_factory=dict)


class Anthropologist:
    """
    Chief anthropologist — orchestrates all specialist analysts.

    Steps:
        1  — Agent behavior annotation   (AgentAnalyst)
        2  — Interaction graph + communities  (GraphAnalyst)
        3  — Group/community annotation   (GroupAnalyst, needs step 2)
        4  — Artifact novelty + classification  (ArtifactAnalyst)
        5  — Artifact phylogeny   (ArtifactAnalyst, needs step 4)
    """

    # Max wall-clock seconds for one phylogeny artifact's LLM call.
    # On timeout, that artifact is marked failed and the worker continues
    # with the rest of the batch — no stuck calls blocking siblings.
    PHYLOGENY_BATCH_TIMEOUT_S = 300
    PHYLOGENY_MAX_FAILURES = 3
    # Cap on concurrent in-flight phylogeny LLM calls. Prevents a batch of N
    # same-timestep artifacts from firing N parallel calls and hitting
    # provider rate limits. Env-overridable.
    PHYLOGENY_MAX_CONCURRENT = int(os.environ.get("ANTHRO_MAX_PHYLOGENY", "16"))
    # Same idea for artifact classification — caps in-flight LLM calls
    # during the step 3 catch-up classification.
    CLASSIFY_MAX_CONCURRENT = int(os.environ.get("ANTHRO_MAX_CLASSIFY", "16"))
    # Maximum number of detection cycles a deferred phylogeny request waits
    # for the artifact index to catch up before it's logged as a permanent
    # miss and dropped.
    PHYLOGENY_MAX_DEFER_CYCLES = 3

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        provider: str = "anthropic",
        audit: bool = True,
        parallel: bool = True,
        force_long_context: bool = False,
        agent_annotation_ttl: int = 20,
        postmortem_model: str = "claude-sonnet-4-5-20250929",
        edge_weight_decay: float = 0.98,
        fallback_model: str | None = None,
    ):
        self.model = model
        self.provider = provider
        self.audit = audit
        self.parallel = parallel
        self.force_long_context = force_long_context
        self.postmortem_model = postmortem_model
        # Fallback model used when an LLM call fails (e.g. content-policy
        # refusal from the primary). Live-cycle analysts use this as fallback;
        # the postmortem analyst (typically Sonnet) uses self.model as fallback
        # so a sonnet refusal automatically retries with haiku.
        self.fallback_model = fallback_model

        self.agent_analyst = AgentAnalyst(
            model,
            provider,
            audit=audit,
            parallel=parallel,
            force_long_context=force_long_context,
            fallback_model=fallback_model,
        )
        self._postmortem_agent_analyst = AgentAnalyst(
            postmortem_model,
            provider,
            audit=True,
            parallel=parallel,
            force_long_context=force_long_context,
            # If postmortem (typically heavier) refuses, retry with the
            # everyday model. Explicit override via fallback_model still wins.
            fallback_model=fallback_model if fallback_model is not None else model,
        )
        self.group_analyst = GroupAnalyst(
            model,
            provider,
            audit=audit,
            parallel=parallel,
            force_long_context=force_long_context,
            fallback_model=fallback_model,
        )
        self.graph_analyst = GraphAnalyst(model, provider)
        self.artifact_analyst = ArtifactAnalyst(
            model,
            provider,
            parallel=parallel,
            classify_model="claude-haiku-4-5-20251001",
            fallback_model=fallback_model,
        )

        # Live detection state — lazily initialised on first run_live_detection() call.
        self._log_scanner: LogScanner | None = None
        self._event_detector: EventDetector | None = None
        self._experiment_artifacts: ExperimentArtifacts | None = None
        self._artifact_categories_cache: dict[str, str] = {}
        self._agent_behaviors_cache: dict[str, list[str]] = {}
        self._agent_descriptions_cache: dict[str, str] = {}  # tag → free-text comment
        self._agent_annotations_cache: dict[
            str, AgentAnnotation
        ] = {}  # tag → full annotation
        self._agent_last_annotated_step: dict[str, int] = {}
        self._agent_annotation_ttl: int = agent_annotation_ttl
        self._edge_weight_decay: float = edge_weight_decay
        # Edge weights for live agents only. Decayed by edge_weight_decay^steps_elapsed
        # each detection cycle and pruned when they drop below 0.01, so the dict
        # stays bounded even for long-running experiments.
        self._live_edge_weights: dict[tuple[str, str], float] = {}
        self._last_graph_step: int = -1

        # Phylogeny processing state. The actual worker loop lives in the
        # server (next to the postmortem and detector loops); the orchestrator
        # exposes process_phylogeny_batch(names, exp_path). All fields below
        # except _phylogeny_done / _phylogeny_fail_counts / _phylogeny_deferred
        # are lazy-initialised on the first batch call.
        self._phylogeny_done: set[int] = set()
        self._phylogeny_fail_counts: dict[int, int] = {}
        # name → number of detection cycles a request has waited for the
        # artifact index to catch up. Bumped each cycle by retry_deferred_phylogeny.
        self._phylogeny_deferred: dict[str, int] = {}
        self._phylogeny_results: dict = {}
        self._phylogeny_save_path: Path | None = None
        self._phylogeny_max_history: int = 10
        self._phylogeny_agent_logs: dict = {}
        self._phylogeny_sem: asyncio.Semaphore = asyncio.Semaphore(
            self.PHYLOGENY_MAX_CONCURRENT
        )

        self._prev_agent_fingerprints: dict[str, dict[str, float]] = {}
        self._dead_agents: set[str] = set()  # accumulated from AGENT_DIED events
        self._postmortem_done: set[str] = set()
        self._recent_events: list[Event] = []  # rolling buffer for dispatch context
        self._agent_emergence_signals: dict[str, list[dict]] = {}
        self._llm_event_debounce: dict[str, int] = {}
        self._pending_log_scanner_state: dict | None = None

    def flag_done_from_disk(self, exp_path: Path | str) -> None:
        """Add to dedup sets anything present on disk, idempotently.

        Cheap and additive — only inserts entries, never overwrites. Used both
        as the no-state-file fallback (combined with full hydration below) and
        as a top-up after a successful state-file load (so a stale state file
        doesn't cause already-done work to be redone).
        """
        exp_path = Path(exp_path)
        pm_dir = exp_path / "annotations"
        if pm_dir.is_dir():
            seeded = 0
            for f in pm_dir.glob("*.json"):
                if f.name == "anthropologist_notes.json":
                    continue
                if f.stem not in self._postmortem_done:
                    self._postmortem_done.add(f.stem)
                    seeded += 1
            if seeded:
                log.info(
                    "[Anthropologist] Flagged %d postmortem(s) as done from disk",
                    seeded,
                )
        ph_file = exp_path / "artifact_analysis" / "artifact_phylogeny.json"
        if ph_file.exists():
            try:
                raw = json.loads(ph_file.read_text())
                seeded = 0
                for k in raw.keys():
                    try:
                        art_id = int(k)
                    except (ValueError, TypeError):
                        continue
                    if art_id not in self._phylogeny_done:
                        self._phylogeny_done.add(art_id)
                        seeded += 1
                if seeded:
                    log.info(
                        "[Anthropologist] Flagged %d phylogeny entr(ies) as done from disk",
                        seeded,
                    )
            except Exception as e:
                log.warning(
                    "[Anthropologist] Could not flag phylogeny from disk: %s", e
                )

    def hydrate_dedup_from_disk(self, exp_path: Path | str) -> None:
        """Full hydration when no state file is available.

        Calls flag_done_from_disk for the dedup sets, then additionally
        rehydrates live-annotation caches and artifact classifications so live
        analysis doesn't re-annotate every alive agent on first cycle. Live
        annotation hydration tags `last_annotated_step` with the `-1` sentinel
        which is converted to the current step on first detection cycle.
        """
        exp_path = Path(exp_path)
        self.flag_done_from_disk(exp_path)

        live_dir = exp_path / "annotations" / "live"
        if live_dir.is_dir():
            names_path = exp_path / "agent_names.json"
            names: dict = {}
            if names_path.exists():
                try:
                    names = json.loads(names_path.read_text())
                except Exception:
                    pass
            seeded = 0
            for f in live_dir.glob("*.json"):
                tag = f.stem
                try:
                    ann = json.loads(f.read_text())
                except Exception:
                    continue
                if not isinstance(ann, dict):
                    continue
                behaviors = [
                    b
                    for b in ann.get("behaviors", [])
                    if isinstance(b, dict) and b.get("behavior")
                ]
                events = [
                    e
                    for e in ann.get("events", [])
                    if isinstance(e, dict) and e.get("event")
                ]
                self._agent_behaviors_cache[tag] = [b["behavior"] for b in behaviors]
                self._agent_descriptions_cache[tag] = ann.get("comment", "")
                self._agent_annotations_cache[tag] = AgentAnnotation(
                    tag=tag,
                    name=names.get(tag, tag),
                    events=events,
                    behaviors=behaviors,
                    comment=ann.get("comment", ""),
                    emergence=ann.get("emergence", {}),
                    anthropologist=ann.get("anthropologist", ""),
                )
                self._agent_last_annotated_step[tag] = -1
                seeded += 1
            if seeded:
                log.info(
                    "[Anthropologist] Hydrated %d live annotation(s) from disk", seeded
                )

        cat_file = exp_path / "artifact_analysis" / "artifact_categories.json"
        if cat_file.exists():
            try:
                self._artifact_categories_cache = json.loads(cat_file.read_text())
                log.info(
                    "[Anthropologist] Hydrated %d artifact classification(s) from disk",
                    len(self._artifact_categories_cache),
                )
            except Exception as e:
                log.warning("[Anthropologist] Could not hydrate classifications: %s", e)

    # State is persisted to <exp>/anthropologist_state.json on graceful
    # shutdown and every CHECKPOINT_EVERY_N_CYCLES detection cycles. The
    # previous slot is a rotating backup so a crash mid-write leaves at
    # least one valid snapshot.
    STATE_FILE_NAME = "anthropologist_state.json"
    STATE_FILE_PREVIOUS_NAME = "anthropologist_state.previous.json"

    def _build_state_dict(self) -> dict:
        return {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "model": self.model,
            "postmortem_model": self.postmortem_model,
            "postmortem_done": sorted(self._postmortem_done),
            "phylogeny_done": sorted(self._phylogeny_done),
            "phylogeny_fail_counts": {
                str(k): v for k, v in self._phylogeny_fail_counts.items()
            },
            "phylogeny_deferred": dict(self._phylogeny_deferred),
            "dead_agents": sorted(self._dead_agents),
            "agent_last_annotated_step": dict(self._agent_last_annotated_step),
            "agent_behaviors_cache": dict(self._agent_behaviors_cache),
            "agent_descriptions_cache": dict(self._agent_descriptions_cache),
            "agent_annotations_cache": {
                tag: dataclasses.asdict(ann)
                for tag, ann in self._agent_annotations_cache.items()
            },
            "prev_agent_fingerprints": dict(self._prev_agent_fingerprints),
            "artifact_categories_cache": dict(self._artifact_categories_cache),
            "recent_events": [dataclasses.asdict(e) for e in self._recent_events],
            "agent_emergence_signals": dict(self._agent_emergence_signals),
            "llm_event_debounce": dict(self._llm_event_debounce),
            "live_edge_weights": [
                {"source": s, "target": t, "weight": w}
                for (s, t), w in self._live_edge_weights.items()
            ],
            "last_graph_step": self._last_graph_step,
            "log_scanner_state": (
                self._log_scanner.export_state()
                if self._log_scanner is not None
                else None
            ),
        }

    def save_state(self, exp_path: Path | str) -> None:
        """Atomically dump orchestrator state to ``<exp>/anthropologist_state.json``.

        Rotation: existing current → previous (if present), new state written
        to ``.tmp`` then renamed to current. Crash-safe — at least one valid
        snapshot is always on disk.
        """
        exp_path = Path(exp_path)
        state_path = exp_path / self.STATE_FILE_NAME
        previous_path = exp_path / self.STATE_FILE_PREVIOUS_NAME
        tmp_path = exp_path / (self.STATE_FILE_NAME + ".tmp")
        state = self._build_state_dict()
        try:
            with open(tmp_path, "w") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            if state_path.exists():
                state_path.replace(previous_path)
            tmp_path.replace(state_path)
            log.info(
                "[Anthropologist] State saved to %s (%d postmortems, %d phylogeny, "
                "%d agents tracked)",
                state_path,
                len(self._postmortem_done),
                len(self._phylogeny_done),
                len(self._agent_last_annotated_step),
            )
        except Exception as e:
            log.exception("[Anthropologist] Failed to save state: %s", e)
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass

    def load_state(self, exp_path: Path | str) -> None:
        """Restore state from disk, trying the current slot then the previous.

        After a successful state-file load, also tops up dedup sets from
        on-disk annotation/phylogeny files — handles the case where the state
        file is stale (e.g., saved before postmortems were ever run). When no
        state file is loadable, falls back to full hydration.
        """
        exp_path = Path(exp_path)
        for path in [
            exp_path / self.STATE_FILE_NAME,
            exp_path / self.STATE_FILE_PREVIOUS_NAME,
        ]:
            if not path.exists():
                continue
            try:
                state = json.loads(path.read_text())
                self._apply_state(state)
            except Exception as e:
                log.warning(
                    "[Anthropologist] State file %s unusable (%s); trying previous slot",
                    path,
                    e,
                )
                continue
            log.info(
                "[Anthropologist] Restored state from %s (saved %s; %d postmortems, "
                "%d phylogeny, %d agents)",
                path,
                state.get("saved_at"),
                len(self._postmortem_done),
                len(self._phylogeny_done),
                len(self._agent_last_annotated_step),
            )
            # Top up dedup from disk — covers stale state files.
            self.flag_done_from_disk(exp_path)
            return
        log.info("[Anthropologist] No state file found; hydrating dedup from disk.")
        self.hydrate_dedup_from_disk(exp_path)

    def _apply_state(self, state: dict) -> None:
        """Restore each field from a state dict produced by ``_build_state_dict``."""
        self._postmortem_done = set(state.get("postmortem_done", []))
        # phylogeny_done keys may have been written as strings via JSON
        self._phylogeny_done = {int(x) for x in state.get("phylogeny_done", [])}
        self._phylogeny_fail_counts = {
            int(k): v for k, v in (state.get("phylogeny_fail_counts") or {}).items()
        }
        self._phylogeny_deferred = dict(state.get("phylogeny_deferred") or {})
        self._dead_agents = set(state.get("dead_agents", []))
        self._agent_last_annotated_step = dict(
            state.get("agent_last_annotated_step") or {}
        )
        self._agent_behaviors_cache = dict(state.get("agent_behaviors_cache") or {})
        self._agent_descriptions_cache = dict(
            state.get("agent_descriptions_cache") or {}
        )
        self._agent_annotations_cache = {
            tag: AgentAnnotation(**ann)
            for tag, ann in (state.get("agent_annotations_cache") or {}).items()
        }
        self._prev_agent_fingerprints = dict(state.get("prev_agent_fingerprints") or {})
        self._artifact_categories_cache = dict(
            state.get("artifact_categories_cache") or {}
        )
        self._recent_events = [Event(**e) for e in (state.get("recent_events") or [])]
        self._agent_emergence_signals = dict(state.get("agent_emergence_signals") or {})
        self._llm_event_debounce = dict(state.get("llm_event_debounce") or {})
        self._live_edge_weights = {
            (e["source"], e["target"]): e["weight"]
            for e in (state.get("live_edge_weights") or [])
        }
        self._last_graph_step = state.get("last_graph_step", -1)
        # LogScanner is lazily constructed in run_live_detection — stash its
        # state and apply it as soon as the scanner is built.
        self._pending_log_scanner_state = state.get("log_scanner_state") or None

    def analyze(
        self,
        exp_path: Path | str,
        steps: list[int] | None = None,
        time_range: tuple[int, int] | None = None,
        save: bool = True,
        error_tracker=None,
    ) -> AnthropologistReport:
        """
        Run selected analysis steps in dependency order.

        Args:
            exp_path: Experiment directory.
            steps: Which steps to run (default: all). Step 3 requires step 2; step 5 requires step 4.
            time_range: Optional (start_ts, end_ts) to restrict agent/group annotation.
            save: Whether specialists should save outputs to disk.
            error_tracker: Optional ErrorTracker for recording failures.

        Returns:
            AnthropologistReport with all results.
        """
        exp_path = Path(exp_path)
        steps = sorted(set(steps or _DEFAULT_STEPS))

        # Enforce dependencies
        if 3 in steps and 2 not in steps:
            log.info("[Anthropologist] Step 3 requires step 2. Adding step 2.")
            steps = sorted(set(steps) | {2})
        if 5 in steps and 4 not in steps:
            log.info(
                "[Anthropologist] Step 5 requires step 4 for artifact loading. Adding step 4."
            )
            steps = sorted(set(steps) | {4})

        try:
            sim_step = get_last_ts(exp_path / "open_gridworld.log")
        except Exception:
            sim_step = -1

        report = AnthropologistReport(
            experiment=exp_path.name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            step=sim_step,
        )

        graph_data: GraphData | None = None
        communities: dict[int, set] | None = None
        save_path = exp_path / "annotations" if save else None

        for step in steps:
            log.info("[Anthropologist] Running step %d...", step)

            if step == 1:
                report.agent_annotations = self.agent_analyst.annotate_experiment(
                    exp_path=exp_path,
                    save_path=save_path,
                    time_range=time_range,
                    error_tracker=error_tracker,
                )

            elif step == 2:
                graph_data = self.graph_analyst.build_graph(
                    exp_path=exp_path, save=save
                )
                report.graph_data = graph_data
                communities = {
                    i: set(c)
                    for i, c in enumerate(graph_data.graph_data.get("communities", []))
                }

            elif step == 3:
                if communities is None and graph_data is None:
                    # Load from disk if step 2 was not run in this call
                    graph_data = self._load_graph(exp_path)
                    if graph_data is not None:
                        communities = {
                            i: set(c)
                            for i, c in enumerate(
                                graph_data.graph_data.get("communities", [])
                            )
                        }
                report.group_annotations = self.group_analyst.annotate_groups(
                    exp_path=exp_path,
                    communities=communities,
                    save_path=save_path,
                    time_range=time_range,
                    error_tracker=error_tracker,
                )

            elif step == 4:
                report.artifact_novelties = self.artifact_analyst.analyze_novelty(
                    exp_path=exp_path,
                    error_tracker=error_tracker,
                )
                report.artifact_categories = self.artifact_analyst.classify_artifacts(
                    exp_path=exp_path,
                    error_tracker=error_tracker,
                )

            elif step == 5:
                report.artifact_phylogeny = self.artifact_analyst.trace_phylogeny(
                    exp_path=exp_path,
                    error_tracker=error_tracker,
                )

        return report

    async def run_live_detection(
        self,
        exp_path: Path | str,
        window: int = 50,
        steps_elapsed: int = 1,
        detect_interval: int = 10,
        resolve_key_fn=None,
        cross_agent_key_fn=None,
    ) -> tuple[ScanResult, list[Event], GraphData, list[str]]:
        """
        One detection cycle for the live anthropologist.

        Reads new log data, builds a graph snapshot, classifies new/modified
        artifacts, re-annotates behaviorally-shifted agents, then runs the
        event detector. All caches are updated in-place so consecutive calls
        accumulate state correctly.

        Returns:
            (scan_result, events, graph_data, phylogeny_retry_names)
            - scan_result.died_agents     → trigger annotate_postmortem
            - events                      → call dispatch() for high-severity ones
            - graph_data                  → context for dispatch() narratives
            - phylogeny_retry_names       → previously-deferred user requests that
                                            became resolvable this cycle (the
                                            caller should re-enqueue them on
                                            Redis). New artifacts found this cycle
                                            are NOT auto-enqueued — phylogeny is
                                            on-demand only.
        """
        exp_path = Path(exp_path)

        # ── Lazy init ─────────────────────────────────────────────────
        if self._log_scanner is None:
            self._log_scanner = LogScanner(exp_path)
            if self._pending_log_scanner_state is not None:
                self._log_scanner.restore_state(self._pending_log_scanner_state)
                self._pending_log_scanner_state = None
        if self._event_detector is None:
            self._event_detector = EventDetector(detect_interval=detect_interval)
        if self._experiment_artifacts is None:
            self._experiment_artifacts = ExperimentArtifacts(exp_path=exp_path)
        if not self._artifact_categories_cache:
            _cat_file = exp_path / "artifact_analysis" / "artifact_categories.json"
            if _cat_file.exists():
                with open(_cat_file) as f:
                    self._artifact_categories_cache = json.load(f)
                log.info(
                    "[Anthropologist] Restored %d artifact classification(s) from disk",
                    len(self._artifact_categories_cache),
                )

        # ── 1. Scan ───────────────────────────────────────────────────
        log.info("[Anthropologist] step 1: scanning log...")
        scan_result = self._log_scanner.scan(window=window)
        current_step = scan_result.steps_covered[1] if scan_result.steps_covered else 0
        # Accumulate dead agents so they are never queued for re-annotation.
        self._dead_agents.update(scan_result.died_agents)
        # Remove newly-dead agents from the live graph mirror.
        for tag in scan_result.died_agents:
            dead_keys = [k for k in self._live_edge_weights if tag in k]
            for k in dead_keys:
                del self._live_edge_weights[k]
        _names = self._log_scanner.get_agent_name_registry()
        died_labels = [
            f"{_names[t]}({t})" if t in _names else t for t in scan_result.died_agents
        ]
        log.info(
            "[Anthropologist] step 1 done: step=%d, interactions=%d, died=%s, total_dead=%d",
            current_step,
            len(scan_result.interactions),
            died_labels,
            len(self._dead_agents),
        )

        # ── 1b. Update artifact index ─────────────────────────────────
        # Always fresh (never buffered), so no dedup needed here.
        log.info("[Anthropologist] step 1b: updating artifact index...")
        new_artifact_tags = self._experiment_artifacts.update_from_events(
            raw_events=scan_result.raw_artifact_world_events,
            up_to_ts=current_step,
        )
        # Phylogeny is on-demand only: new artifacts are NOT auto-enqueued.
        # Only previously-deferred user requests get retried here.
        phylogeny_retry_names = self.retry_deferred_phylogeny()
        active_count = len(
            self._experiment_artifacts.artifacts_by_ts.get(
                self._experiment_artifacts.last_ts, []
            )
        )
        log.info(
            "[Anthropologist] step 1b done: %d new version(s) this cycle, active=%d, total_versions=%d",
            len(new_artifact_tags),
            active_count,
            len(self._experiment_artifacts.all_artifacts),
        )

        # ── 2. Graph snapshot ─────────────────────────────────────────
        # Decay existing edge weights by decay^steps_elapsed, then accumulate
        # new interactions (step > _last_graph_step only, to avoid double-counting
        # from overlapping scan windows). Entries below 0.01 are pruned.
        log.info("[Anthropologist] step 2: building graph snapshot...")
        decay = self._edge_weight_decay**steps_elapsed
        stale = [k for k, w in self._live_edge_weights.items() if w * decay < 0.01]
        for k in stale:
            del self._live_edge_weights[k]
        for k in self._live_edge_weights:
            self._live_edge_weights[k] *= decay
        for interaction in scan_result.interactions:
            if interaction["step"] > self._last_graph_step:
                key = (interaction["source"], interaction["target"])
                w = interaction["weight"]
                if key[0] not in self._dead_agents and key[1] not in self._dead_agents:
                    self._live_edge_weights[key] = (
                        self._live_edge_weights.get(key, 0.0) + w
                    )
        self._last_graph_step = current_step
        graph_data = self.graph_analyst.build_graph_from_edge_weights(
            self._live_edge_weights
        )
        log.info(
            "[Anthropologist] step 2 done: %d live edge(s), %d community(s)",
            len(self._live_edge_weights),
            len(graph_data.graph_data.get("communities", [])),
        )

        # ── 3. Artifact classification ────────────────────────────────
        log.info("[Anthropologist] step 3: classifying artifacts...")
        latest_artifact_registry = self._log_scanner.get_latest_artifact_registry()
        seen_artifacts = {e["artifact_id"] for e in scan_result.artifact_events}
        modified_artifacts = {
            e["artifact_id"]
            for e in scan_result.artifact_events
            if e["action"] == "modify"
            and e["artifact_id"] in self._artifact_categories_cache
        }
        new_artifacts = seen_artifacts - set(self._artifact_categories_cache)
        # Also catch any registry artifact that was never classified (e.g. artifacts
        # that fell outside every scan window and were never seen in artifact_events).
        uncategorized_existing = set(latest_artifact_registry.keys()) - set(
            self._artifact_categories_cache
        )
        to_classify = new_artifacts | modified_artifacts | uncategorized_existing
        if to_classify:
            _cat_file = exp_path / "artifact_analysis" / "artifact_categories.json"
            _cat_file.parent.mkdir(parents=True, exist_ok=True)

            def _save_chunk(chunk_cats: dict[str, str]) -> None:
                self._artifact_categories_cache.update(chunk_cats)
                try:
                    with open(_cat_file, "w") as f:
                        json.dump(self._artifact_categories_cache, f, indent=2)
                except Exception as e:
                    log.warning("[Anthropologist] Failed to save classifications: %s", e)

            await self.artifact_analyst.classify_artifacts_incremental(
                to_classify, latest_artifact_registry,
                max_concurrent=self.CLASSIFY_MAX_CONCURRENT,
                on_batch=_save_chunk,
                resolve_key_fn=resolve_key_fn,
            )
        log.info(
            "[Anthropologist] step 3 done: %d classified, total=%d",
            len(to_classify),
            len(self._artifact_categories_cache),
        )

        # ── 4. Agent annotation (fingerprint shift + TTL) ─────────────
        # Must run BEFORE event detection: the event detector consumes
        # _agent_behaviors_cache to build behavioral_drift snapshots.
        # Agents named in fired events are re-annotated AFTER (step 7) to
        # ensure get_llm_narrative has fresh context for them, even if they
        # didn't cross the fingerprint/TTL threshold this cycle.
        # NOTE: agent_annotation_ttl must be > detect_interval, otherwise
        # the TTL condition is always true and fingerprint checks are wasted.
        fingerprints = self._log_scanner.get_agent_fingerprints()
        pre_event_agents = self._get_agents_to_annotate(fingerprints, current_step)
        live_save_path = exp_path / "annotations" / "live"
        pre_event_labels = [
            f"{_names[t]}({t})" if t in _names else t for t in pre_event_agents
        ]
        log.info(
            "[Anthropologist] step 4: annotating %s...", pre_event_labels or "(none)"
        )
        await self._annotate_and_record(
            pre_event_agents, exp_path, live_save_path, current_step,
            resolve_key_fn=resolve_key_fn,
        )
        log.info("[Anthropologist] step 4 done")

        # ── 5. Artifact access dicts ──────────────────────────────────
        log.info("[Anthropologist] step 5: building artifact access maps...")
        artifact_access_agents, artifact_last_access = self._build_artifact_maps(
            scan_result.artifact_events
        )
        log.info("[Anthropologist] step 5 done")

        # ── 6. Event detection ────────────────────────────────────────
        log.info("[Anthropologist] step 6: detecting events...")
        destroyed_artifacts = [
            {"artifact_id": e["artifact_id"], "agent_tag": e["agent_tag"]}
            for e in scan_result.artifact_events
            if e["action"] == "destroy"
        ]
        events = self._event_detector.detect(
            graph_data=graph_data.graph_data,
            artifact_categories=self._artifact_categories_cache,
            artifact_access_agents=artifact_access_agents,
            artifact_last_access=artifact_last_access,
            agent_tags=self._agent_behaviors_cache,
            current_step=current_step,
            graph=graph_data.graph,
            destroyed_artifacts=destroyed_artifacts,
            expired_artifact_ids=scan_result.expired_artifact_ids,
        )
        log.info("[Anthropologist] step 6 done: %d event(s) detected", len(events))

        # ── 7. On-demand annotation for agents named in fired events ──
        event_agents: set[str] = set()
        for e in events:
            if "agent_tag" in e.data:
                event_agents.add(e.data["agent_tag"])
            if "agent_tags" in e.data:
                event_agents.update(e.data["agent_tags"])
        post_event_agents = [
            tag
            for tag in event_agents
            if tag not in self._dead_agents
            and (
                tag not in self._agent_behaviors_cache
                or current_step - self._agent_last_annotated_step.get(tag, 0)
                >= self._agent_annotation_ttl
            )
        ]
        post_event_labels = [
            f"{_names[t]}({t})" if t in _names else t for t in post_event_agents
        ]
        log.info(
            "[Anthropologist] step 7: on-demand annotation for %s...",
            post_event_labels or "(none)",
        )
        await self._annotate_and_record(
            post_event_agents, exp_path, live_save_path, current_step,
            resolve_key_fn=resolve_key_fn,
        )
        log.info("[Anthropologist] step 7 done — detection cycle complete")

        # ── 8. LLM emergent event synthesis ──────────────────────────────
        log.info("[Anthropologist] step 8: synthesizing emergent events...")
        single_events = self._detect_single_emergence(current_step)
        # Cross-agent synthesis: pay with the first available key among the
        # agents whose signals are in the buffer.
        synth_key: str | None = None
        if cross_agent_key_fn is not None:
            buffered_tags = [
                s["agent_tag"]
                for sigs in self._agent_emergence_signals.values()
                for s in sigs
            ]
            synth_key = cross_agent_key_fn(buffered_tags)
        multi_events = await self._synthesize_emergent_events(
            current_step, graph_data, api_key=synth_key,
        )
        llm_events = single_events + multi_events
        events.extend(llm_events)
        log.info(
            "[Anthropologist] step 8 done: %d single-agent, %d multi-agent",
            len(single_events),
            len(multi_events),
        )

        self._recent_events = (self._recent_events + events)[-20:]

        return scan_result, events, graph_data, phylogeny_retry_names

    def _build_narrative_prompt(
        self, event: Event, graph_snapshot: GraphData
    ) -> str | None:
        """Build the LLM prompt for a narrative call, or None if already narrated."""
        if event.narrative:
            return None

        gd = graph_snapshot.graph_data
        name_reg = (
            self._log_scanner.get_agent_name_registry() if self._log_scanner else {}
        )

        involved_tags: set[str] = set()
        if "agent_tag" in event.data:
            involved_tags.add(event.data["agent_tag"])
        if "agent_tags" in event.data:
            involved_tags.update(event.data["agent_tags"])

        recent_context = (
            "\n".join(
                f"  step {e.step} [{e.kind}] {e.data}"
                for e in self._recent_events[-6:]
                if e is not event
            )
            or "  (none)"
        )

        if event.kind in _AGENT_EVENT_KINDS:
            scene_hdr = (
                "An event just occurred that concerns a specific agent's trajectory."
            )
            scene_body = f"Agent profile(s):\n{self._agent_profile_block(involved_tags, gd, name_reg)}"
            tail = "State what changed and what it signals. Reference actual behaviors. Do not speculate."
        elif event.kind in _ARTIFACT_EVENT_KINDS:
            scene_hdr = (
                "Agents create, modify, exchange, and destroy artifacts — "
                "artifacts are the material record of their culture."
            )
            scene_body = (
                f"Artifact:\n{self._artifact_block(event, name_reg)}\n\n"
                f"Agents involved:\n{self._agent_profile_block(involved_tags, gd, name_reg)}"
            )
            tail = "State what the artifact event means and what it reveals. Reference actual content. Do not speculate."
        else:
            scene_hdr = "An event just occurred reflecting a shift in the population's social structure."
            scene_body = self._network_block(gd, name_reg, involved_tags)
            tail = "State what shifted and which communities or agents are affected. Do not speculate."

        return (
            f"You are an anthropologist observing a multi-agent simulation.\n{scene_hdr}\n\n"
            f"Event: {event.kind} (severity {event.severity}/2)\n"
            f"Step: {event.step} | Data: {event.data}\n\n"
            f"{scene_body}\n\n"
            f"Recent simulation events:\n{recent_context}\n\n"
            f"Write 1-2 sentences max (~30 words). No heading, no bullet points, no markdown.\n{tail}"
        )

    def _agent_profile_block(self, tags: set[str], gd: dict, name_reg: dict) -> str:
        """Per-agent profile (behaviors, comment, anthropologist note, community)."""
        comms = gd.get("communities", [])
        node2com = gd.get("node2com", {})
        sections = []
        for tag in tags:
            name = name_reg.get(tag, tag)
            ann = self._agent_annotations_cache.get(tag)
            if not ann:
                sections.append(
                    f"  {name} ({tag}): "
                    f"{self._agent_descriptions_cache.get(tag, 'not yet annotated')}"
                )
                continue
            behaviors = (
                ", ".join(b["behavior"] for b in ann.behaviors) or "none recorded"
            )
            valid_ids = [
                c for c in (node2com.get(tag) or []) if c != -1 and c < len(comms)
            ]
            if valid_ids:
                comm_str = "; ".join(
                    f"community {cid} ({len(comms[cid])} members: "
                    f"{', '.join(str(name_reg.get(m, m)) for m in comms[cid])})"
                    for cid in valid_ids
                )
            else:
                comm_str = "unknown community"
            sections.append(
                f"  {name} ({tag}):\n"
                f"    Behaviors: {behaviors}\n"
                f"    Life summary (factual recap of what this agent did): {ann.comment}\n"
                f"    Anthropological reading (novel or culturally significant patterns): "
                f"{ann.anthropologist or 'none'}\n"
                f"    Community: {comm_str}"
            )
        return "\n".join(sections) or "  (no profiles available)"

    def _artifact_block(self, event: Event, name_reg: dict) -> str:
        artifact_id = event.data.get("artifact_id") or event.data.get("name")
        if artifact_id is None or self._experiment_artifacts is None:
            return "  (no artifact details available)"
        art_obj = next(
            (
                a
                for a in self._experiment_artifacts.all_artifacts.values()
                if a.get("name") == artifact_id
            ),
            None,
        )
        if not art_obj:
            return "  (no artifact details available)"
        cat = self._artifact_categories_cache.get(
            art_obj.get("name", str(artifact_id)), "?"
        )
        creator_tag = art_obj.get("original_creator_tag", "unknown")
        return (
            f"  Name: {art_obj.get('name', artifact_id)}\n"
            f"  Category: {cat} ({_CAT_LEGEND})\n"
            f"  Payload: {str(art_obj.get('payload', ''))[:250]}\n"
            f"  Created: step {art_obj.get('original_version_creation', '?')}, "
            f"version {art_obj.get('version', 0)}\n"
            f"  Original creator: {name_reg.get(creator_tag, creator_tag)} ({creator_tag})"
        )

    def _network_block(self, gd: dict, name_reg: dict, involved_tags: set[str]) -> str:
        comms = gd.get("communities", [])
        comm_block = (
            "\n".join(
                f"  Community {i} ({len(m)} members): "
                f"{', '.join(str(name_reg.get(x, x)) for x in m)}"
                for i, m in enumerate(comms)
            )
            or "  (no community data)"
        )
        metrics = (
            f"Metrics: interaction_rate={gd.get('interaction_rate', 0.0):.3f}, "
            f"polarization={gd.get('neg_polarization', 0.0):.3f}, "
            f"reciprocity={gd.get('reciprocity', 0.0):.3f}"
        )
        tail = (
            f"\nKey agents named in event:\n{self._agent_profile_block(involved_tags, gd, name_reg)}"
            if involved_tags
            else ""
        )
        return f"Community structure:\n{comm_block}\n{metrics}{tail}"

    async def get_llm_narrative(
        self,
        event: Event,
        graph_snapshot: GraphData,
        api_key: str | None = None,
    ) -> str:
        """Generate a 2-3 sentence narrative for a high-severity event.

        Context is tailored to the event category:
          - agent events  → full agent profile (behaviors, comment, anthropologist note, community)
          - artifact events → artifact content/history + involved agent profiles
          - network events  → community compositions with named members + key metrics

        Uses Haiku for speed. Stores the result in event.narrative and returns it.
        Async so callers can fan out via ``asyncio.gather`` across multiple events."""
        if event.narrative:
            return event.narrative
        prompt = self._build_narrative_prompt(event, graph_snapshot)
        if prompt is None:
            return event.narrative or ""
        llm = LLMClient(client=self.provider, long_context=False)
        response = await llm.get_response_async(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": prompt}],
            chat_parameters={},
            output_json=False,
            api_key=api_key,
        )
        narrative = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        event.narrative = narrative
        return narrative

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_agents_to_annotate(
        self,
        fingerprints: dict[str, dict[str, float]],
        current_step: int,
        threshold: float = 0.3,
    ) -> list[str]:
        """
        Return agents that need (re-)annotation this cycle.

        An agent is included if any of the following hold:
        - never annotated before
        - TTL since last annotation has elapsed
        - L1 fingerprint distance from previous cycle exceeds threshold

        Updates _prev_agent_fingerprints in-place.
        """
        to_annotate: list[str] = []
        for tag, buckets in fingerprints.items():
            if tag in self._dead_agents:
                continue
            last = self._agent_last_annotated_step.get(tag)
            if last == -1:
                # Hydrated from disk on startup — treat as just-annotated at
                # current_step so TTL/fingerprint logic kicks in normally.
                self._agent_last_annotated_step[tag] = current_step
                continue
            if last is None or current_step - last >= self._agent_annotation_ttl:
                to_annotate.append(tag)
            else:
                prev = self._prev_agent_fingerprints.get(tag)
                if prev is not None:
                    all_keys = set(buckets) | set(prev)
                    l1 = sum(
                        abs(buckets.get(k, 0.0) - prev.get(k, 0.0)) for k in all_keys
                    )
                    if l1 > threshold:
                        to_annotate.append(tag)
        self._prev_agent_fingerprints = fingerprints
        return to_annotate

    async def _annotate_and_record(
        self,
        agent_tags: list[str],
        exp_path: Path,
        live_save_path: Path,
        current_step: int,
        resolve_key_fn=None,
    ) -> None:
        """Run live agent annotation and update caches + emergence signals."""
        if not agent_tags:
            return
        annotations = await self.agent_analyst.annotate_agents(
            agent_tags=agent_tags,
            exp_path=exp_path,
            save_path=live_save_path,
            audit=False,
            verbose=False,
            resolve_key_fn=resolve_key_fn,
        )
        for a in annotations:
            self._agent_behaviors_cache[a.tag] = [b["behavior"] for b in a.behaviors]
            self._agent_descriptions_cache[a.tag] = a.comment
            self._agent_annotations_cache[a.tag] = a
        for tag in agent_tags:
            self._agent_last_annotated_step[tag] = current_step
        self._record_emergence_signals(annotations, current_step)

    def _record_emergence_signals(
        self, annotations: list[AgentAnnotation], current_step: int
    ) -> None:
        """Append non-trivial emergence signals to per-agent storage, prune old ones."""
        for ann in annotations:
            em = ann.emergence or {}
            kws = em.get("keywords", ["none"])
            cmt = em.get("comment", "none")
            if kws == ["none"] and cmt in ("none", ""):
                continue
            self._agent_emergence_signals.setdefault(ann.tag, []).append(
                {
                    "agent_tag": ann.tag,
                    "agent_name": ann.name,
                    "step": current_step,
                    "keywords": kws,
                    "comment": cmt,
                    "behaviors": [b["behavior"] for b in ann.behaviors]
                    if ann.behaviors
                    else [],
                    "anthropologist": ann.anthropologist,
                }
            )
        cutoff = current_step - _EMERGENCE_SIGNAL_RETENTION_STEPS
        for tag in list(self._agent_emergence_signals.keys()):
            kept = [
                s for s in self._agent_emergence_signals[tag] if s["step"] >= cutoff
            ]
            if kept:
                self._agent_emergence_signals[tag] = kept
            else:
                del self._agent_emergence_signals[tag]

    def _detect_single_emergence(self, current_step: int) -> list[Event]:
        """
        Fire a low-severity event when a single agent has shown a persistent
        novel emergence signal across at least _SINGLE_EMERGENCE_MIN_CYCLES
        distinct detection cycles.

        Severity escalates with persistence: ≥ MIN_CYCLES → 0, ≥ 2×MIN_CYCLES → 1.
        Debounced per agent (20 steps) via _llm_event_debounce["single_{tag}"].
        """
        results: list[Event] = []
        for tag, signals in self._agent_emergence_signals.items():
            distinct_steps = len({s["step"] for s in signals})
            if distinct_steps < _SINGLE_EMERGENCE_MIN_CYCLES:
                continue

            debounce_key = f"single_{tag}"
            if current_step - self._llm_event_debounce.get(debounce_key, -999) < 20:
                continue

            severity = 1 if distinct_steps >= _SINGLE_EMERGENCE_MIN_CYCLES * 2 else 0

            # Use the most recent signal for narrative material
            latest = max(signals, key=lambda s: s["step"])
            keywords = latest["keywords"]
            anthropologist = latest.get("anthropologist", "")
            comment = latest.get("comment", "")

            # Build narrative from annotation — no extra LLM call needed.
            # Prefer the free-form anthropological reading over the structured summary.
            if anthropologist:
                narrative = anthropologist
            elif comment not in ("none", ""):
                narrative = comment
            else:
                kw_str = ", ".join(k for k in keywords if k != "none")
                narrative = (
                    f"Persistent novel behavior observed: {kw_str}."
                    if kw_str
                    else "Persistent novel behavior observed."
                )

            self._llm_event_debounce[debounce_key] = current_step
            results.append(
                Event(
                    kind="llm_single_emergence",
                    severity=severity,
                    step=current_step,
                    data={
                        "agent_tag": tag,
                        "keywords": keywords,
                        "cycles_observed": distinct_steps,
                        "description": narrative,
                    },
                    narrative=narrative,
                )
            )

        return results

    async def _synthesize_emergent_events(
        self,
        current_step: int,
        graph_snapshot: GraphData,
        api_key: str | None = None,
    ) -> list[Event]:
        """
        Synthesize population-level emergent events from buffered per-agent signals.

        Only calls Haiku when at least 2 distinct agents have non-trivial
        emergence signals in the buffer.  Each signal carries the full agent
        profile (behaviors, comment, anthropologist note, community) so the
        LLM has concrete evidence to reason from.  Results are debounced per
        kind (50 steps) to prevent floods.  Returned events have their
        narrative pre-filled so get_llm_narrative() skips them.
        """
        all_signals = sorted(
            (s for sigs in self._agent_emergence_signals.values() for s in sigs),
            key=lambda s: s["step"],
        )
        recent_signals = all_signals[-_MULTI_EMERGENCE_PROMPT_MAX_SIGNALS:]
        distinct_agents = {s["agent_tag"] for s in recent_signals}
        if len(distinct_agents) < 2:
            return []

        gd = graph_snapshot.graph_data
        name_reg = (
            self._log_scanner.get_agent_name_registry() if self._log_scanner else {}
        )
        comms = gd.get("communities", [])
        node2com = gd.get("node2com", {})

        # Community overview for structural context
        comm_lines = []
        for i, members in enumerate(comms):
            names = [str(name_reg.get(m, m)) for m in members]
            preview = ", ".join(names)
            comm_lines.append(f"  Community {i} ({len(members)}): {preview}")
        comm_overview = "\n".join(comm_lines) or "  (no community data)"

        # Rich per-agent signal blocks
        signal_blocks = []
        for s in recent_signals:
            tag = s["agent_tag"]
            comm_ids = node2com.get(tag) or []
            valid_ids = [c for c in comm_ids if c != -1 and c < len(comms)]
            if valid_ids:
                comm_str = "; ".join(
                    f"community {c} ({len(comms[c])} members)" for c in valid_ids
                )
            else:
                comm_str = "unknown community"
            behaviors_str = ", ".join(s.get("behaviors", [])) or "none recorded"
            block = (
                f"  Agent: {s['agent_name']} ({tag}) — {comm_str}\n"
                f"  Step observed: {s['step']}\n"
                f"  Current behaviors: {behaviors_str}\n"
                f"  Life summary (factual recap of what this agent did): {s['comment']}\n"
                f"  Emergence keywords: {s['keywords']}"
            )
            if s.get("anthropologist"):
                block += f"\n  Anthropological reading (novel or culturally significant patterns): {s['anthropologist']}"
            signal_blocks.append(block)
        signals_text = "\n\n".join(signal_blocks)

        system_prompt = (
            "You are a population-level emergence detector for a multi-agent simulation.\n\n"
            "You receive per-agent behavioral signals from an ongoing anthropological analysis. "
            "Your role is to identify cross-agent, population-level cultural phenomena that "
            "no single-agent annotation can capture — things like the emergence of a shared "
            "practice, norm, ideology, social role, or economic relationship.\n\n"
            "Rules:\n"
            "1. Only fire if MULTIPLE distinct agents show convergent signals.\n"
            "2. The phenomenon must be genuinely population-level — a pattern in how the "
            "GROUP behaves, not a coincidence of individual quirks.\n"
            "3. Name it precisely in 2-4 snake_case words "
            "(e.g. 'religion_emerged', 'gift_economy', 'territorial_hierarchy', ...).\n"
            "4. Severity scale:\n"
            "   - 0: speculative (weak or ambiguous signals)\n"
            "   - 1: plausible (consistent signals, but could use more evidence)\n"
            "   - 2: strongly evidenced (clear, repeated, multi-agent convergence)\n"
            "   If genuinely uncertain, return [].\n"
            "5. 'description' must be SPECIFIC: refer to agents by their NAME "
            "(not their tag), describe their behaviors, "
            "and cite the concrete evidence — no generic statements.\n\n"
            "Output VALID JSON ONLY — a list of objects with keys:\n"
            "  kind        (str, snake_case prefixed 'llm_')\n"
            "  severity    (int 0-2)\n"
            "  description (str, 1-2 sentences, evidence-specific, use agent names not tags)\n"
            "  agent_tags  (list[str], the raw agent tags for structured lookup)\n"
            "Return at most 1 event — the single strongest, most evidenced phenomenon. "
            "If multiple qualify, pick the one with the broadest cross-agent support. "
            "Return [] if nothing qualifies."
        )

        user_prompt = (
            f"Simulation step: {current_step}\n\n"
            f"Population community structure:\n{comm_overview}\n\n"
            f"Per-agent emergence signals "
            f"({len(recent_signals)} observations, "
            f"{len(distinct_agents)} distinct agents):\n\n"
            f"{signals_text}\n\n"
            "Identify any genuine population-level emergent cultural phenomenon, "
            "or return []."
        )

        llm = LLMClient(client=self.provider, long_context=False)
        try:
            response = await llm.get_response_async(
                model="claude-haiku-4-5-20251001",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                chat_parameters={},
                output_json=True,
                api_key=api_key,
            )
            raw = response.content
        except Exception as e:
            log.warning("[Anthropologist] step 8 LLM error: %s", e)
            return []

        if not isinstance(raw, list):
            return []

        results: list[Event] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind", ""))
            if not kind.startswith("llm_"):
                kind = f"llm_{kind}" if kind else "llm_unknown"
            severity = int(item.get("severity", 0))
            description = str(item.get("description", ""))
            agent_tags = list(item.get("agent_tags", []))

            last_fired = self._llm_event_debounce.get(kind, -999)
            if current_step - last_fired < 50:
                continue

            self._llm_event_debounce[kind] = current_step
            results.append(
                Event(
                    kind=kind,
                    severity=severity,
                    step=current_step,
                    data={"description": description, "agent_tags": agent_tags},
                    narrative=description,
                )
            )

        return results

    def get_live_annotations(self) -> dict[str, dict]:
        """Return the current live agent descriptions (tag → {text, step})."""
        return {
            tag: {"text": text, "step": self._agent_last_annotated_step.get(tag, -1)}
            for tag, text in self._agent_descriptions_cache.items()
        }

    def get_artifact_categories(self) -> dict[str, str]:
        """Return the current artifact classification cache (name → category string)."""
        return dict(self._artifact_categories_cache)

    def list_artifact_names(self) -> list[str]:
        """Return every artifact name seen so far. For --catchup phylogeny enqueue."""
        if self._experiment_artifacts is None:
            return []
        return [
            n
            for art in self._experiment_artifacts.all_artifacts.values()
            if (n := art.get("name"))
        ]

    def get_phylogeny_processed_names(self, exp_path: Path | str) -> set[str]:
        """
        Return the set of artifact names whose phylogeny has been computed
        (regardless of whether ancestors were found).  Reads the disk file so
        results written by the background worker are always reflected.
        """
        if self._experiment_artifacts is None:
            return set()
        save_path = (
            Path(exp_path)
            / "artifact_analysis"
            / "artifact_phylogeny.json"
        )
        try:
            raw: dict = json.load(open(save_path))
        except Exception:
            return set()
        arts = self._experiment_artifacts.all_artifacts
        names: set[str] = set()
        for art_id_str in raw:
            try:
                art_id = int(art_id_str)
            except (ValueError, TypeError):
                continue
            art_name = arts.get(art_id, {}).get("name", "")
            if art_name:
                names.add(art_name)
        return names

    def get_phylogeny_by_name(
        self, exp_path: Path | str
    ) -> dict[str, dict[str, float]]:
        """
        Read the saved phylogeny file from disk and resolve integer artifact IDs
        to artifact names using _experiment_artifacts.

        Returns {artifact_name: {ancestor_name: score}}.
        Returns an empty dict if the file doesn't exist yet or if the artifact
        index hasn't been loaded (first-cycle startup).
        """
        if self._experiment_artifacts is None:
            return {}
        save_path = (
            Path(exp_path)
            / "artifact_analysis"
            / "artifact_phylogeny.json"
        )
        try:
            raw: dict = json.load(open(save_path))
        except Exception:
            return {}
        arts = self._experiment_artifacts.all_artifacts
        id_to_name: dict[int, str] = {k: v.get("name", str(k)) for k, v in arts.items()}
        result: dict[str, dict[str, float]] = {}
        for art_id_str, ancestors in raw.items():
            try:
                art_id = int(art_id_str)
            except (ValueError, TypeError):
                continue
            art_name = id_to_name.get(art_id, str(art_id))
            if not isinstance(ancestors, dict):
                continue
            resolved: dict[str, float] = {}
            for anc_id_str, score in ancestors.items():
                try:
                    anc_id = int(anc_id_str)
                except (ValueError, TypeError):
                    continue
                resolved[id_to_name.get(anc_id, str(anc_id))] = score
            if resolved:
                result[art_name] = resolved
        return result

    async def annotate_postmortem(
        self,
        agent_tag: str,
        exp_path: Path | str,
        death_step: int,
        api_key: str | None = None,
    ) -> AgentAnnotation | None:
        """Run a postmortem for a dead agent, or load it from disk if already done.

        Returns the annotation in both the freshly-computed and cached cases, so
        callers always get a payload to publish. Returns None only when there's
        nothing to annotate (no agent log).
        """
        exp_path = Path(exp_path)
        save_path = exp_path / "annotations"
        cached_file = save_path / f"{agent_tag}.json"
        if cached_file.exists():
            self._postmortem_done.add(agent_tag)
            try:
                data = json.loads(cached_file.read_text())
                names_path = exp_path / "agent_names.json"
                agent_name = agent_tag
                if names_path.exists():
                    agent_name = json.loads(names_path.read_text()).get(
                        agent_tag, agent_tag
                    )
                return AgentAnnotation(
                    tag=agent_tag,
                    name=agent_name,
                    events=[
                        e
                        for e in data.get("events", [])
                        if isinstance(e, dict) and e.get("event")
                    ],
                    behaviors=[
                        b
                        for b in data.get("behaviors", [])
                        if isinstance(b, dict) and b.get("behavior")
                    ],
                    comment=data.get("comment", ""),
                    emergence=data.get("emergence", {}),
                    anthropologist=data.get("anthropologist", ""),
                )
            except Exception as e:
                log.warning(f"Could not load cached postmortem for {agent_tag}: {e}")
                return None
        results = await self._postmortem_agent_analyst.annotate_agents(
            agent_tags=[agent_tag],
            exp_path=exp_path,
            save_path=save_path,
            time_range=(0, death_step),
            audit=True,
            api_key=api_key,
        )
        annotation = results[0] if results else None
        if annotation is not None:
            self._postmortem_done.add(agent_tag)
        return annotation

    def hydrate_artifact_index(self, exp_path: Path | str) -> None:
        """Rebuild ``_experiment_artifacts`` from the full world log.

        Needed on restart: ``LogScanner`` reads via byte offsets and only
        emits NEW events after a resume, so artifacts created before the
        restart never enter the index built by ``update_from_events``.
        Dashboard phylogeny requests for those artifacts otherwise fail to
        resolve and get dropped after PHYLOGENY_MAX_DEFER_CYCLES.
        """
        exp_path = Path(exp_path)
        worldlog = exp_path / "open_gridworld.log"
        if not worldlog.exists():
            log.info("[Anthropologist] No world log to hydrate artifact index from.")
            return
        if self._experiment_artifacts is None:
            self._experiment_artifacts = ExperimentArtifacts(exp_path=exp_path)
        loaded = self._experiment_artifacts.load_full_history()
        log.info(
            "[Anthropologist] Hydrated artifact index from world log: %d artifact(s)",
            loaded,
        )

    def _resolve_artifact_ids(self, names: list[str]) -> list[int]:
        """Translate artifact names to int IDs. Unresolvable names go into
        ``_phylogeny_deferred`` for retry next cycle; resolved names already
        marked done are filtered out. Returns IDs ready for processing."""
        if self._experiment_artifacts is None:
            for name in names:
                self._phylogeny_deferred.setdefault(name, 0)
            log.info(
                "[Anthropologist] Phylogeny resolve: deferred=%d (no artifact index yet)",
                len(names),
            )
            return []
        name_to_id = {
            art.get("name"): art_id
            for art_id, art in self._experiment_artifacts.all_artifacts.items()
            if art.get("name")
        }
        resolved: list[int] = []
        already_done: list[str] = []
        deferred: list[str] = []
        for name in names:
            art_id = name_to_id.get(name)
            if art_id is None:
                self._phylogeny_deferred.setdefault(name, 0)
                deferred.append(name)
                continue
            self._phylogeny_deferred.pop(name, None)
            if art_id in self._phylogeny_done:
                already_done.append(name)
            else:
                resolved.append(art_id)
        log.info(
            "[Anthropologist] Phylogeny resolve: to_process=%d, already_done=%d %s, "
            "deferred=%d %s",
            len(resolved),
            len(already_done),
            already_done or "",
            len(deferred),
            deferred or "",
        )
        return resolved

    def retry_deferred_phylogeny(self) -> list[str]:
        """Re-check the artifact index for previously-deferred names.

        Returns names that are now resolvable (the caller should re-enqueue
        them on Redis). Names still missing have their wait counter bumped;
        those past PHYLOGENY_MAX_DEFER_CYCLES are dropped with a warning."""
        if not self._phylogeny_deferred or self._experiment_artifacts is None:
            return []
        name_to_id = {
            art.get("name"): art_id
            for art_id, art in self._experiment_artifacts.all_artifacts.items()
            if art.get("name")
        }
        resolved: list[str] = []
        for name in list(self._phylogeny_deferred.keys()):
            if name_to_id.get(name) is not None:
                self._phylogeny_deferred.pop(name)
                resolved.append(name)
                continue
            new_count = self._phylogeny_deferred[name] + 1
            if new_count >= self.PHYLOGENY_MAX_DEFER_CYCLES:
                self._phylogeny_deferred.pop(name)
                log.warning(
                    "[Anthropologist] Phylogeny: '%s' still not in artifact index "
                    "after %d detection cycle(s); giving up.",
                    name,
                    self.PHYLOGENY_MAX_DEFER_CYCLES,
                )
            else:
                self._phylogeny_deferred[name] = new_count
        return resolved

    def _phylogeny_lazy_init(self, exp_path: Path) -> None:
        if self._phylogeny_save_path is not None:
            return
        save_dir = exp_path / "artifact_analysis"
        save_dir.mkdir(parents=True, exist_ok=True)
        self._phylogeny_save_path = save_dir / "artifact_phylogeny.json"
        try:
            params = json.load(open(exp_path / "params.json"))
            self._phylogeny_max_history = params.get("max_history") or params.get(
                "agent", {}
            ).get("max_history", 10)
        except Exception:
            self._phylogeny_max_history = 10
        try:
            self._phylogeny_results = (
                json.load(open(self._phylogeny_save_path))
                if self._phylogeny_save_path.exists()
                else {}
            )
        except Exception:
            self._phylogeny_results = {}
        for art_id_str in self._phylogeny_results.keys():
            try:
                self._phylogeny_done.add(int(art_id_str))
            except (ValueError, TypeError):
                continue

    async def process_phylogeny_batch(
        self,
        names: list[str],
        exp_path: Path | str,
        on_save=None,
        api_key: str | None = None,
    ) -> list[str]:
        """Run phylogeny on a batch of artifacts identified by name.

        Same-creation_time artifacts run in parallel under a concurrency
        semaphore. Results are saved incrementally to disk after each
        creation-time group. ``on_save`` is awaited (or called) after each
        save so the server can republish classifications.

        Returns names of artifacts whose transient failure didn't reach the
        max-fail cap — the caller should re-enqueue them on Redis."""
        exp_path = Path(exp_path)
        self._phylogeny_lazy_init(exp_path)
        # Lazy-init guarantees these are populated.
        assert self._phylogeny_results is not None
        assert self._phylogeny_save_path is not None
        assert self._phylogeny_sem is not None
        art_ids = self._resolve_artifact_ids(names)
        if not art_ids:
            return []
        assert self._experiment_artifacts is not None
        artifacts = self._experiment_artifacts.all_artifacts

        art_ids.sort(key=lambda t: artifacts[t].get("creation_time", 0))
        by_ts: dict[int, list[int]] = defaultdict(list)
        for t in art_ids:
            by_ts[int(artifacts[t].get("creation_time", 0))].append(t)

        # Build a creation-time-ordered (art_id → name) view of every artifact
        # ever seen. The bulk trace_phylogeny path passes "everything from prior
        # timesteps" as candidates — we match that here so on-demand requests
        # can find ancestors that haven't themselves been phylogeny-processed yet.
        all_by_creation: list[tuple[int, int, str]] = sorted(
            (
                (int(art.get("creation_time", 0)), aid, art.get("name", "") or "")
                for aid, art in artifacts.items()
            ),
            key=lambda x: (x[0], x[1]),
        )

        failed_ids: list[int] = []
        for ts in sorted(by_ts):
            batch = by_ts[ts]
            # Candidates for this batch: every artifact strictly older than the
            # batch's creation_time. Preserves the same "no peer ancestors within
            # the same ts" rule the bulk path enforces.
            previous_artifacts_for_batch: dict[int, str] = {
                aid: name for (ct, aid, name) in all_by_creation if ct < ts
            }
            tasks_with_ids: list[tuple[int, Any]] = []
            for art_id in batch:
                artifact = artifacts[art_id]
                creator = artifact.get("creator_tag", "")
                if creator not in self._phylogeny_agent_logs:
                    try:
                        self._phylogeny_agent_logs[creator] = load_agent_log(
                            filepath=exp_path / "agent_logs" / f"{creator}.jsonl",
                            reduce=False,
                        )
                    except Exception:
                        self._phylogeny_agent_logs[creator] = {}
                tasks_with_ids.append(
                    (
                        art_id,
                        _process_artifact(
                            art_id,
                            artifact,
                            self._phylogeny_agent_logs,
                            exp_path,
                            self._phylogeny_max_history,
                            self._experiment_artifacts,
                            previous_artifacts_for_batch,
                            self.model,
                            self.provider,
                            False,
                            True,
                            True,
                            {},
                            self.fallback_model,
                            api_key,
                        ),
                    )
                )

            async def _bounded(task):
                async with self._phylogeny_sem:
                    return await asyncio.wait_for(
                        task, timeout=self.PHYLOGENY_BATCH_TIMEOUT_S
                    )

            results = await asyncio.gather(
                *[_bounded(t) for _, t in tasks_with_ids],
                return_exceptions=True,
            )

            for (art_id, _), res in zip(tasks_with_ids, results):
                if isinstance(res, asyncio.TimeoutError):
                    log.warning(
                        "[Anthropologist] Phylogeny timed out for artifact %s after %ds",
                        art_id,
                        self.PHYLOGENY_BATCH_TIMEOUT_S,
                    )
                    failed_ids.append(art_id)
                elif isinstance(res, BaseException):
                    log.warning(
                        "[Anthropologist] Phylogeny failed for artifact %s: %s",
                        art_id,
                        res,
                    )
                    failed_ids.append(art_id)
                else:
                    _id, _, llm_res, _ = res
                    self._phylogeny_results[_id] = llm_res
                    self._phylogeny_done.add(_id)

            try:
                with open(self._phylogeny_save_path, "w") as f:
                    json.dump(self._phylogeny_results, f, indent=4)
            except Exception as e:
                log.warning("[Anthropologist] Failed to save phylogeny: %s", e)
                continue

            if on_save is not None:
                try:
                    if inspect.iscoroutinefunction(on_save):
                        await on_save()
                    else:
                        on_save()
                except Exception as e:
                    log.warning(
                        "[Anthropologist] Phylogeny on_save callback failed: %s", e
                    )

        retry_names: list[str] = []
        for art_id in failed_ids:
            self._phylogeny_fail_counts[art_id] = (
                self._phylogeny_fail_counts.get(art_id, 0) + 1
            )
            if self._phylogeny_fail_counts[art_id] >= self.PHYLOGENY_MAX_FAILURES:
                self._phylogeny_results[art_id] = {"_failed": True}
                self._phylogeny_done.add(art_id)
                log.warning(
                    "[Anthropologist] Phylogeny permanently skipping "
                    "artifact %s after %d failures",
                    art_id,
                    self.PHYLOGENY_MAX_FAILURES,
                )
            else:
                name = artifacts.get(art_id, {}).get("name")
                if name:
                    retry_names.append(name)
        if failed_ids:
            try:
                with open(self._phylogeny_save_path, "w") as f:
                    json.dump(self._phylogeny_results, f, indent=4)
            except Exception:
                pass

        return retry_names

    def _load_graph(self, exp_path: Path) -> GraphData | None:
        """Load graph.pkl from disk if it exists."""
        graph_pkl = exp_path / "graph.pkl"
        if not graph_pkl.exists():
            log.warning(
                "[Anthropologist] graph.pkl not found at %s. Skipping community load.",
                graph_pkl,
            )
            return None
        with open(graph_pkl, "rb") as f:
            data = pkl.load(f)
        return GraphData(
            graph=data["graph"],
            contrib_df=data["contrib_df"],
            graph_data=data["graph_data"],
        )

    def _build_artifact_maps(self, artifact_events: list[dict]) -> tuple[dict, dict]:
        """
        Build two lookup dicts from a list of artifact events.

        Returns:
            access_agents: artifact_id → set of agent tags that touched it
            last_access:   artifact_id → last simulation step it was touched
        """
        access_agents: dict = defaultdict(set)
        last_access: dict = {}
        for e in artifact_events:
            aid = e["artifact_id"]
            access_agents[aid].add(e["agent_tag"])
            last_access[aid] = max(last_access.get(aid, 0), e["step"])
        return dict(access_agents), last_access
