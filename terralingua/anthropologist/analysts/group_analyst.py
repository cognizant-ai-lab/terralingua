"""
GroupAnalyst: expert in community/group-level behavioral annotation.

Extracted from 003_llm_group_analyser.py.
"""

import copy
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Dict, List, Set, Tuple

log = logging.getLogger(__name__)

from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm

from terralingua.anthropologist.analysts.base import BaseAnalyst
from terralingua.anthropologist.error_tracker import ErrorTracker
from terralingua.anthropologist.graph_utils import build_graph, get_slpa_communities
from terralingua.utils.llm_client import LLMClient
from terralingua.utils.llm_utils import (
    MAX_CONTEXT_TOKENS,
    MAX_OUTPUT_TOKENS,
    count_tokens,
    is_context_enough,
)

load_dotenv(find_dotenv(usecwd=True), override=True)


@dataclass
class GroupAnnotation:
    community_id: int
    agents: list = field(default_factory=list)
    events: list = field(default_factory=list)
    behaviors: list = field(default_factory=list)
    comment: str = ""
    emergence: dict = field(default_factory=dict)
    anthropologist: str = ""
    interval: list = field(default_factory=list)


# Prompts (preserved verbatim from 003_llm_group_analyser.py)
# ---------------------------
_ANNOTATOR_SYSTEM_PROMPT = """You are an extremely good anthropological annotation engine.
You will receive the logs of a group of agents.
Your task is to analyze and annotate the logs.
Output VALID JSON ONLY matching the schema.
Never invent IDs or tags. Only make claims that are directly supported by provided fields.
Lower confidence or omit claims when uncertain.
Note: It is extremely important that you get this right, as this will be used for scientific analysis.
"""

_EVENT_ANNOTATOR_USER_PROMPT = """Analyze the following group behavior.
The group logs are structured as {{timestep0: [agent1_log, agent2_log, ...], timestep1: [agent0_log, agent2_log, ...], ...}}.
Each agent log contains:
- Agent name
- Agent tag
- Performed action
- Action parameters
- Message broadcast by agent
- Internal memory of the agent
- Observation containing: messages received from other agents, agent remaining time and energy, agent's inventory

Note:
- Messages are broadcast and can be perceived by any nearby agent.
- Egocentric coordinates. Each agent reports locations in its own frame where (0,0) is that agent's current cell at that timestep. Thus (0,3) in two different agent logs usually refers to different absolute cells. Do not compare positions across agents unless a shared frame is provided (e.g., an artifact/location name or an explicitly stated global coordinate). Only treat positions as comparable within the same agent's log at a given timestep.
- The content of the elements in the inventory is always visible to the agent and might affect the agent's behavior.

Your tasks:
Analyze the logs and the exchanged messages of the agents in the group and do the following:
1. Events (instantaneous)
    - Highlight important events.
    - STRICT REQUIREMENT: Tag them with one of the following event tags, given as (EVENT_TAG: description):
        {event_tags}
2. Behaviors (spanning multiple timesteps)
    - Identify main behavioral characteristics.
    - STRICT REQUIREMENT: Tag them with one of the following behavioral tags, given as (BEHAVIOR_TAG: description):
        {behavioral_tags}
3. For each annotation (event or behavior) provide:
    - "confidence": <0-10 number>` ("0 = guess, 10 = direct evidence")
    - "description": "<short natural language description>"
    - "reference": [{{"step": <timestep>, "snippet": "<exact short quote>"}}]
    - "agents": [tags of agents involved]
    - For events: "timesteps": [<t1>, ...]
    - For behaviors: "time_span": [<start_step>, <end_step>]
4. Inclusion criteria (STRICT)
   - Treat tag lists as a VOCABULARY, not a checklist. Output ONLY tags that actually occur.
   - For EVENTS:
       • "timesteps": must be a non-empty array (min 1).
       • "reference": must be a non-empty array (min 1), with exact quotes present in the logs.
       • "confidence": must be ≥ 3. If < 3, OMIT the event.
   - For BEHAVIORS:
       • "time_span": must be [start, end] with start ≤ end and both present in the logs.
       • "reference": must be a non-empty array (min 2) from ≥2 distinct timesteps.
       • "confidence": must be ≥ 3. If < 3, OMIT the behavior.
5. Forbidden output (STRICT)
   - Do NOT produce placeholders for tags with no evidence (e.g., "No evidence of X").
   - Do NOT include any event/behavior with empty "timesteps"/"time_span"/"reference", or "confidence": 0.
   - If a tag has no supporting evidence, OUTPUT NOTHING for that tag.
   - Report absences only in "emergence.comment" if relevant, never as empty annotations.
6. References (STRICT):
    - For each reference, quote exact substrings from the logs.
    - Do not paraphrase.
7. Condensation
    - If similar events repeat, merge into one entry.
8. Emergence
    - Identify any emergent properties.
    - Set `"emergence.keywords"` to a list using ONLY these tags: {emergent_tags}. (STRICT)
    - If no emergent behavior is present, set `"emergence.keywords": ["none"]`.
    - Set `"emergence.comment"` to a short, one-sentence explanation. If truly nothing to say, set it to "none".
9. Summary
   - Provide a short 2-3 sentence recap of the group life and trends.

Notes:
- Give particular attention to effects spanning multiple timesteps (e.g., agentX gives energy to agentY, and in the future agentY is friendlier with agentX, or agents setting up exchange protocols, etc.)
- Also note when agents are interacting with agents outside of the group.
- Before emitting the final JSON, self-check and DELETE any event/behavior that violates the inclusion criteria.
- Agents belong to the same group with respect to the number of interactions they had. Such interactions can be BOTH positive or negative. Being in the same group does NOT mean that agents are friendly among themselves.
- Only refer to agents by their tags

Output must be **VALID JSON ONLY**, following exactly this schema:

```json
{{
  "events": [
    {{
      "event": "<event_type>",
      "timesteps": [<t1>, <t2>, ...],
      "confidence": <confidence_value>,
      "description": "<short_description>",
      "reference": [{{"step": <timestep>, "snippet": "<exact short quote>"}}]
    }}
  ],
  "behaviors": [
    {{
      "behavior": "<behavior_type>",
      "time_span": [<start_time>, <end_time>],
      "confidence": <confidence_value>,
      "description": "<short_description>",
      "reference": [{{"step": <timestep>, "snippet": "<exact short quote>"}}]
    }}
  ],
  "comment": "<short recap>",
  "emergence": {{
    "keywords": ["<keyword1>", "<keyword2>", ...],
    "comment": "<short explanation or 'none'>"
  }}
}}
```

Constraints:
- Never invent IDs or tags. Only make claims that are directly supported by provided fields.
- Lower confidence or omit claims when uncertain.

Group data:

Tags of agents in the group:
{community_tags}

Tags to name mapping in the form of agent_tag:agent_name :
{agent_names}

Group Log
{community_data}
"""

_AUDITOR_SYSTEM_PROMPT = """You are an extremely good annotation AUDITOR.
You will receive the logs of a group of agents and a set of annotations made on those logs.
Your job is to VERIFY, not to re-annotate from scratch.
Output VALID JSON ONLY matching the schema.
Verify that each annotation is SUPPORTED by the logs.
Never invent IDs or tags.
Verify that IDs and tags are not invented but match the provided valid tags.
Note: It is extremely important that you get this right, as this will be used for scientific analysis.
"""

_AUDITOR_USER_PROMPT = """Audit the following group annotations.

You are given:
A) The group logs, structured as {{timestep0: [agent1_log, agent2_log, ...], timestep1: [agent0_log, agent2_log, ...], ...}}.
Each agent log contains:
- Agent name
- Agent tag
- Performed action
- Action parameters
- Message broadcast by agent
- Internal memory of the agent
- Observation containing: messages received from other agents, agent remaining time and energy, agent's inventory

B) An annotation with:
   - "events": [{{"event", "timesteps", "confidence", "description", "reference"}}, ...]
   - "behaviors": [{{"behavior", "time_span", "confidence", "description", "reference"}}, ...]
   - "comment": string

Note:
- Messages are broadcast and can be perceived by any nearby agent.
- Egocentric coordinates. Each agent reports locations in its own frame where (0,0) is that agent's current cell at that timestep. Thus (0,3) in two different agent logs usually refers to different absolute cells. Do not compare positions across agents unless a shared frame is provided (e.g., an artifact/location name or an explicitly stated global coordinate). Only treat positions as comparable within the same agent's log at a given timestep.
- The content of the elements in the inventory is always visible to the agent and might affect the agent's behavior.

Your task is to audit the annotations provided based on the logs.

Rules:
- Use ONLY these valid tags (STRICT):
  EVENT_TAGS
  {event_tags}

  BEHAVIOR_TAGS
  {behavior_tags}

- Events = punctual; Behaviors = span multiple timesteps.
- For each item:
  1) TAG FIT: Does the tag semantically match the evidence?
  2) TIME SPAN (if behavior): Are start/end steps consistent with logs?
  3) TIMESTEPS (if event): Are they consistent with logs?
  4) REFERENCE: Do the cited steps/messages/events actually support it?
  5) CONSISTENCY CHECKS:
     - PREDATION/KILL implies a target and causal evidence (attack → death or energy gain).
     - COALITION/COOPERATION implies multi-agent coordination.
     - MISINFORMATION requires contradiction between message content and observed reality.
     - TERRITORIALITY implies area claim/defense over time.

Output VALID JSON ONLY with this schema:

{{
  "events_audit": [
    {{
      "index": <index in input events array>,
      "verdict": "pass" | "fail" | "revise",
      "issues": ["<short issue>", ...],
      "proposed_fix": {{
        "event": "<tag or null>",
        "timesteps": [<timestep or null>, ...],
        "description": "<revised or null>",
        "reference": "<revised or null>",
        "confidence": <number or null>
      }},
      "evidence": [{{"step": <timestep>, "snippet": "<exact short quote>"}}],
      "confidence": <0-10 number>
    }}
  ],
  "behaviors_audit": [
    {{
      "index": <index in input behaviors array>,
      "verdict": "pass" | "fail" | "revise",
      "issues": ["..."],
      "proposed_fix": {{
        "behavior": "<tag or null>",
        "time_span": [<start or null>, <end or null>],
        "description": "<revised or null>",
        "reference": "<revised or null>",
        "confidence": <number or null>
      }},
      "evidence": [{{"step": <timestep>, "snippet": "<quote>"}}],
      "confidence": <0-10 number>
    }}
  ],
  "summary": "<2-3 sentences on overall annotation quality>"
}}

Notes:
- Index must match the input array index (0-based).
- If verdict == pass, do not include proposed_fix or evidence.
- If verdict == fail, do not include proposed_fix (item will be discarded).
- If verdict == revise, proposed_fix must include all keys.
- Keep evidence concise (direct quotes from logs).
- Do not output any explanations outside the JSON.
- Multiple similar events can be grouped into a single entry. Both grouped and non-grouped entries are fine.

Data provided:

Tags of agents in the group:
{community_tags}

Tags to name mapping in the form of agent_tag:agent_name :
{agent_names}

Group Log
{community_data}

Annotations:
{annotations}
"""

_ANTHROPOLOGIST_SYSTEM_PROMPT = """You are an experienced anthropologist studying the life and actions of agents living in a 2D world.
You will receive the logs of a group of agents.
Your task is to identify anything interesting or novel that might emerge from the logs the same way an anthropologist would.
Output a few sentences describing what you discovered.
Keep it short and concise.
Note: It is extremely important that you get this right, as this will be used for scientific analysis.
"""

_ANTHROPOLOGIST_USER_PROMPT = """Analyze the following group behavior.

The rules of the world in which the agents live are:
- At each timestep, they observe:
    - A list of agents, food sources, and artifacts in their field of view
    - Their energy level and time left
    - The content of their inventory
- At each timestep, they produce an action
- They lose 1 energy at each timestep. If energy goes to 0 they die. To recover energy they have to eat food
- They lose 1 unit of time at each timestep. Once time is 0 they die.
- They can broadcast a message to all the agents in their field of view
- They can create, collect, modify, destroy, or exchange artifacts
- They can give energy to or take energy from any agent in their field of view
- They have no set goal.

You want to identify any emergent behaviors in the agents.
You will receive the logs of a group of agents.

The group logs are structured as {{timestep0: [agent1_log, agent2_log, ...], timestep1: [agent0_log, agent2_log, ...], ...}}.
Each agent log contains:
- Agent name
- Agent tag
- Performed action
- Action parameters
- Message broadcast by agent
- Internal memory of the agent
- Observation containing: messages received from other agents, agent remaining time and energy, agent's inventory

Note:
- Messages are broadcast and can be perceived by any nearby agent.
- Egocentric coordinates. Each agent reports locations in its own frame where (0,0) is that agent's current cell at that timestep. Thus (0,3) in two different agent logs usually refers to different absolute cells. Do not compare positions across agents unless a shared frame is provided (e.g., an artifact/location name or an explicitly stated global coordinate). Only treat positions as comparable within the same agent's log at a given timestep.

Your task is to identify anything interesting or novel that might emerge from the logs the same way an anthropologist would.
Output a few sentences describing what you discovered.

Data provided:

Tags of agents in the group:
{community_tags}

Tags to name mapping in the form of agent_tag:agent_name :
{agent_names}

Group Log
{community_data}
"""
# ---------------------------


def _merge_logs(data: Dict[str, Dict[int, dict]]) -> Tuple[dict, Tuple[int, int]]:
    """Merge per-agent logs into timestep-major format."""
    merged_data = {}
    start_ts = min(min(d.keys()) for d in data.values())
    end_ts = max(max(d.keys()) for d in data.values())

    for ts in range(start_ts, end_ts + 1):
        merged_data[ts] = []
        for agent_data in data.values():
            if ts in agent_data:
                data_point = deepcopy(agent_data[ts])
                del data_point["timestamp"]
                if "available_actions" in data_point:
                    del data_point["available_actions"]
                del data_point["observation"]
                data_point["agent_name"] = data_point.pop("agent")
                merged_data[ts].append(data_point)

    return merged_data, (start_ts, end_ts)


def _build_annotator_messages(
    tags, community, community_data, agent_names
) -> list[dict]:
    user_prompt = _EVENT_ANNOTATOR_USER_PROMPT.format(
        event_tags=tags["group_events"],
        behavioral_tags=tags["group_behavior"],
        emergent_tags=tags["group_emergence"],
        community_tags=community,
        community_data=community_data,
        agent_names=agent_names,
    )
    return [
        {"role": "system", "content": _ANNOTATOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _split_community_data(
    community: Set[str],
    community_data: dict,
    agent_names: dict,
    tags: dict,
    model: str,
    overlap_ratio: float = 0.33,
    max_intervals: int = 32,
) -> Tuple[List[dict], list | None, bool]:
    """Split community data into overlapping chunks that fit in context."""
    full_messages = _build_annotator_messages(
        tags, community, community_data, agent_names
    )
    if is_context_enough(
        messages=full_messages,
        max_input_tokens=MAX_CONTEXT_TOKENS[model]["base"],
        model=model,
    ):
        return [community_data], None, False

    if MAX_CONTEXT_TOKENS[model].get("long") is not None:
        if is_context_enough(
            messages=full_messages,
            max_input_tokens=MAX_CONTEXT_TOKENS[model]["long"],
            model=model,
        ):
            return [community_data], None, True

    MAX_CONTEXT = MAX_CONTEXT_TOKENS[model].get(
        "long", MAX_CONTEXT_TOKENS[model]["base"]
    )
    intervals = 2
    final_data = []
    windows: List[Tuple[int, int]] = []
    keys = sorted(community_data.keys())
    start_ts, end_ts = keys[0], keys[-1]
    total_span = end_ts - start_ts

    while intervals <= max_intervals:
        base_window = total_span / intervals
        overlap = max(0, int(base_window * overlap_ratio))
        windows = []
        for i in range(intervals):
            win_start = int(start_ts + i * base_window)
            win_end = int(start_ts + (i + 1) * base_window)
            if i > 0:
                win_start -= overlap
            if i < intervals - 1:
                win_end += overlap
            win_start = max(start_ts, win_start)
            win_end = min(end_ts, win_end)
            if win_end < win_start:
                win_end = win_start
            windows.append((win_start, win_end))

        chunk_data = []
        too_large = False
        for ws, we in windows:
            sub_data = {ts: community_data[ts] for ts in keys if ws <= ts <= we}
            messages = _build_annotator_messages(tags, community, sub_data, agent_names)
            toks = count_tokens(messages, model=model)
            if toks >= MAX_CONTEXT - MAX_OUTPUT_TOKENS[model]:
                too_large = True
                break
            chunk_data.append(sub_data)

        if not too_large:
            log.info(
                "[GroupAnalyst] Split into %d windows: %s",
                len(chunk_data), ", ".join(f"[{a},{b}]" for a, b in windows),
            )
            final_data = chunk_data
            break
        intervals += 1

    return final_data, windows, True


def _merge_annotations(annotations_list: List[dict], merge_fn) -> Tuple[dict, dict]:
    """Merge a list of per-chunk annotation dicts into one."""
    if len(annotations_list) == 1:
        return annotations_list[0], {"input": 0, "output": 0}

    log.info("[GroupAnalyst] Merging annotations...")
    merged = {
        "events": [],
        "behaviors": [],
        "emergence": {"keywords": set(), "comment": []},
        "comment": [],
    }
    total_tokens = {"input": 0, "output": 0}
    for ann in annotations_list:
        merged["events"].extend(ann.get("events", []))
        merged["behaviors"].extend(ann.get("behaviors", []))
        emergence = ann.get("emergence", {})
        merged["emergence"]["keywords"].update(emergence.get("keywords", []))
        merged["emergence"]["comment"].extend(
            [emergence.get("comment", "")]
            if isinstance(emergence.get("comment"), str)
            else emergence.get("comment", [])
        )
        comment = ann.get("comment", "")
        if comment:
            merged["comment"].append(comment)

    merged["emergence"]["keywords"] = list(merged["emergence"]["keywords"])
    merged["comment"], total_tokens = merge_fn(merged["comment"], total_tokens)
    merged["emergence"]["comment"], total_tokens = merge_fn(
        merged["emergence"]["comment"]
        if isinstance(merged["emergence"]["comment"], list)
        else [merged["emergence"]["comment"]],
        total_tokens,
    )
    return merged, total_tokens


class GroupAnalyst(BaseAnalyst):
    """Expert in community/group-level behavioral annotation."""

    def _merge_notes(
        self, notes_list: List[str], total_tokens: dict
    ) -> Tuple[str, dict]:
        """Merge multiple text notes into one via LLM."""
        if len(notes_list) == 1:
            return notes_list[0], total_tokens
        if not notes_list:
            return "", total_tokens

        llm_client = LLMClient(client=self.provider, long_context=False)
        system_prompt = """You are an expert at merging and condensing anthropological notes.
You are given multiple notes of the same group of agents over different time intervals.
Your task is to merge them into a single coherent note."""
        numbered_notes = "".join(
            f"Note {i + 1}:\n{n}\n\n" for i, n in enumerate(notes_list)
        )
        user_prompt = f"""Here are the notes to merge:
    {numbered_notes}
    Provide a single short summary comment of 2-3 sentences that captures the main trends and insights.

    Output a single short summary comment.
    """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        params = copy.deepcopy({})
        response = llm_client.get_response(
            model=self.model,
            messages=messages,
            chat_parameters=params,
            output_json=False,
        )
        total_tokens["input"] += response.input_tokens
        total_tokens["output"] += response.output_tokens
        return str(response.content or "No note available."), total_tokens

    def annotate_group(
        self,
        community: Set[str],
        community_idx: int,
        exp_path: Path,
        save_path: Path | None = None,
        time_range: tuple[int, int] | None = None,
    ) -> tuple[GroupAnnotation, dict]:
        """
        Full annotation pipeline for one community: annotate → audit → narrative.

        Handles context splitting for large communities automatically.
        """
        log.info("[GroupAnalyst] Working on community %d (%d agents)", community_idx, len(community))
        exp_path = Path(exp_path)

        # Load all agent logs for the community
        data = {}
        for agent_tag in community:
            data[agent_tag] = self._load_agent_log_windowed(
                exp_path / "agent_logs" / f"{agent_tag}.jsonl",
                time_range=time_range,
            )
        community_data, community_interval = _merge_logs(data)

        agent_names = {}
        names_path = exp_path / "agent_names.json"
        if names_path.exists():
            with open(names_path) as f:
                agent_names = json.load(f)

        total_tokens = {"input": 0, "output": 0}

        # Split community data if needed
        community_data_chunks, windows, long_context = _split_community_data(
            community=community,
            community_data=community_data,
            agent_names=agent_names,
            tags=self._tags,
            model=self.model,
        )
        if self.force_long_context:
            long_context = True

        llm_client = LLMClient(client=self.provider, long_context=long_context)

        raw_annotations = []
        audited_annotations = []
        anthropologist_responses = []

        for i, data_chunk in enumerate(community_data_chunks):
            if windows is not None:
                log.info(
                    "[GroupAnalyst] Chunk %d/%d: timesteps %d-%d",
                    i + 1, len(community_data_chunks), windows[i][0], windows[i][1],
                )

            # Annotate chunk
            messages = _build_annotator_messages(
                self._tags, community, data_chunk, agent_names
            )
            response = llm_client.get_response(
                model=self.model,
                messages=messages,
                chat_parameters={},
                output_json=True,
            )
            raw_ann_chunk = response.content
            total_tokens["input"] += response.input_tokens
            total_tokens["output"] += response.output_tokens
            raw_annotations.append(raw_ann_chunk)
            assert isinstance(raw_ann_chunk, dict), "Annotations must be a dict"

            # Audit chunk
            if self.audit and raw_ann_chunk is not None:
                log.info("[GroupAnalyst] Auditing chunk %d...", i + 1)
                audited_ann_chunk = deepcopy(raw_ann_chunk)
                audit_prompt = _AUDITOR_USER_PROMPT.format(
                    community_tags=community,
                    agent_names=agent_names,
                    community_data=data_chunk,
                    annotations=raw_ann_chunk,
                    event_tags=self._tags["group_events"],
                    behavior_tags=self._tags["group_behavior"],
                )
                audit_messages = [
                    {"role": "system", "content": _AUDITOR_SYSTEM_PROMPT},
                    {"role": "user", "content": audit_prompt},
                ]
                audit_response = llm_client.get_response(
                    model=self.model,
                    messages=audit_messages,
                    chat_parameters={},
                    output_json=True,
                )
                audits_chunk = audit_response.content
                total_tokens["input"] += audit_response.input_tokens
                total_tokens["output"] += audit_response.output_tokens
                assert isinstance(audits_chunk, dict), "Audits must be a dict"
                audited_ann_chunk = self._apply_audit_revisions(
                    audited_ann_chunk, audits_chunk
                )
                audited_annotations.append(audited_ann_chunk)

                if save_path is not None:
                    os.makedirs(save_path / "raw_annotations", exist_ok=True)
                    with open(
                        save_path
                        / "raw_annotations"
                        / f"community_{community_idx}.json",
                        "w",
                    ) as f:
                        json.dump(raw_annotations, f, indent=4)
                    os.makedirs(save_path / "audits", exist_ok=True)
                    with open(
                        save_path / "audits" / f"community_{community_idx}.json", "w"
                    ) as f:
                        json.dump(audits_chunk, f, indent=4)

            # Anthropologist narrative chunk
            narrative_prompt = _ANTHROPOLOGIST_USER_PROMPT.format(
                community_tags=community,
                community_data=data_chunk,
                agent_names=agent_names,
            )
            narrative_messages = [
                {"role": "system", "content": _ANTHROPOLOGIST_SYSTEM_PROMPT},
                {"role": "user", "content": narrative_prompt},
            ]
            params = {}
            narrative_response = llm_client.get_response(
                model=self.model,
                messages=narrative_messages,
                chat_parameters=params,
                output_json=False,
            )
            anthropologist_responses.append(narrative_response.content)
            total_tokens["input"] += narrative_response.input_tokens
            total_tokens["output"] += narrative_response.output_tokens

        # Merge chunks
        source = (
            audited_annotations
            if (self.audit and audited_annotations)
            else raw_annotations
        )
        final_annotations, merge_tokens = _merge_annotations(source, self._merge_notes)
        total_tokens["input"] += merge_tokens["input"]
        total_tokens["output"] += merge_tokens["output"]

        anthropologist_text, _ = self._merge_notes(
            [r for r in anthropologist_responses if r is not None],
            {"input": 0, "output": 0},
        )
        final_annotations["anthropologist"] = anthropologist_text
        final_annotations["interval"] = list(community_interval)

        if save_path is not None:
            with open(save_path / f"community_{community_idx}.json", "w") as f:
                json.dump(final_annotations, f, indent=4)
            try:
                with open(save_path / "token_counts.jsonl", "a") as f:
                    json.dump({"community_idx": community_idx, **total_tokens}, f)
                    f.write("\n")
            except Exception:
                pass

        behaviors = [
            b for b in final_annotations.get("behaviors", [])
            if isinstance(b, dict) and b.get("behavior")
        ]
        events = [
            e for e in final_annotations.get("events", [])
            if isinstance(e, dict) and e.get("event")
        ]
        return GroupAnnotation(
            community_id=community_idx,
            agents=list(community),
            events=events,
            behaviors=behaviors,
            comment=final_annotations.get("comment", ""),
            emergence=final_annotations.get("emergence", {}),
            anthropologist=anthropologist_text,
            interval=list(community_interval),
        ), total_tokens

    def annotate_groups(
        self,
        exp_path: Path,
        communities: Dict[int, Set[str]] | None = None,
        save_path: Path | None = None,
        time_range: tuple[int, int] | None = None,
        error_tracker: ErrorTracker | None = None,
    ) -> list[GroupAnnotation]:
        """
        Annotate all communities in parallel.

        Args:
            exp_path: Experiment directory.
            communities: Dict of {community_idx: set_of_agent_tags}. If None, loads
                         from communities.json or re-detects via graph.
            save_path: Directory for output JSON files.
            time_range: Optional (start, end) step filter.
            error_tracker: Optional ErrorTracker.

        Returns:
            List of GroupAnnotation objects.
        """
        exp_path = Path(exp_path)

        if communities is None:
            comm_path = exp_path / "community_annotations" / "communities.json"
            if comm_path.exists():
                with open(comm_path) as f:
                    raw = json.load(f)
                communities = {int(k): set(v) for k, v in raw.items()}
                log.info("[GroupAnalyst] Loaded %d communities from file.", len(communities))
            else:
                log.info("[GroupAnalyst] Detecting communities via interaction graph...")
                G, _ = build_graph(run_dir=exp_path)
                comms, _ = get_slpa_communities(G)
                communities = {i: set(c) for i, c in enumerate(comms)}
                if save_path is not None:
                    comm_save = save_path.parent / "communities.json"
                    os.makedirs(comm_save.parent, exist_ok=True)
                    with open(comm_save, "w") as f:
                        json.dump(
                            {k: list(v) for k, v in communities.items()}, f, indent=4
                        )

        log.info("[GroupAnalyst] Annotating %d communities...", len(communities))

        worker = partial(
            self.annotate_group,
            exp_path=exp_path,
            save_path=save_path,
            time_range=time_range,
        )

        results = []
        anthropologist_notes = {}

        if self.parallel:
            with ThreadPoolExecutor(max_workers=4) as ex:
                future_map = {
                    ex.submit(worker, community, idx): idx
                    for idx, community in communities.items()
                }
                for fut in tqdm(
                    as_completed(future_map),
                    total=len(future_map),
                    desc="Annotating groups",
                ):
                    idx = future_map[fut]
                    try:
                        result = fut.result()
                        if result is not None:
                            annotation, _ = result
                            results.append(annotation)
                            if annotation.anthropologist:
                                anthropologist_notes[str(idx)] = (
                                    annotation.anthropologist
                                )
                    except Exception as e:
                        if error_tracker is not None:
                            error_tracker.add_error(
                                f"community_{idx}",
                                e,
                                error_type="error",
                                additional_info={"experiment": str(exp_path)},
                            )
                        else:
                            log.warning("[GroupAnalyst] Error on community %d: %s", idx, e)
        else:
            for idx, community in tqdm(communities.items(), desc="Annotating groups"):
                try:
                    result = worker(community, idx)
                    if result is not None:
                        annotation, _ = result
                        results.append(annotation)
                        if annotation.anthropologist:
                            anthropologist_notes[str(idx)] = annotation.anthropologist
                except Exception as e:
                    if error_tracker is not None:
                        error_tracker.add_error(
                            f"community_{idx}",
                            e,
                            error_type="error",
                            additional_info={"experiment": str(exp_path)},
                        )
                    else:
                        log.warning("[GroupAnalyst] Error on community %d: %s", idx, e)

        if save_path is not None and anthropologist_notes:
            notes_path = save_path / "anthropologist_notes.json"
            try:
                existing = json.load(open(notes_path)) if notes_path.exists() else {}
            except Exception:
                existing = {}
            existing.update(anthropologist_notes)
            with open(notes_path, "w") as f:
                json.dump(existing, f, indent=4, ensure_ascii=False)

        return results
