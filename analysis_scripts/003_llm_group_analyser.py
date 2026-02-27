import copy
import glob
import json
import os
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from functools import partial
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Set, Tuple

import numpy as np
from dotenv import load_dotenv
from graph_utils import build_graph, get_slpa_communities
from tqdm import tqdm

from analysis_scripts.error_tracker import ErrorTracker
from core.utils import ROOT
from core.utils.analysis_utils import load_agent_log
from core.utils.llm_client import LLMClient
from core.utils.llm_utils import (
    MAX_CONTEXT_TOKENS,
    MAX_OUTPUT_TOKENS,
    count_tokens,
    is_context_enough,
)

load_dotenv()

"""
This script analyzes the logs of groups of agents (communities) identified with script 002 using LLMs.
It performs the following steps for each community:
1. Load the logs of all agents in the community and merge them into a single chronological log
2. Annotate the merged logs using an LLM with a detailed prompt (ANNOTATOR)
3. Audit the annotations using a second LLM with a detailed prompt (AUDITOR)
4. Analyze the logs from an anthropological perspective using a third LLM with a detailed prompt (ANTHROPOLOGIST)
5. Save the raw annotations, audits, and final merged annotations to disk for each community.
"""

EXPERIMENTS_NAMES = []  # e.g. ['core_run', 'scarcity_run', ...]

# Setup
# ---------------------------
FOLDER_PATH = Path(__file__).resolve().parent
AGENT_PREFIX = "being"
PERFORM_AUDIT = True
LLM_PROVIDER = "anthropic"
LLM_MODEL = "claude-sonnet-4-5-20250929"
LLM_CHAT_PARAMS = {
    # "response_format": {"type": "json_object"},
}
FORCE_LONG_CONTEXT = False
# ---------------------------

# Prompts
# ---------------------------
ANNOTATOR_SYSTEM_PROMPT = """You are an extremely good anthropological annotation engine. 
You will receive the logs of a group of agents.
Your task is to analyze and annotate the logs.
Output VALID JSON ONLY matching the schema.
Never invent IDs or tags. Only make claims that are directly supported by provided fields. 
Lower confidence or omit claims when uncertain.
Note: It is extremely important that you get this right, as this will be used for scientific analysis.
"""

EVENT_ANNOTATOR_USER_PROMPT = """Analyze the following group behavior.
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
    - "confidence": <0–10 number>` ("0 = guess, 10 = direct evidence")
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
   - Provide a short 2–3 sentence recap of the group life and trends.

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

AUDITOR_SYSTEM_PROMPT = """You are an extremely good annotation AUDITOR.
You will receive the logs of a group of agents and a set of annotations made on those logs.
Your job is to VERIFY, not to re-annotate from scratch.
Output VALID JSON ONLY matching the schema.
Verify that each annotation is SUPPORTED by the logs.
Never invent IDs or tags. 
Verify that IDs and tags are not invented but match the provided valid tags.
Note: It is extremely important that you get this right, as this will be used for scientific analysis.
"""

AUDITOR_USER_PROMPT = """Audit the following group annotations.

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

ANTHROPOLOGIST_SYSTEM_PROMPT = """You are an experienced anthropologist studying the life and actions of agents living in a 2D world.
You will receive the logs of a group of agents.
Your task is to identify anything interesting or novel that might emerge from the logs the same way an anthropologist would.
Output a few sentences describing what you discovered.
Keep it short and concise.
Note: It is extremely important that you get this right, as this will be used for scientific analysis.
"""

ANTHROPOLOGIST_USER_PROMPT = """Analyze the following group behavior.

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


# Utils
# ---------------------------
def merge_logs(data: Dict[str, Dict[int, dict]]):
    merged_data = {}
    start_ts = min(min(d.keys()) for d in data.values())
    end_ts = max(max(d.keys()) for d in data.values())

    for ts in range(start_ts, end_ts + 1):
        merged_data[ts] = []
        for agent_data in data.values():
            if ts in agent_data:
                data_point = deepcopy(agent_data[ts])
                # Clean up useless things
                del data_point["timestamp"]
                if "available_actions" in data_point:
                    del data_point["available_actions"]
                del data_point["observation"]
                data_point["agent_name"] = data_point.pop("agent")

                merged_data[ts].append(data_point)

    return merged_data, (start_ts, end_ts)


def build_annotator_messages(tags, community, community_data, agent_names):
    user_prompt = EVENT_ANNOTATOR_USER_PROMPT.format(
        event_tags=tags["group_events"],
        behavioral_tags=tags["group_behavior"],
        emergent_tags=tags["group_emergence"],
        community_tags=community,
        community_data=community_data,
        agent_names=agent_names,
    )
    return [
        {"role": "system", "content": ANNOTATOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def split_community_data(
    community: Set[str],
    community_data: dict,
    agent_names: dict,
    tags: dict,
    overlap_ratio: float = 0.33,
    max_intervals: int = 32,
):
    """
    Split community_data into overlapping time intervals so each prompt fits in context.

    overlap_ratio: fraction of each window length to overlap with previous/next window.
    Returns: list of message lists (each suitable for one LLM call).
    """
    full_messages = build_annotator_messages(
        tags, community, community_data, agent_names
    )
    if is_context_enough(
        messages=full_messages,
        max_input_tokens=MAX_CONTEXT_TOKENS[LLM_MODEL]["base"],
        model=LLM_MODEL,
    ):
        print("No need to split community data..")
        return [community_data], None, False  # Fits in context

    if MAX_CONTEXT_TOKENS[LLM_MODEL].get("long", None) is not None:
        if is_context_enough(
            messages=full_messages,
            max_input_tokens=MAX_CONTEXT_TOKENS[LLM_MODEL]["long"],
            model=LLM_MODEL,
        ):
            print("No need to split community data. Using long context..")
            return [community_data], None, True

    # If we have to split we use the longest context available...
    MAX_CONTEXT = MAX_CONTEXT_TOKENS[LLM_MODEL].get(
        "long", MAX_CONTEXT_TOKENS[LLM_MODEL]["base"]
    )

    # Need to chunk
    intervals = 2
    final_data = []
    keys = sorted(community_data.keys())
    start_ts, end_ts = keys[0], keys[-1]
    total_span = end_ts - start_ts

    while intervals <= max_intervals:
        base_window = total_span / intervals
        overlap = max(0, int(base_window * overlap_ratio))

        windows: List[Tuple[int, int]] = []
        for i in range(intervals):
            win_start = int(start_ts + i * base_window)
            win_end = int(start_ts + (i + 1) * base_window)
            if i > 0:
                win_start -= overlap
            if i < intervals - 1:
                win_end += overlap
            # Clamp
            win_start = max(start_ts, win_start)
            win_end = min(end_ts, win_end)
            # Ensure monotonic non-empty
            if windows and win_start <= windows[-1][1]:
                # Allow overlap but avoid zero-length regressions
                pass
            if win_end < win_start:
                win_end = win_start
            windows.append((win_start, win_end))

        # Build prompts per window
        chunk_data = []
        too_large = False
        for ws, we in windows:
            sub_data = {ts: community_data[ts] for ts in keys if ws <= ts <= we}
            messages = build_annotator_messages(tags, community, sub_data, agent_names)
            toks = count_tokens(messages, model=LLM_MODEL)
            if toks >= MAX_CONTEXT - MAX_OUTPUT_TOKENS[LLM_MODEL]:
                too_large = True
                break
            chunk_data.append(sub_data)

        if not too_large:
            print(
                f"Split community data into {len(chunk_data)} overlapping windows: "
                + ", ".join(f"[{a},{b}]" for a, b in windows)
            )
            final_data = chunk_data
            break

        intervals += 1  # Try finer split

    return final_data, windows, True  # We use the longest context available


def merge_notes(notes_list: List[str], total_tokens) -> Tuple[str, dict]:
    if len(notes_list) == 1:
        return notes_list[0], total_tokens

    llm_client = LLMClient(client=LLM_PROVIDER, long_context=False)

    # Prompts for merging comments
    # ---------------------------
    system_prompt = """You are an expert at merging and condensing anthropological notes.
You are given multiple notes of the same group of agents over different time intervals.
Your task is to merge them into a single coherent note."""

    user_prompt = """Here are the notes to merge:
    {comments}
    Provide a single short summary comment of 2-3 sentences that captures the main trends and insights.

    Output a single short summary comment.
    """
    # ---------------------------

    numbered_notes = ""
    for i in range(len(notes_list)):
        numbered_notes += f"Note {i + 1}:\n{notes_list[i]}\n\n"

    whole_note_prompt = user_prompt.format(comments=numbered_notes)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": whole_note_prompt},
    ]
    chat_params = copy.deepcopy(LLM_CHAT_PARAMS)
    chat_params.pop("response_format", None)
    response = llm_client.get_response(
        model=LLM_MODEL,
        messages=messages,
        chat_parameters=chat_params,
        output_json=False,
    )
    merged_note = response.content
    total_tokens["input"] += response.input_tokens
    total_tokens["output"] += response.output_tokens

    if merged_note is None:
        merged_note = "No note available."
    return str(merged_note), total_tokens


def merge_anthropologist_responses(
    responses_list: List[str], total_tokens
) -> Tuple[str, dict]:
    if len(responses_list) == 1:
        return responses_list[0], total_tokens

    print("Merging anthropologist responses...")
    merged_response, total_tokens = merge_notes(responses_list, total_tokens)
    return merged_response, total_tokens


def merge_annotations(annotations_list: List[dict], total_tokens) -> Tuple[dict, dict]:
    if len(annotations_list) == 1:
        return annotations_list[0], total_tokens

    print("Merging annotations...")
    merged = {
        "events": [],
        "behaviors": [],
        "emergence": {
            "keywords": set(),
            "comment": [],
        },
        "comment": [],
    }
    for ann in annotations_list:
        events = ann.get("events", [])
        merged["events"].extend(events)

        behaviors = ann.get("behaviors", [])
        merged["behaviors"].extend(behaviors)

        emergence = ann.get("emergence", {})
        keywords = emergence.get("keywords", [])
        merged["emergence"]["keywords"].update(keywords)
        comments = emergence.get("comment", [])
        merged["emergence"]["comment"].extend(comments)

        comment = ann.get("comment", "")
        merged["comment"].append(comment)

    merged["emergence"]["keywords"] = list(merged["emergence"]["keywords"])
    merged["comment"], total_tokens = merge_notes(merged["comment"], total_tokens)
    merged["emergence"]["comment"], total_tokens = merge_notes(
        merged["emergence"]["comment"], total_tokens
    )
    return merged, total_tokens


# -------------------------------


def annotate_and_audit(
    community_idx: int,
    community: Set[str],
    exp_path: Path,
    save_path: Path,
):
    # community_idx, community = community # type: ignore
    print("===============================</>")
    print(f"Working on community {community_idx}")
    print("Loading files")
    data = {}
    for agent_tag in community:
        data[agent_tag] = load_agent_log(
            exp_path / "agent_logs" / f"{agent_tag}.jsonl", reduce=True
        )

    community_data, community_interval = merge_logs(data)
    print("Files loaded")

    agent_names = json.load(open(exp_path / "agent_names.json", "r"))

    total_tokens = {"input": 0, "output": 0}

    # Initialize LLM client

    # Load tags
    tags = json.load(open(FOLDER_PATH / "tags.json", "r"))

    community_data, windows, long_context = split_community_data(
        community=community,
        community_data=community_data,
        agent_names=agent_names,
        tags=tags,
    )
    if FORCE_LONG_CONTEXT:
        long_context = True

    llm_client = LLMClient(client=LLM_PROVIDER, long_context=long_context)

    raw_annotations = []
    audits = []
    audited_annotations = []
    anthropologist_response = []
    for i, data_chunk in enumerate(community_data):
        # Annotate
        # ---------------------------
        if windows is not None:
            print(
                f"Annotating chunk {i + 1}/{len(community_data)}: timesteps {windows[i][0]} to {windows[i][1]}"
            )

        messages = build_annotator_messages(tags, community, data_chunk, agent_names)

        response = llm_client.get_response(
            model=LLM_MODEL,
            messages=messages,
            chat_parameters=LLM_CHAT_PARAMS,
            output_json=True,
        )
        raw_ann_chunk = response.content
        total_tokens["input"] += response.input_tokens
        total_tokens["output"] += response.output_tokens

        raw_annotations.append(raw_ann_chunk)
        assert isinstance(raw_ann_chunk, dict), (
            "Parsed annotations should be a dictionary."
        )
        # ---------------------------

        # Audit
        # ---------------------------
        if PERFORM_AUDIT and raw_ann_chunk is not None:
            print("Auditing...")
            audited_ann_chunk = deepcopy(raw_ann_chunk)

            auditor_user_prompt = AUDITOR_USER_PROMPT.format(
                community_tags=community,
                agent_names=agent_names,
                community_data=data_chunk,
                annotations=raw_ann_chunk,
                event_tags=tags["group_events"],
                behavior_tags=tags["group_behavior"],
            )
            auditor_messages = [
                {"role": "system", "content": AUDITOR_SYSTEM_PROMPT},
                {"role": "user", "content": auditor_user_prompt},
            ]

            response = llm_client.get_response(
                model=LLM_MODEL,
                messages=auditor_messages,
                chat_parameters=LLM_CHAT_PARAMS,
                output_json=True,
            )
            audits_chunk = response.content
            total_tokens["input"] += response.input_tokens
            total_tokens["output"] += response.output_tokens

            assert isinstance(audits_chunk, dict), (
                "Parsed audits should be a dictionary."
            )
            audits.append(audits_chunk)

            # Update annotations
            for ann_type in ["behaviors", "events"]:
                removals = []
                for audit in audits_chunk.get(f"{ann_type}_audit", []):
                    verdict = audit.get("verdict", "pass")
                    confidence = int(audit.get("confidence", 0))
                    print(
                        f"{ann_type} annotation audit verdict: {verdict}. Confidence: {confidence}"
                    )
                    if verdict == "revise" and confidence > 6:
                        proposed_fix = audit.get("proposed_fix", {})
                        index = audit.get("index")
                        if index is not None and 0 <= index < len(
                            audited_ann_chunk[ann_type]
                        ):
                            print("Annotation:")
                            pprint(audited_ann_chunk[ann_type][index])
                            print()
                            print("Audit:")
                            pprint(audit)
                            print()
                            for key in proposed_fix:
                                if (
                                    proposed_fix[key] is not None
                                    and key in audited_ann_chunk[ann_type][index]
                                ):
                                    audited_ann_chunk[ann_type][index][key] = (
                                        proposed_fix[key]
                                    )
                            print("Annotation REVISED")
                    elif verdict == "fail" and confidence > 6:
                        index = audit.get("index")
                        if index is not None:
                            print("Annotation:")
                            pprint(audited_ann_chunk[ann_type][index])
                            print()
                            print("Audit:")
                            pprint(audit)
                            print()
                            removals.append(index)
                            print("ANNOTATION REMOVED")

                for idx in sorted(removals, reverse=True):
                    if 0 <= idx < len(audited_ann_chunk[ann_type]):
                        audited_ann_chunk[ann_type].pop(idx)
            print()

            audited_annotations.append(audited_ann_chunk)
        # ---------------------------

        # Anthropologist
        # ---------------------------
        anthropologist_user_prompt = ANTHROPOLOGIST_USER_PROMPT.format(
            community_tags=community,
            community_data=data_chunk,
            agent_names=agent_names,
        )

        anthropologist_messages = [
            {"role": "system", "content": ANTHROPOLOGIST_SYSTEM_PROMPT},
            {"role": "user", "content": anthropologist_user_prompt},
        ]

        anthropologist_params = deepcopy(LLM_CHAT_PARAMS)
        anthropologist_params.pop("response_format", None)
        print("Anthropologist analysis...")
        response = llm_client.get_response(
            model=LLM_MODEL,
            messages=anthropologist_messages,
            chat_parameters=anthropologist_params,
            output_json=False,
        )
        anth_chunk_response = response.content
        total_tokens["input"] += response.input_tokens
        total_tokens["output"] += response.output_tokens

        anthropologist_response.append(anth_chunk_response)
        # ---------------------------

    # Save data
    # ---------------------------
    anthropologist_response, total_tokens = merge_anthropologist_responses(
        anthropologist_response, total_tokens
    )

    if PERFORM_AUDIT:
        annotations_path = save_path / "raw_annotations"
        os.makedirs(annotations_path, exist_ok=True)
        with open(annotations_path / f"community_{community_idx}.json", "w") as f:
            json.dump(raw_annotations, f, indent=4)

        audits_path = save_path / "audits"
        os.makedirs(audits_path, exist_ok=True)
        with open(audits_path / f"community_{community_idx}.json", "w") as f:
            json.dump(audits, f, indent=4)

        final_annotations, total_tokens = merge_annotations(
            audited_annotations, total_tokens
        )
        final_annotations["anthropologist"] = anthropologist_response
        final_annotations["interval"] = list(community_interval)
        with open(save_path / f"community_{community_idx}.json", "w") as f:
            json.dump(final_annotations, f, indent=4)
    else:
        final_annotations, total_tokens = merge_annotations(
            raw_annotations, total_tokens
        )
        final_annotations["anthropologist"] = anthropologist_response
        final_annotations["interval"] = list(community_interval)
        with open(save_path / f"community_{community_idx}.json", "w") as f:
            json.dump(final_annotations, f, indent=4)
    # ---------------------------
    tok_counter = {
        "community_idx": community_idx,
        "input_tokens": total_tokens["input"],
        "output_tokens": total_tokens["output"],
    }

    return final_annotations, tok_counter


def main(exp_path: Path | str, error_tracker: ErrorTracker):
    exp_path = Path(exp_path)

    save_path = exp_path / "community_annotations" / LLM_MODEL
    os.makedirs(save_path, exist_ok=True)

    if (save_path.parent / "communities.json").exists():
        with open(save_path.parent / "communities.json", "r") as f:
            comm_dict = json.load(f)
        print("Communities already computed, skipping community detection.")
    else:
        print("Building interaction graph and detecting communities...")
        G, _ = build_graph(run_dir=exp_path)
        communities, _ = get_slpa_communities(G)

        comm_dict = {i: list(communities[i]) for i in range(len(communities))}
        with open(save_path.parent / "communities.json", "w") as f:
            json.dump(comm_dict, f, indent=4)

    print(f"Detected {len(comm_dict)} communities.")
    worker = partial(
        annotate_and_audit,
        exp_path=exp_path,
        save_path=save_path,
    )

    annotations = {}

    max_workers = 4
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {
            ex.submit(worker, int(idx), set(comm)): idx
            for idx, comm in comm_dict.items()
        }
        for fut in tqdm(
            as_completed(future_map),
            total=len(future_map),
            desc="Processing communities",
        ):
            comm_idx = future_map[fut]
            tokens = None
            try:
                results = fut.result()
                annotations[comm_idx] = results[0]
                tokens = results[1]
            except Exception as e:
                error_tracker.add_error(
                    f"community_{comm_idx}",
                    e,
                    error_type="error",
                    additional_info={
                        "community_members": comm_dict[comm_idx],
                        "experiment": str(exp_path),
                    },
                )
                continue

            try:
                if tokens is not None:
                    with open(save_path / "token_counts.jsonl", "a") as f:
                        json.dump(tokens, f)
                        f.write("\n")
            except:
                print(f"Cannot save token count for community {comm_idx}")
                print(tokens)

    try:
        anthropologist_notes = json.load(open(save_path / "anthropologist_notes.json"))
    except Exception:
        anthropologist_notes = {}
    for comm in annotations:
        note = annotations[comm].get("anthropologist")
        if note is not None:
            anthropologist_notes[comm] = note
    with open(save_path / "anthropologist_notes.json", "w") as f:
        json.dump(anthropologist_notes, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    failed = []

    main_tracker = ErrorTracker(show_stacktraces=False)

    for exp in EXPERIMENTS_NAMES:
        print(f"Running analysis on experiment: {exp}")
        try:
            main(exp_path=ROOT / "logs" / exp, error_tracker=main_tracker)
        except Exception as e:
            main_tracker.add_experiment_failure(exp, e)
        print()

    main_tracker.print_summary()
    main_tracker.save_to_file(ROOT / "analysis_summary" / "003_summary.json")
