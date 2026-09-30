"""
AgentAnalyst: expert in individual agent behavioral annotation.

Extracted from 001_llm_agent_analyser.py.
"""

import asyncio
import glob
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm

from terralingua.anthropologist.analysts.base import BaseAnalyst
from terralingua.anthropologist.error_tracker import ErrorTracker

load_dotenv(find_dotenv(usecwd=True), override=True)


@dataclass
class AgentAnnotation:
    tag: str
    name: str
    events: list = field(default_factory=list)
    behaviors: list = field(default_factory=list)
    comment: str = ""
    emergence: dict = field(default_factory=dict)
    anthropologist: str = ""


# Prompts (preserved verbatim from 001_llm_agent_analyser.py)
# ---------------------------
_ANNOTATOR_SYSTEM_PROMPT = """You are an extremely good anthropological annotation engine.
You will receive the logs of an agent.
Your task is to analyze and annotate the logs.
Output VALID JSON ONLY matching the schema.
Never invent IDs or tags. Only make claims that are directly supported by provided fields.
Lower confidence or omit claims when uncertain.
Note: It is extremely important that you get this right, as this will be used for scientific analysis.
"""

_EVENT_ANNOTATOR_USER_PROMPT = """Analyze the following agent's behavior.

The agent log contains a line for each timestep. Each line contains:
- Timestep
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
Analyze the logs and the exchanged messages of the agent and do the following:
1. **Events** (instantaneous)
    - Highlight important events.
    - Tag them with one of the following event tags, given as (EVENT_TAG: description):
        {event_tags}
2. **Behaviors** (spanning multiple timesteps)
    - Identify main behavioral characteristics.
    - Tag them with one of the following behavioral tags, given as (BEHAVIOR_TAG: description):
        {behavioral_tags}
3. **For each annotation (event or behavior) provide:**
    - For events: `"timesteps": [<t1>, ...]`
    - For behaviors: `"time_span": [<start_step>, <end_step>]`
    - `"confidence": <0-10 number>` ("0 = guess, 10 = direct evidence")
    - `"description": "<short natural language description>"`
    - `"reference": [{{"step": <timestep>, "snippet": "<exact short quote>"}}]
4. **References:**
    - For each reference, quote an exact substring from one of: sent_message, received_messages[<agent>], or artifact payload.
    - Do not paraphrase.
    - If no exact quote exists, omit that annotation.
5. **Condensation**
    - If similar events repeat, merge into one entry.
6. **Emergence**
    - Identify any emergent properties of this agent's behavior.
    - Set `"emergence.keywords"` to a list of concise snake_case labels you invent
      (e.g. `["religion_emerged", "market_exchange"]`).  Use your own judgment —
      but set an High bar: only include a keyword if it
      describes a genuinely novel, repeatable pattern, not a one-off action.
    - If no emergent behavior is present, set `"emergence.keywords": ["none"]`.
    - Set `"emergence.comment"` to a short, one-sentence explanation. If truly nothing to say, set it to "none".
7. **Summary**
   - Provide a short 2-3 sentence recap of the agent's life and trends.

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

Agent data:

Agent Name: {agent_name} (internal tag: {agent_tag})
Note: In all descriptions, always refer to this agent by their name ({agent_name}), never by their internal tag.

Agent Life Log
{agent_summary}
"""

_AUDITOR_SYSTEM_PROMPT = """You are an extremely good annotation AUDITOR.
You will receive the logs of an agent and a set of annotations made on those logs.
Your job is to VERIFY, not to re-annotate from scratch.
Output VALID JSON ONLY matching the schema.
Verify that each annotation is SUPPORTED by the logs.
Never invent IDs or tags.
Verify that IDs and tags are not invented but match the provided valid tags.
Note: It is extremely important that you get this right, as this will be used for scientific analysis.
"""

_AUDITOR_USER_PROMPT = """Audit agent {agent_name} (internal tag: {agent_tag}) annotation.
Note: In all descriptions, always refer to this agent by their name ({agent_name}), never by their internal tag.

You are given:
A) Agent Life Log with a line for each timestep. Each line contains:
- Timestep
- Agent name
- Agent tag
- Performed action
- Action parameters
- Message broadcast by agent
- Internal memory of the agent
- Observation containing: messages received from other agents, agent remaining time and energy, agent's inventory

B) A set of annotations with:
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

Agent logs:
{agent_logs}

Annotations:
{annotations}
"""

_ANTHROPOLOGIST_SYSTEM_PROMPT = """You are an experienced anthropologist studying the life and actions of agents living in a 2D world.
You will receive the logs of an agent.
Your task is to identify anything interesting or novel that might emerge from the logs the same way an anthropologist would.
Output a few sentences describing what you discovered.
Keep it short and concise.
Note: It is extremely important that you get this right, as this will be used for scientific analysis.
"""

_ANTHROPOLOGIST_USER_PROMPT = """Analyze the following agent behavior.

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
You will receive the logs of an agent.

The agent log contains a line for each timestep. Each line contains:
- Timestep
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

Agent name: {agent_name} (internal tag: {agent_tag})
Note: In your output, always refer to this agent by their name ({agent_name}), never by their internal tag.

Agent logs:
{agent_logs}
"""
# ---------------------------


class AgentAnalyst(BaseAnalyst):
    """Expert in individual agent behavioral annotation."""

    async def annotate_agent(
        self,
        agent_tag: str,
        exp_path: Path,
        save_path: Path | None = None,
        time_range: tuple[int, int] | None = None,
        audit: bool | None = None,
        verbose: bool = True,
        api_key: str | None = None,
    ) -> tuple[AgentAnnotation, dict] | None:
        """Full annotation pipeline for one agent: annotate → audit → narrative.

        Async so multiple agents can be annotated concurrently via
        ``asyncio.gather`` in :meth:`annotate_agents`. Returns
        ``(AgentAnnotation, token_usage)`` or ``None`` on failure."""
        exp_path = Path(exp_path)

        agent_label = agent_tag
        log_file = exp_path / "agent_logs" / f"{agent_tag}.jsonl"
        try:
            with open(log_file) as _f:
                first = json.loads(_f.readline())
            agent_name_str = first.get("agent")
            if agent_name_str:
                agent_label = agent_name_str
        except Exception:
            pass

        if verbose:
            log.info("[AgentAnalyst] Working on agent %s(%s)", agent_label, agent_tag)

        agent_summary = self._load_agent_log_windowed(
            log_file, time_range=time_range,
        )

        total_tokens = {"input": 0, "output": 0}

        if verbose:
            log.info("[AgentAnalyst] Annotating %s(%s)...", agent_label, agent_tag)
        user_prompt = _EVENT_ANNOTATOR_USER_PROMPT.format(
            agent_name=agent_label,
            agent_tag=agent_tag,
            event_tags=self._tags["agent_events"],
            agent_summary=agent_summary,
            behavioral_tags=self._tags["agent_behavior"],
        )
        messages = [
            {"role": "system", "content": _ANNOTATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        response = await self._run_annotation(messages, api_key=api_key)
        annotations = response.content
        total_tokens["input"] += response.input_tokens
        total_tokens["output"] += response.output_tokens

        if not isinstance(annotations, dict):
            if verbose:
                log.warning("[AgentAnalyst] No valid annotation for %s(%s)", agent_label, agent_tag)
            return None

        if save_path is not None:
            raw_path = save_path / "raw_annotations"
            os.makedirs(raw_path, exist_ok=True)
            with open(raw_path / f"{agent_tag}.json", "w") as f:
                json.dump(annotations, f, indent=4)

        if audit if audit is not None else self.audit:
            if verbose:
                log.info("[AgentAnalyst] Auditing %s(%s)...", agent_label, agent_tag)
            audit_prompt = _AUDITOR_USER_PROMPT.format(
                agent_name=agent_label,
                agent_tag=agent_tag,
                agent_logs=agent_summary,
                annotations=annotations,
                event_tags=self._tags["agent_events"],
                behavior_tags=self._tags["agent_behavior"],
            )
            audit_messages = [
                {"role": "system", "content": _AUDITOR_SYSTEM_PROMPT},
                {"role": "user", "content": audit_prompt},
            ]
            audit_response = await self._run_audit(audit_messages, api_key=api_key)
            audits = audit_response.content
            total_tokens["input"] += audit_response.input_tokens
            total_tokens["output"] += audit_response.output_tokens

            if isinstance(audits, dict):
                if save_path is not None:
                    audits_path = save_path / "audits"
                    os.makedirs(audits_path, exist_ok=True)
                    with open(audits_path / f"{agent_tag}.json", "w") as f:
                        json.dump(audits, f, indent=4)
                annotations = self._apply_audit_revisions(annotations, audits)

        if verbose:
            log.info("[AgentAnalyst] Anthropologist analysis for %s(%s)...", agent_label, agent_tag)
        narrative_prompt = _ANTHROPOLOGIST_USER_PROMPT.format(
            agent_name=agent_label,
            agent_tag=agent_tag,
            agent_logs=agent_summary,
        )
        narrative_messages = [
            {"role": "system", "content": _ANTHROPOLOGIST_SYSTEM_PROMPT},
            {"role": "user", "content": narrative_prompt},
        ]
        narrative_response = await self._get_narrative(narrative_messages, api_key=api_key)
        annotations["anthropologist"] = narrative_response.content
        total_tokens["input"] += narrative_response.input_tokens
        total_tokens["output"] += narrative_response.output_tokens

        if save_path is not None:
            # Persist the model that produced this annotation so a future
            # reader knows which provider/model the cost was attributed to.
            # The folder layout no longer carries the model name; this field
            # is the canonical source of that info.
            annotations.setdefault("_meta", {})["model"] = self.model
            with open(save_path / f"{agent_tag}.json", "w") as f:
                json.dump(annotations, f, indent=4, ensure_ascii=False)

        agent_name = agent_tag
        names_path = exp_path / "agent_names.json"
        if names_path.exists():
            with open(names_path) as f:
                names = json.load(f)
            agent_name = names.get(agent_tag, agent_tag)

        behaviors = [
            b for b in annotations.get("behaviors", [])
            if isinstance(b, dict) and b.get("behavior")
        ]
        events = [
            e for e in annotations.get("events", [])
            if isinstance(e, dict) and e.get("event")
        ]
        return AgentAnnotation(
            tag=agent_tag,
            name=agent_name,
            events=events,
            behaviors=behaviors,
            comment=annotations.get("comment", ""),
            emergence=annotations.get("emergence", {}),
            anthropologist=annotations.get("anthropologist", ""),
        ), total_tokens

    async def annotate_agents(
        self,
        agent_tags: list[str],
        exp_path: Path,
        save_path: Path | None = None,
        time_range: tuple[int, int] | None = None,
        error_tracker: ErrorTracker | None = None,
        audit: bool | None = None,
        parallel: bool = True,
        verbose: bool = True,
        api_key: str | None = None,
        resolve_key_fn=None,
    ) -> list[AgentAnnotation]:
        """Annotate specified agents, running in parallel if self.parallel.

        Async-native: awaits each agent's annotate_agent directly. When
        parallel=True, fans out via asyncio.gather; otherwise sequential.

        Per-agent API key resolution: pass either a single ``api_key`` (used
        uniformly — postmortem path) or a ``resolve_key_fn(agent_tag) -> str``
        callback (per-agent — live annotation path).
        """
        exp_path = Path(exp_path)

        def _key_for(tag: str) -> str | None:
            if resolve_key_fn is not None:
                return resolve_key_fn(tag)
            return api_key

        results: list[AgentAnnotation] = []

        if parallel:
            gathered = await asyncio.gather(
                *[
                    self.annotate_agent(
                        tag,
                        exp_path=exp_path,
                        save_path=save_path,
                        time_range=time_range,
                        audit=audit,
                        verbose=verbose,
                        api_key=_key_for(tag),
                    )
                    for tag in agent_tags
                ],
                return_exceptions=True,
            )
            for tag, item in zip(agent_tags, gathered):
                if isinstance(item, BaseException):
                    if error_tracker is not None:
                        error_tracker.add_error(
                            tag,
                            item,
                            error_type="error",
                            additional_info={"experiment": str(exp_path)},
                        )
                    else:
                        log.warning("[AgentAnalyst] Error on %s: %s", tag, item)
                    continue
                if item is None:
                    continue
                annotation, tokens = item
                results.append(annotation)
                if save_path is not None and tokens is not None:
                    try:
                        with open(save_path / "token_usage.jsonl", "a") as f:
                            json.dump({"agent": tag, **tokens}, f)
                            f.write("\n")
                    except Exception:
                        pass
        else:
            for tag in tqdm(agent_tags, desc="Annotating agents"):
                try:
                    result = await self.annotate_agent(
                        tag,
                        exp_path=exp_path,
                        save_path=save_path,
                        time_range=time_range,
                        audit=audit,
                        verbose=verbose,
                        api_key=_key_for(tag),
                    )
                    if result is not None:
                        annotation, tokens = result
                        results.append(annotation)
                        if save_path is not None and tokens is not None:
                            try:
                                with open(save_path / "token_usage.jsonl", "a") as f:
                                    json.dump({"agent": tag, **tokens}, f)
                                    f.write("\n")
                            except Exception:
                                pass
                except Exception as e:
                    if error_tracker is not None:
                        error_tracker.add_error(
                            tag,
                            e,
                            error_type="error",
                            additional_info={"experiment": str(exp_path)},
                        )
                    else:
                        log.warning("[AgentAnalyst] Error on %s: %s", tag, e)

        return results

    def annotate_experiment(
        self,
        exp_path: Path,
        save_path: Path | None = None,
        time_range: tuple[int, int] | None = None,
        error_tracker: ErrorTracker | None = None,
    ) -> list[AgentAnnotation]:
        """
        Annotate all agents in an experiment directory (auto-discovers being*.jsonl).

        Args:
            exp_path: Experiment directory.
            save_path: Directory for saving per-agent JSON files.
            time_range: Optional (start, end) step filter.
            error_tracker: Optional ErrorTracker for recording failures.

        Returns:
            List of AgentAnnotation objects (failed agents are skipped).
        """
        exp_path = Path(exp_path)
        agent_files = glob.glob(str(exp_path / "agent_logs" / "being*.jsonl"))
        agent_tags = [os.path.splitext(os.path.basename(f))[0] for f in agent_files]
        results = self.annotate_agents(
            agent_tags,
            exp_path,
            save_path=save_path,
            time_range=time_range,
            error_tracker=error_tracker,
        )

        # Persist anthropologist notes (merged from all agent annotations)
        if save_path is not None:
            notes = {a.tag: a.anthropologist for a in results if a.anthropologist}
            if notes:
                notes_path = save_path / "anthropologist_notes.json"
                try:
                    if notes_path.exists():
                        with open(notes_path) as f:
                            existing = json.load(f)
                    else:
                        existing = {}
                except Exception:
                    existing = {}
                existing.update(notes)
                with open(notes_path, "w") as f:
                    json.dump(existing, f, indent=4, ensure_ascii=False)

        return results
