import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from functools import partial
from pathlib import Path
from pprint import pprint

from dotenv import load_dotenv
from tqdm import tqdm

from analysis_scripts.error_tracker import ErrorTracker
from core.utils import ROOT
from core.utils.analysis_utils import load_agent_log
from core.utils.llm_client import LLMClient, Response
from core.utils.llm_utils import (
    MAX_CONTEXT_TOKENS,
    is_context_enough,
)

load_dotenv()

"""
This script analyzes the logs of agents using LLMs. 

It performs the following steps for each agent:
1. Load the agent's logs.
2. Generate annotations of the agent's behavior using an LLM.
3. Audit the annotations for consistency and accuracy using another LLM.
4. Optionally, have an "anthropologist" LLM provide insights on the agent's behavior.
5. Save the annotations, audits, and anthropologist notes.
"""

EXPERIMENTS_NAMES = []  # e.g. ['core_run', 'scarcity_run', ...]

# Setup
# ---------------------------
AUDIT = True
SHOW_STACKTRACES = False
LLM_PROVIDER = "anthropic"
LLM_MODEL = "claude-sonnet-4-5-20250929"
LLM_CHAT_PARAMS = {
    # "response_format": {"type": "json_schema"},
}
FORCE_LONG_CONTEXT = False
# ---------------------------


# Prompts
# ---------------------------
ANNOTATOR_SYSTEM_PROMPT = """You are an extremely good anthropological annotation engine. 
You will receive the logs of an agent.
Your task is to analyze and annotate the logs.
Output VALID JSON ONLY matching the schema.
Never invent IDs or tags. Only make claims that are directly supported by provided fields. 
Lower confidence or omit claims when uncertain.
Note: It is extremely important that you get this right, as this will be used for scientific analysis.
"""

EVENT_ANNOTATOR_USER_PROMPT = """Analyze the following agent's behavior.

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
    - For each reference, quote an exact substring from one of: action.message, observation.message[<agent>], or artifact payload. 
    - Do not paraphrase. 
    - If no exact quote exists, omit that annotation.
5. **Condensation**
    - If similar events repeat, merge into one entry.
6. **Emergence**
    - Identify any emergent properties.
    - Set `"emergence.keywords"` to a list using only these tags: {emergent_tags}.
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

Agent Name: {agent_name}

Agent Life Log
{agent_summary}
"""

AUDITOR_SYSTEM_PROMPT = """You are an extremely good annotation AUDITOR. 
You will receive the logs of an agent and a set of annotations made on those logs.
Your job is to VERIFY, not to re-annotate from scratch.
Output VALID JSON ONLY matching the schema.
Verify that each annotation is SUPPORTED by the logs.
Never invent IDs or tags. 
Verify that IDs and tags are not invented but match the provided valid tags.
Note: It is extremely important that you get this right, as this will be used for scientific analysis.
"""

AUDITOR_USER_PROMPT = """Audit agent {agent_name} annotation.

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

ANTHROPOLOGIST_SYSTEM_PROMPT = """You are an experienced anthropologist studying the life and actions of agents living in a 2D world.
You will receive the logs of an agent.
Your task is to identify anything interesting or novel that might emerge from the logs the same way an anthropologist would.
Output a few sentences describing what you discovered.
Keep it short and concise.
Note: It is extremely important that you get this right, as this will be used for scientific analysis.
"""

ANTHROPOLOGIST_USER_PROMPT = """Analyze the following agent behavior.

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

Agent name: {agent_name}

Agent logs:
{agent_logs}
"""
# ---------------------------


def annotate_and_audit(agent_name: str, exp_path: Path, save_path: Path):
    print(f"Working on agent {agent_name}")
    print("Loading files")
    agent_summary = load_agent_log(
        exp_path / "agent_logs" / f"{agent_name}.jsonl", reduce=True
    )
    for data in agent_summary.values():
        data.pop("available_actions", None)
        data["observation"].pop("vision_radius", None)
        data["observation"].pop("observation", None)
        if "reasoning" in data["action"]:
            data["action"].pop("reasoning", None)

    print("Files loaded")
    total_tokens = {"input": 0, "output": 0}

    # Load tags
    tags = json.load(open(ROOT / "analysis_scripts" / "tags.json", "r"))

    # Annotate
    # ---------------------------
    print("Annotating...")
    user_prompt = EVENT_ANNOTATOR_USER_PROMPT.format(
        agent_name=agent_name,
        event_tags=tags["agent_events"],
        agent_summary=agent_summary,
        behavioral_tags=tags["agent_behavior"],
        emergent_tags=tags["agent_emergence"],
    )
    messages = [
        {"role": "system", "content": ANNOTATOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Initialize LLM client
    if is_context_enough(
        messages=messages,
        max_input_tokens=MAX_CONTEXT_TOKENS[LLM_MODEL]["base"],
        model=LLM_MODEL,
    ):
        llm_client = LLMClient(client=LLM_PROVIDER, long_context=False)
    else:
        print(f"Using long context model for annotating agent {agent_name}")
        llm_client = LLMClient(client=LLM_PROVIDER, long_context=True)

    if FORCE_LONG_CONTEXT:
        print(f"Forcing long context model for annotating agent {agent_name}")
        llm_client = LLMClient(client=LLM_PROVIDER, long_context=True)

    response = llm_client.get_response(
        model=LLM_MODEL,
        messages=messages,
        chat_parameters=LLM_CHAT_PARAMS,
        output_json=True,
    )
    annotations = response.content
    total_tokens["input"] += response.input_tokens
    total_tokens["output"] += response.output_tokens

    if annotations is None:
        print(f"No valid annotation for agent {agent_name}")
        return
    else:
        raw_annotations = save_path / "raw_annotations"
        os.makedirs(raw_annotations, exist_ok=True)
        with open(raw_annotations / f"{agent_name}.json", "w") as f:
            json.dump(annotations, f, indent=4)

    assert isinstance(annotations, dict), "Parsed annotations should be a dictionary."
    # ---------------------------

    # Audit
    # ---------------------------
    if AUDIT:
        print("Auditing...")
        user_prompt = AUDITOR_USER_PROMPT.format(
            agent_name=agent_name,
            agent_logs=agent_summary,
            annotations=annotations,
            event_tags=tags["agent_events"],
            behavior_tags=tags["agent_behavior"],
        )
        messages = [
            {"role": "system", "content": AUDITOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Initialize LLM client
        if is_context_enough(
            messages=messages,
            max_input_tokens=MAX_CONTEXT_TOKENS[LLM_MODEL]["base"],
            model=LLM_MODEL,
        ):
            llm_client = LLMClient(client=LLM_PROVIDER, long_context=False)
        else:
            print(f"Using long context model for auditing agent {agent_name}")
            llm_client = LLMClient(client=LLM_PROVIDER, long_context=True)

        if FORCE_LONG_CONTEXT:
            print(f"Forcing long context model for auditing agent {agent_name}")
            llm_client = LLMClient(client=LLM_PROVIDER, long_context=True)

        response = llm_client.get_response(
            model=LLM_MODEL,
            messages=messages,
            chat_parameters=LLM_CHAT_PARAMS,
            output_json=True,
        )
        audits = response.content
        total_tokens["input"] += response.input_tokens
        total_tokens["output"] += response.output_tokens

        if audits is None:
            print(f"No valid audit for {agent_name}")
            return
        else:
            audits_path = save_path / "audits"
            os.makedirs(audits_path, exist_ok=True)
            with open(audits_path / f"{agent_name}.json", "w") as f:
                json.dump(audits, f, indent=4)

        assert isinstance(audits, dict), "Parsed audits should be a dictionary."
        # ---------------------------

        # Update annotations
        to_remove = {"behaviors": [], "events": []}
        for ann_type in ["behaviors", "events"]:
            for audit in audits.get(f"{ann_type}_audit", []):
                verdict = audit.get("verdict", "pass")
                confidence = int(audit.get("confidence", 0))
                print(
                    f"{ann_type} annotation audit verdict: {verdict}. Confidence: {confidence}"
                )
                if verdict == "revise" and confidence > 6:
                    proposed_fix = audit.get("proposed_fix", {})
                    index = audit.get("index")
                    if index is not None:
                        print("Annotation:")
                        pprint(annotations[ann_type][index])
                        print()
                        print("Audit:")
                        pprint(audit)
                        print()
                        for key in proposed_fix:
                            if (
                                proposed_fix[key] is not None
                                and key in annotations[ann_type][index]
                            ):
                                annotations[ann_type][index][key] = proposed_fix[key]
                        print("Annotation REVISED")
                elif verdict == "fail" and confidence > 6:
                    index = audit.get("index")
                    if index is not None:
                        print("Annotation:")
                        pprint(annotations[ann_type][index])
                        print()
                        print("Audit:")
                        pprint(audit)
                        print()
                        to_remove[ann_type].append(index)
                        print("ANNOTATION REMOVED")

        for ann_type in ["behaviors", "events"]:
            for index in sorted(to_remove[ann_type], reverse=True):
                annotations[ann_type].pop(index)

        print()

    # Anthropologist
    # ---------------------------
    user_prompt = ANTHROPOLOGIST_USER_PROMPT.format(
        agent_name=agent_name, agent_logs=agent_summary
    )

    anthropologist_params = deepcopy(LLM_CHAT_PARAMS)
    anthropologist_params.pop("response_format", None)
    print("Anthropologist analysis...")
    messages = [
        {"role": "system", "content": ANTHROPOLOGIST_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Initialize LLM client
    if is_context_enough(
        messages=messages,
        max_input_tokens=MAX_CONTEXT_TOKENS[LLM_MODEL]["base"],
        model=LLM_MODEL,
    ):
        llm_client = LLMClient(client=LLM_PROVIDER, long_context=False)
    else:
        print(
            f"Using long context model for anthropologist analysis of agent {agent_name}"
        )
        llm_client = LLMClient(client=LLM_PROVIDER, long_context=True)

    if FORCE_LONG_CONTEXT:
        print(
            f"Forcing long context model for anthropologist analysis of agent {agent_name}"
        )
        llm_client = LLMClient(client=LLM_PROVIDER, long_context=True)

    response = llm_client.get_response(
        model=LLM_MODEL,
        messages=messages,
        chat_parameters=anthropologist_params,
        output_json=False,
    )
    # ---------------------------

    annotations["anthropologist"] = response.content
    total_tokens["input"] += response.input_tokens
    total_tokens["output"] += response.output_tokens
    with open(save_path / f"{agent_name}.json", "w") as f:
        json.dump(annotations, f, indent=4, ensure_ascii=False)

    tok_counter = {
        "agent": agent_name,
        "input_tokens": total_tokens["input"],
        "output_tokens": total_tokens["output"],
    }
    return annotations, tok_counter


def main(exp_path: Path | str, error_tracker: ErrorTracker):
    print("----------------------------------------")
    print(f"Starting analysis for experiment at {exp_path}")
    exp_path = Path(exp_path)
    assert exp_path.exists(), f"Experiment path {exp_path} does not exist."

    agent_files = glob.glob(str(exp_path / "agent_logs" / "being*.jsonl"))
    agent_names = [
        os.path.splitext(os.path.basename(file_path))[0] for file_path in agent_files
    ]

    save_path = exp_path / "annotations" / LLM_MODEL
    os.makedirs(save_path, exist_ok=True)

    worker = partial(
        annotate_and_audit,
        exp_path=exp_path,
        save_path=save_path,
    )

    annotations = {}
    max_workers = 4
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {
            ex.submit(worker, agent_name): agent_name for agent_name in agent_names
        }
        for fut in tqdm(
            as_completed(future_map), total=len(future_map), desc="Processing agents"
        ):
            agent = future_map[fut]
            try:
                results = fut.result()
            except Exception as e:
                error_tracker.add_error(
                    agent,
                    e,
                    error_type="error",
                    additional_info={"experiment": str(exp_path)},
                )
                continue

            annotations[agent] = results[0]  # type: ignore
            tokens = results[1]  # type: ignore
            try:
                if tokens is not None:
                    with open(save_path / "token_usage.jsonl", "a") as f:
                        json.dump(tokens, f)
                        f.write("\n")
            except:
                print(f"Cannot save token count for {agent}")
                print(tokens)

    try:
        anthropologist_notes = json.load(open(save_path / "anthropologist_notes.json"))
    except Exception:
        anthropologist_notes = {}

    for agent in annotations:
        note = annotations[agent].get("anthropologist", None)
        if note is not None:
            anthropologist_notes[agent] = note
    with open(save_path / "anthropologist_notes.json", "w") as f:
        json.dump(anthropologist_notes, f, indent=4, ensure_ascii=False)
    print("----------------------------------------")


if __name__ == "__main__":
    main_tracker = ErrorTracker(show_stacktraces=SHOW_STACKTRACES)

    for exp in EXPERIMENTS_NAMES:
        try:
            main(exp_path=ROOT / "logs" / exp, error_tracker=main_tracker)
        except Exception as e:
            main_tracker.add_experiment_failure(exp, e)
        print()

    main_tracker.print_summary()
    main_tracker.save_to_file(ROOT / "analysis_summary" / "001_summary.json")
