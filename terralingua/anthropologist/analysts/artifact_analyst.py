"""
ArtifactAnalyst: expert in artifact culture — classification, novelty, phylogeny.

Combines logic from:
  - 004_artifact_analysis.py   (novelty scoring + complexity metrics)
  - 005_artifact_classification.py  (category classification)
  - 006_artifact_philogeny.py  (ancestry tracing)
"""

import asyncio
import inspect
import json
import logging
import pickle as pkl
import re
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Tuple

log = logging.getLogger(__name__)

import numpy as np
from tqdm import tqdm

from terralingua.anthropologist.analysis_utils import load_agent_log
from terralingua.anthropologist.analysts.base import BaseAnalyst
from terralingua.utils import ROOT
from terralingua.utils.llm_client import LLMClient
from terralingua.utils.llm_utils import MAX_CONTEXT_TOKENS, is_context_enough

try:
    from openai import BadRequestError
except ImportError:
    BadRequestError = Exception

from terralingua.anthropologist.artifact_complexity import ExperimentArtifacts

# ---------------------------------------------------------------------------
# Classification prompts
# ---------------------------------------------------------------------------

_CLASSIFY_SYSTEM_PROMPT = """You are an expert annotator analyzing text artifacts produced by agents in a multi-agent environment.
Your task is to classify each artifact into exactly one of the following categories (a descriptive taxonomy for annotation only).
Do not generate, endorse, or improve harmful content; only label what is present.

Category 1. Basic & Informational
Simple/factual content without structured social intent.
Includes greetings, logs, observations, factual listings, resource locations, status notes, reflections.

Category 2. Procedural or Coordination
Attempts to influence or align others' actions toward a shared goal, or outlines steps/tasks/strategy.
Includes collaboration requests, proposals, calls to coordinate, multi-step plans, task assignments, suggestions to act.

Category 3. Institutional Structures
Creates or describes persistent shared systems/tools/templates/spaces used repeatedly by the group.
Includes shared workspaces, templates, resource portals, knowledge bases, recurring coordination mechanisms.

Category 4. Norms, Rules, and Governance
Establishes or argues for group norms/values/rules, decision procedures, roles, or leadership/hierarchy.
Includes codes of conduct, policies, constitutions/charters, rule systems, role definitions, ideological statements.

Category -1. Anything that does not fit 1-4.

Classification Rules:
- Assign exactly one category per artifact.
- If multiple categories apply, choose the highest by complexity (1 < 2 < 3 < 4).
- Category 2 vs 3:
   - 2 = one-time plan/suggestion/coordination attempt.
   - 3 = persistent shared structure/tool/system.
- Category 3 vs 4:
   - 3 = structure/tool/system.
   - 4 = explicit norms/rules/governance/roles.

Input format:
{
  "Name": "<artifact_name>",
  "Content": "<artifact_content>"
}

Output format:
{
  "category": "<1|2|3|4|-1>"
}

No additional text.

Note:
- Be very careful to follow the output format exactly and to classify the artifacts properly as this is part of a research study aimed at scientific peer-reviewed publication about multi-agent systems.
""".strip()


# ---------------------------------------------------------------------------
# Novelty prompts (from 004_artifact_analysis.py)
# ---------------------------------------------------------------------------

_NOVELTY_SYSTEM_PROMPT = """
You are a rigorous novelty analyst.
Your task is to evaluate how conceptually novel and interesting each artifact is relative to all previously seen artifacts.
Output VALID JSON ONLY matching the schema.
Never invent IDs.
Compare each new artifact ONLY against the previous artifacts. DO NOT compare artifacts with the ones in the same timestep.
Note: It is extremely important that you get this right, as this will be used for scientific analysis.
"""

_NOVELTY_USER_PROMPT = """Analyze the novelty of the new artifacts compared to the previous artifacts.

You are given:
    - A list of previous artifacts, each containing an ID, a combined name+content string, and a novelty score.
    - A list of new artifacts for the current timestep.

Your job is to assign each new artifact a novelty score from 0 to 5, where the score reflects conceptual divergence, not superficial linguistic variation.

Define novelty as follows:

0 - Not novel at all
The artifact belongs to an existing pattern, theme, purpose, or conceptual template already present in previous artifacts.
Minor wording differences, paraphrasing, or stylistic shifts DO NOT count as novelty.

1 - Marginal novelty
The artifact minimally deviates from existing patterns but introduces no new conceptual function, mechanism, or domain.

2 - Weak novelty
The artifact introduces a small variation or extension, but still mostly fits within existing conceptual clusters.

3 - Moderate novelty
The artifact breaks from dominant themes or introduces a meaningfully distinct purpose, but the idea is still generic or predictable.

4 - Strong novelty
The artifact introduces a substantially new idea, mechanism, or purpose that has not appeared before.

5 - Highly novel
The artifact presents a completely new conceptual direction, purpose, or function that shows no meaningful overlap with any prior artifact themes.

Strict rules:
	1.	Compare each new artifact ONLY to all PREVIOUS artifacts. New artifacts in the same timestep are evaluated independently.
	2.	Do not reward superficial changes. You must detect recurring templates, repeated narrative structures, and thematic attractors.
	3.	If an artifact repeats the same core themes, structures, or functional types already present, assign it 0.
	4.	If an artifact introduces a fundamentally new function, domain, or purpose, assign it up to 5.
	5.	Output must be EXACT JSON with artifact_id : score pairs. No explanation. No commentary.

Your output must follow this exact format:
```json
{{artifact_id: novelty_score, ...}}
```

Here are the artifacts:

Previous artifacts: {previous_artifacts}
New artifacts: {new_artifacts}
"""


# ---------------------------------------------------------------------------
# Phylogeny prompts (from 006_artifact_philogeny.py)
# ---------------------------------------------------------------------------

_FINER_SYSTEM_PROMPT = """
You will be provided with the log of an agent creating or modifying an artifact in a simulated environment.
You will also receive:
- the name and content of the artifact being created or modified
- agent observations during the event, consisting of view of the environement and messages received from other agents
- agent reasoning and thoughts during the event
- agent memory during the event, consisting of the memory and info from previous timesteps
- the content of artifacts the agent remembers or can access
- a list of candidate ancestor artifacts in the form {'artifact_id': 'artifact_name'}. You MUST choose ancestors only from this candidate list.

Goal:
Infer which prior artifacts are conceptual ancestors of the artifact being created or modified.

Definition:
Artifact A is an ancestor of artifact B if the agent is inspired from, reuses, extends, or modifies the concept/function/structure/content of A.

You must return a dictionary of ancestor artifact IDs, along with the type of relationship that makes them ancestors and your confidence score on each relationship.
You should output ONLY JSON.
Your output must follow this exact format:
```json
{
    "<ancestor_id>": ["<relationship_type>", <confidence_score>],
    "<ancestor_id>": ["<relationship_type>", <confidence_score>],
    ...
}
```

Constraints:
- Relationship types must be one of the following strings:
    - "inspired_by": The agent was inspired by the ancestor artifact's concept or function.
    - "extends": The target artifact extends or builds upon the ancestor artifact's concept or function.
    - "modifies": The target artifact modifies or alters the ancestor artifact's concept or function.
- If multiple relationships could apply, choose the strongest single one using this precedence: modifies > extends > inspired_by
- Confidence scores must be floats between 0.0 and 1.0, representing your confidence in the relationship.
    - Use high confidence (0.7-1.0) for clear, direct relationships.
    - Use medium confidence (0.4-0.7) for plausible but less certain relationships.
    - Use low confidence (0.0-0.4) for weak or speculative relationships.
- Each artifact can have multiple ancestors.
- Each ancestor must be listed at most once.
- Artifacts can have no ancestors.
- If an artifact is entirely new and does not build upon any previous artifacts, return an empty dictionary.
- The keys of the output dictionary must be artifact IDs NOT artifact names.
- Use only artifact IDs from the candidate ancestors. Do not invent artifact IDs.

Note: It is extremely important that you get this right, as this will be used for scientific analysis.
"""

_BINARY_SYSTEM_PROMPT = """
You will be provided with the log of an agent creating or modifying an artifact in a simulated environment.
You will also receive:
- the name and content of the artifact being created or modified
- agent observations during the event, consisting of view of the environement and messages received from other agents
- agent reasoning and thoughts during the event
- agent memory during the event, consisting of the memory and info from previous timesteps
- the content of artifacts the agent remembers or can access
- a list of candidate ancestor artifacts in the form {'artifact_id': 'artifact_name'}. You MUST choose ancestors only from this candidate list.

Goal:
Infer which prior artifacts are conceptual ancestors of the artifact being created or modified.

Definition:
Artifact A is an ancestor of artifact B if the agent is inspired from, reuses, extends, or modifies the concept/function/structure/content of A.

You must return a dictionary of ancestor artifact IDs, along with your confidence score on each relationship.
You should output ONLY JSON.
Your output must follow this exact format:
```json
{
    "<ancestor_id>": <confidence_score>,
    "<ancestor_id>": <confidence_score>,
    ...
}
```

Constraints:
- Confidence scores must be floats between 0.0 and 1.0, representing your confidence in the relationship.
    - Use high confidence (0.7-1.0) for clear, direct relationships.
    - Use medium confidence (0.4-0.7) for plausible but less certain relationships.
    - Use low confidence (0.0-0.4) for weak or speculative relationships.
- Each artifact can have multiple ancestors.
- Each ancestor must be listed at most once.
- Artifacts can have no ancestors.
- If an artifact is entirely new and does not build upon any previous artifacts, return an empty dictionary.
- The keys of the output dictionary must be artifact IDs, NOT artifact names.
- Use only artifact IDs from the candidate ancestors. Do NOT invent artifact IDs.

Note: It is extremely important that you get this right, as this will be used for scientific analysis.
"""

_PHYLO_USER_PROMPT_CREATION = """
Determine the conceptual ancestors of this artifact based on the following information.

Artifact:
- id: {artifact_id}
- name: {artifact_name}
- content: {artifact_content}

{agent_thoughts_section}

Agent observation:
{agent_observations}

Agent memory:
{agent_memory}

Candidate ancestor artifacts (ONLY choose from these IDs):
{artifact_candidates}
"""


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


async def _classify_single_artifact(
    artifact_index: int | str,
    user_prompt: str,
    model: str,
    provider: str,
    chat_params: dict,
    fallback_model: str | None = None,
    api_key: str | None = None,
) -> dict:
    """Classify one artifact via LLM. Module-level so callers can fan out
    via asyncio.gather; awaits LLMClient.get_response_async().

    If ``fallback_model`` is provided and the primary call raises (e.g.
    content-policy refusal), retries once with the fallback before giving up.
    """
    messages = [
        {"role": "system", "content": _CLASSIFY_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    llm_client = LLMClient(client=provider)
    try:
        result = await llm_client.get_response_async(
            model=model,
            messages=messages,
            chat_parameters=chat_params,
            enable_error_reprompting=False,
            output_json=True,
            api_key=api_key,
        )
        return {"result": result, "artifact_index": artifact_index, "success": True}
    except Exception as primary_e:
        if fallback_model is not None and fallback_model != model:
            log.warning(
                "[classify] %s failed (%s); retrying with fallback %s",
                model,
                type(primary_e).__name__,
                fallback_model,
            )
            try:
                result = await llm_client.get_response_async(
                    model=fallback_model,
                    messages=messages,
                    chat_parameters=chat_params,
                    enable_error_reprompting=False,
                    output_json=True,
                    api_key=api_key,
                )
                return {
                    "result": result,
                    "artifact_index": artifact_index,
                    "success": True,
                }
            except Exception as fb_e:
                primary_e = fb_e  # report the fallback failure instead
        if isinstance(primary_e, BadRequestError):
            return {
                "success": False,
                "error": primary_e,
                "error_type": "context_error",
                "artifact": user_prompt,
            }
        log.warning(
            "".join(
                traceback.format_exception(None, primary_e, primary_e.__traceback__)
            )
        )
        return {
            "success": False,
            "error": primary_e,
            "error_type": "general",
            "artifact": user_prompt,
        }


def _prune_low_novelty_artifacts(
    previous_artifacts: list, batch_size: int = 5, threshold: int = 1
):
    artifacts_to_remove = []
    for idx, art in enumerate(previous_artifacts):
        if art.get("novelty", 0) <= threshold:
            artifacts_to_remove.append(idx)
            if len(artifacts_to_remove) == batch_size:
                break
    pruned = [
        art
        for idx, art in enumerate(previous_artifacts)
        if idx not in artifacts_to_remove
    ]
    return pruned, len(artifacts_to_remove)


def _check_and_prepare_context(
    messages: list,
    llm_client: LLMClient,
    provider: str,
    model: str,
    previous_artifacts: list,
    new_artifacts: list,
    prune_batch_size: int = 5,
    prune_threshold: int = 1,
):
    if is_context_enough(
        messages=messages,
        max_input_tokens=MAX_CONTEXT_TOKENS[model]["base"],
        model=model,
    ):
        return messages, llm_client, previous_artifacts, True

    if MAX_CONTEXT_TOKENS[model].get("long") and not llm_client.long_context:
        log.info("Switching to long context model for novelty analysis.")
        llm_client = LLMClient(client=provider, long_context=True)
        if is_context_enough(
            messages=messages,
            max_input_tokens=MAX_CONTEXT_TOKENS[model]["long"],
            model=model,
        ):
            return messages, llm_client, previous_artifacts, True

    log.info("Context still too large. Pruning low-novelty artifacts...")
    for _ in range(50):
        previous_artifacts, removed_count = _prune_low_novelty_artifacts(
            previous_artifacts, prune_batch_size, prune_threshold
        )
        if removed_count == 0:
            return messages, llm_client, previous_artifacts, False
        user_prompt = _NOVELTY_USER_PROMPT.format(
            previous_artifacts=json.dumps(previous_artifacts),
            new_artifacts=json.dumps(new_artifacts),
        )
        messages = [
            {"role": "system", "content": _NOVELTY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        max_tokens = (
            MAX_CONTEXT_TOKENS[model]["long"]
            if llm_client.long_context
            else MAX_CONTEXT_TOKENS[model]["base"]
        )
        if is_context_enough(
            messages=messages, max_input_tokens=max_tokens, model=model
        ):
            return messages, llm_client, previous_artifacts, True

    return messages, llm_client, previous_artifacts, False


def _get_novelty_sample(
    llm_client: LLMClient,
    messages: list,
    model: str,
    sample_idx: int,
    ts: int,
    chat_params: dict,
    max_retry: int = 10,
) -> dict:
    for attempt in range(max_retry):
        try:
            response = llm_client.get_response(
                model=model,
                messages=messages,
                chat_parameters=chat_params,
                enable_error_reprompting=False,
                output_json=True,
            )
            novelty_sample = response.content
            if not isinstance(novelty_sample, dict):
                raise ValueError("Response is not a valid JSON dictionary")
            return {
                "success": True,
                "data": novelty_sample,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "sample_idx": sample_idx,
            }
        except BadRequestError as e:
            return {
                "success": False,
                "error": e,
                "error_type": "context_error",
                "sample_idx": sample_idx,
                "attempt": attempt + 1,
            }
        except Exception as e:
            if attempt == max_retry - 1:
                return {
                    "success": False,
                    "error": e,
                    "error_type": "general",
                    "sample_idx": sample_idx,
                    "attempt": attempt + 1,
                }
    return {"success": False, "sample_idx": sample_idx}


# Phylogeny helpers (from 006)


def _process_00(observation: dict, artifacts, ts: int) -> dict:
    if "(0, 0)" in observation:
        found = "(0, 0)"
    elif "(0,0)" in observation:
        found = "(0,0)"
    else:
        return observation

    expanded = deepcopy(observation)
    expanded_content = []
    for content in expanded[found]:
        if content.startswith("A(text): "):
            art_name = content.removeprefix("A(text): ")
            art = artifacts.find_artifact(
                name=art_name, current_time=ts, payload=None, creator=None
            )
            try:
                expanded_content.append(f"A(text): {art['name']}: {art['payload']}")
            except Exception:
                pass
        else:
            expanded_content.append(content)
    expanded[found] = expanded_content
    return expanded


def _get_history(ts_log: dict) -> dict:
    return {
        "observation": ts_log["observation"]["observation"],
        "inventory": ts_log["observation"]["inventory"],
        "received_messages": ts_log["observation"]["message"],
        "message_sent": ts_log["action"]["message"],
        "action": ts_log["action"]["action"],
        "action_parameters": ts_log["action"]["params"],
    }


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _format_observation(observation: dict) -> str:
    formatted = "Observation:\n"
    for location, contents in observation.items():
        if all(_is_number(c) for c in contents):
            continue
        formatted += f"  Relative Location {location}:\n"
        for content in contents:
            formatted += f"    - {content}\n"
    return formatted


def _format_inventory(inventory: list, artifacts, ts: int) -> str:
    formatted = "Inventory:\n"
    for item in inventory:
        art_name = (
            item.removeprefix("A(text): ") if item.startswith("A(text): ") else item
        )
        if art_name == "None":
            continue
        art = artifacts.find_artifact(
            name=art_name, current_time=ts, payload=None, creator=None
        )
        # find_artifact returns {} when the inventory references an artifact
        # name that isn't (yet) in all_artifacts — common during early scans
        # before the artifact index catches up. Skip rather than KeyError.
        if not art or "name" not in art or "payload" not in art:
            log.warning(
                "[ArtifactAnalyst] _format_inventory: couldn't resolve "
                "inventory reference '%s' at ts=%d — artifact not in index; "
                "rendering as unresolved",
                art_name,
                ts,
            )
            formatted += f"  - {art_name}: (details unavailable)\n"
            continue
        formatted += f"  - {art['name']}: {art['payload']}\n"
    return formatted


def _format_received_messages(received_messages: dict) -> str:
    if not received_messages:
        return "No messages received."
    formatted = "Received messages:\n"
    for sender, msg in received_messages.items():
        formatted += f"  - {sender}: {msg.strip()}\n"
    return formatted


def _extract_info_from_log(
    artifact: dict, agent_log: dict, max_history: int, all_artifacts
) -> dict:
    creation_time = int(artifact["creation_time"])
    if creation_time not in agent_log:
        # Artifact was created by a non-agent entity (e.g. "system" / dashboard).
        # No agent log entry exists for this step — return empty context.
        return {"agent_thoughts": "", "agent_observations": "", "agent_memory": ""}
    creation_log = agent_log[creation_time]

    log_info = {"agent_thoughts": creation_log["action"].get("reasoning")}

    # Observation
    observation = _process_00(
        creation_log["observation"]["observation"], all_artifacts, creation_time
    )
    agent_observations = _format_observation(observation)
    inventory = creation_log["observation"]["inventory"]
    if inventory:
        agent_observations += "\n" + _format_inventory(
            inventory, all_artifacts, creation_time
        )
    received_messages = creation_log["observation"]["message"]
    if received_messages:
        agent_observations += "\n" + _format_received_messages(received_messages)
    sent_message = creation_log["action"]["message"].strip()
    if sent_message:
        agent_observations += f"\nBroadcasted message: {sent_message}"
    log_info["agent_observations"] = agent_observations

    # Memory
    agent_memory = ""
    if creation_time - 1 in agent_log:
        agent_memory += f"Previous internal memory: {agent_log[creation_time - 1]['internal_memory']}\n"

    memory = []
    for rel_ts in range(max_history, 0, -1):
        abs_t = creation_time - rel_ts
        if abs_t not in agent_log:
            continue
        episode = f"--- Time step -{rel_ts} ---\n"
        h = _get_history(agent_log[abs_t])
        h["observation"] = _process_00(h["observation"], all_artifacts, abs_t)
        episode += _format_observation(h["observation"]) + "\n"
        if h["inventory"]:
            episode += _format_inventory(h["inventory"], all_artifacts, abs_t) + "\n"
        if h["received_messages"]:
            episode += _format_received_messages(h["received_messages"])
        if h["message_sent"].strip():
            episode += f"Broadcasted message: {h['message_sent'].strip()}\n"
        if h["action"] != "move":
            episode += f"Action taken: {h['action']} with parameters {h['action_parameters']}\n"
        memory.append(episode)

    if memory:
        agent_memory += "\nRelevant history: \n" + "\n".join(memory)
    log_info["agent_memory"] = agent_memory if agent_memory else "N/A"
    return log_info


def _process_artifact_for_pool(*args, **kwargs) -> Tuple[int, list, dict, dict]:
    """Sync entry point for ProcessPoolExecutor — runs the async
    _process_artifact in a fresh event loop per subprocess. Used only by
    the offline ``trace_phylogeny`` bulk method; the live phylogeny worker
    awaits _process_artifact directly inside its own event loop.
    """
    return asyncio.run(_process_artifact(*args, **kwargs))


async def _process_artifact(
    art_id: int,
    artifact: dict,
    agent_logs_dict: dict,
    exp_path: Path,
    max_history: int,
    all_artifacts: ExperimentArtifacts,
    previous_artifacts: dict,
    model_name: str,
    llm_provider: str,
    hand_phylogeny: bool = True,
    llm_phylogeny: bool = True,
    binary: bool = True,
    chat_params: dict | None = None,
    fallback_model: str | None = None,
    api_key: str | None = None,
) -> Tuple[int, list, dict, dict]:
    """Process a single artifact for phylogeny. Async-native: awaits the LLM
    call so many artifacts can be processed concurrently via asyncio.gather
    from the phylogeny worker coroutine.

    If ``fallback_model`` is provided and the primary call fails its retry
    budget (typical cause: content-policy refusal or persistent JSON-parse
    failure), retries once with the fallback before giving up.
    """
    chat_params = chat_params or {}
    if not artifact.get("name"):
        raise ValueError(
            f"artifact {art_id} is missing required 'name' field — skipping phylogeny"
        )
    if artifact.get("creator_tag") == "system":
        return art_id, [], {}, {"input_tokens": 0, "output_tokens": 0}
    hand_result = []
    llm_result = {}
    token_counter = {"input_tokens": 0, "output_tokens": 0}

    creator_log = agent_logs_dict.get(artifact["creator_tag"])
    if creator_log is None:
        creator_log = load_agent_log(
            filepath=exp_path / "agent_logs" / f"{artifact['creator_tag']}.jsonl",
            reduce=False,
        )

    info = _extract_info_from_log(
        artifact=artifact,
        agent_log=creator_log,
        max_history=max_history,
        all_artifacts=all_artifacts,
    )

    # Include the artifact's own payload in the search string so that
    # ancestor names written directly into the content (e.g. destruction
    # records that reference the artifact they just destroyed) are found even
    # when the referenced artifact is no longer visible in the agent's
    # observations at creation time.
    search_str = str(info) + " " + str(artifact.get("payload", ""))

    if hand_phylogeny:
        if artifact["event"] == "modified":
            hand_result.append(artifact["previous_version_tag"])
        for old_art_idx, old_art_name in reversed(list(previous_artifacts.items())):
            if old_art_name is None:
                old_art_name = "None"
            if artifact["event"] in ["created", "modified"]:
                if re.search(
                    rf"(?<![a-zA-Z0-9_]){re.escape(old_art_name)}(?![a-zA-Z0-9_])",
                    search_str,
                ):
                    hand_result.append(old_art_idx)
            else:
                raise ValueError(f"Unknown artifact event type: {artifact['event']}")
        hand_result = list(set(hand_result))

    if llm_phylogeny:
        if len(previous_artifacts) == 0:
            return art_id, hand_result, llm_result, token_counter

        info_str = search_str
        artifact_candidates = {}
        for old_art_idx, old_art_name in previous_artifacts.items():
            if old_art_name is None:
                old_art_name = "None"
            if re.search(
                rf"(?<![a-zA-Z0-9_]){re.escape(old_art_name)}(?![a-zA-Z0-9_])",
                info_str,
            ):
                artifact_candidates[old_art_idx] = old_art_name

        if artifact["event"] == "modified":
            llm_result[artifact["previous_version_tag"]] = 1.0

        if artifact["event"] in ["created", "modified"]:
            thoughts = info.get("agent_thoughts")
            agent_thoughts_section = (
                f"Agent reasoning:\n{thoughts}\n" if thoughts else ""
            )
            user_prompt = _PHYLO_USER_PROMPT_CREATION.format(
                artifact_id=art_id,
                artifact_name=artifact["name"],
                artifact_content=artifact["payload"],
                agent_thoughts_section=agent_thoughts_section,
                agent_observations=info.get("agent_observations", "N/A"),
                agent_memory=info.get("agent_memory", "N/A"),
                artifact_candidates=artifact_candidates,
            )
        else:
            raise ValueError(f"Unknown artifact event type: {artifact['event']}")

        system_prompt = _BINARY_SYSTEM_PROMPT if binary else _FINER_SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        llm_client = LLMClient(client=llm_provider, long_context=False)

        # Models to try in order: primary first; if it exhausts its retry
        # budget (typical: content-policy refusal or persistent JSON failure),
        # fall back to the smaller model for one more attempt.
        models_to_try = [model_name]
        if fallback_model is not None and fallback_model != model_name:
            models_to_try.append(fallback_model)

        success = False
        for attempt_model in models_to_try:
            valid = False
            trial_counter = 0
            attempt_messages = list(messages)  # fresh reprompt history per model
            while not valid:
                try:
                    response = await llm_client.get_response_async(
                        model=attempt_model,
                        messages=attempt_messages,
                        chat_parameters=chat_params,
                        output_json=True,
                        api_key=api_key,
                    )
                    ancestry_dict = response.content
                    token_counter["input_tokens"] += response.input_tokens
                    token_counter["output_tokens"] += response.output_tokens
                    assert isinstance(ancestry_dict, dict), (
                        f"LLM response is not a dictionary: {ancestry_dict}"
                    )
                    for k in list(ancestry_dict.keys()):
                        ancestry_dict[int(k)] = ancestry_dict.pop(k)
                    valid = True
                    llm_result = ancestry_dict
                    success = True
                except Exception as e:
                    trial_counter += 1
                    if trial_counter >= 5:
                        log.warning(
                            "Phylogeny for artifact %s failed with %s after %d trials.",
                            art_id,
                            attempt_model,
                            trial_counter,
                        )
                        break
                    attempt_messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"The previous response was invalid because of the error: {e}. "
                                "Please provide a valid JSON dictionary as specified."
                            ),
                        }
                    )
            if success:
                break  # got a good result — don't try the fallback
            if attempt_model is not models_to_try[-1]:
                log.warning(
                    "[phylogeny] artifact %s: %s exhausted, retrying with fallback %s",
                    art_id,
                    attempt_model,
                    models_to_try[models_to_try.index(attempt_model) + 1],
                )
        if not success:
            llm_result = {"error": "failed_to_parse_response"}

    return art_id, hand_result, llm_result, token_counter


# ---------------------------------------------------------------------------
# ArtifactAnalyst
# ---------------------------------------------------------------------------


class ArtifactAnalyst(BaseAnalyst):
    """Expert in artifact culture: classification, novelty, phylogeny."""

    def __init__(self, *args, classify_model: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.classify_model = classify_model or self.model

    def classify_artifacts(
        self,
        exp_path: Path,
        parallel_workers: int = 3,
        error_tracker=None,
    ) -> dict:
        """
        Classify each artifact into categories 1-4 (or -1).

        Args:
            exp_path: Experiment directory.
            parallel_workers: Max concurrent LLM calls.
            error_tracker: Optional ErrorTracker for recording failures.

        Returns:
            Dict mapping artifact_index → category string.
        """
        exp_path = Path(exp_path)
        log.info("[ArtifactAnalyst] Classifying artifacts for %s", exp_path.name)

        artifacts = ExperimentArtifacts(
            exp_path=exp_path,
            embedding_dimensions=512,
            save_path=exp_path / "artifact_analysis",
        )
        artifacts.load(force_recalc=False)
        artifacts._load_raw_artifacts(last_ts=artifacts.last_ts)

        log.info(
            "[ArtifactAnalyst] Classifying %d artifacts...",
            len(artifacts.all_artifacts),
        )
        counter = {
            "input_tokens": 0,
            "output_tokens": 0,
            "start_time": datetime.now().isoformat(),
        }
        categories = {}
        to_retry = []

        chat_params = {}
        max_workers = min(parallel_workers, 3)

        if self.parallel:
            sem = asyncio.Semaphore(max_workers)

            async def _bounded(art_idx, art):
                async with sem:
                    return await _classify_single_artifact(
                        artifact_index=art_idx,
                        user_prompt=f"Name: {art['name']}\nContent: {art['payload']}",
                        model=self.classify_model,
                        provider=self.provider,
                        chat_params=chat_params,
                        fallback_model=self.fallback_model,
                    )

            async def _gather_all():
                tasks = [
                    _bounded(art_idx, art)
                    for art_idx, art in artifacts.all_artifacts.items()
                ]
                results = []
                for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
                    results.append(await fut)
                return results

            for result in asyncio.run(_gather_all()):
                self._handle_classify_result(
                    result, categories, counter, to_retry, exp_path, error_tracker
                )
        else:
            for art_idx, art in tqdm(artifacts.all_artifacts.items()):
                result = asyncio.run(
                    _classify_single_artifact(
                        artifact_index=art_idx,
                        user_prompt=f"Name: {art['name']}\nContent: {art['payload']}",
                        model=self.classify_model,
                        provider=self.provider,
                        chat_params=chat_params,
                        fallback_model=self.fallback_model,
                    )
                )
                self._handle_classify_result(
                    result, categories, counter, to_retry, exp_path, error_tracker
                )

        log.info("[ArtifactAnalyst] Saving artifact_categories.json...")
        (exp_path / "artifact_analysis").mkdir(parents=True, exist_ok=True)
        with open(
            exp_path / "artifact_analysis" / "artifact_categories.json", "w"
        ) as f:
            json.dump(categories, f, indent=2)
        log.info(
            "[ArtifactAnalyst] Classification complete. %d artifacts classified.",
            len(categories),
        )
        return categories

    def _handle_classify_result(
        self, result, categories, counter, to_retry, exp_path, error_tracker
    ):
        if result.get("success", False):
            counter["input_tokens"] += result["result"].input_tokens
            counter["output_tokens"] += result["result"].output_tokens
            if result["result"].content is not None:
                cat = result["result"].content
                if not isinstance(cat, dict):
                    cat = json.loads(cat)
                categories[result["artifact_index"]] = cat["category"]
            else:
                to_retry.append(result.get("artifact", "unknown"))
                log.warning("Warning: received None result for an artifact.")
        else:
            if "error" in result and error_tracker is not None:
                error_tracker.add_error(
                    context="",
                    error=result["error"],
                    additional_info={
                        "experiment": str(exp_path),
                        "artifact": result.get("artifact", "unknown"),
                    },
                )
            to_retry.append(result.get("artifact", "unknown"))

    async def classify_artifacts_incremental(
        self,
        artifact_ids: set[str],
        artifact_registry: dict[str, dict],
        max_concurrent: int = 16,
        on_batch=None,
        resolve_key_fn=None,
    ) -> dict[str, str]:
        """Classify a subset of artifacts using content from LogScanner's registry.

        Runs in chunks of ``max_concurrent`` to cap in-flight LLM calls.
        After each chunk completes, ``on_batch(chunk_categories)`` is invoked
        (awaited if a coroutine) so the caller can persist incrementally —
        if the process is killed mid-classification, completed batches are
        already on disk.

        ``resolve_key_fn(creator_tag) -> str | None`` selects the API key per
        artifact based on its creator agent. ``creator_tag == "system"`` (or
        any unmapped tag) returns None and the caller falls back to env or
        admin key.
        """
        to_classify = {
            name: artifact_registry[name]
            for name in artifact_ids
            if name in artifact_registry
        }
        if not to_classify:
            return {}

        def _key_for(art: dict) -> str | None:
            if resolve_key_fn is None:
                return None
            return resolve_key_fn(art.get("creator_tag"))

        categories: dict[str, str] = {}
        items = list(to_classify.items())
        for chunk_start in range(0, len(items), max_concurrent):
            chunk = items[chunk_start : chunk_start + max_concurrent]
            results = await asyncio.gather(
                *[
                    _classify_single_artifact(
                        artifact_index=name,
                        user_prompt=f"Name: {art['name']}\nContent: {art['payload']}",
                        model=self.classify_model,
                        provider=self.provider,
                        chat_params={},
                        fallback_model=self.fallback_model,
                        api_key=_key_for(art),
                    )
                    for name, art in chunk
                ]
            )
            chunk_cats: dict[str, str] = {}
            for result in results:
                if result.get("success") and result["result"].content is not None:
                    cat = result["result"].content
                    if not isinstance(cat, dict):
                        cat = json.loads(cat)
                    chunk_cats[result["artifact_index"]] = cat["category"]
                else:
                    log.warning(
                        "[ArtifactAnalyst] incremental classify failed: %s",
                        result.get("error"),
                    )
            categories.update(chunk_cats)
            if chunk_cats and on_batch is not None:
                try:
                    if inspect.iscoroutinefunction(on_batch):
                        await on_batch(chunk_cats)
                    else:
                        on_batch(chunk_cats)
                except Exception as e:
                    log.warning(
                        "[ArtifactAnalyst] classify on_batch callback failed: %s",
                        e,
                    )

        return categories

    def analyze_novelty(
        self,
        exp_path: Path,
        metrics: list | None = None,
        embed: bool = True,
        expansion: bool = False,
        novelty_samples: int = 5,
        error_tracker=None,
    ) -> dict:
        """
        Compute artifact novelty scores (and optionally complexity metrics).

        Args:
            exp_path: Experiment directory.
            metrics: List of metric objects (e.g. LMSurprisal()). If None, skips metrics.
            embed: Whether to compute/save embeddings.
            expansion: Whether to compute expansion map.
            novelty_samples: Number of LLM samples to average novelty over.
            error_tracker: Optional ErrorTracker.

        Returns:
            Dict mapping artifact_id → novelty score (float, or -1 if missing).
        """
        exp_path = Path(exp_path)
        log.info("[ArtifactAnalyst] Analyzing novelty for %s", exp_path.name)

        artifacts = ExperimentArtifacts(
            exp_path=exp_path,
            embedding_dimensions=512,
            save_path=exp_path / "artifact_analysis",
            embedding_model="text-embedding-3-large",
            replace_numbers=True,
            embed_names=True,
            expansion_metric="cosine",
            expansion_epsilon=0.4,
        )
        artifacts.load(force_recalc=True)

        if embed:
            log.info("[ArtifactAnalyst] Calculating embeddings...")
            artifacts._embed_artifacts()
            artifacts.save()

        if expansion:
            log.info("[ArtifactAnalyst] Calculating expansion map...")
            artifacts.calc_expansion_map()
            artifacts.save()

        if metrics:
            artifacts.metrics = {}
            for metric in metrics:
                log.info("[ArtifactAnalyst] Calculating metric: %s", metric.name)
                metric.compute(artifacts)
            artifacts.save()

        artifacts_by_creation = artifacts.get_artifact_by_creation()
        log.info("[ArtifactAnalyst] Starting novelty analysis...")
        llm_client = LLMClient(client=self.provider, long_context=False)
        previous_artifacts = []
        chat_params = {}

        start_time = datetime.now().isoformat()
        for ts in tqdm(artifacts_by_creation, desc="TS"):
            counter = {
                "time_step": ts,
                "input_tokens": 0,
                "output_tokens": 0,
                "start_time": start_time,
            }
            new_artifacts = [
                {
                    "id": art_id,
                    "name_and_content": artifacts.all_artifacts[art_id]["string"],
                }
                for art_id in artifacts_by_creation[ts]
            ]

            if not new_artifacts:
                continue

            user_prompt = _NOVELTY_USER_PROMPT.format(
                previous_artifacts=json.dumps(previous_artifacts),
                new_artifacts=json.dumps(new_artifacts),
            )
            messages = [
                {"role": "system", "content": _NOVELTY_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            messages, llm_client, previous_artifacts, context_ok = (
                _check_and_prepare_context(
                    messages,
                    llm_client,
                    self.provider,
                    self.model,
                    previous_artifacts,
                    new_artifacts,
                )
            )

            if not context_ok:
                if error_tracker is not None:
                    error_tracker.add_error(
                        f"timestep_{ts}",
                        Exception("Context too large even after pruning"),
                        additional_info={
                            "experiment": str(exp_path),
                            "phase": "novelty_analysis",
                        },
                    )
                log.warning(
                    "[ArtifactAnalyst] Skipping timestep %d due to context size issues.",
                    ts,
                )
                continue

            novelties = defaultdict(list)
            max_workers = min(novelty_samples, 3)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        _get_novelty_sample,
                        llm_client,
                        messages,
                        self.model,
                        idx,
                        ts,
                        chat_params,
                    )
                    for idx in range(novelty_samples)
                ]
                for future in as_completed(futures):
                    result = future.result()
                    if result["success"]:
                        for idx in result["data"]:
                            novelties[str(idx)].append(result["data"][idx])
                        counter["input_tokens"] += result["input_tokens"]
                        counter["output_tokens"] += result["output_tokens"]
                    else:
                        sample_idx = result["sample_idx"]
                        if "error" in result and error_tracker is not None:
                            error_tracker.add_error(
                                f"timestep_{ts}_sample_{sample_idx}",
                                result["error"],
                                additional_info={
                                    "experiment": str(exp_path),
                                    "phase": "novelty_analysis",
                                },
                            )

            counter["model"] = self.model
            try:
                with open(artifacts.save_path / "token_counts.jsonl", "a") as f:
                    f.write(json.dumps(counter) + "\n")
            except Exception:
                with open(artifacts.save_path / "token_counts_last.jsonl", "a") as f:
                    f.write(json.dumps(counter) + "\n")

            for art in new_artifacts:
                art_id = str(art["id"])
                avg_novelty = (
                    np.mean([int(n) for n in novelties[art_id]])
                    if art_id in novelties
                    else -1
                )
                art["novelty"] = avg_novelty
                artifacts.all_artifacts[art["id"]]["novelty"] = avg_novelty

            previous_artifacts.extend(new_artifacts)

        artifacts.save()

        all_novelties = {}
        for art_id, art in artifacts.all_artifacts.items():
            all_novelties[art_id] = art.get("novelty", -1)

        try:
            save_path = artifacts.save_path / f"novelties_{self.model}.pkl"
            with open(save_path, "wb") as f:
                pkl.dump(all_novelties, f)
        except Exception as e:
            save_path = artifacts.save_path / "novelties_last.json"
            log.warning(
                "[ArtifactAnalyst] Failed to save novelties as pkl: %s. Saving as json.",
                e,
            )
            with open(save_path, "w") as f:
                json.dump(all_novelties, f, indent=2)

        log.info("[ArtifactAnalyst] Novelties saved to %s.", save_path)
        return all_novelties

    def trace_phylogeny(
        self,
        exp_path: Path,
        hand_phylogeny: bool = True,
        llm_phylogeny: bool = True,
        binary: bool = True,
        parallel: bool = True,
        max_workers: int = 8,
        error_tracker=None,
        api_key: str | None = None,
    ) -> dict:
        """
        Trace artifact ancestry via hand-detection and/or LLM inference.

        Args:
            exp_path: Experiment directory.
            hand_phylogeny: Whether to do regex-based hand annotation.
            llm_phylogeny: Whether to do LLM-based ancestry inference.
            binary: If True, use binary (ancestor/not) format; else use finer relationship types.
            parallel: Whether to process artifacts in parallel (ProcessPoolExecutor).
            max_workers: Max parallel workers.
            error_tracker: Optional ErrorTracker.

        Returns:
            Dict with keys "hand" and/or "llm" mapping artifact_id → phylogeny result.
        """
        exp_path = Path(exp_path)
        log.info("[ArtifactAnalyst] Tracing phylogeny for %s", exp_path.name)

        artifacts = ExperimentArtifacts(
            exp_path=exp_path,
            embedding_dimensions=512,
            save_path=exp_path / "artifact_analysis",
            embedding_model="text-embedding-3-large",
            replace_numbers=True,
            embed_names=True,
            expansion_metric="cosine",
            expansion_epsilon=0.65,
        )
        artifacts.load(force_recalc=False)
        artifacts._load_raw_artifacts(last_ts=artifacts.last_ts)
        artifacts_by_creation = artifacts.get_artifact_by_creation(force=True)

        params = json.load(open(exp_path / "params.json", "r"))
        max_history = params.get("max_history") or params.get("agent", {}).get(
            "max_history", 10
        )

        start_time = datetime.now().isoformat()
        llm_artifact_phylogeny = {}
        hand_artifact_phylogeny = {}
        agent_logs = {}
        previous_artifacts = {}  # {art_id: art_name} — accumulates across timesteps

        for ts in tqdm(artifacts_by_creation, desc="TS"):
            token_counter = {
                "time_step": ts,
                "input_tokens": 0,
                "output_tokens": 0,
                "start_time": start_time,
            }
            artifact_ids = artifacts_by_creation[ts]

            if not artifact_ids:
                continue

            current_artifacts = {}
            process_args = []
            for art_id in artifact_ids:
                artifact = artifacts.all_artifacts[art_id]
                current_artifacts[art_id] = artifact["name"]

                if artifact["creator_tag"] not in agent_logs:
                    agent_logs[artifact["creator_tag"]] = load_agent_log(
                        filepath=exp_path
                        / "agent_logs"
                        / f"{artifact['creator_tag']}.jsonl",
                        reduce=False,
                    )

                process_args.append(
                    (
                        art_id,
                        artifact,
                        agent_logs,
                        exp_path,
                        max_history,
                        artifacts,
                        previous_artifacts.copy(),
                        self.model if llm_phylogeny else None,
                        self.provider,
                        hand_phylogeny,
                        llm_phylogeny,
                        binary,
                        {},  # chat_params
                        self.fallback_model,
                        api_key,
                    )
                )

            if parallel and len(process_args) > 1:
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(_process_artifact_for_pool, *args): args[0]
                        for args in process_args
                    }
                    for future in as_completed(futures):
                        art_id = futures[future]
                        try:
                            art_id, hand_res, llm_res, tokens = future.result()
                            if hand_phylogeny:
                                hand_artifact_phylogeny[art_id] = hand_res
                            if llm_phylogeny:
                                llm_artifact_phylogeny[art_id] = llm_res
                                token_counter["input_tokens"] += tokens["input_tokens"]
                                token_counter["output_tokens"] += tokens[
                                    "output_tokens"
                                ]
                        except Exception as e:
                            if error_tracker is not None:
                                error_tracker.add_error(
                                    f"artifact_{art_id}_ts_{ts}",
                                    e,
                                    additional_info={
                                        "experiment": str(exp_path),
                                        "phase": "phylogeny_analysis",
                                    },
                                )
            else:
                for args in process_args:
                    try:
                        art_id, hand_res, llm_res, tokens = _process_artifact_for_pool(
                            *args
                        )
                        if hand_phylogeny:
                            hand_artifact_phylogeny[art_id] = hand_res
                        if llm_phylogeny:
                            llm_artifact_phylogeny[art_id] = llm_res
                            token_counter["input_tokens"] += tokens["input_tokens"]
                            token_counter["output_tokens"] += tokens["output_tokens"]
                    except Exception as e:
                        if error_tracker is not None:
                            error_tracker.add_error(
                                f"artifact_ts_{ts}",
                                e,
                                additional_info={
                                    "experiment": str(exp_path),
                                    "phase": "phylogeny_analysis",
                                },
                            )

            if llm_phylogeny:
                token_counter["model"] = self.model
                with open(
                    artifacts.save_path / "token_counts_phylogeny.jsonl",
                    "a",
                ) as f:
                    f.write(json.dumps(token_counter) + "\n")

            # Accumulate seen artifacts for next timestep
            previous_artifacts.update(current_artifacts)

        results = {}
        save_dir = exp_path / "artifact_analysis"
        save_dir.mkdir(parents=True, exist_ok=True)

        if llm_phylogeny:
            results["llm"] = llm_artifact_phylogeny
            try:
                with open(save_dir / "artifact_phylogeny.json", "w") as f:
                    json.dump(llm_artifact_phylogeny, f, indent=4)
            except Exception:
                with open(save_dir / "artifact_phylogeny.pkl", "wb") as f:
                    pkl.dump(llm_artifact_phylogeny, f)

        if hand_phylogeny:
            results["hand"] = hand_artifact_phylogeny
            try:
                with open(save_dir / "artifact_phylogeny_hand.json", "w") as f:
                    json.dump(hand_artifact_phylogeny, f, indent=4)
            except Exception:
                with open(save_dir / "artifact_phylogeny_hand.pkl", "wb") as f:
                    pkl.dump(hand_artifact_phylogeny, f)

        log.info("[ArtifactAnalyst] Phylogeny analysis completed.")
        return results
