import json
import math
import os
import pickle as pkl
import re
import sys
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import tiktoken

# Single artifact metric
from artifact_complexity import ExperimentArtifacts
from tqdm import tqdm

from analysis_scripts.error_tracker import ErrorTracker
from core.utils import ROOT
from core.utils.analysis_utils import load_agent_log
from core.utils.llm_client import LLMClient

"""
This script analyzes the phylogeny of artifacts created by agents in the experiments. 
For each artifact, it tries to determine which previous artifacts are its conceptual ancestors, 
meaning that the agent was inspired by, reused, extended, or modified those previous artifacts when creating the new one.

The analysis is done in two ways:
1. Hand annotation: We look for explicit mentions of previous artifacts in the agent's reasoning, observations, and memory at the time of creation/modification. 
    This is a more conservative approach that only considers direct references to previous artifacts.
2. LLM inference: We use a language model to infer the conceptual ancestry of artifacts based on the agent's logs and the content of previous artifacts. 
    This approach can capture more subtle relationships that are not explicitly mentioned. 
"""

EXPERIMENTS_NAMES = []  # e.g. ['core_run', 'scarcity_run', ...]


# Setup
# ---------------------------
SHOW_STACKTRACES = False
BYNARY = True  # Whether to use the binary ancestor detection or the finer one
LLM_PROVIDER = "anthropic"
LLM_MODEL = "claude-haiku-4-5"
LLM_CHAT_PARAMS = {}

PARALLEL = True
MAX_PARALLEL_WORKERS = 8
HAND_PHYLOGENY = True
LLM_PHYLOGENY = True
# ---------------------------

# Prompts
# ---------------------------
# This one asks to infer the type of ancestry relationship
FINER_SYSTEM_PROMPT = """
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

# This one only asks to name the ancestors
BYNARY_SYSTEM_PROMPT = """
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


USER_PROMPT_CREATION = """
Determine the conceptual ancestors of this artifact based on the following information.

Artifact:
- id: {artifact_id}
- name: {artifact_name}
- content: {artifact_content}

Agent reasoning:
{agent_thoughts}

Agent observation:
{agent_observations}

Agent memory:
{agent_memory}

Candidate ancestor artifacts (ONLY choose from these IDs):
{artifact_candidates}
"""
# ----------------------------


# Utils
# ---------------------------
def process_00(observation: dict, artifacts: ExperimentArtifacts, ts: int) -> dict:
    """Process the observation at (0,0) to extract artifact contents.

    Args:
        observation (dict): The observation at (0,0).
        artifacts (dict): All artifacts in the experiment.

    Returns:
        dict: Mapping from artifact name to content.
    """
    if "(0, 0)" in observation.keys():
        found = "(0, 0)"
    elif "(0,0)" in observation.keys():
        found = "(0,0)"
    else:
        found = None

    if found is not None:
        expanded_observation = deepcopy(observation)
        expanded_content = []
        for content in expanded_observation[found]:
            if content.startswith("A(text): "):
                art_name = content.removeprefix("A(text): ")
                art = artifacts.find_artifact(
                    name=art_name, current_time=ts, payload=None, creator=None
                )
                try:
                    expanded_content.append(f"A(text): {art['name']}: {art['payload']}")
                except:
                    print()
            else:
                expanded_content.append(content)
        expanded_observation[found] = expanded_content
        return expanded_observation
    else:
        return observation


def get_history(ts_log: dict) -> dict:
    """Extract relevant info from a timestep log for history.

    Args:
        ts_log (dict): _description_

    Returns:
        dict: _description_
    """
    sent_message = ts_log["action"]["message"]
    observation = ts_log["observation"]["observation"]
    inventory = ts_log["observation"]["inventory"]
    received_messages = ts_log["observation"]["message"]
    action = ts_log["action"]["action"]
    action_params = ts_log["action"]["params"]

    return {
        "observation": observation,
        "inventory": inventory,
        "received_messages": received_messages,
        "message_sent": sent_message,
        "action": action,
        "action_parameters": action_params,
    }


def format_inventory(inventory: list, artifacts: ExperimentArtifacts, ts: int) -> str:
    """Format the inventory into a readable string.

    Args:
        inventory (list): The agent's inventory.
        artifacts (ExperimentArtifacts): All artifacts in the experiment.
        ts (int): Current timestep.

    Returns:
        str: Formatted inventory string.
    """
    formatted_inv = "Inventory:\n"
    for item in inventory:
        if item.startswith("A(text): "):
            art_name = item.removeprefix("A(text): ")
        else:
            art_name = item

        if art_name == "None":
            continue

        art = artifacts.find_artifact(
            name=art_name, current_time=ts, payload=None, creator=None
        )
        formatted_inv += f"  - {art['name']}: {art['payload']}\n"
    return formatted_inv


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def format_observation(observation: dict) -> str:
    """Format the observation and inventory into a readable string.

    Args:
        observation (dict): The agent's observation.

    Returns:
        str: Formatted observation string.
    """
    formatted_obs = "Observation:\n"
    for location, contents in observation.items():
        # This way we don't give it lines with just food
        skip = True
        for content in contents:
            if not is_number(content):
                skip = False
                break
        if skip:
            continue
        formatted_obs += f"  Relative Location {location}:\n"
        for content in contents:
            formatted_obs += f"    - {content}\n"
    return formatted_obs


def format_received_messages(received_messages: dict) -> str:
    """Format received messages into a readable string.

    Args:
        received_messages (dict): Messages received by the agent.

    Returns:
        str: Formatted received messages string.
    """
    if not received_messages:
        return "No messages received."

    formatted_msgs = "Received messages:\n"
    for sender, msg in received_messages.items():
        formatted_msgs += f"  - {sender}: {msg.strip()}\n"

    return formatted_msgs


def extract_info_from_log(
    artifact: dict,
    agent_log: dict,
    max_history: int,
    all_artifacts: ExperimentArtifacts,
) -> dict:
    """
    Extract info from the log at the artifact creation time and at max_history steps previous to that.

    The info for the creeation time includes the reasoning and previous ts internal memory.
    The history info just contain observations and actions, as the agent does not perceive its own reasoning again.
    """
    creation_time = int(artifact["creation_time"])
    creation_log = agent_log[creation_time]

    log_info = {}

    log_info["agent_thoughts"] = creation_log["action"]["reasoning"]

    # Process observation
    # ---------------------------
    observation = creation_log["observation"]["observation"]
    observation = process_00(
        observation=observation, artifacts=all_artifacts, ts=creation_time
    )
    agent_observations = format_observation(observation)

    inventory = creation_log["observation"]["inventory"]
    if inventory:
        agent_observations += "\n" + format_inventory(
            inventory, all_artifacts, creation_time
        )

    received_messages = creation_log["observation"]["message"]
    if received_messages:
        agent_observations += "\n" + format_received_messages(received_messages)

    sent_message = creation_log["action"]["message"].strip()
    if len(sent_message) > 0:
        agent_observations += f"\nBroadcasted message: {sent_message}"

    log_info["agent_observations"] = agent_observations
    # ---------------------------

    # Process memory
    # ---------------------------
    agent_memory = ""
    if creation_time - 1 in agent_log:
        prev_internal_memory = agent_log[creation_time - 1]["internal_memory"]
        agent_memory += f"Previous internal memory: {prev_internal_memory}\n"

    memory = []
    for rel_ts in range(max_history, 0, -1):
        abs_t = creation_time - rel_ts
        if abs_t not in agent_log:
            continue
        history_episode = "--- Time step -" + str(rel_ts) + " ---\n"
        history_dict = get_history(agent_log[abs_t])

        history_dict["observation"] = process_00(
            history_dict["observation"], artifacts=all_artifacts, ts=abs_t
        )
        history_obs = format_observation(history_dict["observation"])
        history_episode += history_obs + "\n"

        inventory = history_dict["inventory"]
        if inventory:
            history_episode += format_inventory(inventory, all_artifacts, abs_t) + "\n"

        if history_dict["received_messages"]:
            history_episode += format_received_messages(
                history_dict["received_messages"]
            )

        if history_dict["message_sent"].strip():
            history_episode += (
                f"Broadcasted message: {history_dict['message_sent'].strip()}\n"
            )

        if history_dict["action"] != "move":
            history_episode += f"Action taken: {history_dict['action']} with parameters {history_dict['action_parameters']}\n"

        memory.append(history_episode)
    if memory:
        relevant_history = "\n".join(memory)
        agent_memory += f"\nRelevant history: \n{relevant_history}"

    log_info["agent_memory"] = agent_memory if agent_memory else "N/A"
    # ---------------------------

    return log_info


# ----------------------------


def process_artifact(
    art_id: int,
    artifact: dict,
    agent_logs_dict: dict,
    exp_path: Path,
    max_history: int,
    all_artifacts: ExperimentArtifacts,
    previous_artifacts: dict,
    model_name: str,
    llm_provider: str,
) -> Tuple[int, list, dict, dict]:
    """Process a single artifact to determine its phylogeny.

    Returns:
        Tuple of (art_id, hand_phylogeny_result, llm_phylogeny_result, token_counter)
    """
    hand_result = []
    llm_result = {}
    token_counter = {"input_tokens": 0, "output_tokens": 0}

    # Load agent log if needed
    creator_log = agent_logs_dict.get(artifact["creator_tag"])
    if creator_log is None:
        creator_tag = artifact["creator_tag"]
        creator_log = load_agent_log(
            filepath=exp_path / "agent_logs" / f"{creator_tag}.jsonl",
            reduce=False,
        )

    info = extract_info_from_log(
        artifact=artifact,
        agent_log=creator_log,
        max_history=max_history,
        all_artifacts=all_artifacts,
    )

    if HAND_PHYLOGENY:
        if artifact["event"] == "modified":
            hand_result.append(artifact["previous_version_tag"])

        for old_art_idx, old_art_name in reversed(previous_artifacts.items()):
            if old_art_name is None:
                old_art_name = "None"
            if artifact["event"] in ["created", "modified"]:
                info_str = str(info)
                if re.search(
                    rf"(?<![a-zA-Z0-9_]){re.escape(old_art_name)}(?![a-zA-Z0-9_])",
                    info_str,
                ):
                    hand_result.append(old_art_idx)
            else:
                raise ValueError(f"Unknown artifact event type: {artifact['event']}")

        hand_result = list(set(hand_result))

    if LLM_PHYLOGENY:
        llm_result = {}

        if len(previous_artifacts) == 0:
            return art_id, hand_result, llm_result, token_counter

        info_str = str(info)
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
            user_prompt = USER_PROMPT_CREATION.format(
                artifact_id=art_id,
                artifact_name=artifact["name"],
                artifact_content=artifact["payload"],
                agent_thoughts=info.get("agent_thoughts", "N/A"),
                agent_observations=info.get("agent_observations", "N/A"),
                agent_memory=info.get("agent_memory", "N/A"),
                artifact_candidates=artifact_candidates,
            )
        else:
            raise ValueError(f"Unknown artifact event type: {artifact['event']}")

        llm_client = LLMClient(client=llm_provider, long_context=False)
        SYSTEM_PROMPT = BYNARY_SYSTEM_PROMPT if BYNARY else FINER_SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        valid = False
        trial_counter = 0
        while not valid:
            try:
                response = llm_client.get_response(
                    model=model_name,
                    messages=messages,
                    chat_parameters=LLM_CHAT_PARAMS,
                    output_json=True,
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
            except Exception as e:
                trial_counter += 1
                if trial_counter >= 5:
                    print(f"Failed for artifact {art_id} after {trial_counter} trials.")
                    llm_result = {"error": "failed_to_parse_response"}
                    break
                print(f"Invalid response for artifact {art_id}: {e}. Retrying...")
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"The previous response was invalid because of the error: {e}. "
                            "Please provide a valid JSON dictionary as specified."
                        ),
                    }
                )

    return art_id, hand_result, llm_result, token_counter


def main(
    exp_path: Path,
    error_tracker: ErrorTracker,
):
    artifacts = ExperimentArtifacts(
        exp_path=exp_path,
        embedding_dimensions=512,
        save_path=exp_path / "artifact_analysis",
        embedding_model="text-embedding-3-large",
        replace_numbers=True,
        embed_names=True,
    )
    artifacts.load(force_recalc=False)
    artifacts._load_raw_artifacts()
    artifacts_by_creation = artifacts.get_artifact_by_creation(force=True)

    params = json.load(open(exp_path / "params.json", "r"))
    if "max_history" in params:
        max_history = params["max_history"]
    else:
        max_history = params["agent"]["max_history"]

    print(f"Starting phylogeny analysis on {exp_path.name}")
    previous_artifacts = {}
    agent_logs = {}  # We load them when necessary, but then store them so not to reload again

    start_time = datetime.now().isoformat()
    if LLM_PHYLOGENY:
        model_name = LLM_MODEL
        print(f"Using model {model_name}...")
        llm_artifact_phylogeny = {}
    if HAND_PHYLOGENY:
        hand_artifact_phylogeny = {}

    for ts in tqdm(artifacts_by_creation, desc="TS"):
        token_counter = {
            "time_step": ts,
            "input_tokens": 0,
            "output_tokens": 0,
            "start_time": start_time,
        }

        current_artifacts = {}
        artifact_ids = artifacts_by_creation[ts]

        if len(artifact_ids) == 0:
            continue

        # Prepare arguments for parallel processing
        process_args = []
        for art_id in artifact_ids:
            artifact = artifacts.all_artifacts[art_id]
            current_artifacts[art_id] = artifact["name"]

            # Load agent log if not already done
            if artifact["creator_tag"] not in agent_logs:
                creator_tag = artifact["creator_tag"]
                creator_log = load_agent_log(
                    filepath=exp_path / "agent_logs" / f"{creator_tag}.jsonl",
                    reduce=False,
                )
                agent_logs[artifact["creator_tag"]] = creator_log

            process_args.append(
                (
                    art_id,
                    artifact,
                    agent_logs,
                    exp_path,
                    max_history,
                    artifacts,
                    previous_artifacts.copy(),
                    model_name if LLM_PHYLOGENY else None,
                    LLM_PROVIDER,
                )
            )

        # Process artifacts in parallel
        if PARALLEL:
            with ProcessPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
                futures = {
                    executor.submit(process_artifact, *args): args[0]
                    for args in process_args
                }

                for future in as_completed(futures):
                    art_id = futures[future]
                    try:
                        art_id, hand_result, llm_result, tokens = future.result()

                        if HAND_PHYLOGENY:
                            hand_artifact_phylogeny[art_id] = hand_result

                        if LLM_PHYLOGENY:
                            llm_artifact_phylogeny[art_id] = llm_result
                            token_counter["input_tokens"] += tokens["input_tokens"]
                            token_counter["output_tokens"] += tokens["output_tokens"]
                    except Exception as e:
                        error_tracker.add_error(
                            f"artifact_{art_id}_ts_{ts}",
                            e,
                            additional_info={
                                "experiment": str(exp_path),
                                "phase": "phylogeny_analysis",
                                "timestep": ts,
                            },
                        )
        else:
            for args in process_args:
                try:
                    art_id, hand_result, llm_result, tokens = process_artifact(*args)
                except:
                    print()

                if HAND_PHYLOGENY:
                    hand_artifact_phylogeny[art_id] = hand_result

                if LLM_PHYLOGENY:
                    llm_artifact_phylogeny[art_id] = llm_result
                    token_counter["input_tokens"] += tokens["input_tokens"]
                    token_counter["output_tokens"] += tokens["output_tokens"]

        if LLM_PHYLOGENY:
            with open(
                artifacts.save_path / f"token_counts_phylogeny_{model_name}.jsonl", "a"
            ) as f:
                f.write(json.dumps(token_counter) + "\n")

        previous_artifacts.update(current_artifacts)

    if LLM_PHYLOGENY:
        try:
            with open(
                artifacts.save_path / f"artifact_phylogeny_{model_name}.json", "w"
            ) as f:
                json.dump(llm_artifact_phylogeny, f, indent=4)
        except Exception:
            print("Failed to save json. Trying pickle...")
            with open(
                artifacts.save_path / f"artifact_phylogeny_{model_name}.pkl", "wb"
            ) as f:
                pkl.dump(llm_artifact_phylogeny, f)

    if HAND_PHYLOGENY:
        try:
            with open(artifacts.save_path / "artifact_phylogeny_hand.json", "w") as f:
                json.dump(hand_artifact_phylogeny, f, indent=4)
        except Exception:
            print("Failed to save json. Trying pickle...")
            with open(artifacts.save_path / "artifact_phylogeny_hand.pkl", "wb") as f:
                pkl.dump(hand_artifact_phylogeny, f)

    print("Phylogeny analysis completed.")


if __name__ == "__main__":
    failed = []

    main_tracker = ErrorTracker(show_stacktraces=SHOW_STACKTRACES)

    for exp in EXPERIMENTS_NAMES:
        print(f"Running analysis on experiment: {exp}")
        try:
            main(
                exp_path=ROOT / "logs" / exp,
                error_tracker=main_tracker,
            )
        except Exception as e:
            print(f"Experiment {exp} failed with error: {e}")
            main_tracker.add_experiment_failure(exp, e)

    main_tracker.print_summary()
    main_tracker.save_to_file(ROOT / "analysis_summary" / "006_summary.json")
