import json
import math
import os
import pickle as pkl
import sys
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np

# Single artifact metric
from artifact_complexity import (
    CompressedSize,
    ExperimentArtifacts,
    InverseCompressionRate,
    LexicalSophistication,
    LMSurprisal,
    SyntacticDepth,
)
from openai import BadRequestError
from tqdm import tqdm

from analysis_scripts.error_tracker import ErrorTracker
from core.utils import ROOT
from core.utils.llm_client import LLMClient
from core.utils.llm_utils import (
    MAX_CONTEXT_TOKENS,
    MAX_OUTPUT_TOKENS,
    count_tokens,
    is_context_enough,
)

"""
This script performs artifact analysis, including novelty scoring and metric calculations.
It processes artifacts from experiments, calculates embeddings, computes metrics, and evaluates novelty using an LLM.
Novelty analysis is done by comparing new artifacts at each timestep against all previously seen artifacts, assigning a novelty score from 0 to 5 based on conceptual divergence.
It performs the following steps for each artifact:
1. Load artifacts and their metadata.
2. Optionally calculate embeddings for artifacts.
3. Optionally compute various complexity and novelty metrics.
4. For novelty analysis, iteratively evaluate new artifacts against previous ones using an LLM, assigning novelty scores, and handle context size issues by pruning low-novelty artifacts if necessary.
5. Save all results, including novelty scores and metrics, for further analysis and visualization.
"""

EXPERIMENTS_NAMES = []  # e.g. ['core_run', 'scarcity_run', ...]

# Setup
# ---------------------------
NOVELTY = True
METRICS = True
EMBED = True
NOVELTY_SAMPLES = 5

PARALLEL = True
LLM_PROVIDER = "anthropic"
LLM_MODEL = "claude-sonnet-4-5-20250929"
LLM_CHAT_PARAMS = {
    # "response_format": {"type": "json_object"},
}

# Novelty analysis configuration
MAX_RETRY_ATTEMPTS = 10
PRUNE_BATCH_SIZE = (
    5  # Number of artifacts to remove per iteration when context is too large
)
PRUNE_NOVELTY_THRESHOLD = 1  # Only prune artifacts with novelty <= this value

# ---------------------------


# Prompts
# ---------------------------
SYSTEM_PROMPT = """
You are a rigorous novelty analyst.
Your task is to evaluate how conceptually novel and interesting each artifact is relative to all previously seen artifacts.
Output VALID JSON ONLY matching the schema.
Never invent IDs.
Compare each new artifact ONLY against the previous artifacts. DO NOT compare artifacts with the ones in the same timestep.
Note: It is extremely important that you get this right, as this will be used for scientific analysis.
"""

USER_PROMPT = """Analyze the novelty of the new artifacts compared to the previous artifacts.

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
# ---------------------------


# Utils
# ---------------------------
def prune_low_novelty_artifacts(previous_artifacts, batch_size=PRUNE_BATCH_SIZE):
    """
    Remove up to batch_size artifacts with novelty <= PRUNE_NOVELTY_THRESHOLD.
    Returns the pruned list and the number of artifacts removed.
    """
    artifacts_to_remove = []
    for idx, art in enumerate(previous_artifacts):
        if art.get("novelty", 0) <= PRUNE_NOVELTY_THRESHOLD:
            artifacts_to_remove.append(idx)
            if len(artifacts_to_remove) == batch_size:
                break

    pruned_artifacts = [
        art
        for idx, art in enumerate(previous_artifacts)
        if idx not in artifacts_to_remove
    ]

    return pruned_artifacts, len(artifacts_to_remove)


def check_and_prepare_context(messages, llm_client, previous_artifacts, new_artifacts):
    """
    Check if messages fit in context. If not, try long context or prune artifacts.
    Returns: (messages, llm_client, previous_artifacts, context_ok)
    """
    # First check if it fits in base context
    if is_context_enough(
        messages=messages,
        max_input_tokens=MAX_CONTEXT_TOKENS[LLM_MODEL]["base"],
        model=LLM_MODEL,
    ):
        return messages, llm_client, previous_artifacts, True

    # Try long context if available
    if (
        MAX_CONTEXT_TOKENS[LLM_MODEL].get("long", None) is not None
        and not llm_client.long_context
    ):
        print("Switching to long context model for novelty analysis.")
        llm_client = LLMClient(client=LLM_PROVIDER, long_context=True)

        if is_context_enough(
            messages=messages,
            max_input_tokens=MAX_CONTEXT_TOKENS[LLM_MODEL]["long"],
            model=LLM_MODEL,
        ):
            return messages, llm_client, previous_artifacts, True

    # If still too large, prune artifacts iteratively
    print("Context still too large. Pruning low-novelty artifacts...")
    max_prune_iterations = 50  # Safety limit
    for iteration in range(max_prune_iterations):
        previous_artifacts, removed_count = prune_low_novelty_artifacts(
            previous_artifacts, batch_size=PRUNE_BATCH_SIZE
        )

        if removed_count == 0:
            print("No more low-novelty artifacts to prune.")
            return messages, llm_client, previous_artifacts, False

        print(f"Pruned {removed_count} artifacts (iteration {iteration + 1})")

        # Rebuild messages with pruned artifacts
        user_prompt = USER_PROMPT.format(
            previous_artifacts=json.dumps(previous_artifacts),
            new_artifacts=json.dumps(new_artifacts),
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Check if it fits now
        max_tokens = (
            MAX_CONTEXT_TOKENS[LLM_MODEL]["long"]
            if llm_client.long_context
            else MAX_CONTEXT_TOKENS[LLM_MODEL]["base"]
        )

        if is_context_enough(
            messages=messages,
            max_input_tokens=max_tokens,
            model=LLM_MODEL,
        ):
            print("Context now fits after pruning.")
            return messages, llm_client, previous_artifacts, True

    print("Warning: Could not fit context even after maximum pruning iterations.")
    return messages, llm_client, previous_artifacts, False


def get_novelty_sample(llm_client, messages, sample_idx, ts):
    """Helper function to get a single novelty sample."""
    for attempt in range(MAX_RETRY_ATTEMPTS):
        try:
            response = llm_client.get_response(
                model=LLM_MODEL,
                messages=messages,
                chat_parameters=LLM_CHAT_PARAMS,
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
            if attempt == MAX_RETRY_ATTEMPTS - 1:
                return {
                    "success": False,
                    "error": e,
                    "error_type": "general",
                    "sample_idx": sample_idx,
                    "attempt": attempt + 1,
                }

    return {"success": False, "sample_idx": sample_idx}


# ----------------------------


def main(
    exp_path: Path,
    metrics: list,
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
    artifacts.load(force_recalc=True)

    if EMBED:
        print("Calculating embeddings...")
        artifacts._embed_artifacts()
        artifacts.save()
        print("Embeddings saved. ")

    if METRICS:
        artifacts.metrics = {}
        for metric in metrics:
            print(f"Calculating metric: {metric.name}")
            metric.compute(artifacts)

        artifacts.save()
        print("Metrics saved. ")

    if NOVELTY:
        artifacts_by_creation = artifacts.get_artifact_by_creation()

        print("Starting novelty analysis...")
        llm_client = LLMClient(client=LLM_PROVIDER, long_context=False)
        previous_artifacts = []

        start_time = datetime.now().isoformat()
        for ts in tqdm(artifacts_by_creation, desc="TS"):
            counter = {
                "time_step": ts,
                "input_tokens": 0,
                "output_tokens": 0,
                "start_time": start_time,
            }
            new_artifacts = []
            for art_id in artifacts_by_creation[ts]:
                new_artifacts.append(
                    {
                        "id": art_id,
                        "name_and_content": f"{artifacts.all_artifacts[art_id]['string']}",
                    }
                )

            if new_artifacts:
                user_prompt = USER_PROMPT.format(
                    previous_artifacts=json.dumps(previous_artifacts),
                    new_artifacts=json.dumps(new_artifacts),
                )

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]

                # Proactively check and prepare context before making LLM calls
                messages, llm_client, previous_artifacts, context_ok = (
                    check_and_prepare_context(
                        messages, llm_client, previous_artifacts, new_artifacts
                    )
                )

                if not context_ok:
                    error_tracker.add_error(
                        f"timestep_{ts}",
                        Exception("Context too large even after pruning"),
                        additional_info={
                            "experiment": str(exp_path),
                            "phase": "novelty_analysis",
                            "previous_artifacts_count": len(previous_artifacts),
                        },
                    )
                    print(f"Skipping timestep {ts} due to context size issues.")
                    continue

                novelties = defaultdict(list)
                if PARALLEL:
                    max_workers = min(NOVELTY_SAMPLES, 3)  # Limit concurrent API calls
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = [
                            executor.submit(
                                get_novelty_sample, llm_client, messages, idx, ts
                            )
                            for idx in range(NOVELTY_SAMPLES)
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
                                if "error" in result:
                                    error_type = result.get("error_type", "unknown")
                                    error_tracker.add_error(
                                        f"timestep_{ts}_sample_{sample_idx}",
                                        result["error"],
                                        error_type=error_type,
                                        additional_info={
                                            "experiment": str(exp_path),
                                            "phase": "novelty_analysis",
                                            "attempt": result.get("attempt", "unknown"),
                                        },
                                    )
                                    if error_type == "context_error":
                                        print(
                                            f"⚠️ BadRequestError at timestep {ts}, sample {sample_idx + 1}"
                                        )
                                    else:
                                        print(
                                            f"⚠️ Error at timestep {ts}, sample {sample_idx + 1}"
                                        )
                                print(
                                    f"Skipping sample {sample_idx + 1} for timestep {ts}"
                                )
                else:
                    for sample_idx in range(NOVELTY_SAMPLES):
                        success = False
                        for attempt in range(MAX_RETRY_ATTEMPTS):
                            try:
                                response = llm_client.get_response(
                                    model=LLM_MODEL,
                                    messages=messages,
                                    chat_parameters=LLM_CHAT_PARAMS,
                                    enable_error_reprompting=False,
                                    output_json=True,
                                )
                                novelty_sample = response.content
                                counter["input_tokens"] += response.input_tokens
                                counter["output_tokens"] += response.output_tokens

                                if not isinstance(novelty_sample, dict):
                                    raise ValueError(
                                        "Response is not a valid JSON dictionary"
                                    )

                                for idx in novelty_sample:
                                    novelties[str(idx)].append(novelty_sample[idx])

                                success = True
                                break

                            except BadRequestError as e:
                                print(
                                    f"⚠️ BadRequestError at timestep {ts}, sample {sample_idx + 1}, attempt {attempt + 1}: {e}"
                                )
                                error_tracker.add_error(
                                    f"timestep_{ts}_sample_{sample_idx}",
                                    e,
                                    error_type="context_error",
                                    additional_info={
                                        "experiment": str(exp_path),
                                        "phase": "novelty_analysis",
                                        "attempt": attempt + 1,
                                    },
                                )
                                # Context should have been handled preemptively, this shouldn't happen
                                break

                            except Exception as e:
                                print(
                                    f"⚠️ Error at timestep {ts}, sample {sample_idx + 1}, attempt {attempt + 1}: {e}"
                                )
                                if attempt == MAX_RETRY_ATTEMPTS - 1:
                                    error_tracker.add_error(
                                        f"timestep_{ts}_sample_{sample_idx}",
                                        e,
                                        additional_info={
                                            "experiment": str(exp_path),
                                            "phase": "novelty_analysis",
                                        },
                                    )
                                    print(
                                        f"Failed after {MAX_RETRY_ATTEMPTS} attempts for timestep {ts}, sample {sample_idx + 1}"
                                    )

                        if not success:
                            print(f"Skipping sample {sample_idx + 1} for timestep {ts}")

                try:
                    with open(
                        artifacts.save_path / f"token_counts_{LLM_MODEL}.jsonl", "a"
                    ) as f:
                        f.write(json.dumps(counter) + "\n")
                except Exception:
                    with open(
                        artifacts.save_path / "token_counts_last.jsonl", "a"
                    ) as f:
                        f.write(json.dumps(counter) + "\n")

                for art in new_artifacts:
                    art_id = str(art["id"])
                    if art_id in novelties:
                        avg_novelty = np.mean([int(n) for n in novelties[art_id]])
                    else:
                        avg_novelty = -1

                    if avg_novelty == -1:
                        print(f"Warning: Artifact {art_id} has no novelty score.")
                        print("Artifact name:", art.get("name", "Unknown"))
                        print("Artifact content:", art.get("payload", "Unknown"))

                    art["novelty"] = avg_novelty
                    artifacts.all_artifacts[art["id"]]["novelty"] = avg_novelty

                previous_artifacts.extend(new_artifacts)

        artifacts.save()

        print("Novelty analysis completed. Verifying results...")
        all_novelties = {}
        for art_id, art in tqdm(
            artifacts.all_artifacts.items(), desc="Saving novelties"
        ):
            novelty = art.get("novelty", None)
            if novelty is None:
                all_novelties[art_id] = -1
                print(f"Warning: Artifact {art_id} has no novelty score.")
                print("Artifact name:", art.get("name", "Unknown"))
                print("Artifact content:", art.get("payload", "Unknown"))
            else:
                all_novelties[art_id] = novelty

        try:
            save_path = artifacts.save_path / f"novelties_{LLM_MODEL}.pkl"
            with open(save_path, "wb") as f:
                pkl.dump(all_novelties, f)
        except Exception as e:
            save_path = artifacts.save_path / "novelties_last.json"
            print(f"Failed to save novelties: {e}. Trying saving at {save_path}")
            with open(save_path, "w") as f:
                json.dump(all_novelties, f)

        print(f"Novelties saved to {save_path}.")


if __name__ == "__main__":
    if METRICS:
        metrics = [
            LMSurprisal(),
            CompressedSize(),
            InverseCompressionRate(),
            SyntacticDepth(),
            LexicalSophistication(),
        ]
    else:
        metrics = []

    main_tracker = ErrorTracker(show_stacktraces=False)

    for exp in EXPERIMENTS_NAMES:
        print(f"Running analysis on experiment: {exp}")
        try:
            main(
                exp_path=ROOT / "logs" / exp,
                metrics=metrics,
                error_tracker=main_tracker,
            )
        except Exception as e:
            main_tracker.add_experiment_failure(exp, e)
        print()

    main_tracker.print_summary()
    main_tracker.save_to_file(ROOT / "analysis_summary" / "004_summary.json")
