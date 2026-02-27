import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from artifact_complexity import ExperimentArtifacts
from openai import BadRequestError
from tqdm import tqdm

from analysis_scripts.error_tracker import ErrorTracker
from core.utils import ROOT
from core.utils.llm_client import LLMClient, Response

"""
This script performs classification of artifacts.
It classifies each artifact into one of the following categories:
Category 1. Basic & Informational
Category 2. Procedural or Coordination
Category 3. Institutional Structures
Category 4. Norms, Rules, and Governance
Category -1. Anything that does not fit 1-4.

It performs the following steps for each artifact:
1. Load the artifact data.
2. Generate a classification prompt.
3. Send the prompt to the LLM for classification.
4. Parse the LLM response.
5. Save the classification result.
The script is designed to run in parallel for multiple artifacts to speed up the classification process.
"""


EXPERIMENTS_NAMES = []  # e.g. ['core_run', 'scarcity_run', ...]

# Setup
# ---------------------------
LLM_PROVIDER = "anthropic"
LLM_MODEL = "claude-haiku-4-5"
LLM_CHAT_PARAMS = {}
PARALLEL_WORKERS = 8
PARALLEL = True
# ---------------------------

# Prompt
# ---------------------------
SYSTEM_PROMPT = """You are an expert annotator analyzing text artifacts produced by agents in a multi-agent environment.
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
# ---------------------------


def classify_artifact(
    artifact_index: int,
    user_prompt: str,
) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    llm_client = LLMClient(client=LLM_PROVIDER)

    try:
        result = llm_client.get_response(
            model=LLM_MODEL,
            messages=messages,
            chat_parameters=LLM_CHAT_PARAMS,
            enable_error_reprompting=False,
            output_json=True,
        )
        return {"result": result, "artifact_index": artifact_index, "success": True}

    except BadRequestError as e:
        return {
            "success": False,
            "error": e,
            "error_type": "context_error",
            "artifact": user_prompt,
            "prompt": messages,
        }

    except Exception as e:
        print("---------------------")
        print("".join(traceback.format_exception(None, e, e.__traceback__)))
        print("--")
        print(messages)
        print("---------------------")
        return {
            "success": False,
            "error": e,
            "error_type": "general",
            "artifact": user_prompt,
            "prompt": messages,
        }


def main(
    exp_path: Path,
    error_tracker: ErrorTracker,
):
    print("Starting artifact clustering...")

    artifacts = ExperimentArtifacts(
        exp_path=exp_path,
        embedding_dimensions=512,
        save_path=exp_path / "artifact_analysis",
    )
    artifacts.load(force_recalc=False)
    artifacts._load_raw_artifacts()

    # prepare user prompts
    print(f"Classifying {len(artifacts.all_artifacts)} artifacts...")
    max_workers = min(PARALLEL_WORKERS, 3)  # Limit concurrent API calls
    counter = {
        "input_tokens": 0,
        "output_tokens": 0,
        "start_time": datetime.now().isoformat(),
    }

    categories = {}
    to_retry = []
    if PARALLEL:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    classify_artifact,
                    user_prompt=f"Name: {art['name']}\nContent: {art['payload']}",
                    artifact_index=art_idx,
                )
                for art_idx, art in artifacts.all_artifacts.items()
            ]

            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
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
                        print("Warning: received None result for an artifact.")
                else:
                    if "error" in result:
                        error_type = result.get("error_type", "unknown")
                        error_tracker.add_error(
                            context="",
                            error=result["error"],
                            error_type=error_type,
                            additional_info={
                                "experiment": str(exp_path),
                                "attempt": result.get("attempt", "unknown"),
                                "artifact": result.get("artifact", "unknown"),
                            },
                        )
                        if error_type == "context_error":
                            print(
                                f"⚠️ BadRequestError from artifact {result.get('artifact', 'unknown')} - likely due to context length. Skipping."
                            )
                        else:
                            print(
                                f"⚠️ Error from artifact {result.get('artifact', 'unknown')}\n {result['error']}"
                            )
                    print(
                        f"Skipping artifact {result.get('artifact', 'unknown')} due to error {result.get('error', 'unknown')}."
                    )
                    to_retry.append(result.get("artifact", "unknown"))
    else:
        for art_idx, art in tqdm(artifacts.all_artifacts.items()):
            result = classify_artifact(
                user_prompt=f"Name: {art['name']}\nContent: {art['payload']}",
                artifact_index=art_idx,
            )
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
                    print("Warning: received None result for an artifact.")
            else:
                if "error" in result:
                    error_type = result.get("error_type", "unknown")
                    error_tracker.add_error(
                        context="",
                        error=result["error"],
                        error_type=error_type,
                        additional_info={
                            "experiment": str(exp_path),
                            "attempt": result.get("attempt", "unknown"),
                            "artifact": result.get("artifact", "unknown"),
                        },
                    )
                    if error_type == "context_error":
                        print(
                            f"⚠️ BadRequestError from artifact {result.get('artifact', 'unknown')} - likely due to context length. Skipping."
                        )
                    else:
                        print(
                            f"⚠️ Error from artifact {result.get('artifact', 'unknown')}\n {result['error']}"
                        )
                print(
                    f"Skipping artifact {result.get('artifact', 'unknown')} due to error {result.get('error', 'unknown')}."
                )
                to_retry.append(result.get("artifact", "unknown"))

    print("Saving results...")
    (exp_path / "artifact_analysis").mkdir(parents=True, exist_ok=True)
    with open(exp_path / "artifact_analysis" / "artifact_categories.json", "w") as f:
        json.dump(categories, f, indent=2)
    print(
        f"Classification complete. {len(categories)} results saved at {exp_path / 'artifact_analysis' / 'artifact_categories.json'}"
    )


if __name__ == "__main__":
    failed = []

    main_tracker = ErrorTracker(show_stacktraces=False)

    for exp in EXPERIMENTS_NAMES:
        print(f"Running analysis on experiment: {exp}")
        try:
            main(
                exp_path=ROOT / "logs" / exp,
                error_tracker=main_tracker,
            )
        except Exception as e:
            main_tracker.add_experiment_failure(exp, e)
        print()

    main_tracker.print_summary()
    main_tracker.save_to_file(ROOT / "analysis_summary" / "005_summary.json")
