import asyncio
import json
import random
import re
import traceback
from typing import Any, Dict, Optional

import tiktoken
from openai import APIConnectionError, BadRequestError, OpenAIError, RateLimitError

from terralingua.utils.models import context_table, output_table

MAX_CONTEXT_TOKENS = context_table()
MAX_OUTPUT_TOKENS = output_table()



_STAY_ACTION = {"action": "move", "message": "", "params": {"direction": "stay"}}


async def async_select_with_retry(
    agent,
    observation,
    available_actions,
    reward,
    info,
    ts,
    llm_client,
    llm_chat_params,
    retries=3,
    backoff_base=1.5,
) -> dict:
    """Async outer retry wrapper — handles network/API errors.

    Returns the action dict only (no need_refresh flag).
    Inner parse-failure retries are handled inside agent.select_action_async().
    """
    original_history_len = agent.max_history
    stay_action = getattr(agent, "stay_action", None) or _STAY_ACTION
    try:
        for attempt in range(retries):
            try:
                action = await agent.select_action_async(
                    obs=observation,
                    reward=reward,
                    info=info,
                    time=ts,
                    available_actions=available_actions,
                    chat_params=llm_chat_params,
                    client=llm_client,
                )
                return action
            except BadRequestError as e:
                print(
                    f"⚠️ Retry {attempt + 1}/{retries} {agent.agent_name}({agent.agent_tag}) Bad Request: {e}. Likely cause is too many tokens"
                )
                agent.max_history = max(0, agent.max_history - 1)
            except (RateLimitError, APIConnectionError, OpenAIError) as e:
                wait_time = backoff_base**attempt + random.uniform(0, 0.5)
                print(
                    f"⚠️ Retry {attempt + 1}/{retries} {agent.agent_name}({agent.agent_tag}) due to error: {e}. Sleep {wait_time:.2f}s"
                )
                await asyncio.sleep(wait_time)
            except Exception as e:
                print(f"❌ Unexpected failure {agent.agent_name}({agent.agent_tag}): {e}")
                traceback.print_exc()
                return {**stay_action, "source": "default", "default_reason": "unexpected_error"}

        print(f"❌ Failed after {retries} retries {agent.agent_name}({agent.agent_tag}).")
        return {**stay_action, "source": "default", "default_reason": "max_retries"}
    finally:
        agent.max_history = original_history_len


def build_output_schema(action_description: dict) -> dict:
    actions = list(action_description.keys())
    schema = {
        "type": "object",
        "properties": {
            "ACTION": {"type": "string", "enum": actions},
            "MESSAGE": {"type": "string"},
            "PARAMS": {"type": "object"},
        },
        "required": ["ACTION", "PARAMS"],
        "additionalProperties": False,
        "allOf": [],
    }

    for action, description in action_description.items():
        params = description.get("params", {}) or {}
        params = list(params.keys())
        # We only enforce that these keys exist in PARAMS; values can be any JSON type.
        # (Each param gets an empty schema {} to mean "present, any type".)
        props = {name: {"type": "string"} for name in params}

        then_obj = {
            "properties": {
                "PARAMS": {
                    "type": "object",
                    "properties": props,
                    "required": params,
                    "additionalProperties": False,
                }
            }
        }

        schema["allOf"].append(
            {"if": {"properties": {"ACTION": {"const": action}}}, "then": then_obj}
        )

    return schema


def strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def extract_json_obj(s: str) -> Optional[Dict[str, Any]]:
    """Try to recover a JSON object from model output (handles code fences,
    quoted JSON, and 'JSON-as-string' cases). Returns dict with lowercase keys or None."""
    s = strip_code_fences(s)

    def lower_keys(d: Dict[str, Any]) -> Dict[str, Any]:
        return {str(k).lower(): v for k, v in d.items()}

    # If the whole thing is quoted:  '{...}'  or  "{\"A\":1}"
    if (s.startswith("'") and s.endswith("'")) or (
        s.startswith('"') and s.endswith('"')
    ):
        try:
            s1 = json.loads(s)  # unquote
            if isinstance(s1, str):
                try:
                    obj = json.loads(s1)  # double-encoded
                    if isinstance(obj, dict):
                        return lower_keys(obj)
                except json.JSONDecodeError:
                    pass
            if isinstance(s1, dict):
                return lower_keys(s1)
        except json.JSONDecodeError:
            pass

    # Direct parse
    try:
        obj = json.loads(s)
        if isinstance(obj, str):
            obj = json.loads(obj)  # JSON string containing JSON
        if isinstance(obj, dict):
            return lower_keys(obj)
    except json.JSONDecodeError:
        return None

    return None


def count_tokens(messages, model="gpt-4o-mini"):
    """
    Count tokens for a given model and messages.

    For Claude models, uses tiktoken with a correction factor as approximation.
    For OpenAI models, uses tiktoken directly.

    Note: Claude token counts are approximate. Tiktoken tokenization is typically
    within 10-15% of Claude's actual tokenizer. We add a 15% buffer for safety.

    Args:
        messages: List of message dicts with "role" and "content" keys
        model: Model identifier string

    Returns:
        Approximate token count (slightly overestimated for safety)
    """
    # Use tiktoken for tokenization (works for both OpenAI and as approximation for Claude)
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        # cl100k_base is used by GPT-4 and is a reasonable approximation for Claude
        enc = tiktoken.get_encoding("cl100k_base")

    total_tokens = 0
    for msg in messages:
        total_tokens += len(enc.encode(msg["role"]))
        total_tokens += len(enc.encode(msg["content"]))

    # For Claude models, add a 15% safety buffer since tiktoken is an approximation
    # This ensures we don't exceed context limits due to tokenization differences
    if model.startswith("claude"):
        total_tokens = int(total_tokens * 1.15)

    return total_tokens


def is_context_enough(messages, max_input_tokens, model) -> bool:
    input_tokens = count_tokens(messages, model=model)
    print("Input tokens:", input_tokens)
    print("Max allowed tokens:", max_input_tokens - MAX_OUTPUT_TOKENS[model])
    if input_tokens < max_input_tokens - MAX_OUTPUT_TOKENS[model]:
        return True
    return False
