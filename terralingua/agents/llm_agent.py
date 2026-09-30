# survival_parallel_llm_agent.py

import asyncio
import copy
import importlib
import json
import logging
import re
from pathlib import Path
from typing import Dict, Tuple

import tiktoken

from terralingua.agents.agent_logger import AgentLogger
from terralingua.agents.agent_mixin import AgentMixin
from terralingua.agents.prompt_templates import (
    ERROR_MSG,
    render_system_prompt,
    resolve_instructions,
)
from terralingua.genome.base_genome import Genome
from terralingua.genome.ocean_5 import Genome as Ocean5Genome
from terralingua.utils.llm_client import AgentClient
from terralingua.utils.logging_setup import logger_level

log = logging.getLogger(__name__)


class LLMAgent(AgentMixin):
    def __init__(
        self,
        agent_name: str,
        agent_tag: str,
        logger: AgentLogger | None = None,
        genome: Genome | None = None,
        log_dir: Path | str | None = None,
        max_history: int = 10,
        system_prompt_template: str = "grid.j2",
        debug: bool = False,
        verbose: int = 2,
        use_inventory: bool = True,
        artifact_creation: bool = True,
        food_mechanism: bool = True,
        energy_death: bool | None = None,
        external_actions: bool = False,
        hidden_external_keys: list[str] | None = None,
        scenario_specific_instructions: str = "base",
        internal_memory_size: int = 500,
        max_message_length: int = 200,
        stay_action: dict | None = None,
        server_instructions: str = "",
        solo: bool = False,
        persona: str = "",
        spawn_allowed: bool = True,
        max_connections: int | None = None,
        finite_energy: bool = True,
        finite_lifespan: bool = True,
    ):
        self.verbose = verbose
        logging.getLogger(__name__).setLevel(logger_level(verbose))
        self.agent_name = agent_name
        self.agent_tag = agent_tag
        self.internal_memory_size = internal_memory_size
        self.use_inventory = use_inventory
        self.artifact_creation = artifact_creation
        self.food_mechanism = food_mechanism
        self.energy_death = food_mechanism if energy_death is None else energy_death
        self.finite_energy = finite_energy
        self.finite_lifespan = finite_lifespan
        # World's canonical pass-the-turn action, supplied by the env (social_graph
        # -> `noop`, grid/graph -> `move(direction=stay)`). Used as the parse-
        # failure fallback so it is always a valid action for the world type.
        self.stay_action = stay_action or {
            "action": "move",
            "message": "",
            "params": {"direction": "stay"},
        }
        self.external_actions = external_actions
        # Top-level keys stripped from external-server responses before they are
        # shown to the LLM (e.g. a "difficulty" field); the runner keeps the
        # full response for logging. Sourced from servers.json `hidden_keys`.
        self.hidden_external_keys = list(hidden_external_keys or [])
        self.scenario_specific_instructions = scenario_specific_instructions
        self.max_message_length = max_message_length
        self.spawn_allowed = spawn_allowed
        self.max_connections = max_connections
        # Instructions the connected MCP servers sent at connection time.
        self.server_instructions = server_instructions
        # The population can never exceed 1 (env.max_agents == 1): social
        # prompt content and the broadcast side-channel are suppressed —
        # there is nobody to hear (v15: 491 broadcasts to an empty world).
        self.solo = solo
        self.persona = persona

        self.system_prompt_template = system_prompt_template
        self.history = []
        self.internal_memory = ""
        log_dir = Path(log_dir) / "agent_logs" if log_dir is not None else log_dir
        self.logger = (
            logger
            if logger is not None
            else AgentLogger(agent_tag=self.agent_tag, log_dir=log_dir)
        )
        self.genome = Ocean5Genome().random() if genome is None else genome
        self.max_history = max_history
        self.internal_memory_encoder = tiktoken.get_encoding("cl100k_base")

        self.logger.save_genome(agent_tag=self.agent_tag, genome=self.genome.as_dict())
        self.debug = debug
        self.update_system_prompt()

    def update_system_prompt(self):
        """Render the system instructions for this agent's current resources."""
        scenario_instructions = resolve_instructions(
            self.scenario_specific_instructions,
            finite_energy=self.finite_energy,
            finite_lifespan=self.finite_lifespan,
        )
        self.system_prompt = render_system_prompt(
            self.system_prompt_template,
            agent_name=self.agent_name,
            use_internal_memory=self.use_internal_memory,
            use_inventory=self.use_inventory,
            artifact_creation=self.artifact_creation,
            food_mechanism=self.food_mechanism,
            energy_death=self.energy_death,
            finite_energy=self.finite_energy,
            finite_lifespan=self.finite_lifespan,
            external_actions=self.external_actions,
            scenario_specific_instructions=scenario_instructions,
            internal_memory_size=self.internal_memory_size,
            max_message_length=self.max_message_length,
            genome_string=self.genome.as_string(),
            debug=self.debug,
            server_instructions=self.server_instructions,
            solo=self.solo,
            persona=self.persona,
            spawn_allowed=self.spawn_allowed,
            max_connections=self.max_connections,
        ).strip()

    async def select_action_async(
        self,
        obs: dict,
        available_actions: dict,
        reward: int,
        info: dict | None,
        time: int,
        chat_params: dict,
        client: AgentClient,
        max_attempts=5,
    ) -> Dict[str, str]:
        """Async version of select_action — uses get_response_async() for non-blocking LLM calls."""
        # Run CPU-bound prompt building in a thread so the event loop stays
        # free to handle I/O while prompts are being assembled.
        formatted_obs, prompt = await asyncio.to_thread(
            self._build_prompt_sync, obs, available_actions, info
        )

        # Copy to avoid mutating the shared dict from llm_router.next()
        chat_params = (
            dict(chat_params) if chat_params is not None else {"model": "o4-mini"}
        )
        post_prompt = chat_params.pop("post_prompt", None)
        if post_prompt is not None:
            prompt += "\n\n" + post_prompt

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0.0
        action = None

        for attempt in range(max_attempts):
            resp = await client.get_response_async(
                messages=messages, chat_params=chat_params
            )

            # A provider can return an empty/None completion; treat it as a
            # failed attempt instead of crashing the retry loop.
            text = (resp.content or "").strip()  # type: ignore
            total_input_tokens += resp.input_tokens
            total_output_tokens += resp.output_tokens
            total_cost += resp.cost
            if not text:
                log.warning(
                    "Empty LLM reply for %s(%s); retrying (attempt %d)",
                    self.agent_name,
                    self.agent_tag,
                    attempt + 1,
                )
                continue
            log.debug(
                "++++++++++++++\n%s(%s)\nRESPONSE\n%s\n++++++++++++++",
                self.agent_name,
                self.agent_tag,
                text,
            )

            try:
                act, message, params, internal_memory, reasoning = (
                    self._parse_response(text, available_actions=available_actions)
                )
                self.history.append((formatted_obs, act, message, params, info))
                self.history = self.history[-self.max_history :] if self.max_history > 0 else []
                self.internal_memory = internal_memory

                action = {
                    "action": act,
                    "message": message,
                    "params": params,
                    "source": "llm",
                }
                if reasoning is not None:
                    action["reasoning"] = reasoning
                break

            except Exception as e:
                log.warning(
                    "Error occurred while parsing response of agent %s(%s): %s",
                    self.agent_name,
                    self.agent_tag,
                    e,
                )
                log.debug("Retrying attempt: %d", attempt + 1)
                error_msg = ERROR_MSG.render(
                    error=e,
                    action_keys=available_actions.keys(),
                    use_internal_memory=self.use_internal_memory,
                    solo=self.solo,
                )
                no_reason_text = (
                    text.split("</think>")[1] if "</think>" in text else text
                )
                messages.append({"role": "assistant", "content": no_reason_text})
                messages.append({"role": "user", "content": error_msg.strip()})

        if action is None:
            log.warning(
                "LLM failed to return a valid response after %d attempts. STAYING",
                max_attempts,
            )
            # Fall back to the world's canonical pass-the-turn action (see
            # self.stay_action). Hardcoding `move` here made social_graph agents
            # emit an unknown `move` action on every parse failure.
            move = self.stay_action.get("action", "move")
            message = self.stay_action.get("message", "")
            params = self.stay_action.get("params", {})
            self.history.append((formatted_obs, move, message, params, info))
            self.history = self.history[-self.max_history :] if self.max_history > 0 else []
            action = {
                "action": move,
                "message": message,
                "params": params,
                "source": "default",
                "default_reason": "parse_failure",
            }

        # Carry this step's USD cost and token usage out with the action so the
        # runner can aggregate a per-timestep total across all agents (costs.csv).
        action["cost"] = total_cost
        action["input_tokens"] = total_input_tokens
        action["output_tokens"] = total_output_tokens

        self.logger.log(
            agent_name=self.agent_name,
            agent_tag=self.agent_tag,
            available_actions=available_actions,
            observation=obs,
            action=action,
            internal_memory=self.internal_memory,
            time=str(time),
            input_prompt=prompt,
        )
        return action

    @staticmethod
    def _response_object(text: str) -> tuple[dict, str | None]:
        response = text.split("</think>")
        if len(response) == 2:
            reasoning = response[0].strip()
            response = response[1]
        else:
            response = text
            reasoning = None

        # Get json output
        CODE_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)
        FIRST_JSON = re.compile(r"\{.*\}", re.S)

        visible = response.strip()
        json_str = CODE_FENCE.search(visible)
        if json_str:
            json_str = json_str.group(1).strip()
        else:
            json_str = FIRST_JSON.search(visible)
            assert json_str is not None, "No JSON object found in response."
            json_str = json_str.group(0).strip()

        try:
            json_obj = json.loads(json_str)
        except json.JSONDecodeError:
            json_str = re.sub(
                r"([{\s,])([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', json_str
            )
            json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
            json_obj = json.loads(json_str)

        if not isinstance(json_obj, dict):
            raise ValueError("The response must be a JSON object.")
        return {k.lower(): v for k, v in json_obj.items()}, reasoning

    def _parse_response(
        self, text: str, available_actions: dict
    ) -> Tuple[str, str, Dict, str, str | None]:
        data, reasoning = self._response_object(text)
        action = data.get("action", "")
        message = data.get("message", "")
        params = data.get("params", {})
        internal_memory = data.get("internal_memory", "")
        internal_memory = self.validate_internal_memory(internal_memory)

        if not isinstance(action, str):
            raise ValueError("Field 'action' must be a string.")
        if action not in available_actions:
            raise ValueError(
                f"Incorrect action. Expected one of {list(available_actions.keys())}; got '{action}'."
            )

        if not isinstance(message, str):
            message = str(message)

        if not isinstance(params, dict):
            raise ValueError("Field 'params' must be a JSON object.")
        expected = available_actions[action].get("params", {})
        exp_keys = set(expected.keys()) if isinstance(expected, dict) else set()
        # MCP omissions must survive to the server so its own defaults apply.
        # Internal actions retain their existing empty-string optional values.
        external_schema = available_actions[action].get("input_schema")
        optional = (
            exp_keys - set(external_schema.get("required", []))
            if isinstance(external_schema, dict)
            else set(available_actions[action].get("optional", []) or [])
        )
        required = exp_keys - optional
        sent = set(params.keys())
        if (required - sent) or (sent - exp_keys):
            missing = sorted(required - sent)
            extra = sorted(sent - exp_keys)
            raise ValueError(
                f"Incorrect action parameters for '{action}'. "
                + (f"MISSING keys: {missing}. " if missing else "")
                + (f"UNEXPECTED keys: {extra}. " if extra else "")
                + f"Send at least these keys: {sorted(required)}"
                + (f"; optional: {sorted(optional)}" if optional else "")
                + (
                    ". Omit optional fields to use the tool's defaults; "
                    "do not substitute empty strings for missing values."
                    if isinstance(external_schema, dict)
                    else ' (use "" for fields you have nothing for).'
                )
            )
        if not isinstance(external_schema, dict):
            for key in exp_keys - sent:
                params[key] = ""

        return action, message, params, internal_memory, reasoning

    def close(self):
        self.logger.close()

    def get_state_ckpt(self) -> dict:
        state_ckpt = {
            "name": self.agent_name,
            "tag": self.agent_tag,
            "type": "LLMAgent",
            "system_prompt": self.system_prompt,
            "system_prompt_template": self.system_prompt_template,
            "use_inventory": self.use_inventory,
            "artifact_creation": self.artifact_creation,
            "food_mechanism": self.food_mechanism,
            "energy_death": self.energy_death,
            "finite_energy": self.finite_energy,
            "finite_lifespan": self.finite_lifespan,
            "external_actions": self.external_actions,
            "hidden_external_keys": self.hidden_external_keys,
            "scenario_specific_instructions": self.scenario_specific_instructions,
            "genome": self.genome.as_dict(),
            "genome_class": f"{self.genome.__class__.__module__}:{self.genome.__class__.__name__}",
            "internal_memory_size": self.internal_memory_size,
            "max_history": self.max_history,
            "max_message_length": self.max_message_length,
            "verbose": self.verbose,
            "debug": self.debug,
            "internal_memory": self.internal_memory,
            "history": self.history,
            "log_dir": str(self.logger.log_dir),
            "solo": self.solo,
            "persona": self.persona,
            "spawn_allowed": self.spawn_allowed,
            "max_connections": self.max_connections,
        }
        return state_ckpt

    def set_state_ckpt(self, state_ckpt: dict):
        self.agent_name = state_ckpt["name"]
        self.agent_tag = state_ckpt["tag"]
        self.system_prompt = state_ckpt["system_prompt"]
        self.finite_energy = state_ckpt["finite_energy"]
        self.finite_lifespan = state_ckpt["finite_lifespan"]
        self.system_prompt_template = state_ckpt["system_prompt_template"]
        self.use_inventory = state_ckpt["use_inventory"]
        self.artifact_creation = state_ckpt["artifact_creation"]
        self.food_mechanism = state_ckpt["food_mechanism"]
        self.energy_death = state_ckpt.get("energy_death", self.food_mechanism)
        self.external_actions = state_ckpt.get("external_actions", False)
        self.hidden_external_keys = state_ckpt.get("hidden_external_keys", [])
        self.scenario_specific_instructions = state_ckpt["scenario_specific_instructions"]
        self.max_history = state_ckpt["max_history"]
        self.max_message_length = state_ckpt["max_message_length"]
        self.verbose = state_ckpt["verbose"]
        self.debug = state_ckpt["debug"]
        self.internal_memory_size = state_ckpt["internal_memory_size"]
        self.persona = state_ckpt.get("persona", "")
        self.internal_memory = self.validate_internal_memory(state_ckpt.get("internal_memory", ""))
        self.history = state_ckpt["history"]
        self.solo = state_ckpt.get("solo", False)
        self.spawn_allowed = state_ckpt.get("spawn_allowed", self.spawn_allowed)
        self.max_connections = state_ckpt.get("max_connections", self.max_connections)

        genome_cls_spec = state_ckpt.get("genome_class")
        if genome_cls_spec:
            mod_name, cls_name = genome_cls_spec.split(":")
            mod = importlib.import_module(mod_name)
            genome_cls = getattr(mod, cls_name)
        else:
            raise ValueError("Genome class specification missing in checkpoint.")
        self.genome = genome_cls().from_dict(state_ckpt["genome"])
        log_dir = Path(state_ckpt["log_dir"])
        self.logger = AgentLogger(agent_tag=self.agent_tag, log_dir=log_dir)
        # Resave the genome as during init the random genome is saved
        self.logger.save_genome(agent_tag=self.agent_tag, genome=self.genome.as_dict())
