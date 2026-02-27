# survival_parallel_llm_agent.py

import importlib
import json
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import tiktoken

from core.agents.agent_logger import AgentLogger
from core.agents.prompt_templates import (
    AGENT_PROMPT,
    AVAILABLE_EX_MOTIVATIONS,
    DEBUG_PROMPT,
    ERROR_MSG,
    MOTIVATIONS,
    OBS_STYLE,
    SYS_PROMPT,
)
from core.genome.base_genome import Genome
from core.genome.ocean_5 import Genome as Ocean5Genome
from core.utils.llm_client import AgentClient


class LLMAgent:
    def __init__(
        self,
        agent_name: str,
        agent_tag: str,
        system_prompt: str | None = None,
        logger: AgentLogger | None = None,
        genome: Genome | None = None,
        log_dir: Path | str | None = None,
        max_history: int = 50,
        obs_style: str = "list",
        debug: bool = False,
        verbose: int = 2,
        use_internal_memory: bool = True,
        use_inventory: bool = True,
        artifact_creation: bool = True,
        food_mechanism: bool = True,
        exogenous_motivation: str = "base",
        internal_memory_size: int = 150,
    ):
        """
        name: the agent’s name in the env
        system_prompt: high-level instructions for the LLM
        model: which chat model to call
        """
        self.verbose = verbose
        self.agent_name = agent_name
        self.agent_tag = agent_tag
        self.use_internal_memory = use_internal_memory
        self.internal_memory_size = internal_memory_size
        self.use_inventory = use_inventory
        self.artifact_creation = artifact_creation
        self.food_mechanism = food_mechanism
        self.exogenous_motivation = exogenous_motivation

        assert obs_style in OBS_STYLE, (
            f"Obs style {obs_style} invalid - Available: {list(OBS_STYLE.keys())}"
        )
        self.obs_style = obs_style
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

        try:
            exogenous_motivation = MOTIVATIONS[self.exogenous_motivation]
        except KeyError:
            raise ValueError(
                f"Invalid exogenous_motivation: {self.exogenous_motivation}"
            )

        if system_prompt:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = SYS_PROMPT.render(
                agent_name=self.agent_name,
                short_obs_descr=OBS_STYLE[self.obs_style]["short"],
                obs_style=self.obs_style,
                detailed_obs_descr=OBS_STYLE[self.obs_style]["detail"],
                use_internal_memory=self.use_internal_memory,
                use_inventory=self.use_inventory,
                artifact_creation=self.artifact_creation,
                food_mechanism=self.food_mechanism,
                exogenous_motivation=exogenous_motivation,
                internal_memory_size=self.internal_memory_size,
            ).strip()

        if self.debug:
            self.system_prompt += "\n" + DEBUG_PROMPT.strip()

    def select_action(
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
        """
        obs: {"grid": np.ndarray of shape (2r+1,2r+1) of str,
            "message": dict sender->str or vector}
        returns (action, message)
        """
        formatted_obs = self._format_observation(obs)
        prompt = self._make_prompt(
            formatted_obs=formatted_obs,
            internal_memory=self.internal_memory,
            info=info,
            available_actions=available_actions,
        )

        chat_params = chat_params if chat_params is not None else {"model": "o4-mini"}
        post_prompt = chat_params.pop("post_prompt", None)
        if post_prompt is not None:
            prompt += "\n\n" + post_prompt

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        total_input_tokens = 0
        total_output_tokens = 0
        # Get action with retries on parsing errors
        # ================================
        for attempt in range(max_attempts):
            action = None
            resp = client.get_response(messages=messages, chat_params=chat_params)

            text = resp.content.strip()  # type: ignore
            total_input_tokens += resp.input_tokens
            total_output_tokens += resp.output_tokens
            if self.verbose >= 2:
                print("++++++++++++++")
                print(f"{self.agent_name}({self.agent_tag})")
                print("RESPONSE")
                print(text)
                print("++++++++++++++")

            # Parse action
            # ---------------------------
            try:
                act, message, params, internal_memory, reasoning = self._parse_response(
                    text, available_actions=available_actions
                )
                self.history.append((formatted_obs, act, message, params, info))
                self.history = self.history[-self.max_history :]
                self.internal_memory = internal_memory

                action = {"action": act, "message": message, "params": params}
                if reasoning is not None:
                    action["reasoning"] = reasoning
                break

            except Exception as e:
                if self.verbose >= 1:
                    print(
                        f"Error occurred while parsing response of agent {self.agent_name}({self.agent_tag}): {e}"
                    )
                if self.verbose >= 2:
                    print(f"Retrying attempt: {attempt + 1}")
                error_msg = ERROR_MSG.render(
                    error=e,
                    action_keys=available_actions.keys(),
                    use_internal_memory=self.internal_memory,
                )
                no_reason_text = (
                    text.split("</think>")[1] if "</think>" in text else text
                )
                messages.append({"role": "assistant", "content": no_reason_text})
                messages.append({"role": "user", "content": error_msg.strip()})
            # ---------------------------
        # ================================

        if action is None:
            if self.verbose >= 1:
                print(
                    f"LLM failed to return a valid response after {max_attempts} attempts. STAYING"
                )
            move = "move"
            message = ""
            params = {"direction": "stay"}
            self.history.append((formatted_obs, move, message, params, info))
            action = {"action": move, "message": message, "params": params}

        # Log
        # ---------------------------
        with open(self.logger.log_dir / "token_counts.jsonl", "a") as f:
            json.dump(
                {
                    "timestep": time,
                    "agent_tag": self.agent_tag,
                    "total_input_tokens": total_input_tokens,
                    "total_output_tokens": total_output_tokens,
                },
                f,
            )
            f.write("\n")

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
        # ---------------------------
        return action

    @staticmethod
    def _make_grid(obs_dict: dict, grid_radius: int):
        grid_size = 2 * grid_radius + 1
        formatted_grid = np.full((grid_size, grid_size), ".", dtype="U32")

        for dx in range(-grid_radius, grid_radius + 1):
            for dy in range(-grid_radius, grid_radius + 1):
                if (dx, dy) in obs_dict:
                    gx = dx + grid_radius
                    gy = dy + grid_radius
                    formatted_grid[gx, gy] = " | ".join(map(str, obs_dict[(dx, dy)]))
        return formatted_grid

    def _format_grid(self, obs_dict: dict, grid_radius: int):
        formatted_grid = self._make_grid(obs_dict=obs_dict, grid_radius=grid_radius)
        return "\n".join(" ".join(row) for row in formatted_grid)

    def _format_grid_with_directions(self, obs_dict: dict, grid_radius: int):
        grid_size = 2 * grid_radius + 1
        formatted_grid = self._make_grid(obs_dict=obs_dict, grid_radius=grid_radius)
        direction_grid = np.full((grid_size + 2, grid_size + 2), " ", dtype="U32")

        direction_grid[1:-1, 1:-1] = formatted_grid
        direction_grid[0, 1:-1] = "up"
        direction_grid[-1, 1:-1] = "down"
        direction_grid[1:-1, 0] = "left"
        direction_grid[1:-1, -1] = "right"

        return "\n".join(" ".join(row) for row in formatted_grid)

    def _format_list(self, obs_dict: dict):
        formatted = ""
        for coords, content in obs_dict.items():
            rx, ry = coords
            if (rx, ry) == (0, 0):
                content = ["yourself"] + content
            list_coords = f"({ry}, {-rx})"
            list_content = " | ".join(map(str, content))
            formatted += f"{list_coords}: {list_content}\n "
        return formatted

    def _format_messages(self, msg_dict):
        if not msg_dict:
            return "<none>"
        msg_lines = []
        for sender, msg in msg_dict.items():
            msg_str = msg if isinstance(msg, str) else str(np.array(msg))
            msg_lines.append(f"{sender}: {msg_str}")
        return " \n".join(msg_lines)

    def _format_observation(self, obs: dict) -> dict:
        """This selects only the features of observation that the agent sees"""
        formatted_obs = {}
        if self.obs_style == "grid":
            formatted_obs["observation"] = self._format_grid_with_directions(
                obs_dict=obs["observation"], grid_radius=obs["vision_radius"]
            )
        elif self.obs_style == "list":
            formatted_obs["observation"] = self._format_list(obs["observation"])
        formatted_obs["message"] = self._format_messages(obs["message"])
        if obs["inventory"]:
            formatted_obs["inventory"] = "\n".join(obs["inventory"])
        else:
            formatted_obs["inventory"] = "<empty>"
        formatted_obs["energy"] = obs["energy"]
        formatted_obs["time"] = obs["time"]
        return formatted_obs

    def _make_prompt(self, formatted_obs, available_actions, internal_memory, info):
        # Build history section
        history_txt = ""
        if self.history:
            history_txt = f"=== History (last {min(len(self.history), self.max_history)} steps) ===\n"
            for i, (
                past_obs,
                past_action,
                past_msg,
                past_params,
                past_info,
            ) in enumerate(self.history[-self.max_history :], 1):
                history_txt += (
                    f"Step {i}:\n"
                    f"\tEnergy: {past_obs['energy']}\n"
                    f"\tIncoming msgs: {past_obs['message']}\n"
                    f"\tObservation:\n{past_obs['observation']}\n"
                )
                if past_info is not None and len(past_info):
                    history_txt += f"\tAdditional info:\n{past_info}\n"
                history_txt += (
                    f"\tAction taken: {past_action}\n"
                    f"\tAction parameters: {past_params}\n"
                    f"\tSent message: {past_msg or '<none>'}\n\n"
                )

        additional_info = ""
        if info is not None and len(info):
            info_list = [f"{key}: {value}" for key, value in info.items()]
            info_list = "\n".join(info_list)
            print(f"ADDING INFO: {info}")
            additional_info += f"""
=== Additional info from the environment ===
{info_list}
"""

        prompt = AGENT_PROMPT.render(
            history=history_txt,
            genome=self.genome.as_string(),
            observation=formatted_obs["observation"],
            messages=formatted_obs["message"],
            energy=formatted_obs["energy"],
            time=formatted_obs["time"],
            inventory=formatted_obs["inventory"],
            additional_info=additional_info.strip(),
            actions=json.dumps(available_actions, indent=4),
            action_keys=", ".join(available_actions.keys()),
            memory=internal_memory,
            use_internal_memory=self.use_internal_memory,
            use_inventory=self.use_inventory,
            food_mechanism=self.food_mechanism,
        )
        return prompt

    def _parse_response(
        self, text: str, available_actions: dict
    ) -> Tuple[str, str, Dict, str, str | None]:
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

        # tolerant key casing
        data = {k.lower(): v for k, v in json_obj.items()}
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
        if set(params.keys()) != exp_keys:
            raise ValueError(
                f"Incorrect action parameters. Expected keys {exp_keys}; got {set(params.keys())}. Params: {params}"
            )

        return action, message, params, internal_memory, reasoning

    def validate_internal_memory(self, internal_memory: str) -> str:
        """Ensure internal memory is within token limits."""
        tokens = self.internal_memory_encoder.encode(str(internal_memory))
        if len(tokens) > self.internal_memory_size + 100:
            # Truncate to last (internal_memory_size + 100) tokens
            # We cut to (internal_memory_size + 100) rather than internal_memory_size to leave some margin
            tokens = tokens[-(self.internal_memory_size + 100) :]
            internal_memory = self.internal_memory_encoder.decode(tokens)
        return internal_memory

    def close(self):
        self.logger.close()

    def get_state_ckpt(self) -> dict:
        state_ckpt = {
            "name": self.agent_name,
            "tag": self.agent_tag,
            "type": "LLMAgent",
            "system_prompt": self.system_prompt,
            "obs_style": self.obs_style,
            "use_internal_memory": self.use_internal_memory,
            "use_inventory": self.use_inventory,
            "artifact_creation": self.artifact_creation,
            "food_mechanism": self.food_mechanism,
            "exogenous_motivation": self.exogenous_motivation,
            "genome": self.genome.as_dict(),
            "genome_class": f"{self.genome.__class__.__module__}:{self.genome.__class__.__name__}",
            "max_history": self.max_history,
            "verbose": self.verbose,
            "debug": self.debug,
            "internal_memory": self.internal_memory,
            "history": self.history,
            "log_dir": str(self.logger.log_dir),
        }
        return state_ckpt

    def set_state_ckpt(self, state_ckpt: dict):
        self.agent_name = state_ckpt["name"]
        self.agent_tag = state_ckpt["tag"]
        self.system_prompt = state_ckpt["system_prompt"]
        self.obs_style = state_ckpt["obs_style"]
        self.use_internal_memory = state_ckpt["use_internal_memory"]
        self.use_inventory = state_ckpt["use_inventory"]
        self.artifact_creation = state_ckpt["artifact_creation"]
        self.food_mechanism = state_ckpt["food_mechanism"]
        self.exogenous_motivation = state_ckpt["exogenous_motivation"]
        self.max_history = state_ckpt["max_history"]
        self.verbose = state_ckpt["verbose"]
        self.debug = state_ckpt["debug"]
        self.internal_memory = state_ckpt["internal_memory"]
        self.history = state_ckpt["history"]

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


if __name__ == "__main__":
    agent = LLMAgent(
        agent_name="TestAgent",
        agent_tag="test_agent_001",
        log_dir="./logs",
        debug=True,
    )

    ckpt = agent.get_state_ckpt()
    with open("agent_ckpt.pkl", "wb") as f:
        import pickle

        pickle.dump(ckpt, f)

    with open("agent_ckpt.pkl", "rb") as f:
        import pickle

        loaded_ckpt = pickle.load(f)
    agent2 = LLMAgent(
        agent_name="Temp",
        agent_tag="temp_agent_001",
    )
    agent2.set_state_ckpt(loaded_ckpt)
    assert agent2.agent_name == "TestAgent"
    assert agent2.genome.as_dict() == agent.genome.as_dict()
    print("State save/load works correctly.")
