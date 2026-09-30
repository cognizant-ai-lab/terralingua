# Mock (non-LLM) agent for debugging environments and action plumbing.

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tiktoken

from terralingua.agents.agent_logger import AgentLogger
from terralingua.agents.agent_mixin import AgentMixin
from terralingua.genome.base_genome import Genome
from terralingua.genome.no_traits import Genome as NoTraitsGenome


class HumanAgent(AgentMixin):
    def __init__(
        self,
        agent_name: str,
        agent_tag: str,
        logger: AgentLogger | None = None,
        genome: Genome | None = None,
        log_dir: Path | str | None = None,
        max_history: int = 10,
        use_inventory: bool = True,
        artifact_creation: bool = True,
        food_mechanism: bool = True,
        verbose: int = 1,
        internal_memory_size: int = 500,
    ):
        self.agent_name = agent_name
        self.agent_tag = agent_tag
        self.internal_memory_size = internal_memory_size
        self.internal_memory_encoder = tiktoken.get_encoding("cl100k_base")
        self.use_inventory = use_inventory
        self.artifact_creation = artifact_creation
        self.food_mechanism = food_mechanism
        self.verbose = verbose

        self.history: List[Tuple[dict, str, str, dict, dict | None]] = []
        self.internal_memory: str = ""
        self.max_history = max_history

        log_dir = Path(log_dir) / "agent_logs" if log_dir is not None else log_dir
        self.logger = (
            logger
            if logger is not None
            else AgentLogger(agent_tag=self.agent_tag, log_dir=log_dir)
        )
        self.genome = NoTraitsGenome() if genome is None else genome
        self.logger.save_genome(agent_tag=self.agent_tag, genome=self.genome.as_dict())

    def _select_action(
        self,
        obs: dict,
        available_actions: dict,
        reward: int,
        info: dict | None,
        time: int,
        chat_params: dict | None = None,
        client=None,
        max_attempts: int = 1,
    ) -> Dict[str, str]:
        formatted_obs = self._format_observation(obs, info)

        action_key = self._prompt_for_action(formatted_obs, available_actions)
        params = self._prompt_for_params(action_key, available_actions)
        message = self._prompt_for_message()

        if self.use_internal_memory:
            self.internal_memory = self.validate_internal_memory(
                self._update_memory(formatted_obs, action_key, params)
            )

        self.history.append((formatted_obs, action_key, message, params, info))
        self.history = self.history[-self.max_history :] if self.max_history > 0 else []

        action_obj = {"action": action_key, "message": message, "params": params}

        self.logger.log(
            agent_name=self.agent_name,
            agent_tag=self.agent_tag,
            available_actions=available_actions,
            observation=obs,
            action=action_obj,
            internal_memory=self.internal_memory,
            time=str(time),
            input_prompt="N/A (human input)",
        )
        if self.verbose >= 2:
            print(f"[MockAgent:{self.agent_name}] chose {action_obj}")
        return action_obj

    async def select_action_async(
        self,
        obs: dict,
        available_actions: dict,
        reward: int,
        info: dict | None,
        time: int,
        chat_params: dict | None = None,
        client=None,
        max_attempts: int = 1,
    ) -> Dict[str, str]:
        """Async version — runs blocking input() calls in a worker thread."""
        return await asyncio.to_thread(
            self._select_action,
            obs=obs,
            available_actions=available_actions,
            reward=reward,
            info=info,
            time=time,
            chat_params=chat_params,
            client=client,
            max_attempts=max_attempts,
        )

    def _format_observation(self, obs: dict, infos: dict | None) -> dict:
        formatted = {}
        vision = obs.get("observation", {})
        formatted["observation"] = vision
        formatted["message"] = obs.get("incoming_broadcasts", {})
        if "incoming_dms" in obs:
            formatted["direct_messages"] = obs.get("incoming_dms", {})
        formatted["inventory"] = obs.get("inventory", [])
        formatted["energy"] = obs.get("energy")
        formatted["time"] = obs.get("time")
        if "external_response" in obs:
            formatted["external_response"] = obs["external_response"]
        if infos:
            formatted["info"] = infos
        return formatted

    def _prompt_for_action(self, formatted_obs: dict, available_actions: dict) -> str:
        print("\n=== Manual Action Selection ===")
        print(f"Agent: {self.agent_name}  Tag: {self.agent_tag}")
        if np.isfinite(formatted_obs["energy"]):
            print(f"Energy: {formatted_obs['energy']}")
        if np.isfinite(formatted_obs["time"]):
            print(f"Remaining lifespan (steps): {formatted_obs['time']}")
        inv = formatted_obs.get("inventory") or []
        if isinstance(inv, list):
            print(f"Inventory: {inv if inv else '<empty>'}")
        else:
            print(f"Inventory: {inv}")
        msgs = formatted_obs.get("message", {})
        if msgs:
            print("Incoming broadcasts:")
            for s, m in msgs.items():
                print(f"  {s}: {m}")
        else:
            print("Incoming broadcasts: <none>")
        if "direct_messages" in formatted_obs:
            dms = formatted_obs.get("direct_messages", {})
            if dms:
                print("Direct messages:")
                for s, m in dms.items():
                    print(f"  {s}: {m}")
            else:
                print("Direct messages: <none>")
        print("Raw observation (relative cells):")
        vision = formatted_obs.get("observation", {})
        for coords, contents in vision.items():
            print(f"  {coords}: {contents}")

        if "external_response" in formatted_obs:
            print(f"External response: \n{formatted_obs['external_response']}")
        if "info" in formatted_obs:
            print("Additional info:")
            for k, v in formatted_obs["info"].items():
                print(f"  {k}: {v}")

        print("\nAvailable actions:")
        for k, spec in available_actions.items():
            params = spec.get("params", {})
            print(f" - {k} params={params}")

        while True:
            choice = input("Enter action name > ").strip()
            if choice in available_actions:
                return choice
            print("Invalid action. Try again.")

    def _prompt_for_params(self, action_key: str, available_actions: dict) -> dict:
        spec = available_actions[action_key].get("params", {})
        if not isinstance(spec, dict) or not spec:
            return {}
        params = {}
        print(f"\nProvide params for action '{action_key}' (press Enter for default):")
        for k, v in spec.items():
            default = None
            options = None
            expected_type = None
            if isinstance(v, list) and v:
                options = v
                default = v[0]
            elif isinstance(v, dict):
                _opts = v.get("choices") or v.get("options")
                if isinstance(_opts, list) and _opts:
                    options = _opts
                    default = options[0]
                if "default" in v:
                    default = v["default"]
                expected_type = v.get("type")
            elif isinstance(v, str):
                default = ""
            if k == "direction" and default is None:
                default = "stay"

            prompt = f"  {k}"
            if options:
                prompt += f" options={options}"
            if default is not None:
                prompt += f" [default={default}]"
            val = input(prompt + " > ").strip()
            if val == "" and default is not None:
                val = default
            # basic type casting
            if expected_type == "int":
                try:
                    val = int(val)
                except ValueError:
                    val = 0
            params[k] = val
        return params

    def _prompt_for_message(self) -> str:
        val = input("Broadcast message (blank for none) > ").strip()
        return val

    def _update_memory(self, formatted_obs: dict, action_key: str, params: dict) -> str:
        recent_msgs = list(formatted_obs.get("message", {}).keys())
        mem = {
            "steps_recorded": len(self.history) + 1,
            "last_action": action_key,
            "last_params": params,
            "seen_senders": recent_msgs,
        }
        if np.isfinite(formatted_obs["energy"]):
            mem["energy"] = formatted_obs["energy"]
        if np.isfinite(formatted_obs["time"]):
            mem["time_left"] = formatted_obs["time"]
        return f"{mem}"

    def close(self):
        self.logger.close()

    def get_state_ckpt(self) -> dict:
        state_ckpt = {
            "name": self.agent_name,
            "tag": self.agent_tag,
            "type": "HumanAgent",
            "internal_memory_size": self.internal_memory_size,
            "use_inventory": self.use_inventory,
            "artifact_creation": self.artifact_creation,
            "food_mechanism": self.food_mechanism,
            "genome": self.genome.as_dict(),
            "genome_class": f"{self.genome.__class__.__module__}:{self.genome.__class__.__name__}",
            "max_history": self.max_history,
            "verbose": self.verbose,
            "internal_memory": self.internal_memory,
            "history": self.history,
            "log_dir": str(self.logger.log_dir),
        }
        return state_ckpt

    def set_state_ckpt(self, state_ckpt: dict):
        self.agent_name = state_ckpt["name"]
        self.agent_tag = state_ckpt["tag"]
        self.use_inventory = state_ckpt["use_inventory"]
        self.artifact_creation = state_ckpt["artifact_creation"]
        self.food_mechanism = state_ckpt["food_mechanism"]
        self.max_history = state_ckpt["max_history"]
        self.verbose = state_ckpt["verbose"]
        self.internal_memory_size = state_ckpt.get("internal_memory_size", self.internal_memory_size)
        self.internal_memory = self.validate_internal_memory(state_ckpt.get("internal_memory", ""))
        self.history = state_ckpt["history"]

        genome_cls_spec = state_ckpt.get("genome_class")
        if genome_cls_spec:
            mod_name, cls_name = genome_cls_spec.split(":")
            try:
                mod = importlib.import_module(mod_name)
                genome_cls = getattr(mod, cls_name)
            except Exception:
                genome_cls = Genome  # fallback
        else:
            genome_cls = Genome  # legacy checkpoints
        self.genome = genome_cls().from_dict(state_ckpt["genome"])
        log_dir = Path(state_ckpt["log_dir"])
        self.logger = AgentLogger(agent_tag=self.agent_tag, log_dir=log_dir)
        # Resave the genome as during init the random genome is saved
        self.logger.save_genome(agent_tag=self.agent_tag, genome=self.genome.as_dict())
