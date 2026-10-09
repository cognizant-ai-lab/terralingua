"""
RemoteAgent: an agent whose decisions are made by an external client over WebSocket.

The runner calls build_payload() to get the prompt dict to send to the client.
The client responds with {action, message, params} JSON, which the ConnectionManager
stores in its action buffer. The runner drains the buffer after the window expires.

This agent does NOT have a select_action() method — it is not dispatched via
ThreadPoolExecutor. Instead, the runner handles remote agents separately:
  1. Build payloads for all remote agents
  2. Send via ConnectionManager.send_all_prompts()
  3. Sleep for the window
  4. Drain action buffer with ConnectionManager.drain_action_buffer()
"""

import asyncio
import importlib
import logging
from pathlib import Path
from typing import Optional

import tiktoken

from terralingua.agents.agent_logger import AgentLogger
from terralingua.agents.agent_mixin import AgentMixin
from terralingua.agents.prompt_templates import render_system_prompt
from terralingua.genome.base_genome import Genome
from terralingua.genome.ocean_5 import Genome as Ocean5Genome

logger = logging.getLogger(__name__)

_STAY_ACTION = {"action": "move", "message": "", "params": {"direction": "stay"}}


class RemoteAgent(AgentMixin):
    """
    Drop-in agent for externally-controlled participants.

    Prompt assembly is identical to LLMAgent (same Jinja templates).
    The motivation_prompt replaces the built-in scenario_specific_instructions text.
    """

    def __init__(
        self,
        agent_name: str,
        agent_tag: str,
        motivation_prompt: str,
        logger: Optional[AgentLogger] = None,
        log_dir: Optional[Path | str] = None,
        genome: Optional[Genome] = None,
        max_history: int = 10,
        system_prompt_template: str = "grid.j2",
        use_inventory: bool = True,
        artifact_creation: bool = True,
        food_mechanism: bool = True,
        show_nearby_artifact_names: bool = True,
        external_actions: bool = False,
        internal_memory_size: int = 500,
        verbose: int = 1,
        connection_manager=None,
        timeout_s: float = 30.0,
        server_instructions: str = "",
        spawn_allowed: bool = True,
        max_connections: int | None = None,
        max_message_length: int = 200,
        energy_death: bool | None = None,
        energy_upkeep: int | None = None,
        finite_energy: bool = True,
        finite_lifespan: bool = True,
    ):
        self.agent_name = agent_name
        self.agent_tag = agent_tag
        self.motivation_prompt = motivation_prompt
        self.agent_type: str = "remote_llm"
        self.connection_manager = connection_manager
        self.timeout_s = timeout_s
        self.system_prompt_template = system_prompt_template
        self.use_inventory = use_inventory
        self.artifact_creation = artifact_creation
        self.food_mechanism = food_mechanism
        self.show_nearby_artifact_names = show_nearby_artifact_names
        self._energy_upkeep_setting = None if energy_upkeep is None else int(energy_upkeep)
        self.energy_death = (food_mechanism or self.energy_upkeep > 0) if energy_death is None else energy_death
        self.finite_energy = finite_energy
        self.finite_lifespan = finite_lifespan
        self.external_actions = external_actions
        self.internal_memory_size = internal_memory_size
        self.verbose = verbose
        self.max_history = max_history
        self.server_instructions = server_instructions
        self.spawn_allowed = spawn_allowed
        self.max_connections = max_connections
        self.max_message_length = max_message_length

        self.genome = Ocean5Genome().random() if genome is None else genome
        self.history: list = []
        self.internal_memory: str = ""
        self.internal_memory_encoder = tiktoken.get_encoding("cl100k_base")

        log_dir = Path(log_dir) / "agent_logs" if log_dir is not None else log_dir
        self.logger = (
            logger
            if logger is not None
            else AgentLogger(agent_tag=self.agent_tag, log_dir=log_dir)
        )
        self.logger.save_genome(agent_tag=self.agent_tag, genome=self.genome.as_dict())

        self.update_system_prompt()

    @property
    def energy_upkeep(self) -> int:
        """Energy lost per step: the world's setting, else 1 with food_mechanism and 0 without."""
        setting = getattr(self, "_energy_upkeep_setting", None)
        if setting is None:
            return 1 if self.food_mechanism else 0
        return setting

    def update_system_prompt(self):
        """(Re-)render `self.system_prompt` from the current attribute values.

        Call after mutating any field that feeds the template — most commonly
        `motivation_prompt` or `genome` from the dashboard's update_agent path —
        so the prompt sent to the LLM reflects the latest state.
        """
        self.system_prompt = render_system_prompt(
            self.system_prompt_template,
            agent_name=self.agent_name,
            use_internal_memory=self.use_internal_memory,
            use_inventory=self.use_inventory,
            artifact_creation=self.artifact_creation,
            food_mechanism=self.food_mechanism,
            show_nearby_artifact_names=self.show_nearby_artifact_names,
            energy_death=self.energy_death,
            energy_upkeep=self.energy_upkeep,
            finite_energy=self.finite_energy,
            finite_lifespan=self.finite_lifespan,
            external_actions=self.external_actions,
            scenario_specific_instructions=self.motivation_prompt,
            internal_memory_size=self.internal_memory_size,
            genome_string=self.genome.as_string(),
            server_instructions=self.server_instructions,
            spawn_allowed=self.spawn_allowed,
            max_connections=self.max_connections,
            max_message_length=self.max_message_length,
        ).strip()

    def build_payload(
        self,
        obs: dict,
        available_actions: dict,
        reward: int,
        info: Optional[dict],
        time: int,
    ) -> dict:
        """
        Build the payload dict that gets sent to the external client over WebSocket.
        Contains both the rendered text prompts and the raw structured data.
        """
        formatted_obs = self._format_observation(obs)
        user_prompt = self._make_prompt(
            formatted_obs=formatted_obs,
            available_actions=available_actions,
            internal_memory=self.internal_memory,
            info=info,
        )
        obs_raw = {
            **obs,
            "observation": {
                f"{k[0]},{k[1]}": v for k, v in obs["observation"].items()
            },
        }
        if not self.finite_energy:
            obs_raw.pop("energy", None)
        if not self.finite_lifespan:
            obs_raw.pop("time", None)
        payload = {
            "step": time,
            "agent_tag": self.agent_tag,
            "system_prompt": self.system_prompt,
            "user_prompt": user_prompt,
            "available_actions": available_actions,
            "observation_raw": obs_raw,
        }
        return payload

    def record_action(self, obs: dict, action: dict, info: Optional[dict]):
        """
        Update history and internal_memory after an action is applied.
        Called by the runner after drain_action_buffer().
        """
        formatted_obs = self._format_observation(obs)
        act = action.get("action", "move")
        msg = action.get("message", "")
        params = action.get("params", {"direction": "stay"})
        internal_memory = action.get("internal_memory", "")
        self.internal_memory = self.validate_internal_memory(internal_memory)

        self.history.append((formatted_obs, act, msg, params, info))
        self.history = self.history[-self.max_history:] if self.max_history > 0 else []

    async def select_action_async(
        self,
        obs: dict,
        available_actions: dict,
        reward: int,
        info: Optional[dict],
        time: int,
        chat_params=None,  # noqa: ARG002 — uniform interface, ignored by RemoteAgent
        client=None,       # noqa: ARG002 — uniform interface, ignored by RemoteAgent
    ) -> dict:
        """Build payload → send over WebSocket → await response → return action."""
        cm = self.connection_manager
        if cm is None:
            return {**_STAY_ACTION, "source": "default", "default_reason": "no_connection"}

        payload = await asyncio.to_thread(
            self.build_payload, obs, available_actions, reward, info, time
        )
        await cm.send_prompt(self.agent_tag, payload)
        action = await cm.wait_for_response(self.agent_tag, self.timeout_s)
        self.record_action(obs, action, info)
        self.logger.log(
            agent_name=self.agent_name,
            agent_tag=self.agent_tag,
            observation=obs,
            available_actions=available_actions,
            action=action,
            time=str(time),
            internal_memory=self.internal_memory,
            input_prompt=payload.get("user_prompt", ""),
        )
        return action

    # ------------------------------------------------------------------
    # Checkpoint support
    # ------------------------------------------------------------------

    def get_state_ckpt(self) -> dict:
        return {
            "type": "RemoteAgent",
            "name": self.agent_name,
            "tag": self.agent_tag,
            "motivation_prompt": self.motivation_prompt,
            "system_prompt": self.system_prompt,
            "system_prompt_template": self.system_prompt_template,
            "use_inventory": self.use_inventory,
            "artifact_creation": self.artifact_creation,
            "food_mechanism": self.food_mechanism,
            "show_nearby_artifact_names": self.show_nearby_artifact_names,
            "energy_death": self.energy_death,
            "energy_upkeep": self.energy_upkeep,
            "finite_energy": self.finite_energy,
            "finite_lifespan": self.finite_lifespan,
            "external_actions": self.external_actions,
            "spawn_allowed": self.spawn_allowed,
            "max_connections": self.max_connections,
            "max_message_length": self.max_message_length,
            "genome": self.genome.as_dict(),
            "genome_class": f"{self.genome.__class__.__module__}:{self.genome.__class__.__name__}",
            "internal_memory_size": self.internal_memory_size,
            "max_history": self.max_history,
            "verbose": self.verbose,
            "internal_memory": self.internal_memory,
            "history": self.history,
            "log_dir": str(self.logger.log_dir),
        }

    def set_state_ckpt(self, state_ckpt: dict):
        self.agent_name = state_ckpt["name"]
        self.agent_tag = state_ckpt["tag"]
        self.motivation_prompt = state_ckpt["motivation_prompt"]
        self.system_prompt = state_ckpt["system_prompt"]
        self.finite_energy = state_ckpt["finite_energy"]
        self.finite_lifespan = state_ckpt["finite_lifespan"]
        self.system_prompt_template = state_ckpt["system_prompt_template"]
        self.use_inventory = state_ckpt["use_inventory"]
        self.artifact_creation = state_ckpt["artifact_creation"]
        self.food_mechanism = state_ckpt["food_mechanism"]
        self.show_nearby_artifact_names = state_ckpt.get("show_nearby_artifact_names", True)
        self.energy_death = state_ckpt.get("energy_death", self.energy_death)
        self._energy_upkeep_setting = state_ckpt.get("energy_upkeep")
        self.external_actions = state_ckpt.get("external_actions", False)
        self.spawn_allowed = state_ckpt.get("spawn_allowed", self.spawn_allowed)
        self.max_connections = state_ckpt.get("max_connections", self.max_connections)
        self.max_message_length = state_ckpt.get("max_message_length", self.max_message_length)
        self.internal_memory_size = state_ckpt["internal_memory_size"]
        self.internal_memory = self.validate_internal_memory(state_ckpt.get("internal_memory", ""))
        self.history = state_ckpt.get("history", [])
        self.max_history = state_ckpt["max_history"]
        self.verbose = state_ckpt.get("verbose", 1)

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
        self.logger.save_genome(agent_tag=self.agent_tag, genome=self.genome.as_dict())

    def close(self):
        self.logger.close()
