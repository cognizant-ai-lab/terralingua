import asyncio
import copy
import json
import logging
import os
import signal
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import count
from pathlib import Path
from typing import Any, Dict, List, Type

import numpy as np
from PIL import Image
from wonderwords import RandomWord

from terralingua.agents import agent_logger as _agent_logger_mod
from terralingua.agents.human_agent import HumanAgent
from terralingua.agents.llm_agent import LLMAgent
from terralingua.agents.remote_agent import RemoteAgent
from terralingua.config.models import ExperimentConfig
from terralingua.environment.env_logger import Event
from terralingua.environment.graph_env import OpenGraphWorld
from terralingua.environment.grid_env import OpenGridWorld
from terralingua.environment.social_graph_env import OpenSocialGraphWorld
from terralingua.experiment.checkpoint import CheckpointManager
from terralingua.experiment.execution_receipts import ExecutionReceiptsMixin
from terralingua.experiment.llm_router import LLMRouter
from terralingua.experiment.neuro_san_hocon import (
    load_neuro_san_agent_network,
)
from terralingua.experiment.run_status import RunStatus
from terralingua.experiment.scenario_loader import import_scenario, load_scenario
from terralingua.experiment.social_graph_history import SocialGraphHistoryRecorder
from terralingua.external.external_state import ExternalStateHandler, make_state_handler
from terralingua.external.request_manager import BATCH_TOOL_NAME, ExternalRequestManager
from terralingua.external.servers_config import (
    ServerSpec,
    apply_port_overrides,
    load_servers_config,
)
from terralingua.genome.base_genome import Genome
from terralingua.genome.no_traits import Genome as NoTraitsGenome
from terralingua.genome.ocean_5 import Genome as Ocean5Genome
from terralingua.genome.sentence_directed import Genome as SentenceDirectedGenome
from terralingua.genome.sentence_mutate import Genome as SentenceMutateGenome
from terralingua.server.redis_bus import CHANNEL_AGENT_EVENT, get_redis
from terralingua.server.redis_connection_manager import RedisConnectionManager
from terralingua.server.redis_publisher import RedisPublisher
from terralingua.server.redis_registry import RedisAgentRegistry
from terralingua.utils.generic import (
    LOGS_DIR,
    create_video,
    get_graph_render_state,
    get_render_state,
)
from terralingua.utils.llm_utils import async_select_with_retry
from terralingua.voting import (
    DiasRewards,
    DirectRewards,
    IndependentVotingManager,
    UnanimousVotingManager,
    VotingManager,
)

logger = logging.getLogger(__name__)


def _dashboard_resource(value: int | float | None) -> int | float | None:
    """Use null for unlimited resources at the dashboard JSON boundary."""
    return value if value is None or np.isfinite(value) else None


@dataclass
class _ExternalState:
    """This step's external bookkeeping, threaded prestep → poststep."""

    # agent_tag → the structured tool args it submitted (for the message log).
    requests: Dict[str, dict] = field(default_factory=dict)
    sent_statuses: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)
    # representative tag → coalition peers that share its response/status.
    broadcasts: Dict[str, set] = field(default_factory=dict)
    # agent_tag → (server, tool) for every requester — captured before the
    # election override erases dissenters' actions.
    tools: Dict[str, tuple] = field(default_factory=dict)
    # Tags whose calls were actually sent (all callers, or elected reps).
    dispatched: set = field(default_factory=set)
    call_contexts: Dict[str, dict] = field(default_factory=dict)
    # representative tag → (topic, VoteOutcome), unanimous mode only.
    vote_outcomes: Dict[str, tuple] = field(default_factory=dict)


def _params_from_input_schema(schema: dict | None) -> dict:
    """Translate an MCP tool's JSON-Schema inputSchema into the env action-param
    format: ``{name: description}`` or ``{name: {description, choices}}`` for
    enums. The auto-injected ``agent_tag`` is omitted from agent-facing params.
    """
    props = (schema or {}).get("properties", {})
    params: dict = {}
    for pname, pinfo in props.items():
        if pname == "agent_tag":
            continue
        pinfo = pinfo or {}
        desc = pinfo.get("description", "")
        if "enum" in pinfo:
            params[pname] = {"description": desc, "choices": list(pinfo["enum"])}
        else:
            params[pname] = desc
    return params


def render_server_instructions(instructions: dict[str, str]) -> str:
    """System-prompt text for the instructions MCP servers send at connection."""
    return "\n\n".join(
        f"=== Instructions from the {name} service ===\n"
        f"Its actions are the ones whose names start with {name}_.\n{text.strip()}"
        for name, text in instructions.items()
        if (text or "").strip()
    )


_rw = RandomWord()


def _random_agent_name(existing: set[str]) -> str:
    """Return a unique random-word name not already in *existing*."""
    for _ in range(100):
        name = _rw.word(word_max_length=8)
        if name not in existing:
            return name
    # Extremely unlikely fallback
    i = len(existing)
    while f"agent{i}" in existing:
        i += 1
    return f"agent{i}"


def get_genome_class(genome_type: str) -> Type[Genome]:
    if genome_type == "ocean_5":
        return Ocean5Genome
    elif genome_type == "no_traits":
        return NoTraitsGenome
    elif genome_type == "sentence_mutate":
        return SentenceMutateGenome
    elif genome_type == "sentence_directed":
        return SentenceDirectedGenome
    else:
        raise ValueError(f"Unsupported genome type: {genome_type}")


class SimulationRunner(ExecutionReceiptsMixin):
    def __init__(self, params: ExperimentConfig, resume: bool = False):
        logger.info("Initializing SimulationRunner...")
        self.params = params
        self.resume = resume
        self._run_id = uuid.uuid4().hex[:8]

        run_cfg = self.params.run
        if resume and not run_cfg.exp_name:
            raise ValueError("Resume requires exp_name to locate saved parameters")
        if run_cfg.exp_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_cfg.exp_name = f"run_{timestamp}_{self._run_id}"
        self.exp_logdir = LOGS_DIR / run_cfg.exp_name
        os.makedirs(self.exp_logdir, exist_ok=True)
        self.checkpointer = CheckpointManager(self.exp_logdir)
        if resume:
            self.params = self.checkpointer.update_parameters()
            run_cfg = self.params.run
        self.run_status = RunStatus(self.exp_logdir / "run_status.json")
        self._pending_external_requests = {}

        # Initialize external request managers, one per declared MCP server.
        self.external_specs: dict[str, ServerSpec] = {}
        # Union of every server's `hidden_keys`: top-level response fields stripped
        # from agent prompts while the full response is still logged (built in
        # _build_external_managers).
        self.hidden_external_keys: list[str] = []
        self.external_managers: dict[str, ExternalRequestManager] = {}
        # Flattened `<server>_<tool>` action specs, built from tool discovery.
        self.external_actions_spec: dict[str, dict] = {}
        if run_cfg.scenario:
            import_scenario(run_cfg.scenario)
        self._build_external_managers()
        self._build_external_actions_spec()
        self.server_instructions = render_server_instructions(
            {name: manager.instructions for name, manager in self.external_managers.items()}
        )

        # Restore checkpoint state using the resolved run parameters.
        self.ckpt = None
        self.env: OpenGridWorld | OpenGraphWorld
        self.obs: dict[str, Any] = {}
        self.infos: dict[str, Any] = {}
        self.dones: dict[str, bool] = {}
        self.rewards: dict[str, float] = {}
        self.start_ts: int = 0
        self.agents: Dict[str, LLMAgent | HumanAgent | RemoteAgent] = {}
        self.connection_manager: RedisConnectionManager | None = None
        self.registry: RedisAgentRegistry | None = None
        self.dashboard_manager: RedisPublisher | None = None
        self._use_redis: bool = False  # Set to True in Redis/multi-process mode
        self._cmd_loop_task: asyncio.Task | None = None
        self.last_spawn_idx = -1
        self.last_actions: dict = {}
        self.pre_step_obs: dict = {}
        self.dead_agent_traits: dict[str, dict] = {}
        if resume:
            self._load_state()
        else:
            self._init_state()

        # Initialize LLM router
        self.llm_router = LLMRouter(
            model_short=self.params.agent.model,
            ports=self.params.run.ports,
            instances=self.params.run.max_parallel_workers,
            max_tokens=self.params.agent.max_tokens,
            reasoning_effort=self.params.agent.reasoning_effort,
        )
        self.last_refresh = datetime.now()
        self.refresh_interval = timedelta(hours=1)

        self.terminate = False
        self._llm_semaphore: asyncio.Semaphore | None = None

        self._ext_msg_log_path = self.exp_logdir / "external_messages.jsonl"
        self._cost_csv_path = self.exp_logdir / "costs.csv"

        with open(self.exp_logdir / "params.json", "w") as f:
            json.dump(self.params.to_json(), f, indent=4)

        logger.info("###### Launching with Parameters ######")
        logger.info("%s", self.params.to_json())
        logger.info("#######################################")

    def _build_external_managers(self) -> None:
        """Build one ExternalRequestManager per declared server.

        Each server's Flask listener binds the ``callback_port`` declared for it
        in servers.json; the loader guarantees these are unique so they don't
        collide.
        """
        for spec in self._resolve_server_specs():
            self.external_specs[spec.name] = spec
            self.external_managers[spec.name] = ExternalRequestManager(
                server_url=spec.url,
                listen_port=spec.callback_port,
                request_timeout_s=None,
                sync_send=self.params.run.external_sync_send,
            )
        self.hidden_external_keys = sorted(
            {k for spec in self.external_specs.values() for k in spec.hidden_keys}
        )
        # One voting manager per server: unanimous (consensus + shared DIAS
        # reward) or independent (each caller keeps its own reward). Add a branch
        # here when a new voting scenario needs a different topic key/strategy.
        coeff = self.params.run.reward_to_energy_coefficient
        self.voting_managers: dict[str, VotingManager] = {}
        for name, spec in self.external_specs.items():
            if spec.mode == "unanimous":
                self.voting_managers[name] = UnanimousVotingManager(
                    rewards=DiasRewards(cost=spec.cost, coefficient=coeff),
                    vote_key=spec.vote_key,
                )
            else:
                self.voting_managers[name] = IndependentVotingManager(
                    rewards=DirectRewards(coefficient=coeff),
                )
        # One shared-state handler per server that opts in via servers.json `state`.
        self.external_state_handlers: dict[str, ExternalStateHandler] = {
            name: make_state_handler(spec.state)
            for name, spec in self.external_specs.items()
            if spec.state
        }

    def _build_external_actions_spec(self) -> None:
        """Flatten every server's discovered MCP tools into `<server>_<tool>`
        env-action specs carrying per-server cost/mode, a description (tool doc +
        energy cost), and structured params translated from the tool's
        inputSchema.
        """
        self.external_actions_spec = {}
        for name, manager in self.external_managers.items():
            spec = self.external_specs[name]
            for tool in manager.list_tools():
                if spec.batch and tool.name == BATCH_TOOL_NAME:
                    # Runner-facing round envelope, not an agent action.
                    continue
                action_name = f"{name}_{tool.name}"
                tool_doc = (getattr(tool, "description", "") or "").strip()
                description = (
                    f"{tool_doc} Costs {spec.cost} energy."
                    if tool_doc
                    else f"Call the '{name}' external server. Costs {spec.cost} energy."
                )
                # Keep the MCP schema across discovery, prompting and parsing.
                # Omitted optional values belong to the server: filling them
                # with empty strings would replace its defaults with new input.
                input_schema = copy.deepcopy(getattr(tool, "inputSchema", None) or {})
                input_schema.setdefault("properties", {}).pop("agent_tag", None)
                input_schema["required"] = [
                    key for key in input_schema.get("required", []) if key != "agent_tag"
                ]
                self.external_actions_spec[action_name] = {
                    "server": name,
                    "tool": tool.name,
                    "cost": spec.cost,
                    "mode": spec.mode,
                    "description": description,
                    "description_without_cost": tool_doc or f"Call the '{name}' external server.",
                    "params": _params_from_input_schema(input_schema),
                    "optional": sorted(
                        set(input_schema.get("properties", {}))
                        - set(input_schema["required"])
                    ),
                    "input_schema": input_schema,
                }

    def _resolve_server_specs(self) -> list[ServerSpec]:
        """Load the external MCP servers declared in ``--external_servers_config``
        (a servers.json), or none if unset. ``external_server_port`` /
        ``external_server_callback_port`` (if set) supersede the sole server's URL
        port / callback_port so parallel runs can each use their own ports."""
        if not self.params.run.external_servers_config:
            return []
        specs = load_servers_config(self.params.run.external_servers_config)
        return apply_port_overrides(
            specs,
            url_port=self.params.run.external_server_port,
            callback_port=self.params.run.external_server_callback_port,
        )

    def _sample_agent_lifespan(self) -> int | float:
        """Sample a per-agent lifespan within ±20% of the configured baseline."""
        base = int(self.params.env.agent_lifespan)
        if base == -1:
            return float("inf")
        if base <= 1:
            return max(1, base)
        low = max(1, int(round(base * 0.8)))
        high = max(low, int(round(base * 1.2)))
        if low == high:
            return low
        if self.env.rng is None:
            self.env.rng = np.random.default_rng()
        return int(self.env.rng.integers(low, high + 1))

    def _assign_sampled_lifespan(self, agent_tag: str) -> int | float | None:
        """Overwrite env-side remaining lifetime for a freshly spawned agent."""
        if agent_tag not in getattr(self.env, "agent_time", {}):
            return None
        sampled = self._sample_agent_lifespan()
        self.env.agent_time[agent_tag] = sampled
        if agent_tag in self.obs and isinstance(self.obs[agent_tag], dict):
            self.obs[agent_tag]["time"] = sampled
        return sampled

    def _make_llm_agent(self, tag: str, name: str, genome: Genome, persona: str = "") -> LLMAgent:
        return LLMAgent(
            agent_tag=tag,
            agent_name=name,
            log_dir=self.exp_logdir,
            max_history=self.params.agent.max_history,
            system_prompt_template=self.env.system_prompt_template,
            internal_memory_size=self.params.agent.internal_memory_size,
            use_inventory=self.params.agent.use_inventory,
            artifact_creation=self.params.env.artifact_creation_cost >= 0,
            food_mechanism=self.params.env.food_mechanism,
            energy_death=self.env.energy_death,
            finite_energy=bool(np.isfinite(self.env.agent_energy.get(tag, self.env.init_agent_energy))),
            finite_lifespan=self.params.env.agent_lifespan != -1,
            spawn_allowed=self.env.spawn_allowed,
            max_connections=getattr(self.env, "max_connections", None),
            external_actions=bool(self.external_managers),
            hidden_external_keys=self.hidden_external_keys,
            stay_action=self.env.stay_action,
            genome=genome,
            persona=persona,
            scenario_specific_instructions=self.params.agent.scenario_specific_instructions,
            max_message_length=self.params.env.max_message_length,
            server_instructions=getattr(self, "server_instructions", ""),
            # Population permanently 1 is a static invariant only when the
            # cap AND the initial population agree (max_agents alone just
            # stops spawn from being offered — it never shrinks a larger
            # initial population): social prompt content and the broadcast
            # side-channel would address nobody.
            solo=(
                self.params.env.max_agents == 1
                and self.params.env.init_agents <= 1
                and self.params.env.init_human_agents == 0
            ),
        )

    def _make_human_agent(self, tag: str) -> HumanAgent:
        return HumanAgent(
            agent_name=tag,
            agent_tag=tag,
            log_dir=self.exp_logdir,
            max_history=self.params.agent.max_history,
            internal_memory_size=self.params.agent.internal_memory_size,
            use_inventory=self.params.agent.use_inventory,
            food_mechanism=self.params.env.food_mechanism,
            genome=NoTraitsGenome(),
        )

    def _make_remote_agent(
        self, tag: str, name: str, motivation_prompt: str, genome: Genome
    ) -> RemoteAgent:
        return RemoteAgent(
            agent_tag=tag,
            agent_name=name,
            motivation_prompt=motivation_prompt,
            log_dir=self.exp_logdir,
            genome=genome,
            max_history=self.params.agent.max_history,
            system_prompt_template=self.env.system_prompt_template,
            internal_memory_size=self.params.agent.internal_memory_size,
            use_inventory=self.params.agent.use_inventory,
            artifact_creation=self.params.env.artifact_creation_cost >= 0,
            food_mechanism=self.params.env.food_mechanism,
            energy_death=self.env.energy_death,
            finite_energy=bool(np.isfinite(self.env.agent_energy.get(tag, self.env.init_agent_energy))),
            finite_lifespan=self.params.env.agent_lifespan != -1,
            spawn_allowed=self.env.spawn_allowed,
            max_connections=getattr(self.env, "max_connections", None),
            max_message_length=self.params.env.max_message_length,
            connection_manager=self.connection_manager,
            timeout_s=self.params.run.env_heartbeat,
            server_instructions=getattr(self, "server_instructions", ""),
        )

    def register_remote_agent(
        self,
        agent_tag: str,
        agent_name: str,
        motivation_prompt: str,
        position: tuple | None = None,
        genome_type: str | None = None,
        genome_data: dict | None = None,
        excluded_actions: list | None = None,
        agent_type: str = "remote_llm",
    ) -> RemoteAgent:
        """
        Called by the API layer when an external user registers.
        Creates a RemoteAgent, adds it to the environment, and tracks it.
        Thread-safe: can be called while the simulation is running.
        """

        if genome_type is None:
            genome_type = self.params.agent.genome
        genome_cls = get_genome_class(genome_type)

        if genome_data is None:
            genome = genome_cls().random()
        else:
            genome = genome_cls().from_dict(genome_data)

        agent = self._make_remote_agent(
            tag=agent_tag,
            name=agent_name,
            motivation_prompt=motivation_prompt,
            genome=genome,
        )

        agent.agent_type = agent_type
        if agent_type == "remote_human":
            agent.timeout_s = -1  # wait indefinitely for human input
        obs, infos = self.env.add_agent(
            agent_tag=agent_tag,
            agent_name=agent_name,
            genome_type=genome.genome_type,
            position=position,
            excluded_actions=excluded_actions,
        )
        sampled_lifespan = self._assign_sampled_lifespan(agent_tag)
        if sampled_lifespan is not None:
            obs["time"] = sampled_lifespan
        # Update runner state atomically-ish (GIL protects dict writes)
        self.obs[agent_tag] = obs
        self.infos[agent_tag] = infos
        self.dones[agent_tag] = False
        self.rewards[agent_tag] = 0
        self.agents[agent_tag] = agent
        logger.info("[Runner] Remote agent registered: %s (%s)", agent_name, agent_tag)
        return agent

    def _make_env(self):
        """Creates the environment instance."""
        shared_kwargs = dict(
            init_agent_energy=self.params.env.init_agent_energy,
            init_food=self.params.env.init_food,
            food_decay_rate=self.params.env.food_decay_rate,
            food_spawn_rate=self.params.env.food_spawn_rate,
            log_path=self.exp_logdir,
            use_inventory=self.params.agent.use_inventory,
            use_library=self.params.env.use_library,
            library_preview=self.params.env.library_preview,
            allow_fixed_artifacts=self.params.env.allow_fixed_artifacts,
            max_artifact_tokens=self.params.env.max_artifact_tokens,
            use_colors=self.params.agent.use_colors,
            reproduction_cost=self.params.env.reproduction_cost,
            artifact_creation_cost=self.params.env.artifact_creation_cost,
            two_parent_spawn=self.params.env.two_parent_spawn,
            max_agents=self.params.env.max_agents,
            lifespan=self.params.env.agent_lifespan,
            static_food=self.params.env.static_food,
            food_zones=self.params.env.food_zones,
            food_mechanism=self.params.env.food_mechanism,
            energy_death=self.params.env.energy_death,
            drop_food_on_death=self.params.env.drop_food_on_death,
            inert_artifacts=self.params.env.inert_artifacts,
            external_actions_spec=self.external_actions_spec,
            excluded_actions=self.params.env.excluded_actions,
            headless=not self.params.run.live_render,
            max_message_length=self.params.env.max_message_length,
            roles_hocon_path=self.params.env.roles_hocon_path,
            affordances_file_path=self.params.env.affordances_file_path,
        )

        if self.params.env.world_type == "graph":
            self.env = OpenGraphWorld(
                graph_cfg=self.params.env.graph,
                **shared_kwargs,  # type: ignore
            )
        elif self.params.env.world_type == "social_graph":
            self.env = OpenSocialGraphWorld(
                graph_cfg=self.params.env.graph,
                **shared_kwargs,  # type: ignore
            )
        else:
            self.env = OpenGridWorld(
                grid_size=self.params.env.grid_size,
                vision_radius=self.params.env.vision_radius,
                **shared_kwargs,  # type: ignore
            )
        if self.params.run.scenario:
            for mechanic in load_scenario(
                self.params.run.scenario, self.params.run.scenario_options
            ):
                self.env.attach(mechanic)
        self._stay_action = self.env.stay_action

    def _seed_artifacts_from_path(self, path: str):
        """Loads JSON artifact files from a directory and seeds them into the environment."""
        artifact_dir = Path(path)
        if not artifact_dir.exists():
            logger.warning(
                "Warning: init_artifacts_path '%s' does not exist, skipping.", path
            )
            return
        json_files = sorted(artifact_dir.glob("*.json"))
        if not json_files:
            logger.warning("Warning: no JSON files found in '%s', skipping.", path)
            return
        for json_file in json_files:
            try:
                with open(json_file) as f:
                    data = json.load(f)
                entries = data if isinstance(data, list) else [data]
                for entry in entries:
                    pose = (
                        entry["pose"]
                        if self.params.env.world_type in ("graph", "social_graph")
                        else tuple(entry["pose"])
                    )
                    message, _ = self.env.seed_artifact(
                        pose=pose,
                        art_type=entry["art_type"],
                        art_name=entry["name"],
                        payload=entry["payload"],
                        lifespan=entry.get("lifespan", -1),
                        movable=entry.get("movable", True),
                        **entry.get("params", {}),
                    )
                    logger.info("  [artifact seed] %s", message)
            except Exception as e:
                logger.warning(
                    "Warning: failed to load artifact from %s: %s", json_file, e
                )

    def _init_state(self):
        """Initializes state and environment from scratch."""
        self._make_env()

        genome_cls = get_genome_class(self.params.agent.genome)
        agent_poses = {}

        if self.params.env.graph.agent_network_hocon_path:
            # genome == 'sentence_directed' is guaranteed by ExperimentConfig validation.
            network = load_neuro_san_agent_network(
                self.params.env.graph.agent_network_hocon_path
            )
            for spec in network.agents:
                agent_tag = spec.name
                agent_name = spec.display_as or spec.name
                genome_sentence = "\n\n".join(
                    [text for text in (spec.description, spec.instructions) if text]
                ).strip()
                agent = self._make_llm_agent(
                    tag=agent_tag,
                    name=agent_name,
                    genome=SentenceDirectedGenome(sentence=genome_sentence),
                )
                self.agents[agent_tag] = agent
                self.env.add_agent(
                    agent_tag=agent_tag,
                    agent_name=agent_name,
                    genome_type=agent.genome.genome_type,
                    position=spec.name,
                )
                agent_poses[agent_tag] = spec.name

            self.last_spawn_idx = len(network.agents) - 1
            logger.info(
                "[Runner] Imported Neuro-SAN HOCON network %s with %d agents",
                network.path,
                len(network.agents),
            )
        else:
            prefix = self.params.agent.agents_name_prefix
            init_agents = {
                f"{prefix}{i}": "text" for i in range(self.params.env.init_agents)
            }
            init_human_agents = {
                f"human{i}": "human" for i in range(self.params.env.init_human_agents)
            }
            init_agents.update(init_human_agents)

            self.last_spawn_idx = self.params.env.init_agents - 1

            use_random_names = self.params.agent.random_names
            used_names: set[str] = set()

            for agent_tag, agent_type in init_agents.items():
                if agent_type == "text":
                    if use_random_names:
                        agent_name = _random_agent_name(used_names)
                        used_names.add(agent_name)
                    else:
                        agent_name = agent_tag
                    identity = self.env.agent_identity(agent_tag)
                    agent_name = identity.get("name") or agent_name
                    used_names.add(agent_name)
                    self.agents[agent_tag] = self._make_llm_agent(
                        tag=agent_tag,
                        name=agent_name,
                        genome=genome_cls().random(),
                        **({"persona": identity["persona"]} if identity.get("persona") else {}),
                    )
                    self.env.add_agent(
                        agent_tag=agent_tag,
                        agent_name=agent_name,
                        genome_type=self.agents[agent_tag].genome.genome_type,
                    )
                elif agent_type == "human":
                    self.agents[agent_tag] = self._make_human_agent(
                        tag=agent_tag,
                    )
                    self.env.add_agent(
                        agent_tag=agent_tag,
                        agent_name=agent_tag,
                        genome_type=self.agents[agent_tag].genome.genome_type,
                    )
                else:
                    raise NotImplementedError(
                        f"{agent_tag} is non-text agent. Not implemented yet."
                    )

        self.obs, self.infos = self.env.restart_env(
            seed=self.params.run.seed, agent_poses=agent_poses
        )
        for agent_tag in self.obs:
            sampled_lifespan = self._assign_sampled_lifespan(agent_tag)
            if sampled_lifespan is not None:
                self.obs[agent_tag]["time"] = sampled_lifespan

        if self.params.env.init_artifacts_path:
            self._seed_artifacts_from_path(self.params.env.init_artifacts_path)

        self.start_ts = 0
        self.rewards = {}
        self.dones = {a: False for a in self.agents.keys()}

    def _load_state(self):
        """Loads state and environment from checkpoint."""
        self.ckpt = self.checkpointer.load_checkpoint()

        self._make_env()
        self.last_spawn_idx = self.ckpt["last_spawn_idx"]

        # Load agents
        for agent_tag, agent_ckpt in self.ckpt["agents"].items():
            if agent_ckpt["type"] == "LLMAgent":
                agent = self._make_llm_agent(
                    tag=agent_ckpt["tag"],
                    name=agent_ckpt["name"],
                    genome=NoTraitsGenome(),  # Temporary genome, will be loaded
                )
            elif agent_ckpt["type"] == "HumanAgent":
                agent = self._make_human_agent(
                    tag=agent_tag,
                )
            elif agent_ckpt["type"] == "RemoteAgent":
                agent = self._make_remote_agent(
                    tag=agent_ckpt["tag"],
                    name=agent_ckpt["name"],
                    motivation_prompt=agent_ckpt["motivation_prompt"],
                    genome=NoTraitsGenome(),  # Temporary genome, will be loaded
                )
            else:
                raise NotImplementedError(
                    f"Unsupported agent type in checkpoint: {agent_ckpt['type']}"
                )
            agent.set_state_ckpt(agent_ckpt)
            self.agents[agent_tag] = agent

        self.env.set_state_ckpt(self.ckpt["env"])
        self.env.agent_genomes = {
            tag: agent.genome.genome_type for tag, agent in self.agents.items()
        }

        # Evict any tag that exists in env state but has no corresponding agent
        # object — this happens when a previous run crashed inside env.step()
        # after a spawned child was added but before _handle_spawn
        # registered the runner-side agent.
        orphans = [
            tag for tag in list(self.env.agent_registry) if tag not in self.agents
        ]
        for tag in orphans:
            self.env.evict_orphan(tag)

        self.start_ts = self.ckpt["ts"]
        if "run_id" in self.ckpt:
            self._run_id = self.ckpt["run_id"]
        self.obs = self.ckpt["env_outs"]["obs"]
        self.infos = self.ckpt["env_outs"]["infos"]
        self.rewards = self.ckpt["env_outs"]["rewards"]
        self.dones = self.ckpt["env_outs"]["dones"]
        self._pending_external_requests = self.ckpt["env_outs"].get("pending_external_requests", {})
        self._restore_reward_state(self.ckpt["env_outs"].get("external_reward_states", {}))

        # Build menus from restored state and the current action rules.
        for tag in self.agents:
            if tag not in self.env.agent_registry:
                continue
            if tag in self.obs:
                self.obs[tag]["time"] = self.env.agent_time[tag]
                self.obs[tag]["energy"] = self.env.agent_energy[tag]
            actions = self.env._get_avail_actions(tag)
            self.env.agent_avail_actions[tag] = actions
            self.infos.setdefault(tag, {})["available_actions"] = actions
        logger.info(
            "Resumed environment and agents from checkpoint. Current timestep: %s",
            self.start_ts,
        )

    def _build_frame(self) -> dict | None:
        """Build and return the current simulation state as a plain dict.

        Returns None if the environment is not yet initialised.
        """
        if not hasattr(self, "env") or self.env is None:
            return None

        if isinstance(self.env, OpenGraphWorld):
            grid_state = get_graph_render_state(self.env)
        else:
            grid_state = get_render_state(self.env)
        connected = (
            set(self.connection_manager.connected_agents())
            if self.connection_manager
            else set()
        )
        pre_obs = self.pre_step_obs
        child_to_parent = {
            child: parent
            for parent, children in self.env.agent_spawn.items()
            for child in children
        }
        agents = []
        for tag, agent in self.agents.items():
            if self.dones.get(tag):
                continue
            agent_obs = pre_obs.get(tag)
            energy = agent_obs.get("energy") if isinstance(agent_obs, dict) else None
            time_left = agent_obs.get("time") if isinstance(agent_obs, dict) else None
            inventory = (
                agent_obs.get("inventory", []) if isinstance(agent_obs, dict) else []
            )
            last = self.last_actions.get(tag, {})
            agents.append(
                {
                    "tag": tag,
                    "name": agent.agent_name,
                    "energy": _dashboard_resource(energy),
                    "time_left": _dashboard_resource(time_left),
                    "inventory": inventory,
                    "motivation": getattr(agent, "motivation_prompt", ""),
                    "last_action": last.get("action"),
                    "last_params": last.get("params"),
                    "connected": tag in connected,
                    "type": getattr(agent, "agent_type", type(agent).__name__),
                    "parent_tag": child_to_parent.get(tag),
                    "genome_type": agent.genome.genome_type,
                    "genome": agent.genome.as_dict(),
                }
            )
        artifacts = []
        for art_name, art in self.env.artifacts.items():
            try:
                serialized = art.serialize()
                # art.pose is set at creation and never updated on pickup/drop.
                # Use artifact_location to get the real current position.
                loc = self.env.artifact_location.get(art_name)
                if loc:
                    kind, where = loc
                    is_graph = isinstance(self.env, OpenGraphWorld)
                    if kind == "map":
                        serialized["pose"] = (
                            where if is_graph else (int(where[0]), int(where[1]))
                        )
                        serialized["carrier"] = None
                    elif kind == "inv":
                        agent_pos = self.env.agent_pos.get(where)
                        if agent_pos:
                            serialized["pose"] = (
                                agent_pos
                                if is_graph
                                else (int(agent_pos[0]), int(agent_pos[1]))
                            )
                        serialized["carrier"] = where
                    elif kind == "library":
                        serialized["pose"] = "Library"
                        serialized["carrier"] = None
                artifacts.append(serialized)
            except Exception:
                pass
        alive_tags = {a["tag"] for a in agents}
        dead_agents = [
            {"tag": tag, "name": name, **self.dead_agent_traits.get(tag, {})}
            for tag, name in self.env.agent_names.items()
            if tag not in alive_tags
        ]
        return {
            "run_id": self._run_id,
            "step": int(self.env.step_count),
            "grid_state": grid_state,
            "agents": agents,
            "dead_agents": dead_agents,
            "artifacts": artifacts,
            "expired_artifacts": [a.serialize() for a in self.env.expired_artifacts],
            "genealogy": child_to_parent,
            "total_agents_ever": len(self.env.agent_names),
            "total_artifacts_ever": len(self.env.artifacts)
            + len(self.env.expired_artifacts),
        }

    async def _push_dashboard_frame(self):
        """Push the current frame to all dashboard WebSocket clients."""
        frame = self._build_frame()
        if frame is None:
            return
        if self.dashboard_manager is not None:
            await self.dashboard_manager.broadcast_frame(frame)

    def _render(self, ts: int):
        """Renders the current frame and saves it as an image."""
        if self.params.run.live_render:
            try:
                self.env.render(mode="human")
            except Exception as e:
                logger.warning("Error during live rendering: %s", e)

        if self.params.run.save_video:
            try:
                frame = self.env.render(mode="rgb_array")
            except Exception as e:
                logger.warning("Error during frame rendering: %s", e)
                frame = None

            if frame is not None:
                try:
                    out_dir = self.exp_logdir / "frames"
                    out_dir.mkdir(exist_ok=True, parents=True)
                    img = Image.fromarray(frame.astype(np.uint8))
                    img.save(out_dir / f"{ts:05d}.png")
                except Exception as e:
                    logger.warning("Error during saving frame: %s", e)

    def _handle_term(self, *args):
        self.terminate = True
        logger.info("⚠️ Caught termination signal. Will terminate after current step. ⚠️")

    def _watch_stdin(self):
        """Watches stdin for EOF (Ctrl+D) and force kills the process immediately."""
        try:
            while not self.terminate:
                if sys.stdin.read(1) == "":
                    logger.warning(
                        "\n💀 Force kill (Ctrl+D). Terminating immediately... 💀"
                    )
                    os._exit(1)
        except Exception:
            pass

    async def _save_checkpoint(self, ts: int):
        await self.checkpointer.save_checkpoint(
            ts=ts,
            env=self.env,
            agents=self.agents,
            last_spawn_idx=self.last_spawn_idx,
            run_id=self._run_id,
            env_outs={
                "obs": self.obs,
                "infos": self.infos,
                "rewards": self.rewards,
                "dones": self.dones,
                "pending_external_requests": getattr(self, "_pending_external_requests", {}),
                "external_reward_states": self._reward_state_checkpoint(),
            },
        )

    async def _handle_spawn(self):
        """Creates agents based on environment spawn infos."""
        for agent_tag, info in self.infos.items():
            if "spawn" in info:
                spawn_status = info["spawn"].get("status", "failed")
                if spawn_status == "successful":
                    parent = self.agents[agent_tag]
                    child_name = info["spawn"]["child_name"]
                    child_tag = info["spawn"]["child_tag"]
                    parent_b_tag = info["spawn"].get("parent_b_tag")

                    # If spawning is two-parent, crossover the genomes of both parents to create the child genome.
                    if parent_b_tag is not None and parent_b_tag in self.agents:
                        parent_b = self.agents[parent_b_tag]
                        child_genome = parent.genome.crossover(parent_b.genome)  # type: ignore
                        spawn_type = "paired"

                    # If spawning is solo, mutate the genome of the parent.
                    # In the case of sentence_directed, it's the parent that provides the genome
                    elif parent.genome.genome_type == "sentence_directed":
                        offspring_genome_str = str(
                            info["spawn"].get("offspring_genome", "").strip()
                        )
                        child_genome = SentenceDirectedGenome(
                            sentence=offspring_genome_str or parent.genome.sentence  # type: ignore
                        )
                        spawn_type = "solo"
                    else:
                        child_genome = parent.genome.mutate(
                            rate=self.params.env.genome_mutation_rate
                        )
                        spawn_type = "solo"

                    if isinstance(parent, RemoteAgent) and self.registry is not None:
                        parent_record = await self.registry.get_by_tag(agent_tag)
                        model_info = parent_record.model_info if parent_record else ""
                        record = await self.registry.register_with_tag(
                            agent_tag=child_tag,
                            agent_name=child_name,
                            motivation_prompt=parent.motivation_prompt,
                            model_info=model_info,
                        )
                        self.agents[child_tag] = self._make_remote_agent(
                            tag=child_tag,
                            name=child_name,
                            motivation_prompt=parent.motivation_prompt,
                            genome=child_genome,
                        )
                        child_push = {
                            "type": "child_agent",
                            "agent_tag": child_tag,
                            "token": record.token,
                            "parent_tag": agent_tag,
                        }
                        if self.dashboard_manager is not None:
                            await self.dashboard_manager.broadcast_push(child_push)
                    else:
                        self.agents[child_tag] = self._make_llm_agent(
                            tag=child_tag,
                            name=child_name,
                            genome=child_genome,
                        )
                    self._assign_sampled_lifespan(child_tag)
                    # Social-graph worlds: a newborn without followers
                    # broadcasts into the void — wire it to its parent.
                    if hasattr(self.env, "bootstrap_follows"):
                        self.env.bootstrap_follows(child_tag, parent=agent_tag)
                    logger.info(
                        "🍼 Agent %s spawned (%s). New agent: %s 🍼",
                        agent_tag,
                        spawn_type,
                        child_tag,
                    )
                    # No need to register agent in environment, as env already did so.

    def _snapshot_dead_traits(self, tag: str, agent) -> None:
        """Capture the traits an alive agent exposes in the frame so the
        dashboard can keep showing them after the agent is removed."""
        try:
            self.dead_agent_traits[tag] = {
                "motivation": getattr(agent, "motivation_prompt", ""),
                "type": getattr(agent, "agent_type", type(agent).__name__),
                "genome_type": agent.genome.genome_type,
                "genome": agent.genome.as_dict(),
            }
        except Exception:
            pass

    def _cleanup_dead(self):
        """Cleans up agents that are done/dead."""
        dead_agents = [k for k, v in self.dones.items() if v]
        if dead_agents:
            logger.info("💀 Cleaning up dead agents: %s 💀", dead_agents)
        for dead in dead_agents:
            dead_agent = self.agents.pop(dead)
            self._snapshot_dead_traits(dead, dead_agent)
            dead_agent.close()
        if dead_agents and self._use_redis:

            async def _notify(tags):
                from terralingua.server.redis_bus import KEY_CONNECTED_AGENTS

                r = await get_redis()
                try:
                    for tag in tags:
                        await r.srem(KEY_CONNECTED_AGENTS, tag)  # type: ignore[misc]
                        await r.publish(
                            CHANNEL_AGENT_EVENT,
                            json.dumps({"event": "died", "tag": tag}),
                        )
                        if self.registry is not None:
                            await self.registry.unregister(tag)
                finally:
                    await r.aclose()

            asyncio.ensure_future(_notify(dead_agents))

    def _maybe_resize_grid(self) -> None:
        """Expand the grid by a fixed step when agent density exceeds the configured max."""
        if self.params.env.world_type in ("graph", "social_graph"):
            # Grid-resize doesn't apply to graph-family worlds. (Graph: dynamic
            # node scaling is a TODO. Social graph: nodes are added/removed
            # automatically on spawn/death.)
            return
        if not self.params.env.dynamic_grid_scaling:
            return
        assert isinstance(self.env, OpenGridWorld)
        total = len(self.agents)
        if total != 0:
            cells_per_agent = self.env.grid_size**2 / total
            logger.info(
                "[Runner] Density: %.1f cells/agent (%d agents, %d×%d grid)",
                cells_per_agent,
                total,
                self.env.grid_size,
                self.env.grid_size,
            )
            if cells_per_agent < self.params.env.dynamic_grid_min_cells_per_agent:
                new_size = self.env.grid_size + self.params.env.dynamic_grid_step
                logger.info(
                    "[Runner] Expanding grid: %d → %d (%.1f cells/agent < %d)",
                    self.env.grid_size,
                    new_size,
                    cells_per_agent,
                    self.params.env.dynamic_grid_min_cells_per_agent,
                )
                self.env.resize_grid(new_size)
        else:
            logger.info("[Runner] No agents to calculate density")

    def inject_event(self, event_type: str, **params) -> None:
        """Inject a world event mid-simulation.

        Supported event types:
        - ``"artifact"``: place an artifact at a node/position.
          Required keys: ``art_name``, ``payload``.
          Graph world: ``node`` (str).
          Grid world: ``pose`` (tuple[int, int]).
        - ``"kill_agent"``: immediately kill an agent.
          Required keys: ``agent_tag``.
        - ``"sever_edge"``: remove an edge (graph world only).
          Required keys: ``u``, ``v``.
        """
        if event_type == "artifact":
            pose = (
                params["node"]
                if self.params.env.world_type in ("graph", "social_graph")
                else params["pose"]
            )
            message, _ = self.env.seed_artifact(
                pose=pose,
                art_type=params.get("art_type", "text"),
                art_name=params["art_name"],
                payload=params["payload"],
                lifespan=params.get("lifespan", -1),
                movable=params.get("movable", True),
            )
            logger.info("[inject_event] %s", message)
        elif event_type == "kill_agent":
            tag = params["agent_tag"]
            if tag in self.env.agent_registry:
                self.env._kill(tag)
            if tag in self.agents:
                self._snapshot_dead_traits(tag, self.agents[tag])
                self.agents[tag].close()
                del self.agents[tag]
                if self._use_redis:

                    async def _notify_kill(t=tag):
                        from terralingua.server.redis_bus import KEY_CONNECTED_AGENTS

                        r = await get_redis()
                        await r.srem(KEY_CONNECTED_AGENTS, t)  # type: ignore[misc]
                        await r.publish(
                            CHANNEL_AGENT_EVENT, json.dumps({"event": "died", "tag": t})
                        )
                        await r.aclose()

                    asyncio.ensure_future(_notify_kill())
        elif event_type == "sever_edge":
            assert isinstance(self.env, OpenGraphWorld), (
                "sever_edge requires graph world"
            )
            self.env.world_graph.remove_edge(params["u"], params["v"])
        else:
            logger.warning("[inject_event] Unknown event type: %r", event_type)

    def _respawn_if_needed(self):
        """Respawns new agents if number of alive agents is below minimum"""
        while len(self.agents) < self.params.env.min_agents:
            # Ensure unique agent tag
            while True:
                self.last_spawn_idx += 1
                new_agent_tag = (
                    f"{self.params.agent.agents_name_prefix}{self.last_spawn_idx}"
                )
                # We check env.agent_names so to verify against dead agents as well.
                if new_agent_tag not in self.env.agent_names:
                    break
            if self.params.agent.random_names:
                new_agent_name = _random_agent_name(set(self.env.agent_names.values()))
            else:
                new_agent_name = new_agent_tag
            identity = self.env.agent_identity(new_agent_tag)
            new_agent_name = identity.get("name") or new_agent_name

            genome_cls = get_genome_class(self.params.agent.genome)
            self.agents[new_agent_tag] = self._make_llm_agent(
                tag=new_agent_tag,
                name=new_agent_name,
                genome=genome_cls().random(),
                **({"persona": identity["persona"]} if identity.get("persona") else {}),
            )
            # Register in environment
            obs, infos = self.env.add_agent(
                agent_tag=new_agent_tag,
                agent_name=new_agent_name,
                genome_type=self.agents[new_agent_tag].genome.genome_type,
            )
            sampled_lifespan = self._assign_sampled_lifespan(new_agent_tag)
            if sampled_lifespan is not None:
                obs["time"] = sampled_lifespan
            # Respawns have no live parent: anchor them to the org's most
            # followed agent so their broadcasts land from turn one.
            if hasattr(self.env, "bootstrap_follows"):
                self.env.bootstrap_follows(new_agent_tag)
            self.obs[new_agent_tag] = obs
            self.infos[new_agent_tag] = infos
            self.dones[new_agent_tag] = False

    def _handle_incoming_external_messages(
        self, sent_status: Dict[str, List[Dict[str, str]]]
    ) -> Dict[str, Dict[str, list]]:
        """Handles incoming messages from all external servers.

        Returns the per-agent, per-server drained messages — the clean
        per-call responses, captured before broadcasts and vote notes are
        mixed into the obs lists.
        """
        if not self.external_managers:
            return {}

        # Queue reads are destructive, so drain each server once and reuse the
        # results for both per-agent obs injection and the state handlers.
        broadcast_by_server = {
            name: mgr.get_broadcast_messages()
            for name, mgr in self.external_managers.items()
        }
        incoming_by_agent = {
            agent_tag: {
                name: mgr.get_incoming_records(agent_tag)
                for name, mgr in self.external_managers.items()
            }
            for agent_tag in self.obs.keys()
        }
        def visible_record(record):
            if record.get("tool_error") or record.get("delivery_failed"):
                return json.dumps({"external_failure": {
                    "request_id": record.get("message_id", ""),
                    "tool_error": bool(record.get("tool_error")),
                    "delivery_failed": bool(record.get("delivery_failed")),
                    "detail": record.get("text", ""),
                }})
            return record.get("text", "")

        all_broadcasts = [m for msgs in broadcast_by_server.values() for m in msgs]

        # Feed each opted-in server's state handler its own responses.
        for name, handler in self.external_state_handlers.items():
            server_msgs = list(broadcast_by_server.get(name, []))
            for per_server in incoming_by_agent.values():
                server_msgs.extend(r["text"] for r in per_server.get(name, [])
                                   if r.get("text") and not r.get("tool_error") and not r.get("delivery_failed"))
            if server_msgs:
                handler.observe(server_msgs)

        logged_responses: set = set()
        for agent_tag in self.obs.keys():
            incoming: list[str] = []
            for name in self.external_managers:
                incoming.extend(visible_record(r) for r in incoming_by_agent[agent_tag][name]
                                if r.get("text") or r.get("tool_error") or r.get("delivery_failed"))
            incoming = incoming + all_broadcasts
            if incoming:
                if "external_response" not in self.obs[agent_tag]:
                    self.obs[agent_tag]["external_response"] = []
                self.obs[agent_tag]["external_response"].extend(incoming)  # type: ignore
                for msg in incoming:
                    if msg not in logged_responses:
                        logged_responses.add(msg)
                        self.env.logger.log(
                            time=self.env.step_count,
                            event_type=Event.EXTERNAL_RESPONSE,
                            response=msg,
                        )

            ext_mess_status = sent_status.get(agent_tag, None)
            if ext_mess_status:
                if "external_message_status" not in self.infos[agent_tag]:
                    self.infos[agent_tag]["external_message_status"] = []
                self.infos[agent_tag]["external_message_status"].extend(ext_mess_status)

        return incoming_by_agent

    def _inject_external_state(self) -> None:
        """Let each server's state handler augment every agent's info dict."""
        if not self.infos:
            return
        for handler in self.external_state_handlers.values():
            handler.inject(self.infos)

    async def _refresh_external_context(self) -> None:
        """Show each agent the context that MCP servers publish for it."""
        tags = [tag for tag in self.agents if tag in self.obs]
        names = list(self.external_managers)
        published = await asyncio.gather(*(
            asyncio.to_thread(self.external_managers[name].agent_contexts, tags)
            for name in names
        ))
        for tag in tags:
            contexts = {name: texts[tag] for name, texts in zip(names, published) if texts.get(tag)}
            if contexts:
                self.infos[tag]["external_context"] = contexts
            else:
                self.infos[tag].pop("external_context", None)

    # ------------------------------------------------------------------
    # run_async helpers
    # ------------------------------------------------------------------

    def _setup_signal_handlers(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._handle_term)
        if sys.stdin.isatty():
            threading.Thread(target=self._watch_stdin, daemon=True).start()
            logger.info(
                "ℹ️  Press Ctrl+D to force kill immediately, Ctrl+C to stop after current step."
            )

    async def _maybe_checkpoint(self, ts: int) -> None:
        ckpt_interval = self.params.run.ckpt_interval
        if ckpt_interval and ts % ckpt_interval == 0 and ts != self.start_ts:
            await self._save_checkpoint(ts)

    def _maybe_refresh_llm_router(self) -> None:
        if datetime.now() - self.last_refresh > self.refresh_interval:
            logger.info("🔄 Refreshing LLM clients... 🔄")
            self.llm_router.refresh(
                ports=self.params.run.ports,
                instances=self.params.run.max_parallel_workers,
            )
            self.last_refresh = datetime.now()

    def _refresh_role_context(self):
        if not hasattr(getattr(self, "env", None), "refresh_role_context"):
            return
        context = {"run_id": self._run_id}
        for server, handler in getattr(self, "external_state_handlers", {}).items():
            if hasattr(handler, "execution_context"):
                world_context = handler.execution_context()
                context[f"{server}/context"] = json.dumps(world_context, sort_keys=True)
                context.update({f"{server}/{key}": value for key, value in world_context.items()})
        self.env.execution_context = context
        for tag, observation in self.obs.items():
            self.env.refresh_role_context(tag, observation)

    async def _collect_actions(self, ts: int) -> dict:
        """Dispatch all agents simultaneously and return their chosen actions."""
        self._refresh_role_context()
        await self._refresh_external_context()
        self._history_before_actions = {
            tag: agent.history[-1] if getattr(agent, "history", None) else None
            for tag, agent in self.agents.items()
        }
        # Read available_actions without mutating self.infos so that the dict
        # remains valid for checkpointing even if a crash occurs before env.step()
        # replaces it with the next step's infos.
        available_actions = {
            tag: self.infos[tag]["available_actions"] for tag in self.agents.keys()
        }
        window = self.params.run.env_heartbeat

        async def _bounded(tag, agent):
            # Pass infos to the agent without the available_actions key — it is
            # provided as a dedicated argument, so including it here is redundant.
            tag_info = self.infos.get(tag)
            agent_info = (
                {k: v for k, v in tag_info.items() if k != "available_actions"}
                if tag_info is not None
                else None
            )
            coro = async_select_with_retry(
                agent,
                self.obs[tag],
                available_actions[tag],
                self.rewards.get(tag, 0),
                agent_info,
                ts,
                *self.llm_router.next(),
            )
            if isinstance(agent, (RemoteAgent, HumanAgent)):
                # Remote/human agents don't call a local LLM — skip the semaphore
                # so they're never queued behind LLM agents and always get the
                # full window.
                return await coro
            if self._llm_semaphore is not None:
                async with self._llm_semaphore:
                    return await coro
            return await coro

        tasks = {
            tag: asyncio.create_task(_bounded(tag, agent))
            for tag, agent in self.agents.items()
            if tag in self.obs
        }
        if not tasks:
            if window > 0:
                await asyncio.sleep(window)
            return {}
        if window == -1:
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        else:
            results, _ = await asyncio.gather(
                asyncio.gather(
                    *[asyncio.wait_for(t, timeout=window) for t in tasks.values()],
                    return_exceptions=True,
                ),
                asyncio.sleep(window),
            )
        actions = {
            tag: (
                result
                if not isinstance(result, BaseException)
                else {
                    **self._stay_action,
                    "source": "default",
                    "default_reason": "agent_timeout"
                    if isinstance(result, asyncio.TimeoutError)
                    else "agent_error",
                }
            )
            for tag, result in zip(tasks.keys(), results)
        }
        return actions

    def _log_step_usage(self, ts: int, actions: dict) -> None:
        """Append this timestep's total LLM cost (USD) and token usage to costs.csv.

        Each agent action carries the cost and input/output token counts of the
        LLM call(s) that produced it (summed across retries). Cost is computed by
        litellm, accounting for the model and Anthropic prompt-cache tokens; models
        litellm can't price (e.g. local vLLM) contribute 0.0. Here we sum across all
        agents that acted this timestep and write one row. This supersedes the old
        per-agent agent_logs/token_counts.jsonl."""
        acted = [a for a in actions.values() if isinstance(a, dict)]
        step_cost = sum(a.get("cost", 0.0) for a in acted)
        input_tokens = sum(a.get("input_tokens", 0) for a in acted)
        output_tokens = sum(a.get("output_tokens", 0) for a in acted)
        write_header = not self._cost_csv_path.exists()
        with open(self._cost_csv_path, "a") as f:
            if write_header:
                f.write("timestep,cost_usd,input_tokens,output_tokens\n")
            f.write(f"{ts},{step_cost:.6f},{input_tokens},{output_tokens}\n")

    def _process_external_prestep(self, actions: dict) -> _ExternalState:
        """Vote on agents' external actions and dispatch the validated ones.

        Each server's voting manager decides which requests are enacted (and
        suppresses the rest to no-ops); the runner then dispatches the surviving
        actions to the MCP servers and records this step's bookkeeping.
        """
        if not self.external_managers:
            return _ExternalState()

        # Collect flattened <server>_<tool> actions: agent_tag -> (server, tool, args)
        ext_by_tag: dict[str, tuple[str, str, dict]] = {}
        for agent_tag, action_data in actions.items():
            name = action_data.get("action")
            if name in self.external_actions_spec:
                spec = self.external_actions_spec[name]
                args = dict(action_data.get("params", {}) or {})
                ext_by_tag[agent_tag] = (spec["server"], spec["tool"], args)

        if not ext_by_tag:
            return _ExternalState()

        state = _ExternalState()
        state.requests = {tag: args for tag, (_, _, args) in ext_by_tag.items()}
        state.tools = {tag: (srv, tool) for tag, (srv, tool, _) in ext_by_tag.items()}

        # Per server, let the voting manager validate which actions are enacted.
        requests_by_server: dict[str, dict[str, dict]] = {}
        for tag, (server, _, args) in ext_by_tag.items():
            requests_by_server.setdefault(server, {})[tag] = args
        to_dispatch: dict[str, dict] = {}
        for server, requests in requests_by_server.items():
            validation = self.voting_managers[server].validate(
                actions, requests, self._stay_action, rng=self.env.rng
            )
            to_dispatch.update(validation.to_dispatch)
            state.broadcasts.update(validation.broadcasts)
            for topic, outcome in validation.outcomes.items():
                state.vote_outcomes[outcome.representative] = (topic, outcome)
        state.dispatched = set(to_dispatch)
        for tag in state.dispatched:
            server, tool = state.tools[tag]
            handler = getattr(self, "external_state_handlers", {}).get(server)
            if hasattr(handler, "capture_call_context"):
                state.call_contexts[tag] = handler.capture_call_context(tool, state.requests[tag])
                state.call_contexts[tag]["concurrent_calls"] = sum(
                    state.tools[peer][0] == server for peer in state.dispatched
                )

        # Dispatch the validated actions. A batch server gets ONE call carrying
        # every agent's request (the call itself marks the end of the round);
        # other servers get one call per agent, grouped by (server, tool).
        batch_requests: dict[str, dict[str, dict]] = {}
        batch_tools: dict[str, dict[str, str]] = {}
        dispatch_groups: dict[tuple[str, str], dict[str, dict]] = {}
        for tag, args in to_dispatch.items():
            server, tool, _ = ext_by_tag[tag]
            if self.external_specs[server].batch:
                batch_requests.setdefault(server, {})[tag] = args
                batch_tools.setdefault(server, {})[tag] = tool
            else:
                dispatch_groups.setdefault((server, tool), {})[tag] = args
        for server, msgs in batch_requests.items():
            statuses = self.external_managers[server].send_outgoing_batch(
                msgs, tools=batch_tools[server]
            )
            state.sent_statuses.update(statuses)
        for (server, tool), msgs in dispatch_groups.items():
            statuses = self.external_managers[server].send_outgoing_messages(
                msgs, tool_name=tool
            )
            state.sent_statuses.update(statuses)

        # Coalition peers inherit their representative's delivery status.
        for rep, peers in state.broadcasts.items():
            rep_status = state.sent_statuses.get(
                rep, [{"message_id": "shared", "status": "awaiting_broadcast"}]
            )
            for tag in peers:
                state.sent_statuses[tag] = rep_status

        # Stamp sent_status onto actions so the env can log it.
        for tag, status in state.sent_statuses.items():
            if tag in actions:
                actions[tag]["sent_status"] = status

        return state

    def _process_external_poststep(self, state: _ExternalState, ts: int) -> None:
        """Inject incoming messages, distribute reward, record, and log."""
        if not self.external_managers:
            return

        incoming_by_agent = self._handle_incoming_external_messages(
            state.sent_statuses
        )

        # Broadcast each representative's response to its coalition peers, then
        # tell every voter (representative + peers) which action the election
        # applied vs. the one it proposed — so an overridden voter learns it was
        # overridden and by what. The note carries no "reward" and no state
        # fields, so reward parsing and state handlers skip it.
        for rep, peers in state.broadcasts.items():
            rep_responses = self.obs.get(rep, {}).get("external_response", [])
            for tag in peers:
                if tag in self.obs and rep_responses:
                    self.obs[tag].setdefault("external_response", []).extend(
                        rep_responses
                    )
            selected_action = state.requests.get(rep, {})
            for tag in {rep, *peers}:
                obs_tag = self.obs.get(tag)
                if obs_tag is None:
                    continue
                your_action = state.requests.get(tag, {})
                selected = your_action == selected_action
                vote_outcome = {
                    "your_action": your_action,
                    "selected_action": selected_action,
                    "you_were_selected": selected,
                }
                if not selected:
                    # Agents kept reading election losses as world mechanics
                    # ("my research fire did nothing") — state the mechanics
                    # plainly: the proposal was never sent, never refused.
                    vote_outcome["note"] = (
                        "your proposed action was outvoted by the team and "
                        "was NEVER SENT to the world — it was not executed "
                        "and not refused; the response above belongs to the "
                        "selected action only"
                    )
                obs_tag.setdefault("external_response", []).append(
                    json.dumps({"vote_outcome": vote_outcome})
                )

        # Reward via each server's voting manager (DIAS for unanimous, direct for
        # independent), then update idle tracking for shared-state handlers.
        for server, manager in self.voting_managers.items():
            reward_observations = {}
            for tag in state.dispatched:
                if state.tools[tag][0] != server:
                    continue
                request_id = (state.sent_statuses.get(tag) or [{}])[0].get("message_id")
                responses = [record["text"] for record in incoming_by_agent.get(tag, {}).get(server, [])
                             if request_id and record.get("message_id") == request_id
                             and record.get("text") and not record.get("tool_error")
                             and not record.get("delivery_failed")
                             and record.get("response_complete") is not False]
                reward_observations[tag] = {"external_response": responses}
            manager.settle(reward_observations, self.env)
        for name, handler in self.external_state_handlers.items():
            manager = self.voting_managers.get(name)
            handler.note_step(manager.active_topics() if manager else set())

        self._record_external_outcomes(state, incoming_by_agent, ts)

        with open(self._ext_msg_log_path, "a") as _f:
            for _agent_tag, _statuses in state.sent_statuses.items():
                _status = _statuses[0] if _statuses else {}
                _f.write(
                    json.dumps(
                        {
                            "timestep": ts,
                            "agent": _agent_tag,
                            "message_id": _status.get("message_id"),
                            "request": state.requests.get(_agent_tag, ""),
                            "response": self.obs.get(_agent_tag, {}).get(
                                "external_response", []
                            ),
                            "status": _status.get("status"),
                            "timestamp": _status.get("timestamp"),
                        }
                    )
                    + "\n"
                )

    def _record_social_graph_snapshot(self) -> None:
        """Capture actual topology after population updates, including headless runs."""
        if not isinstance(getattr(self, "env", None), OpenSocialGraphWorld):
            return
        recorder = getattr(self, "_social_graph_recorder", None)
        if recorder is None:
            recorder = SocialGraphHistoryRecorder(
                self.exp_logdir / "social_graph.jsonl", self._run_id,
            )
            self._social_graph_recorder = recorder
            recorder.record(
                self.env, segment_start=True,
                reason="resume" if getattr(self, "resume", False) else "start",
            )
        else:
            recorder.record(self.env)

    async def _advance_population(self) -> None:
        """Run spawning, dead-agent cleanup, and respawn for this timestep."""
        await self._handle_spawn()
        self._cleanup_dead()
        self._respawn_if_needed()
        self._maybe_resize_grid()

    async def _teardown(self, ts: int) -> None:
        """Save final checkpoint, close resources, consolidate logs, create video."""
        if self._cmd_loop_task is not None:
            self._cmd_loop_task.cancel()
            try:
                await self._cmd_loop_task
            except asyncio.CancelledError:
                pass
            self._cmd_loop_task = None
        await self._save_checkpoint(ts=ts)
        self._render(ts=ts)

        self.env.close()
        for agent in self.agents.values():
            agent.close()
        for manager in self.external_managers.values():
            manager.close()

        # Unregister surviving remote agents and publish "died" events so that
        # server_llm_loop tasks in the API process clean up their ogw:bg:* keys.
        if self._use_redis and self.agents:
            remote_tags = [
                tag for tag, a in self.agents.items() if isinstance(a, RemoteAgent)
            ]
            if remote_tags:

                async def _farewell(tags):
                    r = await get_redis()
                    try:
                        for tag in tags:
                            if self.registry is not None:
                                await self.registry.unregister(tag)
                            await r.publish(
                                CHANNEL_AGENT_EVENT,
                                json.dumps({"event": "died", "tag": tag}),
                            )
                    finally:
                        await r.aclose()

                await _farewell(remote_tags)

        if self._ext_msg_log_path.exists():
            ext_msg_log: Dict = {}
            with open(self._ext_msg_log_path) as _f:
                for line in _f:
                    entry = json.loads(line)
                    ts_key = str(entry.pop("timestep"))
                    ext_msg_log.setdefault(ts_key, []).append(entry)
            with open(self.exp_logdir / "external_messages.json", "w") as _f:
                json.dump(ext_msg_log, _f, indent=2)

        if self.params.run.save_video:
            self.run_status.export_started("video")
            try:
                create_video(
                    str(self.exp_logdir / "frames" / "%05d.png"),
                    output_file=str(self.exp_logdir / "video.mp4"),
                    fps=self.params.run.video_fps,
                )
            except Exception as exc:
                self.run_status.export_finished("video", error=exc)
                logger.warning("Simulation saved; optional video export failed: %s", exc)
            else:
                self.run_status.export_finished("video")
        else:
            self.run_status.export_finished("video", skipped=True)

    def _on_simulation_finished(self, completed_steps, reason, error=None):
        self.run_status.simulation_finished(
            completed_steps=completed_steps, reason=reason, error=error,
            completed=error is None and reason in {"step_budget", "external_solved", "empty_population"},
        )

    # ------------------------------------------------------------------
    # Redis command loop (multi-process mode)
    # ------------------------------------------------------------------

    async def _dispatch_cmd(self, cmd: str, params: dict) -> dict:
        """Handle a single command from the API process and return a response dict."""
        try:
            if cmd == "get_config":
                # OpenSocialGraphWorld is a subclass of OpenGraphWorld, so it
                # must be checked first.
                if isinstance(self.env, OpenSocialGraphWorld):
                    world = {
                        "world_type": "social_graph",
                        "topology": self.env.graph_cfg.topology,
                        "hop_radius": self.env.hop_radius,
                        "nodes": len(self.env.world_graph.all_nodes()),
                        "max_connections": self.env.max_connections,
                        "random_intro_k": self.env.random_intro_k,
                        "edge_decay_steps": self.env.edge_decay_steps,
                    }
                elif isinstance(self.env, OpenGraphWorld):
                    world = {
                        "world_type": "graph",
                        "topology": self.env.graph_cfg.topology,
                        "hop_radius": self.env.hop_radius,
                        "nodes": len(self.env.world_graph.all_nodes()),
                    }
                    if self.params.env.graph.agent_network_hocon_path:
                        world["agent_network_hocon_path"] = (
                            self.params.env.graph.agent_network_hocon_path
                        )
                else:
                    world = {
                        "world_type": "grid",
                        "grid_size": self.env.grid_size,
                        "vision_radius": self.env.vision_radius,
                    }
                return {
                    "ok": True,
                    "data": {
                        "allow_human_agents": self.params.run.allow_human_agents,
                        "world": world,
                        "population": {
                            "min_agents": self.params.env.min_agents,
                            "agent_lifespan": _dashboard_resource(self.env.lifespan),
                            "init_agent_energy": _dashboard_resource(self.env.init_agent_energy),
                            "spawn": self.env.spawn_allowed,
                            "spawn_cost": self.env.reproduction_cost
                            if self.env.spawn_allowed
                            else None,
                        },
                        "food": {
                            "enabled": self.env.food_mechanism,
                            "spawn_rate": self.env._food_spawn_rate,
                            "decay_rate": self.env._food_decay_rate,
                        },
                        "artifacts": {
                            "creation": self.env.artifact_creation,
                            "creation_cost": self.env.artifact_creation_cost
                            if self.env.artifact_creation
                            else None,
                            "inert": self.env.inert_artifacts,
                        },
                        "agents": {
                            "max_history": self.params.agent.max_history,
                            "internal_memory": (self.params.agent.internal_memory_size > 0),
                        },
                    },
                }

            elif cmd == "get_genome_info":
                from dataclasses import fields as dc_fields

                from terralingua.genome.ocean_5 import Genome as Ocean5Genome

                ocean_fields = []
                for f in dc_fields(Ocean5Genome):
                    lo, hi = f.metadata["range"]
                    ocean_fields.append(
                        {
                            "name": f.name,
                            "field_type": "float",
                            "range": [lo, hi],
                            "default": f.default,
                            "descr": f.metadata["descr"],
                            "trait_type": f.metadata.get("trait_type", "personality"),
                        }
                    )
                return {
                    "ok": True,
                    "data": {
                        "active": self.params.agent.genome,
                        "types": {
                            "no_traits": {"fields": []},
                            "sentence_mutate": {
                                "fields": [
                                    {
                                        "name": "words",
                                        "field_type": "list",
                                        "default": [],
                                        "descr": "A list of words forming the personality sentence",
                                    }
                                ]
                            },
                            "sentence_directed": {
                                "fields": [
                                    {
                                        "name": "sentence",
                                        "field_type": "text",
                                        "default": "",
                                        "descr": "A sentence describing the agent's personality",
                                    }
                                ]
                            },
                            "ocean_5": {"fields": ocean_fields},
                        },
                    },
                }

            elif cmd == "get_status":
                agents = []
                connected = (
                    set(self.connection_manager.connected_agents())
                    if self.connection_manager
                    else set()
                )
                for tag, agent in self.agents.items():
                    obs = self.obs.get(tag, {}) if hasattr(self, "obs") else {}
                    agents.append(
                        {
                            "tag": tag,
                            "name": agent.agent_name,
                            "type": getattr(agent, "agent_type", type(agent).__name__),
                            "connected": tag in connected,
                            "energy": _dashboard_resource(obs.get("energy"))
                            if isinstance(obs, dict)
                            else None,
                        }
                    )
                allow_human = self.params.run.allow_human_agents
                return {
                    "ok": True,
                    "data": {
                        "timestep": getattr(self, "start_ts", 0),
                        "agent_count": len(agents),
                        "agents": agents,
                        "allow_human_agents": allow_human,
                    },
                }

            elif cmd == "get_messages":
                since = int(params.get("since", 0))
                result = []
                chat = getattr(self.env, "chat", {}) or {}
                for step in sorted(chat.keys()):
                    if int(step) < since:
                        continue
                    msgs = chat[step]
                    if msgs:
                        result.append({"step": int(step), "messages": list(msgs)})
                return {"ok": True, "data": {"messages": result}}

            elif cmd == "get_agent_log":
                tag = params.get("tag", "")
                since = int(params.get("since", 0))
                entries = []
                log_path = self.exp_logdir / "agent_logs" / f"{tag}.jsonl"
                if log_path.exists():
                    with open(log_path) as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                rec = json.loads(line)
                                step = int(rec.get("timestamp", 0))
                                if step < since:
                                    continue
                                action_dict = rec.get("action") or {}
                                obs = rec.get("observation") or {}
                                entries.append(
                                    {
                                        "step": step,
                                        "action": action_dict.get("action"),
                                        "params": action_dict.get("params"),
                                        "message": action_dict.get("message", ""),
                                        "memory": rec.get("internal_memory", ""),
                                        "inventory": obs.get("inventory", []),
                                        "energy": _dashboard_resource(obs.get("energy")),
                                        "time_left": _dashboard_resource(obs.get("time")),
                                    }
                                )
                            except Exception:
                                continue
                return {"ok": True, "data": {"entries": entries}}

            elif cmd == "get_actions":
                from terralingua.environment.actions import ACTION_TEXT

                env = self.env
                allowed = set(ACTION_TEXT.keys())
                if not env.spawn_allowed:
                    allowed.discard("spawn")
                if env.inert_artifacts or not env.artifact_creation:
                    allowed.discard("create_artifact")
                if env.inert_artifacts or not env.use_inventory:
                    allowed.discard("pickup_artifact")
                    allowed.discard("drop_artifact")
                    allowed.discard("give_artifact")
                if not getattr(env, "use_colors", True):
                    allowed.discard("set_color")
                for exc in env.excluded_actions:
                    allowed.discard(exc)
                actions = [
                    {"name": k, "description": ACTION_TEXT[k]["description"]}
                    for k in ACTION_TEXT
                    if k in allowed
                ]
                actions += [
                    {"name": name, "description": spec["description"]}
                    for name, spec in self.external_actions_spec.items()
                ]
                return {"ok": True, "data": {"actions": actions}}

            elif cmd == "register":
                if self.registry is None:
                    return {"ok": False, "error": "Registry not available"}
                agent_name = params.get("agent_name", "agent")
                motivation_prompt = params.get("motivation_prompt", "")
                record = await self.registry.register(
                    agent_name=agent_name,
                    motivation_prompt=motivation_prompt,
                    model_info=params.get("model_info", ""),
                )
                row = params.get("row")
                col = params.get("col")
                position = (row, col) if row is not None and col is not None else None
                try:
                    self.register_remote_agent(
                        agent_tag=record.agent_tag,
                        agent_name=agent_name,
                        motivation_prompt=motivation_prompt,
                        position=position,
                        genome_type=params.get("genome_type"),
                        genome_data=params.get("genome"),
                        excluded_actions=params.get("excluded_actions"),
                        agent_type=params.get("agent_type", "remote_llm"),
                    )
                except ValueError as e:
                    await self.registry.unregister(record.agent_tag)
                    return {"ok": False, "error": str(e)}
                return {
                    "ok": True,
                    "data": {"agent_tag": record.agent_tag, "token": record.token},
                }

            elif cmd == "create_artifact":
                pose = (
                    params["node_id"]
                    if self.params.env.world_type in ("graph", "social_graph")
                    else (params["row"], params["col"])
                )
                message, final_name = self.env.seed_artifact(
                    pose=pose,
                    art_type="text",
                    art_name=params["name"],
                    payload=params.get("payload", ""),
                    lifespan=params.get("lifespan", -1),
                    movable=params.get("movable", True),
                )
                if final_name is None:
                    return {"ok": False, "error": message}
                return {"ok": True, "data": {"status": message, "name": final_name}}

            elif cmd == "update_agent":
                tag = params.get("tag", "")
                token = params.get("token", "")
                record = (
                    await self.registry.get_by_token(token) if self.registry else None
                )
                if record is None or record.agent_tag != tag:
                    return {"ok": False, "error": "Invalid token"}
                if tag in self.agents:
                    agent = self.agents[tag]
                    if not isinstance(agent, RemoteAgent):
                        return {"ok": False, "error": "Agent is not a remote agent"}
                    prompt_dirty = False
                    if params.get("motivation_prompt") is not None:
                        record.motivation_prompt = params["motivation_prompt"]
                        agent.motivation_prompt = params["motivation_prompt"]
                        prompt_dirty = True
                    if params.get("excluded_actions") is not None:
                        self.env.agent_excluded_actions[tag] = set(
                            params["excluded_actions"]
                        )
                    has_genome = (
                        params.get("genome") is not None
                        or params.get("genome_type") is not None
                    )
                    if has_genome:
                        genome_cls = get_genome_class(
                            params.get("genome_type") or agent.genome.genome_type
                        )
                        if params.get("genome"):
                            agent.genome = genome_cls().from_dict(params["genome"])
                        else:
                            agent.genome = genome_cls().random()
                        prompt_dirty = True
                    if prompt_dirty:
                        agent.update_system_prompt()

                return {"ok": True, "data": {"status": "updated"}}

            elif cmd == "kill_agent":
                tag = params.get("tag", "")
                token = params.get("token", "")
                record = (
                    await self.registry.get_by_token(token) if self.registry else None
                )
                if record is None or record.agent_tag != tag:
                    return {"ok": False, "error": "Invalid token"}
                if tag in self.env.agent_energy:
                    self.env._pending_kills[tag] = None
                return {"ok": True, "data": {"status": "killed"}}

            elif cmd == "modify_artifact":
                name = params.get("name", "")
                if name not in self.env.artifacts:
                    return {"ok": False, "error": f"Artifact '{name}' not found"}
                artifact = self.env.artifacts[name]
                new_payload = params.get("payload", artifact.payload)
                if "payload" in params:
                    valid, err = artifact.verify_payload(new_payload)
                    if not valid:
                        return {"ok": False, "error": err}
                new_lifespan = artifact.lifespan
                if "lifespan" in params:
                    try:
                        lifespan = int(params["lifespan"])
                    except (ValueError, TypeError):
                        return {"ok": False, "error": "'lifespan' must be an integer"}
                    new_lifespan = np.inf if lifespan == -1 else lifespan
                # Record current state as a past version before applying changes
                past_version = {
                    "payload": artifact.payload,
                    "lifespan": "inf"
                    if artifact.lifespan == np.inf
                    else artifact.lifespan,
                    "name": artifact.name,
                    "version": artifact.version,
                    "version_creation_time": artifact.version_creation_time,
                    "modified_by": "[user]",
                }
                artifact.past_versions.append(past_version)
                artifact.version += 1
                artifact.version_creation_time = self.env.step_count
                artifact.payload = new_payload
                artifact.lifespan = new_lifespan
                artifact.remaining_time = new_lifespan
                return {
                    "ok": True,
                    "data": {"status": "updated", "version": artifact.version},
                }

            elif cmd == "destroy_artifact":
                name = params.get("name", "")
                if name not in self.env.artifacts:
                    return {"ok": False, "error": f"Artifact '{name}' not found"}
                self.env.artifacts[name].remaining_time = 0
                return {"ok": True, "data": {"status": "destroyed"}}

            elif cmd == "download_messages":
                content = json.dumps(self.env.chat, indent=4)
                return {
                    "ok": True,
                    "data": {"filename": "messages.json", "content": content},
                }

            elif cmd == "get_agent_names":
                tags = params.get("tags", [])
                return {
                    "ok": True,
                    "data": {
                        t: self.env.agent_names[t]
                        for t in tags
                        if t in self.env.agent_names
                    },
                }

            else:
                return {"ok": False, "error": f"Unknown command: {cmd}"}

        except Exception as e:
            logger.exception(f"[Runner] command '{cmd}' raised: {e}")
            return {"ok": False, "error": str(e)}

    async def _run_cmd_loop(self):
        """Subscribe to ogw:cmd, dispatch commands, publish responses."""
        from terralingua.server.redis_bus import CHANNEL_CMD, get_redis, rsp_channel

        r = await get_redis()
        pubsub = r.pubsub()
        await pubsub.subscribe(CHANNEL_CMD)
        logger.info("[Runner] Redis command loop started")
        try:
            async for msg in pubsub.listen():
                if msg["type"] != "message":
                    continue
                try:
                    payload = json.loads(msg["data"])
                    req_id = payload.get("req_id", "")
                    cmd = payload.get("cmd", "")
                    params = payload.get("params", {})
                    response = await self._dispatch_cmd(cmd, params)
                    await r.publish(rsp_channel(req_id), json.dumps(response))
                except Exception as e:
                    logger.error(f"[Runner] cmd loop error: {e}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[Runner] cmd loop fatal error: {e}")
        finally:
            await pubsub.aclose()

    # ------------------------------------------------------------------

    def run(self):
        asyncio.run(self.run_async())

    def _external_task_solved(self) -> bool:
        """True if any external server reported task completion this step — a truthy
        'solved' or 'done' in a response. Used by run.stop_on_external_solved so a
        one-shot task ends the run the moment it is solved rather
        than idling to max_ts."""
        for info in self.obs.values():
            if not isinstance(info, dict):
                continue
            for resp in info.get("external_response", []) or []:
                try:
                    data = json.loads(resp)
                except (json.JSONDecodeError, TypeError):
                    continue
                if isinstance(data, dict) and (data.get("solved") or data.get("done")):
                    return True
        return False

    async def run_async(self):
        self._setup_signal_handlers()
        _agent_logger_mod.start_async_logging()
        self._llm_semaphore = asyncio.Semaphore(
            self.params.run.max_parallel_workers or 64
        )
        if self._use_redis:
            self._cmd_loop_task = asyncio.create_task(self._run_cmd_loop())
            # Publish metadata so the API process can read exp_logdir
            from terralingua.server.redis_bus import KEY_META, get_redis

            _r = await get_redis()
            await _r.hset(KEY_META, "exp_logdir", str(self.exp_logdir))  # type: ignore
        if self.dashboard_manager is not None:
            await self.dashboard_manager.broadcast_status("running")
            await self._push_dashboard_frame()
        empty_countdown = self.params.run.empty_countdown
        ts = self.start_ts
        completed_steps = 0
        reason = "step_budget"
        simulation_error = None
        self.run_status.start(self.start_ts, step_budget=(None if self.params.run.max_ts == -1 else self.params.run.max_ts - self.start_ts))
        try:
            self._record_social_graph_snapshot()
            ts_iter = (
                count(self.start_ts)
                if self.params.run.max_ts == -1
                else range(self.start_ts, self.params.run.max_ts)
            )
            for ts in ts_iter:
                logger.info("\n=== Timestep %d ===", ts)
                self._render(ts)
                await self._maybe_checkpoint(ts)
                self._maybe_refresh_llm_router()
                incoming = self._handle_incoming_external_messages({})
                self._record_external_outcomes(_ExternalState(), incoming, ts)
                self._inject_external_state()

                _decision_start = time.perf_counter()
                actions = await self._collect_actions(ts)
                _decision_ms = (time.perf_counter() - _decision_start) * 1000
                self._capture_action_requests(actions, ts)
                self.last_actions = copy.deepcopy(actions)
                self._log_step_usage(ts, actions)
                self.pre_step_obs = self.obs if isinstance(self.obs, dict) else {}

                ext_state = self._process_external_prestep(actions)
                _step_start = time.perf_counter()
                self.obs, self.rewards, self.dones, _, self.infos = self.env.step(
                    actions
                )
                completed_steps += 1
                _step_ms = (time.perf_counter() - _step_start) * 1000
                logger.info(
                    "[env.step] ts=%d agents=%d artifacts=%d duration_ms=%.1f",
                    ts,
                    len(self.agents),
                    len(self.env.artifacts),
                    _step_ms,
                )
                _maintenance_start = time.perf_counter()
                self._process_external_poststep(ext_state, ts)
                self._record_local_action_receipts(actions, ext_state, ts)
                logger.info(
                    "[step.phases] ts=%d decision_ms=%.1f environment_ms=%.1f maintenance_ms=%.1f",
                    ts, _decision_ms, _step_ms,
                    (time.perf_counter() - _maintenance_start) * 1000,
                )
                await self._advance_population()
                self._record_social_graph_snapshot()

                await self._push_dashboard_frame()

                if (
                    self.params.run.stop_on_external_solved
                    and self._external_task_solved()
                ):
                    reason = "external_solved"
                    logger.info("✅ External task solved — stopping the run.")
                    break

                if len(self.agents) == 0 and empty_countdown != -1:
                    empty_countdown -= 1
                if empty_countdown == 0:
                    reason = "empty_population"
                    break
                if self.terminate:
                    reason = "terminated"
                    logger.info(
                        "⚠️ Terminating as requested. Saving checkpoint and exiting... ⚠️"
                    )
                    break

        except KeyboardInterrupt:
            reason = "interrupted"
            logger.info("⚠️ User interruption ⚠️")
        except asyncio.CancelledError:
            reason = "cancelled"
            logger.info("⚠️ Runner task cancelled — saving data... ⚠️")
        except Exception as e:
            reason = "error"
            simulation_error = e
            logger.exception("Exception: %s", e)
        finally:
            self._on_simulation_finished(completed_steps, reason, simulation_error)
            teardown_error = None
            try:
                if self.dashboard_manager is not None:
                    await self.dashboard_manager.broadcast_status("stopped")
                await self._teardown(self.start_ts + completed_steps)
            except Exception as exc:
                teardown_error = exc
                raise
            finally:
                await _agent_logger_mod.stop_async_logging()
                self.run_status.process_finished(error=teardown_error or simulation_error)
        if simulation_error is not None:
            raise simulation_error
