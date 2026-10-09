"""
base_env.py — BaseWorld: abstract base class for OpenGridWorld and OpenGraphWorld.

Both concrete environments share:
  - All __init__ parameters (except the spatial-topology ones)
  - The full step() loop, including action dispatch and artifact management
  - Agent bookkeeping (energy, time, names, registry, chat)
  - Food decay and respawn skeleton
  - Checkpoint helpers for agent/artifact state
  - _get_avail_actions (all branches except the move action entry)
  - inject_resource, _kill, _build_step_snapshot, _wrap_text, close

Subclasses implement the spatial-specific abstract methods below.

Unified attribute names used by this class:
  self.food          — Dict[pos, float]     food value at each position/node
  self.pos_artifacts — Dict[pos, Set[str]]  artifact names at each position/node
"""

import copy
import hashlib
import json
import logging
import math
import pickle
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Set, Tuple

import numpy as np
import pygame
import tiktoken
from pygame.font import Font

from terralingua.environment.actions import (
    ACTION_TEXT,
    LocationAffordance,
    build_spawn_action,
    create_artifact_text,
)
from terralingua.environment.artifact import (
    ARTIFACT_TYPES,
    MAX_TEXT_ARTIFACT_SIZE,
    Artifact,
    ArtifactCreationError,
    TextArtifact,
    creatable_types,
)
from terralingua.environment.env_logger import Event, JSONLogger
from terralingua.environment.mechanic import Mechanic
from terralingua.environment.roles import RolesMixin
from terralingua.utils.logging_setup import logger_level

log = logging.getLogger(__name__)

_msg_encoder = tiktoken.get_encoding("cl100k_base")


def _collapse_repeats(items: List[str]) -> List[str]:
    """Identical entries, e.g. twenty pieces of one equipment, become one line."""
    if not all(isinstance(item, str) for item in items):
        return items
    counts: Dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return [item if n == 1 else f"{item} (x{n})" for item, n in counts.items()]


class BaseWorld(RolesMixin, ABC):
    """Abstract base for grid-based and graph-based world environments.

    Concrete subclasses must implement all methods decorated with @abstractmethod.
    All agent bookkeeping, artifact management, the main step loop, and
    checkpointing helpers live here.
    """

    system_prompt_template = "_base.j2"
    _LOG_FILENAME: str = "env.log"

    @property
    def stay_action(self) -> dict:
        """Canonical pass-the-turn / no-op action for this world type.

        Grid and graph worlds pass the turn with `move(direction='stay')`.
        Worlds without movement (e.g. social_graph) override this with their
        own no-op. Consumers (runner defaults, voting overrides) read it here
        instead of hardcoding an action, so it always matches the world.
        """
        return {"action": "move", "message": "", "params": {"direction": "stay"}}

    @property
    def energy_upkeep(self) -> int:
        """Energy every agent loses per step: the configured value, else 1 with food_mechanism and 0 without."""
        setting = getattr(self, "_energy_upkeep_setting", None)
        if setting is None:
            return 1 if self.food_mechanism else 0
        return setting

    def __init__(
        self,
        init_agent_energy: int = 100,
        lifespan: int = 100,
        init_food: int = 1250,
        max_food_value: float = 10.0,
        food_decay_rate: float = 0.05,
        food_decay_amount: float = 1.0,
        food_spawn_rate: int = 1,
        log_path: Path | str | None = None,
        drop_food_on_death: bool = True,
        use_inventory: bool = False,
        use_library: bool = False,
        library_preview: bool = True,
        allow_fixed_artifacts: bool = True,
        max_artifact_tokens: int = MAX_TEXT_ARTIFACT_SIZE,
        use_colors: bool = False,
        reproduction_cost: int = 50,
        artifact_creation_cost: int = 0,
        two_parent_spawn: bool = False,
        max_agents: int | None = None,
        food_zones: Any = None,
        food_sigma: float | None = None,
        static_food: bool = False,
        food_mechanism: bool = True,
        energy_upkeep: int | None = None,
        energy_death: bool | None = None,
        verbose: int = 2,
        inert_artifacts: bool = False,
        external_actions_spec: dict | None = None,
        excluded_actions: List[str] | None = None,
        headless: bool = False,
        max_message_length: int = 200,
        exclusive_pos_occupancy: bool = True,
        roles_hocon_path: str | None = None,
        affordances_file_path: str | None = None,
    ):
        self.verbose = verbose
        logging.getLogger(__name__).setLevel(logger_level(verbose))
        if init_agent_energy < 0:
            self.init_agent_energy = np.inf
        else:
            self.init_agent_energy = init_agent_energy
        self.lifespan = lifespan
        self.drop_food_on_death = drop_food_on_death
        self.use_inventory = use_inventory
        self.use_library = use_library
        self.library_preview = library_preview
        self.allow_fixed_artifacts = allow_fixed_artifacts
        self.max_artifact_tokens = max_artifact_tokens
        self.use_colors = use_colors
        if (
            isinstance(reproduction_cost, bool)
            or not isinstance(reproduction_cost, int)
            or reproduction_cost < -1
        ):
            raise ValueError("reproduction_cost must be an integer at least -1.")
        self.reproduction_cost = reproduction_cost
        self.artifact_creation_cost = artifact_creation_cost
        self.inert_artifacts = inert_artifacts
        self.two_parent_spawn = two_parent_spawn
        self.max_agents = max_agents
        self.food_mechanism = food_mechanism
        # Death at energy 0 follows the food mechanism unless set explicitly.
        if energy_upkeep is not None and (
            isinstance(energy_upkeep, bool) or not isinstance(energy_upkeep, int) or energy_upkeep < 0
        ):
            raise ValueError("energy_upkeep must be an integer at least 0.")
        self._energy_upkeep_setting = energy_upkeep
        self.energy_death = (food_mechanism or self.energy_upkeep > 0) if energy_death is None else energy_death
        if not self.food_mechanism:
            self.drop_food_on_death = False

        self._init_food_count = init_food
        self._max_food_value = max_food_value
        self._food_decay_amount = food_decay_amount
        self._food_decay_rate = food_decay_rate
        self._food_spawn_rate = food_spawn_rate
        self.food_zones = food_zones
        self.food_sigma = food_sigma
        self.static_food = static_food
        # Flattened external actions, keyed by `<server>_<tool>` action name. Each
        # value carries {server, tool, cost, mode, description, params}.
        # Built by the runner from servers.json + MCP tool discovery; empty when no
        # external servers are configured.
        self.external_actions_spec: dict = external_actions_spec or {}
        self.excluded_actions = set(excluded_actions) if excluded_actions else set()
        self.max_message_length = max_message_length
        self.agent_genomes: Dict[str, str] = {}
        self.log_path = Path(log_path) if log_path is not None else Path(".")
        self.logger = JSONLogger(self.log_path / self._LOG_FILENAME)

        # Runtime state
        # ---------------------------
        self.rng: np.random.Generator | None = None

        self.food: Dict = {}
        self.food_count: List[float] = []
        self._food_zone_centers = None

        self.artifacts: Dict[str, Artifact] = {}
        self.pos_artifacts: Dict = defaultdict(set)
        self.agent_inventories: Dict[str, Set[str]] = defaultdict(set)
        self.library: Set[str] = set()
        self.artifact_location: Dict[str, tuple] = {}
        self.expired_artifacts: List[Artifact] = []
        self._all_artifact_names: Set[str] = (
            set()
        )  # all names ever used, including expired

        self.exclusive_pos_occupancy = exclusive_pos_occupancy
        self.pos_to_agent: dict = defaultdict(set)

        self.agent_pos: Dict[str, Any] = {}
        self.agent_trajectories: Dict[str, List] = {}
        self.agent_avail_actions: Dict[str, Dict[str, dict]] = {}
        self.agent_excluded_actions: Dict[str, set] = {}
        self.agent_energy: Dict[str, float] = {}
        self.agent_time: Dict[str, float] = {}
        self.agent_spawn: Dict[str, List[str]] = {}
        self.agent_names: Dict[str, str] = {}
        self._pending_kills: Dict[str, str | None] = {}
        self._notes: Dict[str, Dict[str, List[str]]] = {}
        self.mechanics: List[Mechanic] = []
        self.no_appetite: Set[str] = set()
        self.transfers: List[Tuple[str, str, str]] = []
        self.deaths: List[dict] = []
        self._new_deaths: List[dict] = []
        # effect["type"] -> handler(agent, params, infos, affordance).
        self._affordance_effects: Dict[
            str, Callable[[str, dict, dict, LocationAffordance], None]
        ] = {
            "drain_energy": self._handle_drain_energy,
            "add_energy": self._handle_add_energy,
            "move_agent": self._handle_move_agent,
        }
        # Grants and blocks that travel with the agent (role permissions are
        # derived from the assignment, not stored here).
        self.agent_affordances: Dict[str, List[LocationAffordance]] = {}
        self.roles_hocon_path = roles_hocon_path
        self.affordances_file_path = affordances_file_path
        self._init_roles(roles_hocon_path, affordances_file_path)
        self.name_to_tag: Dict[str, str] = {}
        self.agent_colors: Dict[str, str] = {}

        self.msg_raw: Dict[str, Any] = {}
        self.chat: Dict = {}
        # parallel to chat[step]: sender tag for chat[step][i]
        self.chat_senders: Dict[int, List[str]] = {}
        # step -> sender_tag -> set of receiver tags (who had sender in vision that step)
        self.chat_recipients: Dict[int, Dict[str, Set[str]]] = {}
        self.agent_registry: Set[str] = set()
        self.step_count: int = 0
        # ---------------------------

        # Subclass spatial state (initialised here so BaseWorld is self-consistent)
        self.empty_food: list = []

        # Rendering
        self._headless = headless
        self._pygame_inited = False
        self._cell_size = 10
        self._window_size = (500, 500)
        self._screen = None
        self._font: Font

    @property
    def artifact_creation(self) -> bool:
        """Whether agents can create artifacts."""
        return self.artifact_creation_cost >= 0

    @property
    def spawn_allowed(self) -> bool:
        """Whether the configured reproduction rule permits births."""
        return self.reproduction_cost >= 0

    def _reproduction_error(self, agent: str, gift: int = 0) -> str | None:
        """Check reproduction requirements against the current world state."""
        if not self.spawn_allowed:
            return "Reproduction is disabled"
        if self.max_agents is not None and len(self.agent_registry) >= self.max_agents:
            return "Population limit reached"
        if self.agent_energy[agent] < self.reproduction_cost + gift:
            return "Not enough energy"
        return None

    # ---------- public helpers ----------
    def add_agent(
        self,
        agent_tag: str,
        agent_name: str,
        genome_type: str,
        position=None,
        excluded_actions: List[str] | None = None,
        initial_energy: float | None = None,
    ) -> Tuple[dict, dict]:
        """Register an agent and place it in the world."""
        if agent_tag in self.agent_registry:
            raise ValueError(
                f"Agent {self.agent_names[agent_tag]}({agent_tag}) already exists in the environment."
            )
        # Place the agent first so that an occupied-cell error leaves no partial state.
        self._place_agent(agent_tag, pos=position)
        if initial_energy is not None:
            self.agent_energy[agent_tag] = initial_energy
        self.agent_registry.add(agent_tag)
        self.agent_names[agent_tag] = agent_name
        self.name_to_tag[agent_name] = agent_tag
        self.agent_excluded_actions[agent_tag] = set(excluded_actions or [])
        self.agent_genomes[agent_tag] = genome_type
        log.info(
            "Adding agent %s(%s) at position %s.",
            self.agent_names[agent_tag],
            agent_tag,
            self.agent_pos[agent_tag],
        )
        if self.logger:
            self.logger.log(
                time=self.step_count,
                event_type=Event.AGENT_ADDED,
                agent_tag=agent_tag,
                position=self.agent_pos[agent_tag],
                agent_name=self.agent_names[agent_tag],
            )
        observation, has_nearby = self._build_obs(agent_tag)
        avail_actions = self._get_avail_actions(agent_tag, nearby_agents=has_nearby)
        info = {"available_actions": avail_actions}
        return observation, info

    def add_artifact(
        self, pose, art_type, art_name, payload, creator, lifespan, movable: bool = True,
        to_inventory: str | None = None, **artifact_params,
    ) -> str:
        """Adds an artifact created by an agent. Only creatable types are allowed."""
        if not self.artifact_creation:
            return "Failed. Artifact creation is disabled."
        if self.agent_energy[creator] < self.artifact_creation_cost:
            return (
                f"Failed. Agent does not have enough energy. "
                f"Required: {self.artifact_creation_cost}"
            )

        cls = ARTIFACT_TYPES.get(art_type)
        if cls is None or not cls.creatable:
            return (
                f"Artifact type: {art_type} is not a valid type. "
                f"Only artifact valid types are: {list(creatable_types())}"
            )
        art_name = self._unique_artifact_name(art_name)
        try:
            artifact = self._build_artifact(
                cls, art_name, payload, pose, creator, lifespan, movable, artifact_params
            )
        except ArtifactCreationError as e:
            return str(e)

        self.agent_energy[creator] -= self.artifact_creation_cost
        self.artifacts[art_name] = artifact
        self._place_new_artifact(art_name, pose, to_inventory)
        self._all_artifact_names.add(art_name)

        if self.logger:
            self.logger.log(
                time=self.step_count,
                event_type=Event.ARTIFACT_ADDED,
                artifact=artifact.serialize(),
                position=artifact.pose,
                agent_tag=creator,
                agent_name=self.agent_names[creator],
            )
        where = f"in the inventory of {to_inventory}" if to_inventory else f"at position {pose}"
        return f"Created artifact {art_name} of type {art_type} {where}."

    def seed_artifact(
        self, pose, art_type, art_name, payload, lifespan, movable: bool = True,
        to_inventory: str | None = None, **artifact_params,
    ) -> tuple[str, str | None]:
        """Seeds an artifact, bypassing agent energy checks.

        Returns ``(message, final_name)``. On collision, ``art_name`` is
        suffixed (``"X" → "X_1"`` etc.); the suffixed name is returned so
        callers that need to address the artifact later (e.g. recording an
        ownership entry) use the actual stored name. ``final_name`` is None
        when the call failed.
        """
        cls = ARTIFACT_TYPES.get(art_type)
        if cls is None:
            return (
                f"Artifact type: {art_type} is not a valid type. "
                f"Only valid types: {list(ARTIFACT_TYPES)}",
                None,
            )
        art_name = self._unique_artifact_name(art_name)
        try:
            artifact = self._build_artifact(
                cls, art_name, payload, pose, "system", lifespan, movable, artifact_params
            )
        except ArtifactCreationError as e:
            return str(e), None

        self.artifacts[art_name] = artifact
        self._place_new_artifact(art_name, pose, to_inventory)
        self._all_artifact_names.add(art_name)

        if self.logger:
            self.logger.log(
                time=self.step_count,
                event_type=Event.ARTIFACT_ADDED,
                artifact=artifact.serialize(),
                position=artifact.pose,
                agent_tag="system",
                agent_name="system",
            )
        return (
            f"Seeded artifact {art_name} of type {art_type} "
            + (f"in the inventory of {to_inventory}." if to_inventory else f"at position {pose}."),
            art_name,
        )

    # ---------- env lifecycle ----------
    def restart_env(
        self, seed=None, **options
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Fresh world: re-place all agents and seed food."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.agent_pos = {}
        self.agent_trajectories = {}
        self.agent_energy = {}
        self.msg_raw = {}
        self.pos_to_agent = defaultdict(set)
        self.agent_spawn = {}
        self._pending_kills.clear()
        self._notes = {}
        self.no_appetite = set()
        self.transfers = []
        self.deaths = []
        self._new_deaths = []
        self.food.clear()
        self.step_count = 0
        self.chat = {}
        self.chat_senders = {}
        self.chat_recipients = {}
        self.food_count = []
        self.agent_inventories = defaultdict(set)
        self.pos_artifacts = defaultdict(set)
        self.library = set()
        self.artifacts = {}
        self.artifact_location = {}
        self.expired_artifacts = []
        self._all_artifact_names = set()
        self.name_to_tag = {name: tag for tag, name in self.agent_names.items()}

        poses = options.get("agent_poses")
        agent_poses = {tag: None for tag in self.agent_registry}
        if poses is not None:
            agent_poses.update(poses)

        # Sorted: the registry is a set, and each placement draws from the seeded rng,
        # so the iteration order must not depend on Python's hash randomisation.
        for tag in sorted(self.agent_registry):
            self._place_agent(tag, pos=agent_poses[tag])

        if self.food_mechanism:
            self._seed_initial_food()
            self.food_count.append(sum(self.food.values()))

        if self.logger:
            self.logger.log(
                time=self.step_count,
                event_type=Event.ENV_RESET,
                num_agents=len(self.agent_registry),
                agent_poses=self.agent_pos,
                agent_tags=list(self.agent_registry),
                food_count=len(self.food),
                agent_names=self.agent_names,
            )

        infos = {}
        for agent_tag in self.agent_registry:
            avail_actions = self._get_avail_actions(agent_tag=agent_tag)
            infos[agent_tag] = {"available_actions": avail_actions}

        observations = {a: self._build_obs(a)[0] for a in self.agent_registry}
        # One-time role draw once the founding population exists.
        if self._roles and not self._initial_roles_assigned:
            self._ensure_initial_roles()
            self._place_role_occupants()
            for agent_tag in self.agent_registry:
                infos[agent_tag]["available_actions"] = self._get_avail_actions(
                    agent_tag=agent_tag
                )
            observations = {a: self._build_obs(a)[0] for a in self.agent_registry}
        return observations, infos

    # ---------- core mechanics ----------
    def step(
        self, actions
    ) -> Tuple[
        Dict[str, Any],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Any],
    ]:
        """
        `actions` may omit some agents (those on cooldown, etc.).
        Missing agents simply keep previous message and perform no move.
        """

        # ---- default bookkeeping ----
        rewards = {a: 0.0 for a in self.agent_registry}
        infos = {a: {} for a in self.agent_registry}
        done_dict = {a: False for a in self.agent_registry}
        self.transfers = []
        self.deaths, self._new_deaths = self._new_deaths, []
        self._ensure_initial_roles()

        # ---- Process actions from acting agents ----
        # ================================
        for agent, act in actions.items():
            if agent not in self.agent_registry:
                raise ValueError(
                    f"Agent {self.agent_names[agent]}({agent}) not found in the environment."
                )

            # move_params is set when action_name == "move"; None otherwise
            move_params = None

            # Get action name, message and params
            # ---------------------------
            action_name = act.get("action", "move")
            message = act.get("message", "")
            if self.max_message_length > 0:
                tokens = _msg_encoder.encode(message)
                if len(tokens) > self.max_message_length:
                    message = _msg_encoder.decode(tokens[: self.max_message_length])
                    tail = " ".join(message.split()[-6:])
                    infos[agent]["Message outcome"] = (
                        f"Your message was longer than {self.max_message_length} "
                        f'tokens and was cut off after "...{tail}". Only that part '
                        "reached the others."
                    )
            action_params = act.get("params", {})

            # Record message regardless of action validity
            self.msg_raw[agent] = message
            if len(message):
                self.chat.setdefault(self.step_count, list()).append(
                    f"{self.agent_names[agent]}: {message}"
                )
                self.chat_senders.setdefault(self.step_count, list()).append(agent)

            if action_name not in self.agent_avail_actions[agent]:
                if self.agent_avail_actions[agent]:
                    msg = (
                        f"Unknown action: '{action_name}'. "
                        f"Available: {list(self.agent_avail_actions[agent].keys())}"
                    )
                    log.warning("%s(%s) - %s", self.agent_names[agent], agent, msg)
                    infos[agent]["Action error"] = msg
                    self.logger.log(
                        time=self.step_count,
                        event_type=Event.ACTION_REFUSED,
                        agent_tag=agent,
                        agent_name=self.agent_names[agent],
                        action=action_name,
                        reason="unknown_action",
                    )
                continue

            avail_params = self.agent_avail_actions[agent][action_name].get(
                "params", {}
            )

            # Added cause the offspring_genome is not necessary when selecting a partner
            # So the llm might not provide it. Rather than make the action fail, we add a dummy
            if (
                action_name == "spawn"
                and "offspring_genome" in avail_params
                and "offspring_genome" not in action_params
                and action_params.get("partner", "")
            ):
                action_params = {**action_params, "offspring_genome": ""}

            action_spec = self.agent_avail_actions[agent][action_name]
            external_schema = action_spec.get("input_schema")
            optional = (
                set(avail_params) - set(external_schema.get("required", []))
                if isinstance(external_schema, dict)
                else set(action_spec.get("optional", []))
            )
            # Preserve MCP omissions, nulls and explicit values on the wire.
            # Legacy internal actions still receive their empty optional fields.
            if not isinstance(external_schema, dict):
                for pname in optional:
                    if pname in avail_params and pname not in action_params:
                        action_params = {**action_params, pname: ""}

            required = set(avail_params) - optional
            sent = set(action_params)
            if (required - sent) or (sent - set(avail_params)):
                required = sorted(required)
                msg = (
                    f"Wrong params for action '{action_name}': "
                    f"got {list(action_params.keys())}, "
                    f"required {required}"
                    + (f", optional {sorted(optional)}" if optional else "")
                )
                log.warning("%s(%s) - %s", self.agent_names[agent], agent, msg)
                infos[agent]["Action error"] = msg
                self.logger.log(
                    time=self.step_count,
                    event_type=Event.ACTION_REFUSED,
                    agent_tag=agent,
                    agent_name=self.agent_names[agent],
                    action=action_name,
                    reason="wrong_params",
                )
                continue

            outcome = self._mechanic_outcome(agent, action_name, action_params)
            if outcome is not None:
                infos[agent]["Action outcome"] = outcome
                self.logger.log(
                    time=self.step_count,
                    event_type="MECHANIC_ACTION",
                    agent_tag=agent,
                    agent_name=self.agent_names[agent],
                    action=action_name,
                    params=action_params,
                    outcome=outcome,
                )
                continue

            # Per-node location affordances (most specific — a node-attached
            # affordance can override an env-class action of the same name).
            # ---------------------------
            if self._on_location_action(agent, action_name, action_params, infos):
                pass
            # ---------------------------

            # Env-class custom actions (e.g. social_graph's connect/disconnect/
            # noop/send_direct_message). Run when no node affordance claimed
            # the action; supersede any built-in action with the same name.
            # ---------------------------
            elif self._on_env_action(agent, action_name, action_params, infos):
                pass
            # ---------------------------

            # Handle Move
            # ---------------------------
            elif action_name == "move":
                move_params = action_params
            # ---------------------------

            # External action — any flattened `<server>_<tool>` action; deduct
            # that action's own per-server cost.
            # ---------------------------
            elif action_name in self.external_actions_spec:
                self.agent_energy[agent] -= self.external_actions_spec[action_name][
                    "cost"
                ]
                self.logger.log(
                    time=self.step_count,
                    event_type=Event.EXTERNAL_REQUEST,
                    agent_tag=agent,
                    agent_name=self.agent_names[agent],
                    step=self.step_count,
                    request=action_params.get("payload") or action_params,
                    sent_status=act.get("sent_status", ""),
                )
            # ---------------------------

            # Handle Energy exchange
            # ---------------------------
            elif action_name in ("give", "take"):
                nearby_agents = self._get_nearby_agents(agent)
                target_name = action_params["target"]
                target_tag = self.name_to_tag.get(target_name)

                if target_tag is not None and target_tag in nearby_agents:
                    amount = max(float(action_params.get("amount", 0.0)), 0.0)
                    if action_name == "give":
                        energy_source = float(self.agent_energy[agent])
                        xfer = min(amount, energy_source)
                        self.agent_energy[agent] -= xfer
                        self.agent_energy[target_tag] += xfer
                        rewards[agent] += xfer * 0.5
                        rewards[target_tag] += xfer
                        self.transfers.append((agent, target_tag, "give"))
                        infos[target_tag]["Energy received"] = (
                            f"{self.agent_names[agent]} gave you {xfer} energy."
                        )
                        self.logger.log(
                            time=self.step_count,
                            event_type=Event.GIFT_ENERGY,
                            agent_tag=agent,
                            agent_name=self.agent_names[agent],
                            target_tag=target_tag,
                            target_name=self.agent_names[target_tag],
                            amount=xfer,
                            step=self.step_count,
                            target_final_energy=self.agent_energy[target_tag],
                            final_energy=self.agent_energy[agent],
                        )
                    else:  # take
                        energy_target = float(self.agent_energy[target_tag])
                        stolen = min(amount, energy_target)
                        self.agent_energy[target_tag] -= stolen
                        self.agent_energy[agent] += stolen
                        rewards[agent] += stolen
                        rewards[target_tag] -= stolen * 2
                        self.transfers.append((agent, target_tag, "take"))
                        infos[target_tag]["Energy stolen"] = (
                            f"{self.agent_names[agent]} stole {stolen} energy from you."
                        )
                        self.logger.log(
                            time=self.step_count,
                            event_type=Event.TAKE_ENERGY,
                            agent_tag=agent,
                            agent_name=self.agent_names[agent],
                            target_tag=target_tag,
                            target_name=self.agent_names[target_tag],
                            amount=stolen,
                            step=self.step_count,
                            target_final_energy=self.agent_energy[target_tag],
                            final_energy=self.agent_energy[agent],
                        )
                else:
                    infos[agent]["Action outcome"] = (
                        f"Cannot {action_name} energy to {target_name} as not nearby"
                    )
                    rewards[agent] -= 1
            # ---------------------------

            # Change color
            # ---------------------------
            elif action_name == "set_color":
                selected_color = action_params["color"]
                self.agent_colors[agent] = selected_color
                infos[agent]["Color selection"] = (
                    f"Success. Current color {selected_color}"
                )
                self.logger.log(
                    time=self.step_count,
                    event_type=Event.SET_COLOR,
                    agent_tag=agent,
                    agent_name=self.agent_names[agent],
                    color=selected_color,
                )
            # ---------------------------

            # Handle spawning
            # ---------------------------
            elif action_name == "spawn":
                gift = action_params.get("energy", 0)
                gift_error = None
                try:
                    if gift is None or gift == "":
                        gift = 0
                    if isinstance(gift, bool):
                        raise ValueError
                    amount = int(gift)
                    if amount < 0 or (isinstance(gift, float) and amount != gift):
                        raise ValueError
                    gift = amount
                except (TypeError, ValueError, OverflowError):
                    gift_error = "Extra energy must be a nonnegative integer"

                failure = gift_error or self._reproduction_error(agent, gift)
                # Failed attempts pay the base cost. Gifts require a successful birth.
                if self.spawn_allowed:
                    self.agent_energy[agent] -= self.reproduction_cost
                if failure:
                    infos[agent] = {"spawn": {"status": "failed", "reason": failure}}
                    self.logger.log(
                        time=self.step_count,
                        agent_tag=agent,
                        agent_name=self.agent_names[agent],
                        event_type=(
                            Event.AGENT_SPAWNED_PAIRED
                            if self.two_parent_spawn and action_params.get("partner")
                            else Event.AGENT_SPAWNED_SOLO
                        ),
                        step=self.step_count,
                        successful=False,
                        fail_reason=failure,
                    )
                    continue

                # Validate partner if two-parent spawning is enabled
                parent_b_tag = None
                if self.two_parent_spawn:
                    target_name = action_params.get("partner", "")
                    if target_name:
                        parent_b_tag = self.name_to_tag.get(target_name)
                        nearby = self._get_nearby_agents(agent)
                        if parent_b_tag is None or parent_b_tag not in nearby:
                            infos[agent] = {
                                "spawn": {
                                    "status": "failed",
                                    "reason": f"Target agent '{target_name}' not nearby.",
                                }
                            }
                            self.logger.log(
                                time=self.step_count,
                                agent_tag=agent,
                                agent_name=self.agent_names[agent],
                                event_type=Event.AGENT_SPAWNED_PAIRED,
                                step=self.step_count,
                                successful=False,
                                fail_reason="Target not nearby",
                                target_name=target_name,
                            )
                            continue
                        elif (
                            self.agent_genomes[parent_b_tag]
                            != self.agent_genomes[agent]
                        ):
                            # This should never happen if the spawn action is provided correctly
                            infos[agent] = {
                                "spawn": {
                                    "status": "failed",
                                    "reason": f"Target agent '{target_name}' has a different genome type.",
                                }
                            }
                            self.logger.log(
                                time=self.step_count,
                                agent_tag=agent,
                                agent_name=self.agent_names[agent],
                                event_type=Event.AGENT_SPAWNED_PAIRED,
                                step=self.step_count,
                                successful=False,
                                fail_reason="Genome type mismatch",
                                target_name=target_name,
                            )
                            continue

                if parent_b_tag is not None:
                    spawn_event = Event.AGENT_SPAWNED_PAIRED
                else:
                    spawn_event = Event.AGENT_SPAWNED_SOLO

                offspring_name = action_params.get("name", "")[:20]
                already_present = False
                while offspring_name in self.name_to_tag:
                    offspring_name = f"{offspring_name}_1"
                    already_present = True

                spawn_pos = self._offspring_position(
                    center=self.agent_pos[agent], node_id=offspring_name
                )
                if spawn_pos is None:
                    infos[agent] = {
                        "spawn": {
                            "status": "failed",
                            "reason": "No free space around",
                        }
                    }
                    self.logger.log(
                        time=self.step_count,
                        agent_tag=agent,
                        agent_name=self.agent_names[agent],
                        event_type=spawn_event,
                        step=self.step_count,
                        successful=False,
                        fail_reason="No space",
                    )
                else:
                    offspring_idx = 0
                    new_agent_idx = f"{agent}_{offspring_idx}"
                    while new_agent_idx in self.agent_spawn[agent]:
                        offspring_idx += 1
                        new_agent_idx = f"{agent}_{offspring_idx}"

                    # Offspring inherit their parent's per-agent action
                    # restrictions; two-parent spawns inherit the union of both
                    # parents'. Without this, add_agent resets exclusions to
                    # empty and the child silently regains every action.
                    inherited_exclusions = set(
                        self.agent_excluded_actions.get(agent, set())
                    )
                    if parent_b_tag is not None:
                        inherited_exclusions |= self.agent_excluded_actions.get(
                            parent_b_tag, set()
                        )
                    self.add_agent(
                        agent_tag=new_agent_idx,
                        agent_name=offspring_name,
                        genome_type=self.agent_genomes[agent],
                        position=spawn_pos,
                        excluded_actions=inherited_exclusions,
                        initial_energy=(
                            self.reproduction_cost if self.reproduction_cost > 0
                            else self.init_agent_energy
                        ) + gift,
                    )
                    self.agent_spawn[agent].append(new_agent_idx)
                    self.agent_energy[agent] -= gift

                    log.info(
                        "Agent %s(%s) is born from %s(%s).\n"
                        "%s(%s) energy: %s\n"
                        "%s(%s) energy: %s",
                        offspring_name,
                        new_agent_idx,
                        self.agent_names[agent],
                        agent,
                        offspring_name,
                        new_agent_idx,
                        self.agent_energy[new_agent_idx],
                        self.agent_names[agent],
                        agent,
                        self.agent_energy[agent],
                    )

                    spawn_info = {
                        "status": "successful",
                        "child_name": offspring_name,
                        "child_tag": new_agent_idx,
                    }
                    if parent_b_tag is not None:
                        spawn_info["parent_b_tag"] = parent_b_tag
                        spawn_info["parent_b_name"] = self.agent_names[parent_b_tag]

                    if self.agent_genomes[agent] == "sentence_directed":
                        spawn_info["offspring_genome"] = action_params.get(
                            "offspring_genome", ""
                        )
                    if already_present:
                        spawn_info["note"] = (
                            f"An agent with name {action_params.get('name')} was already "
                            f"present. Offspring named: {offspring_name}"
                        )

                    infos[agent] = {"spawn": spawn_info}
                    infos[new_agent_idx] = {}
                    log_kwargs: Dict[str, Any] = dict(
                        time=self.step_count,
                        agent_tag=agent,
                        agent_name=self.agent_names[agent],
                        event_type=spawn_event,
                        step=self.step_count,
                        child_name=offspring_name,
                        child_tag=new_agent_idx,
                        successful=True,
                        energy_gifted=gift,
                        final_energy=self.agent_energy[agent],
                        child_energy=self.agent_energy[new_agent_idx],
                    )
                    if parent_b_tag is not None:
                        log_kwargs["parent_b_tag"] = parent_b_tag
                        log_kwargs["parent_b_name"] = self.agent_names[parent_b_tag]
                    self.logger.log(**log_kwargs)

            # ---------------------------

            # Artifact creation
            # ---------------------------
            elif action_name == "create_artifact":
                if not self.artifact_creation:
                    infos[agent]["Artifact creation status"] = "Failed. Artifact creation is disabled."
                    continue
                pose = self.agent_pos[agent]
                art_type = action_params.get("type", "text")
                art_name = action_params.get("name", "")
                if not art_name or not str(art_name).strip():
                    infos[agent]["Artifact creation status"] = (
                        "Failed. Artifact name cannot be empty."
                    )
                    continue
                art_name = str(art_name).strip()
                payload = action_params.get("payload", "")
                try:
                    lifespan = int(action_params["lifespan"])
                except ValueError:
                    infos[agent]["Artifact creation status"] = (
                        "Failed. 'lifespan' must be an integer."
                    )
                    continue
                except KeyError:
                    infos[agent]["Artifact creation status"] = (
                        "Failed. 'lifespan' parameter is required for artifact creation."
                    )
                    continue

                if lifespan == -1:
                    lifespan = np.inf
                movable = True
                if self.allow_fixed_artifacts:
                    movable = action_params.get("movable", True)
                    if isinstance(movable, str) and movable.lower() in ("true", "false"):
                        movable = movable.lower() == "true"
                    if not isinstance(movable, bool):
                        infos[agent]["Artifact creation status"] = (
                            "Failed. 'movable' must be true or false."
                        )
                        continue
                existing_artifact = self.artifacts.get(art_name)
                status = self.add_artifact(
                    pose=pose,
                    art_type=art_type,
                    art_name=art_name,
                    payload=payload,
                    creator=agent,
                    lifespan=lifespan,
                    movable=movable,
                )
                infos[agent]["Artifact creation status"] = status
                created = self.artifacts.get(art_name)
                actual_creation = created is not None and created is not existing_artifact
                infos[agent]["artifact_effect"] = {
                    "effect_status": "confirmed" if actual_creation else "not_observed",
                    "effect_evidence": {"kind": "artifact_creation", "name": art_name,
                                        "created": actual_creation,
                                        "version": created.version if actual_creation else None},
                }
            # ---------------------------

            # Artifact pickup
            # ---------------------------
            elif action_name == "pickup_artifact":
                art_to_pickup = action_params.get("name")
                pose = self.agent_pos[agent]

                if art_to_pickup in self.artifacts:
                    if art_to_pickup in self.pos_artifacts[pose]:
                        if not self.artifacts[art_to_pickup].movable:
                            status = f"Failed. Artifact {art_to_pickup} is fixed and cannot be picked up"
                        else:
                            self.artifacts[art_to_pickup].users[agent].add(
                                self.step_count
                            )
                            self.artifacts[art_to_pickup].pose = f"Inv({agent})"
                            self.pos_artifacts[pose].remove(art_to_pickup)
                            self.agent_inventories[agent].add(art_to_pickup)
                            self.artifact_location[art_to_pickup] = ("inv", agent)
                            status = "Success"
                    else:
                        status = f"Failed. No artifact with name {art_to_pickup} at current position"
                else:
                    status = f"Failed. Artifact {art_to_pickup} does not exist"

                infos[agent]["Artifact pickup status"] = status
                self.logger.log(
                    time=self.step_count,
                    event_type=Event.ARTIFACT_PICKUP,
                    agent_tag=agent,
                    agent_name=self.agent_names[agent],
                    artifact_name=art_to_pickup,
                    status=status,
                    pose=pose,
                )
            # ---------------------------

            # Artifact drop
            # ---------------------------
            elif action_name == "drop_artifact":
                art_to_drop = action_params.get("name")
                pose = self.agent_pos[agent]

                if art_to_drop in self.artifacts:
                    if art_to_drop in self.agent_inventories[agent]:
                        self.artifacts[art_to_drop].users[agent].add(self.step_count)
                        self.artifacts[art_to_drop].pose = pose
                        self.agent_inventories[agent].remove(art_to_drop)
                        self.pos_artifacts[pose].add(art_to_drop)
                        self.artifact_location[art_to_drop] = ("map", pose)
                        status = "Success"
                    else:
                        status = (
                            f"Failed. No artifact with name {art_to_drop} in inventory"
                        )
                else:
                    status = f"Failed. Artifact {art_to_drop} does not exist"

                infos[agent]["Artifact drop status"] = status
                self.logger.log(
                    time=self.step_count,
                    event_type=Event.ARTIFACT_DROP,
                    agent_tag=agent,
                    agent_name=self.agent_names[agent],
                    artifact_name=art_to_drop,
                    status=status,
                    pose=pose,
                )
            # ---------------------------

            # Artifact gift
            # ---------------------------
            elif action_name == "give_artifact":
                nearby_agents_list = self._get_nearby_agents(agent)
                art_to_gift = action_params["artifact_name"]

                target_name = action_params["target_agent"]
                target_tag = self.name_to_tag.get(target_name)

                if target_tag is None:
                    status = f"Failed. No being with name {target_name}"
                elif target_tag in nearby_agents_list:
                    giver_node = self.agent_pos[agent]
                    if art_to_gift in self.agent_inventories[agent]:
                        # inventory -> target's inventory
                        self.agent_inventories[agent].remove(art_to_gift)
                        self.agent_inventories[target_tag].add(art_to_gift)
                        self.artifacts[art_to_gift].pose = f"Inv({target_tag})"
                        self.artifact_location[art_to_gift] = ("inv", target_tag)
                        self.artifacts[art_to_gift].users[agent].add(self.step_count)
                        self.artifacts[art_to_gift].users[target_tag].add(
                            self.step_count
                        )
                        status = "Success"
                    elif art_to_gift in self.pos_artifacts[giver_node]:
                        # node -> target's node
                        target_node = self.agent_pos[target_tag]
                        self.pos_artifacts[giver_node].discard(art_to_gift)
                        self.pos_artifacts[target_node].add(art_to_gift)
                        self.artifacts[art_to_gift].pose = target_node
                        self.artifact_location[art_to_gift] = ("map", target_node)
                        self.artifacts[art_to_gift].users[agent].add(self.step_count)
                        self.artifacts[art_to_gift].users[target_tag].add(
                            self.step_count
                        )
                        status = "Success"
                    else:
                        status = (
                            f"Failed. No artifact with name {art_to_gift} on your "
                            "node or in inventory"
                        )
                else:
                    status = f"Failed. Target being {target_name} not nearby."

                infos[agent]["Artifact give status"] = status
                if status == "Success":
                    self.transfers.append((agent, target_tag, "give_artifact"))
                self.logger.log(
                    time=self.step_count,
                    event_type=Event.GIVE_ARTIFACT,
                    agent_tag=agent,
                    agent_name=self.agent_names[agent],
                    target_tag=target_tag,
                    target_name=target_name,
                    artifact_name=art_to_gift,
                    status=status,
                )
            # ---------------------------

            # Node affordances + artifact active interaction
            # ---------------------------
            else:
                pose = self.agent_pos[agent]
                interactable_artifacts = list(self.pos_artifacts[pose])
                interactable_artifacts.extend(self.agent_inventories[agent])

                art_name = str(action_params.get("name", ""))
                artifact_found = (
                    art_name in interactable_artifacts
                    and action_name in self.artifacts[art_name].actions
                )

                if artifact_found:
                    artifact = self.artifacts[art_name]
                    before = copy.deepcopy({"payload": artifact.payload, "version": artifact.version,
                                            "remaining_time": artifact.remaining_time})
                    interaction_result = self.artifacts[art_name].interact(
                        agent_name=agent,
                        action=action_name,
                        params=action_params,
                        timestamp=self.step_count,
                    )
                    infos[agent]["Artifact interaction result"] = interaction_result
                    changed = [key for key, value in before.items() if getattr(artifact, key) != value]
                    infos[agent]["artifact_effect"] = {
                        "effect_status": "confirmed" if changed else "not_observed",
                        "effect_evidence": {
                            "kind": "artifact_state_change", "name": art_name,
                            "changed_fields": changed, "before_version": before["version"],
                            "after_version": artifact.version,
                            "payload_sha256": hashlib.sha256(str(artifact.payload).encode()).hexdigest(),
                            "note": "Observed fields changed; this does not certify the content's truth.",
                        },
                    }
                    self.logger.log(
                        time=self.step_count,
                        agent_tag=agent,
                        agent_name=self.agent_names[agent],
                        event_type=Event.ARTIFACT_INTERACTION,
                        action=action_name,
                        action_params=action_params,
                        result=interaction_result,
                        artifact=self.artifacts[art_name].serialize(),
                    )
                else:
                    infos[agent]["Artifact interaction result"] = (
                        f"No artifact named '{art_name}' with action "
                        f"{action_name} at your position or in your inventory"
                    )
            # ---------------------------

            # Move and eat
            # ---------------------------
            new_pos, infos = self._apply_move(agent, move_params, infos)
            if new_pos in self.food and agent not in self.no_appetite:
                val = self.food.pop(new_pos)
                if self.static_food:
                    self.empty_food.append(new_pos)
                self.agent_energy[agent] += val
                rewards[agent] += val
            # ---------------------------

            # ---------------------------
        # ================================

        # Artifact passive effects
        # ================================
        if not self.inert_artifacts:
            for agent in self.agent_registry:
                pos = self.agent_pos[agent]
                passive_effects = []
                for art_name in self.pos_artifacts[pos]:
                    effect = self.artifacts[art_name].passive_effect(
                        timestamp=self.step_count, agent_name=agent
                    )
                    passive_effects.append(effect)
                    self.logger.log(
                        time=self.step_count,
                        agent_tag=agent,
                        agent_name=self.agent_names[agent],
                        event_type=Event.ARTIFACT_PASSIVE_INTERACTION,
                        result=effect,
                        artifact=self.artifacts[art_name].serialize(),
                    )
                if passive_effects:
                    infos[agent][
                        "Artifacts here"
                    ] = _collapse_repeats(passive_effects)

                passive_effects = []
                for art_name in self.agent_inventories[agent]:
                    effect = self.artifacts[art_name].passive_effect(
                        timestamp=self.step_count, agent_name=agent
                    )
                    passive_effects.append(effect)
                if passive_effects:
                    infos[agent][
                        "Artifacts in your inventory"
                    ] = _collapse_repeats(passive_effects)
        # ================================

        # World events
        # ================================
        # Handle artifact expiration
        # ---------------------------
        # Map
        updated_artifacts = defaultdict(set)
        for cell, artifacts in self.pos_artifacts.items():
            for art_name in artifacts:
                artifact = self.artifacts[art_name]
                if artifact.creation_time != self.step_count:
                    artifact.remaining_time -= 1
                if artifact.remaining_time <= 0:
                    self.artifacts.pop(art_name)
                    self.artifact_location.pop(art_name, None)
                    artifact.deletion_time = self.step_count
                    self.expired_artifacts.append(artifact)
                    self.logger.log(
                        time=self.step_count,
                        event_type=Event.ARTIFACT_REMOVED,
                        artifact=artifact.serialize(),
                        pose=cell,
                    )
                else:
                    updated_artifacts[cell].add(art_name)
        self.pos_artifacts = updated_artifacts

        # Inventory
        updated_inventory = defaultdict(set)
        for agent_tag, inventory in self.agent_inventories.items():
            for art_name in inventory:
                artifact = self.artifacts[art_name]
                if artifact.creation_time != self.step_count:
                    artifact.remaining_time -= 1
                if artifact.remaining_time <= 0:
                    self.artifacts.pop(art_name)
                    self.artifact_location.pop(art_name, None)
                    artifact.deletion_time = self.step_count
                    self.expired_artifacts.append(artifact)
                    self.logger.log(
                        time=self.step_count,
                        event_type=Event.ARTIFACT_REMOVED,
                        artifact=artifact.serialize(),
                        possessor_tag=agent_tag,
                        possessor_name=self.agent_names[agent_tag],
                    )
                else:
                    updated_inventory[agent_tag].add(art_name)
        self.agent_inventories = updated_inventory

        # Library
        updated_library: Set[str] = set()
        for art_name in self.library:
            artifact = self.artifacts[art_name]
            if artifact.creation_time != self.step_count:
                artifact.remaining_time -= 1
            if artifact.remaining_time <= 0:
                self.artifacts.pop(art_name)
                self.artifact_location.pop(art_name, None)
                artifact.deletion_time = self.step_count
                self.expired_artifacts.append(artifact)
                self.logger.log(
                    time=self.step_count,
                    event_type=Event.ARTIFACT_REMOVED,
                    artifact=artifact.serialize(),
                )
            else:
                updated_library.add(art_name)
        self.library = updated_library
        # ---------------------------

        # ---- Scenario mechanics ----
        if self.mechanics:
            for a in self.agent_registry:
                infos.setdefault(a, {})
            for mechanic in self.mechanics:
                mechanic.on_step(self, infos)

        # ---- Handle energy and time loss for all agents ----
        # Energy only drains over time when the food mechanism is on. Without
        # food, energy is a pure action-cost currency (no passive starvation).
        for a in self.agent_registry:
            if self.energy_upkeep:
                self.agent_energy[a] -= self.energy_upkeep
            self.agent_time[a] -= 1

        # ---- handle deaths ----
        # Energy death is independently configurable; lifespan and explicit
        # kills apply regardless of the food setting.
        dead = {
            a: self._build_obs(a)[0]
            for a in self.agent_registry
            if (self.energy_death and self.agent_energy[a] <= 0)
            or self.agent_time[a] <= 0
            or a in self._pending_kills
        }
        for a in dead:
            self._kill(a, reason=self._pending_kills.get(a))
            done_dict[a] = True
            rewards[a] -= 100
        self._pending_kills.clear()

        # ---- handle food ----
        if self.food_mechanism:
            self._decay_and_respawn_food()
        # ================================

        # Build observations and available actions in one scan per agent.
        # The step snapshot pre-computes food/artifact strings once for all agents.
        food_snap, art_snap = self._build_step_snapshot()
        observations = {}
        for agent_tag in self.agent_registry:
            obs, has_nearby = self._build_obs(
                agent_tag, food_snapshot=food_snap, artifact_snapshot=art_snap
            )
            observations[agent_tag] = obs
            avail_actions = self._get_avail_actions(agent_tag, nearby_agents=has_nearby)
            if agent_tag not in infos:
                infos[agent_tag] = {}
            infos[agent_tag]["available_actions"] = avail_actions
            for key, lines in self._notes.pop(agent_tag, {}).items():
                merged = "\n".join(lines)
                existing = infos[agent_tag].get(key)
                if existing is not None:
                    merged = f"{existing}\n{merged}"
                infos[agent_tag][key] = merged
            if self.use_colors:
                infos[agent_tag]["Your color"] = self.agent_colors.get(
                    agent_tag, "no color"
                )
        observations.update(dead)  # include dead agents' last obs
        self._notes.clear()

        self.step_count += 1
        self.food_count.append(sum(self.food.values()))

        return observations, rewards, done_dict, done_dict, infos

    # ---------- internal helpers ----------
    def inject_resource(
        self,
        amount: float,
        coefficient: float,
        target: str | None,
    ) -> None:
        """Inject or withdraw a resource from an external source.

        Converts an external quantity into in-simulation resources via a
        conversion coefficient, then routes the result to the chosen target.

            effective = amount * coefficient

        target=None  — place/remove food totalling `effective` value.
        target="all" — distribute `effective` energy equally across all agents.
        target=tag   — give `effective` energy directly to the named agent.

        Negative `amount` (or `coefficient`) withdraws resources.
        Every call is logged as a RESOURCE_SIGNAL event.
        """
        if target is not None and target != "all" and target not in self.agent_registry:
            log.warning(
                "[inject_resource] unknown target %r — "
                "must be None, 'all', or a valid agent tag. Skipping.",
                target,
            )
            return

        effective = amount * coefficient

        if target is None:
            if self.rng is None:
                self.rng = np.random.default_rng()
            if effective > 0:
                n_full = int(effective // self._max_food_value)
                remainder = effective % self._max_food_value
                for _ in range(n_full):
                    self._respawn_food_one()
                if remainder > 0:
                    self._respawn_food_one(value=remainder)
            elif effective < 0:
                to_remove = abs(effective)
                positions = list(self.food.keys())
                self.rng.shuffle(positions)
                for pos in positions:
                    if to_remove <= 0:
                        break
                    tile_val = self.food[pos]
                    if tile_val <= to_remove:
                        self.food.pop(pos)
                        if self.static_food:
                            self.empty_food.append(pos)
                        to_remove -= tile_val
                    else:
                        self.food[pos] = tile_val - to_remove
                        to_remove = 0
        elif target == "all":
            n = len(self.agent_registry)
            if n > 0:
                per_agent = effective / n
                for tag in self.agent_registry:
                    self.agent_energy[tag] += per_agent
        else:
            self.agent_energy[target] += effective

        if self.logger:
            self.logger.log(
                time=self.step_count,
                event_type=Event.RESOURCE_SIGNAL,
                amount=amount,
                coefficient=coefficient,
                target=target,
                effective=effective,
            )

    def _decay_and_respawn_food(self) -> None:
        """Decay food values and respawn expired ones."""
        if self.rng is None:
            self.rng = np.random.default_rng()

        roll = self.rng.random(len(self.food))
        expired = []
        items = list(self.food.items())
        for i, (pos, val) in enumerate(items):
            if roll[i] < self._food_decay_rate:
                new_val = val - self._food_decay_amount
                if new_val <= 0:
                    expired.append(pos)
                else:
                    self.food[pos] = new_val

        for pos in expired:
            self.food.pop(pos, None)
            if self.static_food:
                self.empty_food.append(pos)

        n_new_food = self.rng.poisson(self._food_spawn_rate)
        for _ in range(n_new_food):
            self._respawn_food_one()

    def attach(self, mechanic: Mechanic) -> None:
        """Add a mechanic. Its hooks run after those of mechanics attached before it."""
        if not isinstance(getattr(mechanic, "state", None), dict):
            raise TypeError(f"{type(mechanic).__name__}.__init__ must call super().__init__()")
        if any(m.name == mechanic.name for m in self.mechanics):
            raise ValueError(f"A mechanic named '{mechanic.name}' is already attached")
        self.mechanics.append(mechanic)

    def agent_identity(self, tag: str) -> dict:
        """Name and persona for a new agent, from the first mechanic that answers."""
        for mechanic in self.mechanics:
            identity = mechanic.identity(self, tag)
            if identity:
                return dict(identity)
        return {}

    def agent_identity_settled(self, tag: str, identity: dict) -> dict:
        """The identity of a new agent once its persona is settled, completed by every mechanic."""
        for mechanic in self.mechanics:
            extra = mechanic.on_identity(self, tag, identity)
            if extra:
                identity = {**identity, **{key: value for key, value in extra.items() if value}}
        return identity

    def _mechanic_outcome(self, agent: str, action_name: str, params: dict) -> str | None:
        for mechanic in self.mechanics:
            outcome = mechanic.on_action(self, agent, action_name, params)
            if outcome is not None:
                return outcome
        return None

    def kill(self, tag: str, reason: str) -> None:
        """Queue a death. The world kills the agent in this step's death phase."""
        self._pending_kills[tag] = reason

    def distance(self, pos_a, pos_b) -> float:
        """Distance between two positions in this world's own measure."""
        raise NotImplementedError

    def agents_within(self, tag: str, r: float) -> List[str]:
        """Tags of the other agents within distance r of the agent."""
        origin = self.agent_pos[tag]
        return sorted(
            other
            for other, pos in self.agent_pos.items()
            if other != tag and self.distance(origin, pos) <= r
        )

    def _unique_artifact_name(self, art_name: str) -> str:
        idx_count = 1
        while art_name in self._all_artifact_names:
            art_name = f"{art_name}_{idx_count}"
            idx_count += 1
        return art_name

    def _build_artifact(
        self, cls, art_name, payload, pose, creator, lifespan, movable, params: dict
    ) -> Artifact:
        kwargs = dict(params)
        if issubclass(cls, TextArtifact):
            kwargs.setdefault("max_tokens", self.max_artifact_tokens)
        return cls(
            name=art_name,
            payload=payload,
            pose=pose,
            creator=creator,
            lifespan=lifespan,
            creation_time=self.step_count,
            movable=movable,
            **kwargs,
        )

    @staticmethod
    def artifact_from_ckpt(data: dict) -> Artifact:
        """Rebuild an artifact from its serialized form, by its registered type."""
        art_type = data.get("art_type")
        cls = ARTIFACT_TYPES.get(art_type)
        if cls is None:
            raise ValueError(f"Checkpoint holds an artifact of unknown type '{art_type}'")
        return cls.deserialize(data)

    def _place_new_artifact(self, art_name: str, pose, to_inventory: str | None) -> None:
        if to_inventory is None:
            self.pos_artifacts[pose].add(art_name)
            self.artifact_location[art_name] = ("map", pose)
        else:
            self.agent_inventories[to_inventory].add(art_name)
            self.artifacts[art_name].pose = f"Inv({to_inventory})"
            self.artifact_location[art_name] = ("inv", to_inventory)

    def note(self, tag: str, key: str, text: str) -> None:
        """Append one line to the agent's next observation under `key`."""
        self._notes.setdefault(tag, {}).setdefault(key, []).append(str(text))

    def _kill(self, agent: str, reason: str | None = None) -> None:
        """Remove agent from environment. Dead agents leave food behind.

        `reason` names the cause in the log. Without it the cause is derived
        from the agent's lifespan and energy.
        """
        self._release_role(agent, cause="death")
        self._forget_successor(agent)
        self.agent_affordances.pop(agent, None)
        self.no_appetite.discard(agent)
        if reason is not None:
            announcement = f"☠️ Agent {self.agent_names[agent]}({agent}) died: {reason}."
        elif self.agent_time[agent] <= 0:
            announcement = (
                f"☠️ Agent {self.agent_names[agent]}({agent}) died of old age."
            )
            reason = "old age"
        elif self.energy_death and self.agent_energy[agent] <= 0:
            reason = "hunger" if self.food_mechanism else "energy"
            announcement = (
                f"☠️ Agent {self.agent_names[agent]}({agent}) died: {reason} at 0."
            )
        else:
            announcement = f"☠️ Agent {self.agent_names[agent]}({agent}) died."
            reason = "unknown"
        log.info(announcement)

        self.agent_registry.discard(agent)
        self.agent_genomes.pop(agent, None)
        energy = self.agent_energy.pop(agent, 0)
        self.agent_time.pop(agent, None)
        self.msg_raw.pop(agent, None)
        pos = self.agent_pos.pop(agent, None)

        dropped: List[str] = []
        if pos is not None:
            artifacts = self.agent_inventories.pop(agent, None)
            if artifacts:
                dropped = sorted(artifacts)
                for art_name in artifacts:
                    self.artifacts[art_name].pose = pos
                    self.pos_artifacts[pos].add(art_name)
                    self.artifact_location[art_name] = ("map", pos)

            self.pos_to_agent[pos].discard(agent)

            if self.drop_food_on_death:
                self.food[pos] = self.food.get(pos, 0) + max(
                    self._max_food_value, energy
                )

        self._new_deaths.append({
            "tag": agent,
            "name": self.agent_names[agent],
            "position": pos,
            "reason": reason,
            "artifacts": dropped,
            "step": self.step_count,
        })
        if self.logger:
            self.logger.log(
                time=self.step_count,
                event_type=Event.AGENT_DIED,
                agent_tag=agent,
                agent_name=self.agent_names[agent],
                position=pos,
                step=self.step_count,
                energy=energy,
                reason=reason,
            )

    def _inventory_lines(self, agent: str) -> List[str]:
        """One line per kind of artifact; several of a kind share it, counted."""
        groups: Dict[Tuple[str, bool], List[str]] = {}
        for art in self.agent_inventories[agent]:
            artifact = self.artifacts[art]
            key = (str(artifact.art_type), bool(artifact.movable))
            groups.setdefault(key, []).append(str(artifact.name))
        lines = []
        for (art_type, movable), names in sorted(groups.items()):
            head = f"A({art_type},{'movable' if movable else 'fixed'})"
            names = sorted(names)
            if len(names) == 1:
                lines.append(f"{head}: {names[0]}")
            else:
                lines.append(f"{head} x{len(names)}: {', '.join(names)}")
        return lines

    def _build_step_snapshot(self) -> Tuple[dict, dict]:
        """Pre-compute per-position food and artifact strings once per step."""
        food_snap: dict = {pos: str(val) for pos, val in self.food.items()}
        art_snap: dict = {}
        if not self.inert_artifacts:
            for pos, art_names in self.pos_artifacts.items():
                if art_names:
                    art_snap[pos] = [
                        f"A({self.artifacts[n].art_type},"
                        f"{'movable' if self.artifacts[n].movable else 'fixed'}): "
                        f"{self.artifacts[n].name}"
                        for n in art_names
                    ]
        return food_snap, art_snap

    def _on_env_action(
        self,
        agent: str,
        action_name: str,
        action_params: dict,
        infos: dict,
    ) -> bool:
        """Handle env-class custom actions (those defined by the env type
        itself, not by a per-node affordance).

        The base handles the role actions. Subclasses that add their own action
        vocabulary (e.g. social_graph's `connect`, `disconnect`, `noop`,
        `send_direct_message`) override this and fall back to super(). Return
        True if the action was handled (so the standard dispatch is skipped),
        False otherwise.
        """
        return self._on_roles_action(agent, action_name, action_params, infos)

    # ---------- location affordances ----------
    def _location_affordance_list(self, location: Any) -> List[LocationAffordance]:
        """Affordances attached to *location*. Graph nodes and grid cells override."""
        return []

    def _parse_location(self, text: str):
        """Turn an agent-supplied location string into a position, or None."""
        return text

    def _place_role_occupants(self) -> None:
        """Move role occupants to their start places. Only graph worlds have them."""

    def _get_location_affordance(self, location: Any) -> dict:
        """Returns add-mode affordances at *location* as an agent-facing dict.

        Strips the effect field — the agent only sees action, description and
        params. Remove-mode affordances are handled by _get_location_removals.
        """
        result = {}
        for aff in self._location_affordance_list(location):
            if aff.mode == "add":
                result[aff.action] = {
                    "description": aff.description,
                    "params": aff.params,
                }
        return result

    def _get_location_removals(self, location: Any) -> set:
        """Returns action names to suppress at *location* (remove-mode affordances)."""
        return {
            aff.action
            for aff in self._location_affordance_list(location)
            if aff.mode == "remove"
        }

    def _on_location_action(
        self, agent: str, action_name: str, action_params: dict, infos: dict
    ) -> bool:
        """Handles custom (non-built-in) add-mode affordance actions.

        Built-in ACTION_TEXT actions (move, give, spawn, …) are
        never intercepted here — they fall through to their standard handlers so
        energy deduction and event logging work correctly.

        When the matched affordance carries an ``effect["type"]`` registered in
        ``self._affordance_effects``, the corresponding handler is invoked.
        """
        if action_name in ACTION_TEXT:
            return False
        # Same precedence as the menu: agent entries first, a block anywhere wins.
        candidates = list(self._agent_affordance_list(agent))
        candidates.extend(self._location_affordance_list(self.agent_pos[agent]))
        if any(a.action == action_name and a.mode == "remove" for a in candidates):
            infos.setdefault(agent, {})[action_name] = {
                "status": "failed",
                "reason": f"'{action_name}' is not available to you.",
            }
            return True
        for aff in candidates:
            if aff.action == action_name and aff.mode == "add":
                handler = self._affordance_effects.get(aff.effect.get("type"))
                if handler is not None:
                    handler(agent, action_params, infos, aff)
                return True
        return False

    # ---------- per-agent affordances ----------
    def add_agent_affordance(self, agent: str, affordance: LocationAffordance) -> None:
        self.agent_affordances.setdefault(agent, []).append(affordance)

    def remove_agent_affordance(
        self, agent: str, action: str, mode: str | None = None
    ) -> bool:
        """Drop the stored affordances for *action* (of *mode*, or any mode) on
        *agent*. Returns True if any was removed."""
        current = self.agent_affordances.get(agent, [])
        kept = [
            aff
            for aff in current
            if not (aff.action == action and (mode is None or aff.mode == mode))
        ]
        if kept:
            self.agent_affordances[agent] = kept
        else:
            self.agent_affordances.pop(agent, None)
        return len(kept) != len(current)

    def _agent_affordance_list(self, agent: str) -> List[LocationAffordance]:
        """Stored grants and blocks plus the permissions of the agent's role."""
        return list(self.agent_affordances.get(agent, [])) + self._role_affordances_for(
            agent
        )

    def _agent_has_affordance(self, agent: str, action: str, mode: str) -> bool:
        return any(
            aff.action == action and aff.mode == mode
            for aff in self._agent_affordance_list(agent)
        )

    def _agent_effectively_has(self, agent: str, action: str) -> bool:
        """True when *action* is granted to the agent (by its place, a stored
        grant, or its role) and not blocked by any remove-mode entry."""
        place = self.agent_pos.get(agent)
        place_affs = self._location_affordance_list(place) if place is not None else []
        affs = place_affs + self._agent_affordance_list(agent)
        granted = any(a.action == action and a.mode == "add" for a in affs)
        blocked = any(a.action == action and a.mode == "remove" for a in affs)
        return granted and not blocked

    def _get_agent_affordance(self, agent_tag: str) -> dict:
        result = {}
        for aff in self._agent_affordance_list(agent_tag):
            if aff.mode == "add":
                result[aff.action] = {
                    "description": aff.description,
                    "params": aff.params,
                }
        return result

    def _get_agent_removals(self, agent_tag: str) -> set:
        return {
            aff.action
            for aff in self._agent_affordance_list(agent_tag)
            if aff.mode == "remove"
        }

    def _release_role(self, agent: str, cause: str) -> str | None:
        role_name = self._role_assignment.get(agent)
        granted = set()
        if role_name is not None:
            granted = {
                aff["action"]
                for aff in self._roles[role_name]["affordances"]
                if aff.get("mode", "add") == "add"
            }
        released = super()._release_role(agent, cause)
        # A block on a power the role gave has no meaning once the role is gone.
        for action in granted:
            if not any(
                a.action == action and a.mode == "add"
                for a in self.agent_affordances.get(agent, [])
            ):
                self.remove_agent_affordance(agent, action, mode="remove")
        return released

    # ---------- general affordance effects ----------
    #
    # Effect dicts (see LocationAffordance):
    #   {"type": "drain_energy", "amount": N} or {"amounts": {kind: N}, "floor": F}
    #   {"type": "add_energy",   "amount": N} or {"amounts": {kind: N}, "cap": C}
    #   {"type": "move_agent"}  params: target, destination (next to the target)
    # Optional "target_roles": [...] limits targets to occupants of those roles.
    # An affordance without a "target" param acts on the actor itself.

    def _effect_target(
        self, agent: str, params: dict, action_name: str, effect: dict, infos: dict
    ) -> str | None:
        if "target" not in params:
            return agent
        target_name = params.get("target", "")
        target_tag = self.name_to_tag.get(target_name) if target_name else None
        if target_tag is None or target_tag not in self.agent_registry:
            reason = f"Target '{target_name}' not found."
        elif target_tag == agent:
            reason = "Cannot target yourself."
        elif target_tag not in self._get_nearby_agents(agent):
            reason = f"'{target_name}' is not within your reach."
        elif effect.get("target_roles") and self.role_of(target_tag) not in effect[
            "target_roles"
        ]:
            reason = f"'{target_name}' is not a valid target for {action_name}."
        else:
            return target_tag
        infos.setdefault(agent, {})[action_name] = {"status": "failed", "reason": reason}
        return None

    def _effect_amount(
        self, agent: str, params: dict, action_name: str, effect: dict, infos: dict
    ):
        """Return (amount, kind) or (None, kind) after writing a failure."""
        if "amounts" not in effect:
            return effect.get("amount", 0), ""
        kind = str(params.get("kind", ""))
        table = effect["amounts"]
        if kind not in table:
            infos.setdefault(agent, {})[action_name] = {
                "status": "failed",
                "reason": f"Unknown kind '{kind}'. Kinds: {sorted(table)}.",
            }
            return None, kind
        return table[kind], kind

    def _report_effect(
        self,
        agent: str,
        target: str,
        action_name: str,
        kind: str,
        note: str,
        infos: dict,
        **detail,
    ) -> None:
        """Write the outcome to the actor, a neutral note to the target, and the log."""
        result = {"status": "successful", "target": self.agent_names[target], **detail}
        if "energy_change" in result and not math.isfinite(result["energy_change"]):
            result.pop("energy_change")
        if kind:
            result["kind"] = kind
        infos.setdefault(agent, {})[action_name] = result
        if target != agent:
            infos.setdefault(target, {})[action_name] = note
        if self.logger:
            self.logger.log(
                time=self.step_count,
                event_type=Event.AFFORDANCE_EFFECT,
                agent_tag=agent,
                agent_name=self.agent_names[agent],
                target_tag=target,
                target_name=self.agent_names[target],
                action=action_name,
                kind=kind,
                **detail,
            )

    def _handle_drain_energy(
        self, agent: str, params: dict, infos: dict, aff: LocationAffordance
    ) -> None:
        target = self._effect_target(agent, params, aff.action, aff.effect, infos)
        if target is None:
            return
        amount, kind = self._effect_amount(agent, params, aff.action, aff.effect, infos)
        if amount is None:
            return
        old = self.agent_energy[target]
        new = max(aff.effect.get("floor", 0), old - amount)
        self.agent_energy[target] = new
        label = kind or aff.action
        note = f"{self.agent_names[agent]}: {label}."
        if math.isfinite(old) and math.isfinite(new):
            note += f" Your energy changed by {new - old}."
        self._report_effect(agent, target, aff.action, kind, note, infos=infos, energy_change=new - old)

    def _handle_add_energy(
        self, agent: str, params: dict, infos: dict, aff: LocationAffordance
    ) -> None:
        target = self._effect_target(agent, params, aff.action, aff.effect, infos)
        if target is None:
            return
        amount, kind = self._effect_amount(agent, params, aff.action, aff.effect, infos)
        if amount is None:
            return
        old = self.agent_energy[target]
        new = old + amount
        if "cap" in aff.effect:
            new = min(aff.effect["cap"], new)
        self.agent_energy[target] = new
        label = kind or aff.action
        note = f"{self.agent_names[agent]}: {label}."
        if math.isfinite(old) and math.isfinite(new):
            note += f" Your energy changed by {new - old}."
        self._report_effect(agent, target, aff.action, kind, note, infos=infos, energy_change=new - old)

    def _handle_move_agent(
        self, agent: str, params: dict, infos: dict, aff: LocationAffordance
    ) -> None:
        target = self._effect_target(agent, params, aff.action, aff.effect, infos)
        if target is None:
            return
        destination_text = str(params.get("destination", ""))
        destination = self._parse_location(destination_text)
        origin = self.agent_pos[target]
        # The destination must touch the target's place or the actor's place.
        if destination is None or destination == origin or not (
            self._adjacent(origin, destination)
            or self._adjacent(self.agent_pos[agent], destination)
        ):
            infos.setdefault(agent, {})[aff.action] = {
                "status": "failed",
                "reason": f"'{destination_text}' is not next to {self.agent_names[target]} ({origin}) or to you.",
            }
            return
        if not self._destination_free(destination):
            infos.setdefault(agent, {})[aff.action] = {
                "status": "failed",
                "reason": f"'{destination_text}' is occupied.",
            }
            return
        self._update_agent_pos(target, destination)
        note = f"{self.agent_names[agent]} moved you to {destination_text}."
        self._report_effect(agent, target, aff.action, "", note, infos=infos, origin=origin, destination=destination)

    def _adjacent(self, pos_a, pos_b) -> bool:
        """True when the two places touch. Graph worlds override this with edges."""
        return pos_a != pos_b and self.distance(pos_a, pos_b) == 1

    def _destination_free(self, destination) -> bool:
        """Whether move_agent may place an agent at *destination*. The grid checks occupancy."""
        return True

    # ---------- observation hooks ----------
    def _extend_obs(self, agent: str, obs: dict) -> None:
        """Hook for subclasses to add their own observation keys."""

    def _extra_observation_sections(self, agent: str, obs: dict) -> List[str]:
        """Hook for subclasses to add text sections after the location list."""
        return []

    def _finish_obs(self, agent: str, obs: dict) -> None:
        """Fold the role text and the subclass sections into observation_text."""
        self._extend_obs(agent, obs)
        self._stamp_role_obs(agent, obs)
        # The prompt renders only observation_text, so fold everything in.
        sections = [obs["observation_text"]]
        sections.extend(self._extra_observation_sections(agent, obs))
        if self._roles:
            sections.append(self._render_roles_overview(agent))
        directive_text = self._render_directive(obs)
        if directive_text:
            sections.insert(0, directive_text)
        obs["observation_text"] = "\n\n".join(s for s in sections if s)

    def _build_avail_actions(
        self, agent_tag: str, nearby_agents: bool | None = None
    ) -> dict:
        """Build the raw available-action set for an agent.

        Subclasses override this to add their own actions. Node affordances,
        remove-mode suppressions and excluded actions are applied once by
        `_get_avail_actions`, so overrides must not duplicate that step.
        """
        agent_position = self.agent_pos[agent_tag]
        finite_energy = bool(np.isfinite(self.agent_energy[agent_tag]))

        available_actions = {"move": self._get_move_description(agent_tag)}

        # Colors
        # ---------------------------
        if self.use_colors:
            available_actions["set_color"] = ACTION_TEXT["set_color"]
        # ---------------------------

        # Agent interactions
        # ---------------------------
        if nearby_agents is None:
            nearby_agent_tags = self._get_nearby_agents(agent_tag)
            nearby_agents = bool(nearby_agent_tags)
        elif nearby_agents:
            nearby_agent_tags = self._get_nearby_agents(agent_tag)
        else:
            nearby_agent_tags = []

        nearby_names = [self.agent_names[t] for t in nearby_agent_tags]

        if nearby_agents:
            available_actions["give"] = {
                "description": ACTION_TEXT["give"]["description"],
                "params": {
                    "target": {
                        "description": ACTION_TEXT["give"]["params"]["target"],
                        "choices": nearby_names,
                    },
                    "amount": (
                        ACTION_TEXT["give"]["params"]["amount"]
                        if finite_energy
                        else "Positive integer amount of energy to transfer."
                    ),
                },
            }
            available_actions["take"] = {
                "description": ACTION_TEXT["take"]["description"],
                "params": {
                    "target": {
                        "description": ACTION_TEXT["take"]["params"]["target"],
                        "choices": nearby_names,
                    },
                    "amount": ACTION_TEXT["take"]["params"]["amount"],
                },
            }
        # ---------------------------

        # Artifacts
        # ---------------------------
        if (
            self.artifact_creation
            and self.agent_energy[agent_tag] >= self.artifact_creation_cost
        ):
            entry = create_artifact_text()
            entry["params"] = dict(entry["params"])
            if not self.allow_fixed_artifacts:
                # Every artifact is movable, so don't ask the agent to choose.
                entry["params"].pop("movable", None)
            if self.max_artifact_tokens != MAX_TEXT_ARTIFACT_SIZE:
                # Reflect the configured cap in the agent-facing description.
                entry["params"]["payload"] = entry["params"]["payload"].replace(
                    f"{MAX_TEXT_ARTIFACT_SIZE} tokens",
                    f"{self.max_artifact_tokens} tokens",
                )
            if finite_energy and self.artifact_creation_cost > 0:
                entry["description"] += (
                    f" It costs {self.artifact_creation_cost} energy."
                )
            available_actions["create_artifact"] = entry

        if not self.inert_artifacts:
            interactable = list(self.pos_artifacts[agent_position])
            interactable.extend(self.agent_inventories[agent_tag])
            artifact_actions = {}
            for art_name in interactable:
                for act_name, spec in self.artifacts[art_name].actions.items():
                    if act_name not in artifact_actions:
                        artifact_actions[act_name] = {
                            "description": spec["description"],
                            "params": {
                                "name": {
                                    "description": "Name of the target artifact",
                                    "choices": [],
                                },
                                **spec["params"],
                            },
                        }
                    artifact_actions[act_name]["params"]["name"]["choices"].append(
                        art_name
                    )
            available_actions.update(artifact_actions)

            if self.use_inventory:
                movable_at_pos = [
                    n
                    for n in self.pos_artifacts[agent_position]
                    if self.artifacts[n].movable
                ]
                if movable_at_pos:
                    available_actions["pickup_artifact"] = {
                        "description": ACTION_TEXT["pickup_artifact"]["description"],
                        "params": {
                            "name": {
                                "description": ACTION_TEXT["pickup_artifact"]["params"][
                                    "name"
                                ],
                                "choices": movable_at_pos,
                            },
                        },
                    }

                if self.agent_inventories[agent_tag]:
                    inventory = list(self.agent_inventories[agent_tag])
                    available_actions["drop_artifact"] = {
                        "description": ACTION_TEXT["drop_artifact"]["description"],
                        "params": {
                            "name": {
                                "description": ACTION_TEXT["drop_artifact"]["params"][
                                    "name"
                                ],
                                "choices": inventory,
                            },
                        },
                    }
                # give_artifact: a giveable artifact is one in the agent's
                # inventory OR a movable artifact on its own node. Offered when a
                # peer is nearby.
                givable = list(self.agent_inventories[agent_tag]) + movable_at_pos
                if givable and nearby_agents:
                    available_actions["give_artifact"] = {
                        "description": ACTION_TEXT["give_artifact"]["description"],
                        "params": {
                            "artifact_name": {
                                "description": ACTION_TEXT["give_artifact"]["params"][
                                    "artifact_name"
                                ],
                                "choices": givable,
                            },
                            "target_agent": {
                                "description": ACTION_TEXT["give_artifact"]["params"][
                                    "target_agent"
                                ],
                                "choices": nearby_names,
                            },
                        },
                    }
        # ---------------------------

        # Spawning
        # ---------------------------
        if self._reproduction_error(agent_tag) is None:
            same_type_nearby = [
                self.agent_names[t]
                for t in nearby_agent_tags
                if self.two_parent_spawn
                and self.agent_genomes[t] == self.agent_genomes[agent_tag]
            ]
            available_actions.update(
                [
                    build_spawn_action(
                        reproduction_cost=self.reproduction_cost,
                        init_agent_energy=self.init_agent_energy,
                        genome_type=self.agent_genomes[agent_tag],
                        same_type_nearby_names=same_type_nearby,
                        finite_energy=finite_energy,
                    )
                ]
            )
        # ---------------------------

        # External requests — one flattened `<server>_<tool>` action per spec
        # entry, offered when the agent can afford its per-server energy cost.
        # ---------------------------
        for action_name, spec in self.external_actions_spec.items():
            if self.agent_energy[agent_tag] < spec["cost"]:
                continue
            available_actions[action_name] = {
                "description": (
                    spec["description"]
                    if finite_energy
                    else spec.get("description_without_cost", spec["description"])
                ),
                "params": spec["params"],
            }
            if "input_schema" in spec:
                available_actions[action_name]["input_schema"] = copy.deepcopy(
                    spec["input_schema"]
                )
                available_actions[action_name]["optional"] = list(spec.get("optional", []))
        # ---------------------------

        available_actions.update(self._role_actions(agent_tag))
        return available_actions

    def _get_avail_actions(
        self, agent_tag: str, nearby_agents: bool | None = None
    ) -> dict:
        """Build the action set, then apply node affordances/removals/exclusions
        once — after any subclass additions, so a suppressed action can't reappear.
        """
        available_actions = self._build_avail_actions(agent_tag, nearby_agents)
        agent_position = self.agent_pos[agent_tag]

        # Affordances: add-mode node and agent actions first, then every
        # remove-mode suppression. A removal from either source wins.
        available_actions.update(self._get_location_affordance(agent_position))
        available_actions.update(self._get_agent_affordance(agent_tag))
        for _rm in self._get_location_removals(agent_position) | self._get_agent_removals(
            agent_tag
        ):
            available_actions.pop(_rm, None)

        # Scenario-wide then per-agent excluded actions.
        for exc_set in (
            self.excluded_actions,
            self.agent_excluded_actions.get(agent_tag, set()),
        ):
            if exc_set:
                available_actions = {
                    k: v
                    for k, v in available_actions.items()
                    if not any(
                        k == exc or (k.startswith(exc + "_") and k not in ACTION_TEXT)
                        for exc in exc_set
                    )
                }

        for mechanic in self.mechanics:
            available_actions = mechanic.on_menu(self, agent_tag, available_actions)
        self.agent_avail_actions[agent_tag] = available_actions
        return available_actions

    def _wrap_text(self, text, max_width):
        """Wrap a single string into multiple lines that fit within max_width pixels."""
        words = text.split(" ")
        lines = []
        line = ""
        for word in words:
            test_line = f"{line} {word}".strip()
            if self._font.size(test_line)[0] <= max_width:
                line = test_line
            else:
                if line:
                    lines.append(line)
                line = word
        if line:
            lines.append(line)
        return lines

    # ---------- checkpointing ----------
    def evict_orphan(self, agent_tag: str) -> None:
        """Silently scrub an agent from all env state. Used during checkpoint
        recovery: if a previous run crashed inside env.step() after add_agent
        for a spawned child but before the runner registered it, the
        saved checkpoint has the child in agent_registry with no matching
        agent object. This restores consistency without emitting a death
        announcement or dropping food."""
        name = self.agent_names.get(agent_tag, "?")
        log.warning(
            "Evicting orphan %s(%s) from env state: no matching agent object "
            "in the runner (likely a spawned child from a crashed step).",
            name,
            agent_tag,
        )
        self.agent_registry.discard(agent_tag)
        self.agent_genomes.pop(agent_tag, None)
        self.agent_excluded_actions.pop(agent_tag, None)
        self.agent_energy.pop(agent_tag, None)
        self.agent_time.pop(agent_tag, None)
        self.msg_raw.pop(agent_tag, None)
        self.agent_inventories.pop(agent_tag, None)
        self.agent_trajectories.pop(agent_tag, None)
        self.agent_avail_actions.pop(agent_tag, None)
        self.agent_colors.pop(agent_tag, None)
        pos = self.agent_pos.pop(agent_tag, None)
        if pos is not None and pos in self.pos_to_agent:
            self.pos_to_agent[pos].discard(agent_tag)
            if not self.pos_to_agent[pos]:
                del self.pos_to_agent[pos]
        self.agent_spawn.pop(agent_tag, None)
        for children in self.agent_spawn.values():
            while agent_tag in children:
                children.remove(agent_tag)
        evicted_name = self.agent_names.pop(agent_tag, None)
        if evicted_name is not None:
            self.name_to_tag.pop(evicted_name, None)

    def get_state_ckpt(self) -> dict:
        """Serialize environment state. Stored via pickle so no JSON encoding needed."""
        ckpt = {
            "rng": self._serialize(self.rng) if self.rng is not None else None,
            "food": dict(self.food),
            "food_count": list(self.food_count),
            "_food_zone_centers": self._food_zone_centers,
            "pos_artifacts": dict(self.pos_artifacts),
            "artifacts": {
                name: art.serialize() for name, art in self.artifacts.items()
            },
            "agent_inventories": dict(self.agent_inventories),
            "library": sorted(self.library),
            "artifact_location": dict(self.artifact_location),
            "expired_artifacts": [art.serialize() for art in self.expired_artifacts],
            "_all_artifact_names": sorted(self._all_artifact_names),
            "agent_pos": dict(self.agent_pos),
            "agent_trajectories": dict(self.agent_trajectories),
            "agent_avail_actions": self.agent_avail_actions,
            "pos_to_agent": {
                p: agents for p, agents in self.pos_to_agent.items() if agents
            },
            "agent_energy": dict(self.agent_energy),
            "agent_time": dict(self.agent_time),
            "agent_spawn": dict(self.agent_spawn),
            "agent_names": dict(self.agent_names),
            "agent_colors": dict(self.agent_colors),
            "msg_raw": dict(self.msg_raw),
            "chat": dict(self.chat),
            "agent_registry": sorted(self.agent_registry),
            "agent_excluded_actions": {
                t: sorted(s) for t, s in self.agent_excluded_actions.items()
            },
            "step_count": self.step_count,
            "mechanics": {m.name: copy.deepcopy(m.state) for m in self.mechanics},
            "pending_deaths": copy.deepcopy(self._new_deaths),
            "no_appetite": sorted(self.no_appetite),
            "logger_save_path": str(self.logger.save_path) if self.logger else None,
            "logger_data": (
                self.logger._sanitize(self.logger.data) if self.logger else None
            ),
        }
        ckpt["agent_affordances"] = {
            tag: [asdict(aff) for aff in affs]
            for tag, affs in self.agent_affordances.items()
            if affs
        }
        self._roles_ckpt(ckpt)
        return ckpt

    def set_state_ckpt(self, state_ckpt: dict) -> None:
        """Restore environment state from a checkpoint dict."""
        self.step_count = state_ckpt["step_count"]
        saved_mechanics = state_ckpt.get("mechanics", {})
        for mechanic in self.mechanics:
            if mechanic.name in saved_mechanics:
                mechanic.state.clear()
                mechanic.state.update(copy.deepcopy(saved_mechanics[mechanic.name]))
        self._new_deaths = copy.deepcopy(state_ckpt.get("pending_deaths", []))
        self.no_appetite = set(state_ckpt.get("no_appetite", []))
        self.agent_affordances = {
            tag: [LocationAffordance.from_dict(d) for d in affs]
            for tag, affs in state_ckpt.get("agent_affordances", {}).items()
        }
        self._restore_roles_ckpt(state_ckpt)
        if self.logger:
            self.logger.log(time=self.step_count, event_type=Event.SET_STATE_CKPT)

        rng_data = state_ckpt.get("rng")
        if rng_data is not None:
            self.rng = self._deserialize(rng_data)  # type: ignore[assignment]
        else:
            self.rng = np.random.default_rng()

        self.food = dict(state_ckpt["food"])
        self.food_count = list(state_ckpt["food_count"])
        self._food_zone_centers = state_ckpt.get("_food_zone_centers")
        self.pos_artifacts = defaultdict(set, state_ckpt["pos_artifacts"])

        self.artifacts = {}
        for name, data in state_ckpt["artifacts"].items():
            self.artifacts[name] = self.artifact_from_ckpt(data)

        self.agent_inventories = defaultdict(set, state_ckpt["agent_inventories"])
        self.library = set(state_ckpt.get("library", []))
        self.artifact_location = dict(state_ckpt["artifact_location"])

        self.expired_artifacts = []
        for data in state_ckpt.get("expired_artifacts", []):
            self.expired_artifacts.append(self.artifact_from_ckpt(data))

        # Rebuild _all_artifact_names from checkpoint if present, otherwise
        # derive it from active + expired artifacts (for backwards compatibility
        # with checkpoints saved before this field existed).
        if "_all_artifact_names" in state_ckpt:
            self._all_artifact_names = set(state_ckpt["_all_artifact_names"])
        else:
            self._all_artifact_names = set(self.artifacts) | {
                art.name for art in self.expired_artifacts
            }

        self.agent_pos = dict(state_ckpt["agent_pos"])
        self.agent_trajectories = dict(state_ckpt["agent_trajectories"])
        self.agent_avail_actions = state_ckpt["agent_avail_actions"]
        self.pos_to_agent = defaultdict(set, state_ckpt["pos_to_agent"])
        self.agent_energy = dict(state_ckpt["agent_energy"])
        self.agent_time = dict(state_ckpt["agent_time"])
        self.agent_spawn = dict(state_ckpt["agent_spawn"])
        self.agent_names = dict(state_ckpt["agent_names"])
        self.agent_colors = dict(state_ckpt["agent_colors"])
        self.msg_raw = dict(state_ckpt["msg_raw"])
        self.chat = dict(state_ckpt["chat"])
        # Needed for back-compatibility
        if isinstance(state_ckpt["agent_registry"], list):
            self.agent_registry = set(state_ckpt["agent_registry"])
        elif isinstance(state_ckpt["agent_registry"], dict):
            self.agent_registry = set(state_ckpt["agent_registry"].keys())
        self.name_to_tag = {name: tag for tag, name in self.agent_names.items()}

        if "agent_excluded_actions" in state_ckpt:
            self.agent_excluded_actions = {
                t: set(s) for t, s in state_ckpt["agent_excluded_actions"].items()
            }
        else:
            self.agent_excluded_actions = {t: set() for t in self.agent_registry}

        logger_path = state_ckpt.get("logger_save_path")
        if logger_path is not None:
            self.logger = JSONLogger(logger_path)
            self.logger.data = state_ckpt.get("logger_data", {})

    def close(self) -> None:
        log.info("Saving environment...")
        self.logger.log(time=self.step_count, event_type=Event.END_RUN)
        self.save_state(self.log_path / "env_state.pkl")

        with open(self.logger.save_path.parent / "messages.json", "w") as f:
            json.dump(self.chat, f, indent=4)

        active_artifacts = []
        for art_name, artifact in self.artifacts.items():
            art_dict = artifact.serialize()
            loc = self.artifact_location.get(art_name)
            if loc is not None:
                kind, where = loc
                if kind == "inv":
                    art_dict["pose"] = f"Inv({where})"
                    art_dict["owner"] = where
                elif kind == "library":
                    art_dict["pose"] = "Library"
                else:
                    art_dict["pose"] = str(where)
            active_artifacts.append(art_dict)

        expired_artifacts = [
            artifact.serialize() for artifact in self.expired_artifacts
        ]
        with open(self.logger.save_path.parent / "artifacts.json", "w") as f:
            json.dump(
                {"active": active_artifacts, "expired": expired_artifacts}, f, indent=4
            )

        with open(self.logger.save_path.parent / "food_counts.json", "w") as f:
            json.dump(self.food_count, f, indent=4)

        with open(self.logger.save_path.parent / "agent_names.json", "w") as f:
            json.dump(self.agent_names, f, indent=4)

        with open(self.logger.save_path.parent / "agent_trajectories.pkl", "wb") as f:
            pickle.dump(self.agent_trajectories, f)

        if self.logger:
            self.logger.close()
        if self._pygame_inited:
            pygame.quit()
            self._pygame_inited = False

    def save_state(self, filepath: str | Path) -> None:
        """Saves the environment state to a pickle file."""
        log.info("Saving environment state at: %s", filepath)
        state_ckpt = self.get_state_ckpt()
        with open(filepath, "wb") as f:
            pickle.dump(state_ckpt, f)
        log.info("Environment state saved at: %s", filepath)

    def load_state(self, filepath: str | Path | None = None) -> None:
        """Loads the environment state from a pickle file."""
        if filepath is None:
            filepath = self.log_path / "env_state.pkl"
        assert Path(filepath).exists(), f"State file {filepath} does not exist."
        log.info("Loading environment state from: %s", filepath)
        with open(filepath, "rb") as f:
            state_ckpt = pickle.load(f)
        self.set_state_ckpt(state_ckpt)
        log.info("Environment state loaded from: %s", filepath)

    @staticmethod
    def _serialize(obj):
        """Serialize a single object to a JSON-safe dict."""
        if isinstance(obj, np.random.Generator):
            return {"__type__": "rng", "state": obj.bit_generator.state}
        if isinstance(obj, str):
            return {"__type__": "str", "data": obj}
        if isinstance(obj, np.ndarray):
            return {
                "__type__": "ndarray",
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "data": obj.tolist(),
            }
        if isinstance(obj, Artifact):
            return {"__type__": "artifact", "data": obj.serialize()}
        raise TypeError(f"Type {type(obj)} not serializable")

    @staticmethod
    def _deserialize(data):
        """Deserialize a dict produced by _serialize back to a Python object."""
        if data["__type__"] == "str":
            return data["data"]
        if data["__type__"] == "rng":
            rng = np.random.default_rng()
            rng.bit_generator.state = data["state"]
            return rng
        if data["__type__"] == "ndarray":
            return np.array(data["data"], dtype=data["dtype"]).reshape(data["shape"])
        if data["__type__"] == "artifact":
            return BaseWorld.artifact_from_ckpt(data["data"])
        raise TypeError(f"Type {data['__type__']} not deserializable")

    # ---------- abstract spatial interface ----------
    def _update_agent_pos(self, agent: str, new_pos) -> None:
        """Update agent position, keeping pos_to_agent consistent."""
        if agent not in self.agent_registry:
            raise ValueError(
                f"Agent {self.agent_names[agent]}({agent}) not found in the environment."
            )
        old_pos = self.agent_pos.get(agent)
        self.agent_trajectories[agent].append(new_pos)
        self.agent_pos[agent] = new_pos
        if old_pos is not None:
            self.pos_to_agent[old_pos].discard(agent)
        self.pos_to_agent[new_pos].add(agent)

    def _place_agent(self, name: str, pos=None) -> None:
        """Place agent at pos (or a random free position if None)."""
        if pos is None:
            pos = self._random_free_pos()
        elif self.exclusive_pos_occupancy and self.pos_to_agent[pos]:
            raise ValueError(
                f"Cell {pos} is already occupied. Choose a different spawn position."
            )
        self.agent_trajectories[name] = []
        self.agent_trajectories[name].append(pos)
        self.agent_pos[name] = pos
        self.pos_to_agent[pos].add(name)
        self.agent_energy[name] = self.init_agent_energy
        self.agent_time[name] = np.inf if self.lifespan == -1 else self.lifespan
        self.agent_spawn[name] = list()

    @abstractmethod
    def _random_free_pos(self):
        """Return a random unoccupied position/node."""

    @abstractmethod
    def _offspring_position(self, center, node_id=None):
        """Return an adjacent position for an offspring of center, or None.

        Found among free neighbors (grid/graph) or created (social graph).
        `node_id` is an optional desired node id, honored only where new nodes
        are created (social graph); other envs ignore it.
        """

    @abstractmethod
    def _apply_move(
        self, agent: str, move_params: dict | None, infos: dict
    ) -> Tuple[Any, dict]:
        """Resolve and execute a move action.

        move_params is action_params when action_name == "move", or None when
        the agent chose a non-move action (implying "stay").

        Must update pos_to_agent when the agent moves and return
        (new_position, infos) — infos may have a "Move outcome" entry added.
        """

    @abstractmethod
    def _get_move_description(self, agent_tag: str) -> dict:
        """Return the available_actions entry dict for the 'move' action.

        e.g. {"description": "...", "params": {"direction": "..."}} for grid,
        or   {"description": "...", "params": {"direction": {"choices": [...]}}} for graph.
        """

    @abstractmethod
    def _seed_initial_food(self) -> None:
        """Populate self.food at the start of a new episode."""

    @abstractmethod
    def _respawn_food_one(self, value: "float | None" = None) -> None:
        """Spawn one food item at a valid position/node."""

    @abstractmethod
    def _get_nearby_agents(self, agent_tag: str) -> list:
        """Return list of agent tags within perception range of agent_tag."""

    @abstractmethod
    def _build_obs(
        self,
        agent: str,
        food_snapshot: dict | None = None,
        artifact_snapshot: dict | None = None,
    ) -> Tuple[dict, bool]:
        """Build observation dict for agent.

        Returns (obs_dict, has_nearby_agents).  When food_snapshot and
        artifact_snapshot are provided (built once per step via
        _build_step_snapshot), the implementation should use them to avoid
        redundant dict lookups.
        """

    @abstractmethod
    def render(self, mode: str = "human"):
        """Render the environment."""
