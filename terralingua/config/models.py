"""Pydantic configuration models for an experiment run.

Three sub-configs (`AgentConfig`, `EnvConfig`, `RunConfig`) compose into
`ExperimentConfig`. Graph-world settings live in a single nested `GraphConfig`
(the real, factory-driven API the world classes consume) — there is no flat
`graph_*` mirror. All cross-field couplings are expressed as validators here, in
one place, rather than being scattered across a build step and the runner.
"""

from pathlib import Path
from typing import Literal, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticUndefined

from terralingua.agents.prompt_templates import BUILTIN_INSTRUCTION_NAMES
from terralingua.environment.graph_builder import AVAILABLE_TOPOLOGIES
from terralingua.genome import AVAILABLE_GENOMES

# Valid reasoning-effort levels. OpenAI reasoning models accept minimal/low/
# medium/high; Anthropic adaptive thinking (Opus 4.6+/Sonnet 4.6) accepts
# low/medium/high/xhigh/max. The union is validated here; an effort a given
# provider does not support is rejected by that provider's API.
REASONING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "max")


class ConfigModel(BaseModel):
    """Reject unknown settings without dropping user input."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    @field_validator("*", mode="before")
    @classmethod
    def _reject_boolean_numbers(cls, value, info):
        annotation = cls.model_fields[info.field_name].annotation
        types = get_args(annotation) or (annotation,)
        if isinstance(value, bool) and bool not in types and any(t in (int, float) for t in types):
            raise ValueError("Use a number, not a boolean.")
        return value


class GraphConfig(ConfigModel):
    """Configuration for graph-world topology.

    Use the factory class methods for a self-documenting API:

        GraphConfig.ring(n_nodes=50)
        GraphConfig.tree(n_nodes=100, branching=3)
        GraphConfig.small_world(n_nodes=100, k=4, p=0.1)
        GraphConfig.scale_free(n_nodes=100, m=2)
        GraphConfig.random_graph(n_nodes=100, p=0.15)
        GraphConfig.complete(n_nodes=20)
        GraphConfig.bipartite(n_nodes=50, n1=20, n2=30)
        GraphConfig.from_file("path/to/graph.json")
        GraphConfig.agent_network("path/to/network.hocon")
    """

    model_config = ConfigDict(validate_assignment=False)

    topology: str = Field(default="grid", json_schema_extra={"enum": list(AVAILABLE_TOPOLOGIES)}, description="Topology used to create the initial graph.")
    n_nodes: int = Field(default=25, ge=0, description="Requested initial node count.")
    hop_radius: int = Field(default=2, ge=0, description="Observation radius in graph edges.")
    max_connections: int | None = Field(default=None, ge=0, description="Maximum outgoing social connections per agent. Null means unlimited.")
    edge_decay_steps: int | None = Field(default=None, ge=0, description="Expire inactive social connections after this many steps. Null disables expiry.")
    # Total social suggestions; about two thirds prefer outgoing two-hop neighbors.
    random_intro_k: int = Field(default=3, ge=0, description="Connection suggestions per step. About two thirds prefer outgoing two-hop neighbors.")
    direct_message_cost: int = Field(default=0, ge=0, description="Energy cost per direct message.")
    allow_follow: bool = Field(default=True, description="Allow following other agents.")
    allow_connection: bool = Field(default=True, description="Allow mutual connection requests.")
    connection_request_lifespan: int = Field(default=3, ge=0, description="Steps before an unanswered connection request expires.")
    # --- grid ---
    wrap: bool = Field(default=False, description="Connect opposite boundaries of the grid graph.")
    # --- tree ---
    tree_branching: int = Field(default=3, ge=1, description="Children per internal tree node.")
    tree_height: int | None = Field(default=None, ge=0, description="Tree height. Null derives height from the requested node count.")
    # --- bipartite ---
    bipartite_n1: int = Field(default=0, ge=0, description="First partition size. Zero uses the topology default.")
    bipartite_n2: int = Field(default=0, ge=0, description="Second partition size. Zero uses the topology default.")
    # --- small_world ---
    small_world_k: int = Field(default=4, ge=0, description="Initial neighbors per node in a small-world graph.")
    small_world_p: float = Field(default=0.1, ge=0, le=1, description="Small-world edge rewiring probability.")
    # --- scale_free ---
    scale_free_m: int = Field(default=2, ge=1, description="Edges added with each scale-free node.")
    # --- random ---
    random_p: float = Field(default=0.15, ge=0, le=1, description="Edge probability in a random graph.")
    # --- shared seed (small_world / scale_free / random) ---
    seed: int | None = Field(default=None, description="Seed for randomized topology generation.")
    # --- file ---
    file_path: str | None = Field(default=None, description="Graph file used by the file topology.")
    # --- Neuro-SAN agent-network HOCON ---
    agent_network_hocon_path: str | None = Field(default=None, description="HOCON file defining founding agents and their connections.")
    agent_network_bidirectional_edges: bool = Field(default=False, description="Add the reverse edge for each HOCON connection.")
    # --- custom ---
    custom_factory: str | None = Field(default=None, description="Import path for a custom topology factory.")

    @model_validator(mode="after")
    def _resolve_topology(self) -> "GraphConfig":
        # A HOCON agent-network path implies the agent_network topology.
        if self.agent_network_hocon_path and self.topology != "agent_network":
            self.topology = "agent_network"

        if self.topology not in AVAILABLE_TOPOLOGIES:
            raise ValueError(
                f"topology must be one of {AVAILABLE_TOPOLOGIES}, got '{self.topology}'"
            )

        return self

    @property
    def _topology_params(self) -> dict:
        """Build topology arguments from the current field values."""
        p: dict = {}
        if self.topology == "grid":
            if self.wrap:
                p["wrap"] = True
        elif self.topology == "tree":
            p["branching"] = self.tree_branching
            if self.tree_height is not None:
                p["height"] = self.tree_height
        elif self.topology == "bipartite":
            if self.bipartite_n1:
                p["n1"] = self.bipartite_n1
            if self.bipartite_n2:
                p["n2"] = self.bipartite_n2
        elif self.topology == "small_world":
            p["k"] = self.small_world_k
            p["p"] = self.small_world_p
            if self.seed is not None:
                p["seed"] = self.seed
        elif self.topology == "scale_free":
            p["m"] = self.scale_free_m
            if self.seed is not None:
                p["seed"] = self.seed
        elif self.topology == "random":
            p["p"] = self.random_p
            if self.seed is not None:
                p["seed"] = self.seed
        elif self.topology == "file":
            if not self.file_path:
                raise ValueError("The file topology requires file_path.")
            p["path"] = self.file_path
        elif self.topology == "agent_network":
            if not self.agent_network_hocon_path:
                raise ValueError(
                    "agent_network topology requires agent_network_hocon_path"
                )
            p["path"] = self.agent_network_hocon_path
            p["bidirectional"] = self.agent_network_bidirectional_edges
        elif self.topology == "custom":
            if not self.custom_factory:
                raise ValueError("The custom topology requires custom_factory.")
            p["factory"] = self.custom_factory
        return p

    # ------------------------------------------------------------------
    # Factory class methods — preferred API
    # ------------------------------------------------------------------

    @classmethod
    def grid(
        cls, n_nodes: int = 25, wrap: bool = False, hop_radius: int = 2, **kw
    ) -> "GraphConfig":
        return cls(
            topology="grid", n_nodes=n_nodes, wrap=wrap, hop_radius=hop_radius, **kw
        )

    @classmethod
    def ring(cls, n_nodes: int = 25, hop_radius: int = 2, **kw) -> "GraphConfig":
        return cls(topology="ring", n_nodes=n_nodes, hop_radius=hop_radius, **kw)

    @classmethod
    def tree(
        cls,
        n_nodes: int = 25,
        branching: int = 3,
        height: int | None = None,
        hop_radius: int = 2,
        **kw,
    ) -> "GraphConfig":
        return cls(
            topology="tree",
            n_nodes=n_nodes,
            tree_branching=branching,
            tree_height=height,
            hop_radius=hop_radius,
            **kw,
        )

    @classmethod
    def complete(cls, n_nodes: int = 25, hop_radius: int = 2, **kw) -> "GraphConfig":
        return cls(topology="complete", n_nodes=n_nodes, hop_radius=hop_radius, **kw)

    @classmethod
    def bipartite(
        cls, n_nodes: int = 25, n1: int = 0, n2: int = 0, hop_radius: int = 2, **kw
    ) -> "GraphConfig":
        return cls(
            topology="bipartite",
            n_nodes=n_nodes,
            bipartite_n1=n1,
            bipartite_n2=n2,
            hop_radius=hop_radius,
            **kw,
        )

    @classmethod
    def small_world(
        cls,
        n_nodes: int = 25,
        k: int = 4,
        p: float = 0.1,
        seed: int | None = None,
        hop_radius: int = 2,
        **kw,
    ) -> "GraphConfig":
        return cls(
            topology="small_world",
            n_nodes=n_nodes,
            small_world_k=k,
            small_world_p=p,
            seed=seed,
            hop_radius=hop_radius,
            **kw,
        )

    @classmethod
    def scale_free(
        cls,
        n_nodes: int = 25,
        m: int = 2,
        seed: int | None = None,
        hop_radius: int = 2,
        **kw,
    ) -> "GraphConfig":
        return cls(
            topology="scale_free",
            n_nodes=n_nodes,
            scale_free_m=m,
            seed=seed,
            hop_radius=hop_radius,
            **kw,
        )

    @classmethod
    def random_graph(
        cls,
        n_nodes: int = 25,
        p: float = 0.15,
        seed: int | None = None,
        hop_radius: int = 2,
        **kw,
    ) -> "GraphConfig":
        return cls(
            topology="random",
            n_nodes=n_nodes,
            random_p=p,
            seed=seed,
            hop_radius=hop_radius,
            **kw,
        )

    @classmethod
    def from_file(cls, path: str, hop_radius: int = 2, **kw) -> "GraphConfig":
        return cls(topology="file", file_path=path, hop_radius=hop_radius, **kw)

    @classmethod
    def agent_network(
        cls, path: str, hop_radius: int = 2, bidirectional: bool = False, **kw
    ) -> "GraphConfig":
        return cls(
            topology="agent_network",
            agent_network_hocon_path=path,
            agent_network_bidirectional_edges=bidirectional,
            hop_radius=hop_radius,
            **kw,
        )

    @classmethod
    def custom(cls, factory: str, hop_radius: int = 2, **kw) -> "GraphConfig":
        return cls(
            topology="custom", custom_factory=factory, hop_radius=hop_radius, **kw
        )


class AgentConfig(ConfigModel):
    agents_name_prefix: str = Field(
        default="being", description="Prefix for agent names (e.g. being0, being1)"
    )
    scenario_specific_instructions: str = Field(
        default="none",
        description=(
            "Scenario instructions injected into the system prompt: a builtin "
            f"name {BUILTIN_INSTRUCTION_NAMES} or a path to an instructions file "
            "(a Markdown file inside the scenario folder, rendered with Jinja)."
        ),
    )
    personas_path: str | None = Field(
        default=None,
        description=(
            "JSON file with personas for the first beings, at the initial population and at "
            "respawns, in creation order. Each entry is the persona text, or an object with "
            "'persona', an optional 'name' and an optional 'count' (default 1). A persona "
            "from the scenario wins over the file. Other keys of an entry, such as a role, "
            "travel with the persona to the scenario."
        ),
    )
    genome: str = Field(default="ocean_5", json_schema_extra={"enum": list(AVAILABLE_GENOMES)}, description="Agent genome type")
    internal_memory_size: int = Field(
        ge=0,
        default=500, description="Internal memory token budget. Zero disables private memory."
    )
    max_history: int = Field(
        ge=0,
        default=10, description="Max number of interactions stored per agent"
    )
    model: str = Field(
        default="claude-haiku-4-5", description="Model name used for agent decisions"
    )
    max_tokens: int | None = Field(
        gt=0,
        default=None,
        description=(
            "Max output tokens per agent LLM call. None uses the router's "
            "per-model default (e.g. 20000 for Claude)."
        ),
    )
    reasoning_effort: str | None = Field(
        json_schema_extra={"enum": [*REASONING_EFFORTS, None]},
        default=None,
        description=(
            f"Reasoning effort, one of {REASONING_EFFORTS}. None keeps the "
            "router's per-model default. For OpenAI reasoning models it sets "
            "reasoning_effort; for Anthropic (Opus 4.6+/Sonnet 4.6) it enables "
            "adaptive thinking at this effort level via output_config.effort."
        ),
    )
    use_colors: bool = Field(
        default=False, description="Allow agents to choose their own color"
    )
    use_inventory: bool = Field(default=True, description="Enable inventory system")
    random_names: bool = Field(
        default=False,
        description="Give agents random word names instead of beingX tags",
    )

    @field_validator("genome")
    @classmethod
    def _validate_genome(cls, v: str) -> str:
        if v not in AVAILABLE_GENOMES:
            raise ValueError(f"Genome must be one of {AVAILABLE_GENOMES}, got {v}")
        return v

    @field_validator("reasoning_effort")
    @classmethod
    def _validate_reasoning_effort(cls, v: str | None) -> str | None:
        if v is not None and v not in REASONING_EFFORTS:
            raise ValueError(
                f"reasoning_effort must be one of {REASONING_EFFORTS} or null, got {v}"
            )
        return v

    @field_validator("personas_path")
    @classmethod
    def _validate_personas_path(cls, v: str | None) -> str | None:
        if v is not None and not Path(v).is_file():
            raise ValueError(f"personas_path must be a path to a JSON file, got {v}")
        return v

    @field_validator("scenario_specific_instructions")
    @classmethod
    def _validate_instructions(cls, v: str) -> str:
        if v not in BUILTIN_INSTRUCTION_NAMES and not Path(v).is_file():
            raise ValueError(
                f"scenario_specific_instructions must be one of {BUILTIN_INSTRUCTION_NAMES} "
                f"or a path to an instructions file, got {v}"
            )
        return v


class EnvConfig(ConfigModel):
    agent_lifespan: int = Field(default=100, description="Lifespan baseline. -1 gives unlimited life. Values above one vary by up to 20%; other values last one turn.")
    artifact_creation_cost: int = Field(ge=-1, default=0, description="Artifact creation cost. -1 disables creation, 0 is free, and positive values charge energy.")
    use_library: bool = Field(
        default=True,
        description="Enable the shared artifact library/commons (social-graph world)",
    )
    library_preview: bool = Field(
        default=True,
        description="show_library_content includes a short content preview, not just names",
    )
    allow_fixed_artifacts: bool = Field(
        default=True,
        description=(
            "Allow agents to create fixed artifacts. When false, create_artifact "
            "makes only movable artifacts. Seeded and administrator-created artifacts are independent."
        ),
    )
    max_artifact_tokens: int = Field(
        ge=0,
        default=500,
        description="Maximum token length of a text artifact's payload",
    )
    drop_food_on_death: bool = Field(
        default=True, description="Drop food at agent position on death"
    )
    food_decay_rate: float = Field(ge=0, le=1, default=0.05, description="Food decay rate")
    food_mechanism: bool = Field(default=True, description="Enable natural food generation, decay, and one energy of upkeep per step. Ignored in social graphs.")
    energy_death: bool | None = Field(
        default=None,
        description="Agents die when energy reaches 0. Default: only when food_mechanism is on",
    )
    food_spawn_rate: int = Field(ge=0, default=10, description="Food spawn per step")
    food_zones: int | list[tuple[int, int]] | list[str] | None = Field(
        default=2, description="Food zones: a positive count, grid x,y coordinates, graph node names, or null for uniform placement"
    )
    dynamic_grid_scaling: bool = Field(
        default=True, description="Expand grid when agent density exceeds the max"
    )
    dynamic_grid_min_cells_per_agent: int = Field(
        ge=1,
        default=50, description="Minimum cells per agent before the grid expands"
    )
    dynamic_grid_step: int = Field(
        ge=1,
        default=10, description="Rows/cols added to each side when the grid expands"
    )
    grid_size: int = Field(ge=1, default=50, description="Grid dimension")
    inert_artifacts: bool = Field(
        default=False, description="Artifacts cannot be interacted with"
    )
    init_agents: int = Field(ge=0, default=20, description="Initial agent count")
    init_human_agents: int = Field(ge=0, default=0, description="Initial human agent count")
    init_agent_energy: int = Field(default=50, description="Initial energy per agent")
    init_food: int = Field(ge=0, default=100, description="Initial food energy, limited by available placement space")
    max_message_length: int = Field(
        ge=0,
        default=200,
        description="Max token length for agent broadcast messages (0 = unlimited)",
    )
    min_agents: int = Field(ge=0, default=0, description="Minimum agent population")
    max_agents: int | None = Field(
        ge=0,
        default=None,
        description=(
            "Population threshold that blocks reproduction. Other arrival paths are "
            "not capped by this setting. None means no reproduction threshold."
        ),
    )
    two_parent_spawn: bool = Field(
        default=False,
        description="Allow two-parent spawning (genome crossover with a nearby agent)",
    )
    genome_mutation_rate: float = Field(
        ge=0, le=1,
        default=0.5,
        description="Probability each gene mutates during spawning (0=never, 1=always)",
    )
    reproduction_cost: int = Field(
        default=50,
        ge=-1,
        description=(
            "Reproduction energy: -1 disables it; 0 grants the child initial agent "
            "energy; a positive value funds that much child energy from the parent. "
            "Failed attempts consume the cost. Optional gifts also come from the parent."
        ),
    )
    static_food: bool = Field(
        default=False, description="Food always spawns in same positions"
    )
    excluded_actions: list[str] = Field(
        default_factory=list, description="Action keys to exclude from available actions"
    )
    init_artifacts_path: str | None = Field(
        default=None,
        description="Path to directory containing JSON artifact files to seed at startup",
    )
    # A HOCON file whose agent entries define persistent ROLES: name,
    # instructions (the charter), optional `capacity` (occupant slots, default
    # 1), `on_vacancy` (open | retire | succeed), and `transferable` (bool).
    # Agents are created by the normal init_agents path and get randomly
    # assigned to role slots at the start; extra agents stay roleless. Role
    # permissions come from the entries of affordances_file_path whose keys
    # match role names; they follow the occupant. See terralingua/environment/roles.py.
    roles_hocon_path: str | None = Field(default=None, description="HOCON file defining persistent roles.")
    # Affordance overlay: keys are graph node ids or 'row,col' grid cells, and,
    # in roles mode, role names.
    affordances_file_path: str | None = Field(default=None, description="JSON file defining node, cell, or role permissions.")
    vision_radius: int = Field(ge=0, default=5, description="Vision radius")
    world_type: str = Field(
        json_schema_extra={"enum": ["grid", "graph", "social_graph"]},
        default="grid", description="World type: 'grid', 'graph', or 'social_graph'"
    )
    # Graph-world settings (used when world_type is 'graph' or 'social_graph').
    graph: GraphConfig = Field(default_factory=GraphConfig)

    @model_validator(mode="before")
    @classmethod
    def _world_food_default(cls, values):
        if isinstance(values, dict) and values.get("world_type") == "social_graph":
            values = dict(values)
            values.setdefault("food_mechanism", False)
        return values

    @field_validator("world_type")
    @classmethod
    def _validate_world_type(cls, v: str) -> str:
        allowed = ("grid", "graph", "social_graph")
        if v not in allowed:
            raise ValueError(f"world_type must be one of {allowed}, got {v}")
        return v

    @field_validator("food_zones", mode="before")
    @classmethod
    def _coerce_food_zones(cls, value):
        if isinstance(value, str) and value.lower() in ("none", "null"):
            return None
        return value

    def _resolve_food_zones(self) -> None:
        """Grid zones are coordinates. Graph zones are node names."""
        value = self.food_zones
        if not self.food_mechanism:
            return
        if value is None or isinstance(value, int):
            if isinstance(value, bool) or (value is not None and value < 1):
                raise ValueError("food_zones must be a positive count or null.")
            return
        if not value:
            raise ValueError("food_zones cannot be empty when food is enabled.")
        if self.world_type in ("graph", "social_graph"):
            if not all(isinstance(item, str) for item in value):
                raise ValueError("Graph food_zones must contain node names.")
            self.food_zones = list(value)
            return
        if len(value) == 1 and isinstance(value[0], str):
            if value[0].lower() in ("none", "null"):
                self.food_zones = None
                return
            if value[0].isdigit():
                self.food_zones = int(value[0])
                if self.food_zones < 1:
                    raise ValueError("food_zones must be a positive count or null.")
                return
        coordinates = []
        for item in value:
            if isinstance(item, str):
                try:
                    x, y = item.split(",")
                    item = (int(x), int(y))
                except (ValueError, TypeError) as exc:
                    raise ValueError("Grid food_zones must contain x,y coordinates.") from exc
            if len(item) != 2:
                raise ValueError("Grid food_zones must contain x,y coordinates.")
            coordinates.append(tuple(item))
        self.food_zones = coordinates

    @model_validator(mode="after")
    def _resolve_couplings(self) -> "EnvConfig":
        if self.world_type == "social_graph" and self.food_mechanism:
            # The setting does not apply there; the world ignores it rather than failing.
            self.food_mechanism = False
        if self.min_agents > self.init_agents:
            raise ValueError("min_agents cannot be greater than init_agents")
        # A HOCON agent network forces a graph world (social_graph if the user
        # opted into it, otherwise the plain graph world). GraphConfig itself
        # already forces topology='agent_network' when a hocon path is set.
        if self.graph.agent_network_hocon_path and self.world_type != "social_graph":
            self.world_type = "graph"
        if self.roles_hocon_path and self.graph.agent_network_hocon_path:
            raise ValueError(
                "roles_hocon_path and agent_network_hocon_path are mutually "
                "exclusive: the same HOCON entries would define both the "
                "founding agents and the roles."
            )
        self._resolve_food_zones()
        if self.world_type in ("graph", "social_graph"):
            self.graph._topology_params
        return self


class RunConfig(ConfigModel):
    runner_class: str | None = Field(
        default=None,
        description="Optional scenario runner class, written as module:Class",
    )
    scenario: str | None = Field(
        default=None,
        description="Python module of the scenario to attach, for example scenarios.example. It exposes Options and build(options).",
    )
    scenario_options: dict = Field(
        default_factory=dict,
        description="Settings validated by the selected scenario or scenario runner",
    )
    seed: int | None = Field(
        default=None,
        description="Seed for the world's random generator. Null leaves it unseeded.",
    )
    ckpt_interval: int = Field(ge=0, default=100, description="Checkpoint interval")
    empty_countdown: int = Field(ge=-1,
        default=20,
        description="Steps to wait with no agents before terminating (-1 = never)",
    )
    exp_description: str = Field(default="", description="Experiment description")
    exp_name: str | None = Field(default=None, description="Experiment name")
    external_servers_config: str | None = Field(
        default=None,
        description="Path to a servers.json declaring external MCP servers",
    )
    external_sync_send: bool = Field(
        default=True,
        description="Send messages to external servers sequentially (sync) vs fire-and-forget",
    )
    external_server_port: int | None = Field(
        ge=1, le=65535,
        default=None,
        description="Override the port in the sole external server's URL from servers.json "
        "(lets parallel runs use their own MCP port without editing the file)",
    )
    external_server_callback_port: int | None = Field(
        ge=1, le=65535,
        default=None,
        description="Override the sole external server's callback_port from servers.json "
        "(lets parallel runs use their own callback port without editing the file)",
    )
    reward_to_energy_coefficient: float = Field(
        default=0.0,
        description="Multiply the 'reward' field of an external server response into agent energy (0 = off)",
    )
    live_render: bool = Field(default=False, description="Render simulation live")
    max_parallel_workers: int = Field(ge=1, default=8, description="Maximum concurrent agent decisions")
    max_ts: int = Field(ge=-1,
        default=3000, description="Max simulation timesteps. -1 for infinite"
    )
    ports: tuple[int, ...] = Field(
        default=(9000, 9001, 9002, 9003, 9010, 9011, 9012),
        description="Ports hosting LLM models",
    )
    remote_api_enabled: bool = Field(
        default=False, description="Enable remote agent API server"
    )
    env_heartbeat: float = Field(
        default=30.0,
        description="Seconds to wait for remote agent actions before stepping (-1 = wait for all)",
    )
    stop_on_external_solved: bool = Field(
        default=False,
        description="Stop the run as soon as an external server response signals task "
        "completion (a truthy 'solved' or 'done'). Off by default; tasks that "
        "run to a fixed horizon leave it off.",
    )
    allow_human_agents: bool = Field(
        default=True,
        description="Allow human-controlled remote agents to be registered via the dashboard",
    )
    save_video: bool = Field(default=True, description="Save video")
    video_fps: int = Field(ge=1, default=10, description="FPS for video output")
    # exp_name is intentionally left None when unset; SimulationRunner fills a
    # run_<timestamp>_<id> name in __init__.


class ExperimentConfig(ConfigModel):
    agent: AgentConfig = Field(default_factory=AgentConfig)
    env: EnvConfig = Field(default_factory=EnvConfig)
    run: RunConfig = Field(default_factory=RunConfig)

    @model_validator(mode="after")
    def _validate_cross_config(self) -> "ExperimentConfig":
        prefix = self.agent.agents_name_prefix
        if (
            not self.env.graph.agent_network_hocon_path
            and self.env.init_agents > 0
            and self.env.init_human_agents > 0
            and prefix.startswith("human")
        ):
            suffix = prefix[len("human"):]
            canonical_suffix = not suffix or (
                suffix.isascii() and suffix.isdigit() and not suffix.startswith("0")
            )
            first_index = suffix + "0"
            highest_index = str(self.env.init_human_agents - 1)
            overlaps = len(first_index) < len(highest_index) or (
                len(first_index) == len(highest_index) and first_index <= highest_index
            )
            if canonical_suffix and overlaps:
                raise ValueError("The agent name prefix would overwrite initial human-agent tags.")
        if (
            self.env.graph.agent_network_hocon_path
            and self.agent.genome != "sentence_directed"
        ):
            raise ValueError(
                "agent_network HOCON imports require genome='sentence_directed', "
                f"got '{self.agent.genome}'"
            )
        return self

    def to_json(self) -> dict:
        """Serialize for params.json (nested, JSON-safe)."""
        return self.model_dump(mode="json")


def _type_str(annotation) -> str:
    """A readable type name for a field annotation."""
    name = getattr(annotation, "__name__", None)
    if name:
        return name
    return str(annotation).replace("typing.", "")


def _field_default(field):
    """The concrete default value of a pydantic FieldInfo."""
    if field.default is not PydanticUndefined:
        return field.default
    if field.default_factory is not None:
        try:
            return field.default_factory()
        except TypeError:
            return None
    return None


def iter_config_fields():
    """Yield (section, name, type_str, default, description) for every tunable field.

    Sections are 'agent', 'env', 'run', and 'env.graph' (the nested GraphConfig).
    Powers `terralingua --help` and the generated template_preset.yaml reference.
    """
    for section, model in (("agent", AgentConfig), ("env", EnvConfig), ("run", RunConfig)):
        for fname, field in model.model_fields.items():
            if fname == "graph":
                continue  # nested model — emitted as its own section below
            yield (
                section,
                fname,
                _type_str(field.annotation),
                _field_default(field),
                field.description or "",
            )
    for fname, field in GraphConfig.model_fields.items():
        yield (
            "env.graph",
            fname,
            _type_str(field.annotation),
            _field_default(field),
            field.description or "",
        )
