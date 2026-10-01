"""Parameter applicability and its explanations.

Conditions use JSON Schema against canonical, flat parameter paths.
Configuration values remain stored when their controls are inactive.
"""

from jsonschema import Draft202012Validator


def when(path: str, **schema) -> dict:
    """Match one canonical parameter with a JSON Schema condition."""
    return {"properties": {path: schema}, "required": [path]}


def every(*conditions) -> dict:
    return {"allOf": list(conditions)}


def either(*conditions) -> dict:
    return {"anyOf": list(conditions)}


GRID = when("env.world_type", const="grid")
GRAPH = when("env.world_type", enum=["graph", "social_graph"])
SOCIAL = when("env.world_type", const="social_graph")
PLAIN_GRAPH = when("env.world_type", const="graph")
FOOD = every(when("env.food_mechanism", const=True), {"not": SOCIAL})
REPRODUCTION = when("env.reproduction_cost", minimum=0)
LIBRARY = every(SOCIAL, when("env.use_library", const=True))
AUTHORING = when("env.artifact_creation_cost", minimum=0)
EXTERNAL = when("run.external_servers_config", type="string", minLength=1)
REMOTE = when("run.remote_api_enabled", const=True)


def topology(name: str) -> dict:
    return every(GRAPH, when("env.graph.topology", const=name))


# Independent fields are explicit too. New fields need a reviewed group.
GROUPS = {
    "Identity": [
        "agent.agents_name_prefix", "agent.genome", "agent.random_names",
        "agent.scenario_specific_instructions",
    ],
    "Memory": [
        "agent.internal_memory_size", "agent.max_history",
    ],
    "Model": [
        "agent.model", "agent.max_tokens", "agent.reasoning_effort",
        "run.max_parallel_workers", "run.ports",
    ],
    "Artifacts": [
        "agent.use_inventory", "env.artifact_creation_cost",
        "env.allow_fixed_artifacts", "env.max_artifact_tokens", "env.inert_artifacts",
        "env.init_artifacts_path", "env.use_library", "env.library_preview",
    ],
    "Communication": [
        "env.max_message_length", "env.graph.direct_message_cost",
    ],
    "Population and energy": [
        "env.agent_lifespan", "env.energy_death", "env.init_agents",
        "env.init_human_agents", "env.init_agent_energy", "env.min_agents",
        "env.max_agents", "env.reproduction_cost", "env.two_parent_spawn",
        "env.genome_mutation_rate",
    ],
    "Food": [
        "env.food_mechanism", "env.drop_food_on_death", "env.food_decay_rate",
        "env.food_spawn_rate", "env.food_zones", "env.init_food", "env.static_food",
    ],
    "World": [
        "env.world_type", "env.grid_size", "env.vision_radius",
        "env.dynamic_grid_scaling", "env.dynamic_grid_min_cells_per_agent",
        "env.dynamic_grid_step", "env.excluded_actions", "agent.use_colors",
        "env.roles_hocon_path", "env.affordances_file_path",
    ],
    "Graph": [
        "env.graph.topology", "env.graph.n_nodes", "env.graph.hop_radius",
        "env.graph.wrap", "env.graph.tree_branching", "env.graph.tree_height",
        "env.graph.bipartite_n1", "env.graph.bipartite_n2",
        "env.graph.small_world_k", "env.graph.small_world_p",
        "env.graph.scale_free_m", "env.graph.random_p", "env.graph.seed",
        "env.graph.file_path", "env.graph.custom_factory",
        "env.graph.agent_network_hocon_path", "env.graph.agent_network_bidirectional_edges",
    ],
    "Social connections": [
        "env.graph.max_connections", "env.graph.edge_decay_steps",
        "env.graph.random_intro_k", "env.graph.allow_follow",
        "env.graph.allow_connection", "env.graph.connection_request_lifespan",
    ],
    "External tools": [
        "run.external_servers_config", "run.external_sync_send",
        "run.external_server_port", "run.external_server_callback_port",
        "run.reward_to_energy_coefficient",
        "run.stop_on_external_solved",
    ],
    "Remote agents": [
        "run.remote_api_enabled", "run.env_heartbeat", "run.allow_human_agents",
    ],
    "Run and output": [
        "run.runner_class", "run.scenario", "run.scenario_options", "run.ckpt_interval",
        "run.empty_countdown", "run.exp_description", "run.exp_name",
        "run.live_render", "run.max_ts", "run.save_video", "run.video_fps",
        "run.seed",
    ],
}

# Each reason states the activation requirement, including combined conditions.
APPLICABILITY = {
    "env.food_mechanism": ({"not": SOCIAL}, "Social graphs do not support natural food."),
    **{
        path: (FOOD, "Requires food.")
        for path in (
            "env.drop_food_on_death", "env.food_decay_rate", "env.food_spawn_rate",
            "env.init_food",
        )
    },
    "env.food_zones": ({"not": SOCIAL}, "Requires a spatial world with natural or manually injected food."),
    "env.static_food": ({"not": SOCIAL}, "Requires a spatial world with natural or manually injected food."),
    "env.two_parent_spawn": (REPRODUCTION, "Requires reproduction."),
    "env.genome_mutation_rate": (
        every(REPRODUCTION, when("agent.genome", enum=["ocean_5", "sentence_mutate"])),
        "Requires reproduction with a genome that mutates.",
    ),
    "env.dynamic_grid_scaling": (GRID, "Requires a grid world."),
    "env.dynamic_grid_min_cells_per_agent": (
        every(GRID, when("env.dynamic_grid_scaling", const=True)),
        "Requires automatic grid expansion.",
    ),
    "env.dynamic_grid_step": (
        every(GRID, when("env.dynamic_grid_scaling", const=True)),
        "Requires automatic grid expansion.",
    ),
    "env.grid_size": (GRID, "Requires a grid world."),
    "env.vision_radius": (GRID, "Requires a grid world."),
    "env.allow_fixed_artifacts": (AUTHORING, "Requires artifact creation."),
    "env.use_library": (SOCIAL, "Requires a social graph."),
    "env.library_preview": (every(LIBRARY, when("env.inert_artifacts", const=False)), "Requires an active social graph library."),
    **{
        path: (GRAPH, "Requires a graph or social graph world.")
        for path in (
            "env.graph.topology",
        )
    },
    "env.graph.n_nodes": (
        every(GRAPH, either(
            when("env.graph.topology", enum=["grid", "ring", "complete", "small_world", "scale_free", "random", "star", "wheel", "custom"]),
            every(when("env.graph.topology", const="tree"), when("env.graph.tree_height", const=None)),
            every(when("env.graph.topology", const="bipartite"), either(
                when("env.graph.bipartite_n1", const=0), when("env.graph.bipartite_n2", const=0),
            )),
        )),
        "File and HOCON sources, explicit tree heights, or complete partition sizes determine node counts.",
    ),
    "env.graph.hop_radius": (
        PLAIN_GRAPH, "Plain graph worlds use this radius. Social graphs use one hop.",
    ),
    "env.graph.wrap": (topology("grid"), "Requires the grid graph topology."),
    "env.graph.tree_branching": (topology("tree"), "Requires the tree topology."),
    "env.graph.tree_height": (topology("tree"), "Requires the tree topology."),
    "env.graph.bipartite_n1": (topology("bipartite"), "Requires the bipartite topology."),
    "env.graph.bipartite_n2": (topology("bipartite"), "Requires the bipartite topology."),
    "env.graph.small_world_k": (topology("small_world"), "Requires the small-world topology."),
    "env.graph.small_world_p": (topology("small_world"), "Requires the small-world topology."),
    "env.graph.scale_free_m": (topology("scale_free"), "Requires the scale-free topology."),
    "env.graph.random_p": (topology("random"), "Requires the random topology."),
    "env.graph.seed": (
        every(GRAPH, when("env.graph.topology", enum=["small_world", "scale_free", "random"])),
        "Requires a small-world, scale-free, or random topology.",
    ),
    "env.graph.file_path": (topology("file"), "Requires the file topology."),
    "env.graph.custom_factory": (topology("custom"), "Requires the custom topology."),
    "env.graph.agent_network_bidirectional_edges": (
        topology("agent_network"), "Requires the agent-network topology.",
    ),
    **{
        path: (SOCIAL, "Requires a social graph.")
        for path in (
            "env.graph.max_connections", "env.graph.edge_decay_steps",
            "env.graph.random_intro_k", "env.graph.direct_message_cost",
            "env.graph.allow_follow", "env.graph.allow_connection",
        )
    },
    "env.graph.connection_request_lifespan": (
        every(SOCIAL, when("env.graph.allow_connection", const=True)),
        "Requires social graph connection requests.",
    ),
    **{
        path: (EXTERNAL, "Requires external servers.")
        for path in (
            "run.external_sync_send", "run.external_server_port",
            "run.external_server_callback_port", "run.reward_to_energy_coefficient",
            "run.stop_on_external_solved",
        )
    },

    "run.allow_human_agents": (REMOTE, "Requires the remote-agent API."),
    "run.scenario_options": (
        either(
            when("run.scenario", type="string", minLength=1),
            when("run.runner_class", type="string", minLength=1),
        ),
        "Requires a scenario or a scenario runner.",
    ),
    "run.video_fps": (when("run.save_video", const=True), "Requires video output."),
}

NOTES = {
    "env.food_mechanism": (
        "Food enables natural generation, decay, and one-energy upkeep per step. "
        "Energy balances and reproduction remain available without food."
    ),
    "env.energy_death": (
        "Null follows food_mechanism. An explicit boolean works independently of food."
    ),
    "env.init_agent_energy": (
        "Used for founders, arrivals, replenishment, and free births. "
        "Paid births receive reproduction_cost plus any parent-funded gift."
    ),
    "env.max_agents": "Limits reproduction. Initial populations and other arrival paths need separate checks.",
    "env.reproduction_cost": (
        "Failed attempts consume the base cost. Additional gifts are charged only on success."
    ),
    "agent.max_history": "History length is independent of internal memory.",
    "env.use_library": "Library access does not require artifact creation or inventory.",
    "env.graph.agent_network_hocon_path": (
        "This path selects the agent-network topology and a graph world."
    ),
    "env.graph.n_nodes": (
        "Some topology sources determine the actual node count themselves. "
        "Social graphs add a node when a new agent needs one."
    ),
    "run.ports": "Local model server ports, used by local agent models.",
    "run.env_heartbeat": "Bounds the whole decision batch and sets minimum turn duration. Includes local agents.",
    "run.scenario_options": "The selected scenario or scenario runner validates this dictionary.",
    "run.allow_human_agents": "Controls the dashboard label. Registration does not currently enforce this setting.",
    "env.max_artifact_tokens": "Also applies to seeded and administrator-created text artifacts.",
    "env.food_zones": "Grid worlds use coordinates. Graph worlds use node names. Manual food injection also uses these zones.",
    "env.static_food": "Manual food injection in spatial worlds can use fixed sites without natural food.",
}

DERIVED_DEPENDENCIES = {
    "artifact_creation_enabled": ["env.artifact_creation_cost"],
    "internal_memory_enabled": ["agent.internal_memory_size"],
    "energy_death": ["env.food_mechanism", "env.energy_death"],
    "reproduction_enabled": ["env.reproduction_cost"],
    "newborn_base_energy": ["env.reproduction_cost", "env.init_agent_energy"],
    "failed_birth_cost": ["env.reproduction_cost"],
    "hop_radius": ["env.world_type", "env.graph.hop_radius"],
}

# These descriptions document Pydantic's authoritative validators.
CONSTRAINTS = [
    {
        "id": "social_food",
        "fields": ["env.world_type", "env.food_mechanism"],
        "description": "Social graphs do not support natural food. Energy remains available.",
    },
    {
        "id": "initial_tag_collision",
        "fields": ["agent.agents_name_prefix", "env.init_agents", "env.init_human_agents", "env.graph.agent_network_hocon_path"],
        "description": "Initial LLM and human-agent tags must not collide.",
    },
    {
        "id": "topology_source",
        "fields": ["env.world_type", "env.graph.topology", "env.graph.file_path", "env.graph.custom_factory"],
        "description": "Active file and custom topologies require their source paths.",
    },
    {
        "id": "food_zone_shape",
        "fields": ["env.world_type", "env.food_mechanism", "env.food_zones"],
        "description": "Active food zones need positive counts, grid coordinates, or graph node names.",
    },
    {
        "id": "population_minimum",
        "fields": ["env.min_agents", "env.init_agents"],
        "description": "Minimum population cannot exceed the initial agent count.",
    },
    {
        "id": "hocon_sources",
        "fields": ["env.graph.agent_network_hocon_path", "env.roles_hocon_path"],
        "description": "Founding-agent and role HOCON files are mutually exclusive.",
    },
    {
        "id": "hocon_genome",
        "fields": ["env.graph.agent_network_hocon_path", "agent.genome"],
        "description": "HOCON founding agents require the sentence_directed genome.",
    },
    {
        "id": "hocon_topology",
        "fields": ["env.graph.topology", "env.graph.agent_network_hocon_path"],
        "description": "The agent-network topology requires its HOCON file.",
    },
]


def dependencies(condition) -> list[str]:
    """Read parameter references from applicability conditions."""
    if isinstance(condition, bool):
        return []
    refs = set(condition.get("properties", {}))
    for key in ("allOf", "anyOf", "oneOf"):
        for child in condition.get(key, []):
            refs.update(dependencies(child))
    if "not" in condition:
        refs.update(dependencies(condition["not"]))
    return sorted(refs)


def is_active(condition, values: dict) -> bool:
    return Draft202012Validator(condition).is_valid(values)


def prefixed(condition, prefix: str):
    """The same condition with every parameter path prefixed."""
    if isinstance(condition, bool):
        return condition
    result = {}
    for key, value in condition.items():
        if key == "properties":
            result[key] = {prefix + path: schema for path, schema in value.items()}
        elif key == "required":
            result[key] = [prefix + path for path in value]
        elif key in ("allOf", "anyOf", "oneOf"):
            result[key] = [prefixed(child, prefix) for child in value]
        elif key == "not":
            result[key] = prefixed(value, prefix)
        else:
            result[key] = value
    return result
