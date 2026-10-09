"""Action templates for OpenGridWorld."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Tuple

from terralingua.environment.artifact import creatable_types


@dataclass
class LocationAffordance:
    """Describes a special action available (or explicitly blocked) at a node.

    Fields
    ------
    action      : action name that appears in (or is removed from) the agent's
                  available-actions list.
    mode        : ``"add"`` (default) — expose the action at this node.
                  ``"remove"`` — suppress the action at this node even if it
                  would otherwise be globally available (e.g. spawn).
                  ``description``, ``params``, and ``effect`` are ignored for
                  remove-mode affordances.
    description : human-readable text shown to the agent (add mode only).
    params      : param schema passed to the agent, same format as ACTION_TEXT
                  entries — e.g. ``{"target": "Node ID to jump to"}``.
    effect      : machine-readable effect applied by _on_location_action.
                  Kept out of the agent-facing description on purpose; the
                  description field is what the agent sees.  Dict is kept
                  JSON-serialisable so it round-trips through checkpoints.
                  Every world handles these types (see base_env.py):
                    {"type": "add_energy",   "amount": N | "amounts": {kind: N}, "cap": C}
                    {"type": "drain_energy", "amount": N | "amounts": {kind: N}, "floor": F}
                    {"type": "move_agent"}   (params: target, destination)
                  Any type may add "target_roles": [...] to limit targets.
                  The social graph adds remove_agent, set_role, grant_ability,
                  revoke_ability.

    Usage
    -----
    Attach to a node via::

        world_graph.set_node_attr("hub", affordances=[
            LocationAffordance(
                action="recharge",
                description="Recharge your energy at the hub station.",
                effect={"type": "add_energy", "amount": 20},
            )
        ])

    Remove a globally available action at a specific node::

        world_graph.set_node_attr("spoke_0", affordances=[
            LocationAffordance(action="spawn", mode="remove"),
        ])

    or via build_graph node_props, or as plain dicts in a JSON graph file
    (graph_env will convert dicts to LocationAffordance automatically).
    """

    action: str
    mode: str = "add"
    description: str = ""
    params: dict = field(default_factory=dict)
    effect: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "LocationAffordance":
        if "action" not in d:
            raise ValueError(
                f"LocationAffordance is missing required field 'action'. Got: {d}"
            )
        mode = d.get("mode", "add")
        if mode not in ("add", "remove"):
            raise ValueError(
                f"LocationAffordance 'mode' must be 'add' or 'remove', got '{mode}'. "
                f"Got: {d}"
            )
        if mode == "add" and "description" not in d:
            raise ValueError(
                f"LocationAffordance is missing required field 'description'. Got: {d}"
            )
        return cls(
            action=d["action"],
            mode=mode,
            description=d.get("description", ""),
            params=d.get("params", {}),
            effect=d.get("effect", {}),
        )


# Base action templates - will be formatted with environment-specific parameters
def create_artifact_text() -> dict:
    """Menu text for create_artifact, listing the types agents may create now."""
    types = creatable_types()
    return {
        "description": "Creates a new artifact at the being's location.",
        "params": {
            "name": "The name of the artifact (use **unique** names)",
            "type": {
                "description": "Type of the artifact to create.",
                "choices": list(types),
            },
            "payload": f"Content of the artifact. Depends on the artifact type: {types}",
            "lifespan": "How many time steps the artifact will last (in number of steps, integer > 0. If -1 the artifact will never disappear)",
            "movable": {
                "description": "Whether other agents can pick up this artifact. A fixed artifact stays at its position permanently.",
                "choices": ["true", "false"],
            },
        },
    }


ACTION_TEXT = {
    "move": {
        "description": (
            "Move to a directly connected neighboring location, "
            "or stay in the current position."
        ),
        "params": {"direction": "Node ID of a neighbor to move to, or 'stay'."},
    },
    "give": {
        "description": "Transfer some of your energy to another nearby being.",
        "params": {
            "target": "Name of a being in your field of view to give energy to.",
            "amount": "Integer amount of energy to transfer (1 up to your current energy).",
        },
    },
    "take": {
        "description": "Steal energy from another nearby being.",
        "params": {
            "target": "Name of a being in your field of view to steal energy from.",
            "amount": "Integer amount of energy to steal (1 up to target's current energy).",
        },
    },
    "create_artifact": create_artifact_text(),
    "pickup_artifact": {
        "description": "Picks up the movable artifact and puts it in the being's inventory. It must be in the same position as the being and movable.",
        "params": {"name": "Name of the artifact to pick up. "},
    },
    "drop_artifact": {
        "description": "Drops artifact from the inventory at the being's current position",
        "params": {"name": "Name of the artifact to drop"},
    },
    "give_artifact": {
        "description": "Gives an artifact from the inventory to a nearby being",
        "params": {
            "artifact_name": "The name of the artifact",
            "target_agent": "Name of a being in your field of view to give the artifact to.",
        },
    },
    "set_color": {
        "description": "Change how you appear to other beings by choosing your color",
        "params": {"color": "The color you want to show to other beings"},
    },
}


_SOCIAL_ACTION_TEXT = {
    "give": (
        "Give some of your energy to an agent you mutually follow.",
        {"target": "Name of an agent you mutually follow."},
    ),
    "take": (
        "Take energy from an agent you mutually follow.",
        {"target": "Name of an agent you mutually follow."},
    ),
    "create_artifact": (
        "Create a public artifact. Its name is visible to your followers.",
        {
            "movable": (
                "Whether this artifact can enter inventory, be given, or be deposited. "
                "A fixed artifact remains one of your public artifacts."
            ),
        },
    ),
    "pickup_artifact": (
        "Put one of your movable public artifacts in your private inventory.",
        {},
    ),
    "drop_artifact": (
        "Make an inventory artifact public. Its name becomes visible to your followers.",
        {},
    ),
    "give_artifact": (
        "Give a movable public or private artifact to an agent you mutually follow. "
        "Public artifacts remain public; private artifacts enter the recipient's inventory.",
        {"target_agent": "Name of an agent you mutually follow."},
    ),
}


def describe_social_actions(actions: dict) -> dict:
    """Reword built-in social actions without changing their choices or rules.

    Call before adding affordances. Other action descriptions stay unchanged.
    """
    result = dict(actions)
    for name, (description, parameter_descriptions) in _SOCIAL_ACTION_TEXT.items():
        entry = actions.get(name)
        original = ACTION_TEXT[name]["description"]
        if entry is None or not entry.get("description", "").startswith(original):
            continue
        entry = deepcopy(entry)
        entry["description"] = description + entry["description"][len(original):]
        for parameter, text in parameter_descriptions.items():
            value = entry.get("params", {}).get(parameter)
            if isinstance(value, dict):
                value["description"] = text
            elif isinstance(value, str):
                entry["params"][parameter] = text
        result[name] = entry

    spawn = actions.get("spawn")
    if (
        spawn is not None
        and spawn.get("description", "").startswith("Spawn a being.")
        and isinstance(spawn.get("params", {}).get("partner"), dict)
    ):
        spawn = deepcopy(spawn)
        spawn["params"]["partner"]["description"] = (
            'An agent you follow with the same genome type. Leave empty ("") for solo spawning.'
        )
        result["spawn"] = spawn
    return result


def build_noop_action() -> Tuple[str, dict]:
    """Build the 'noop' available-action entry.

    A no-op action — used by envs (e.g. the social-graph world) where the agent
    cannot move but still needs a way to pass the turn without taking any other
    action. Functionally equivalent to `move(direction='stay')` in worlds that
    have movement.

    Returns `(name, spec)` so callers can do `actions.update([build_noop_action()])`.
    """
    return "noop", {
        "description": "Pass this turn without taking any action.",
        "params": {},
    }


def build_follow_action(
    at_cap: bool,
    current_subscriptions: list,
) -> Tuple[str, dict]:
    """Build the 'follow' available-action entry (social-graph world only).

    Following is unilateral: it creates a one-way link (you → target) so you
    start receiving the target's broadcasts and can direct-message them. The
    target does not have to agree.

    `current_subscriptions` is the list of names the agent already follows.
    When `at_cap`, `replace` is required: the agent must name one of its current
    follows to drop to make room. Otherwise `replace` is optional (`""`).
    """
    if at_cap:
        description = (
            "Follow another agent: you start receiving their broadcasts and can "
            "direct-message them (one-way, no consent needed). You are at your "
            "connection cap, so you MUST set `replace` to one of the agents you "
            "currently follow, which will be dropped to make room."
        )
        replace_choices = current_subscriptions
        replace_desc = (
            "REQUIRED while at the connection cap. "
            "Name of an agent you currently follow, to drop to make room."
        )
    else:
        description = (
            "Follow another agent: you start receiving their broadcasts and can "
            "direct-message them (one-way, no consent needed). Leave `replace` "
            "empty for a plain follow, or name an agent you follow to swap."
        )
        replace_choices = [""] + current_subscriptions
        replace_desc = (
            "Optional. Leave empty for a plain follow, or name an agent you "
            "currently follow to drop while adding the new one."
        )

    params: dict = {
        "target": (
            "Name of an agent to follow. Must be alive and not someone you "
            "already follow or have a pending request to."
        ),
        "replace": {
            "description": replace_desc,
            "choices": replace_choices,
        },
    }
    return "follow", {"description": description, "params": params}


def build_request_connection_action(
    at_cap: bool,
    current_subscriptions: list,
    pending_outgoing_targets: list,
) -> Tuple[str, dict]:
    """Build the 'request_connection' available-action entry (social-graph only).

    A connection request is consent-based and bilateral: the target sees the
    request and may accept (forming a mutual relationship) or reject. While the
    request is pending it reserves one of your slots.

    When `at_cap`, `replace` is required and may name either an agent you
    currently follow OR a pending outgoing request — dropping either frees the
    slot the new request needs.
    """
    if at_cap:
        description = (
            "Request a mutual connection with another agent. They will see the "
            "request and choose to accept or reject it; until then the slot is "
            "reserved on your side. You are at your connection cap, so you MUST "
            "set `replace` to free a slot (an agent you follow OR a pending "
            "request you sent)."
        )
        replace_choices = current_subscriptions + pending_outgoing_targets
        replace_desc = (
            "REQUIRED while at the connection cap. Name of an agent you currently "
            "follow OR a pending outgoing request to drop, to free a slot."
        )
    else:
        description = (
            "Request a mutual connection with another agent. They will see the "
            "request and choose to accept or reject it; until then the slot is "
            "reserved on your side. Leave `replace` empty, or name something to "
            "drop to make room."
        )
        replace_choices = [""] + current_subscriptions + pending_outgoing_targets
        replace_desc = (
            "Optional. Leave empty, or name an agent you follow OR a pending "
            "outgoing request to drop while sending the new one."
        )

    params: dict = {
        "target": (
            "Name of an agent to request a mutual connection with. Must be alive "
            "and not someone you already follow or have a pending request to."
        ),
        "replace": {
            "description": replace_desc,
            "choices": replace_choices,
        },
    }
    return "request_connection", {"description": description, "params": params}


def build_accept_connection_action(
    pending_sender_names: list,
    current_subscriptions: list,
    at_cap: bool,
) -> Tuple[str, dict]:
    """Build the 'accept_connection' available-action entry (social-graph only).

    Accepting a pending incoming request forms a mutual relationship: both
    directions of the link are created. When `at_cap`, `replace` is required and
    names one of your current follows to drop to make room for the new link.
    """
    if at_cap:
        description = (
            "Accept an incoming connection request, forming a mutual connection "
            "with the sender. You are at your connection cap, so you MUST set "
            "`replace` to one of the agents you currently follow, which will be "
            "dropped to make room."
        )
        replace_choices = current_subscriptions
        replace_desc = (
            "REQUIRED while at the connection cap. "
            "Name of an agent you currently follow, to drop to make room."
        )
    else:
        description = (
            "Accept an incoming connection request, forming a mutual connection "
            "with the sender. Leave `replace` empty, or name an agent you follow "
            "to drop while accepting."
        )
        replace_choices = [""] + current_subscriptions
        replace_desc = (
            "Optional. Leave empty, or name an agent you currently follow to "
            "drop while accepting."
        )

    params: dict = {
        "target": {
            "description": "Name of the agent whose connection request to accept.",
            "choices": pending_sender_names,
        },
        "replace": {
            "description": replace_desc,
            "choices": replace_choices,
        },
    }
    return "accept_connection", {"description": description, "params": params}


def build_reject_connection_action(pending_sender_names: list) -> Tuple[str, dict]:
    """Build the 'reject_connection' available-action entry (social-graph only).

    Rejecting drops a pending incoming request; nothing is formed.
    """
    return "reject_connection", {
        "description": (
            "Reject an incoming connection request. The request is dropped and no "
            "connection is formed."
        ),
        "params": {
            "target": {
                "description": "Name of the agent whose connection request to reject.",
                "choices": pending_sender_names,
            },
        },
    }


def build_direct_message_action(
    current_connections: list,
    direct_message_cost: int,
    *,
    finite_energy: bool = True,
) -> Tuple[str, dict]:
    """Build the 'send_direct_message' available-action entry.

    `current_connections` is the list of names the agent is already connected
    to. The action is meaningful only when that list is non-empty (caller is
    responsible for not exposing it otherwise).

    Unlike the broadcast `message:` side-channel, this is a private one-to-one
    message — only the named target sees the content.
    """
    description = (
        "Send a private message to one agent you follow. Only that agent receives it. "
        "This uses your action for the turn."
    )
    if finite_energy and direct_message_cost > 0:
        description += f" Costs {direct_message_cost} energy."
    return "send_direct_message", {
        "description": description,
        "params": {
            "target": {
                "description": "Name of an agent you follow.",
                "choices": current_connections,
            },
            "content": "Plain text content of the private message.",
        },
    }


def build_request_role_action(open_roles: dict) -> Tuple[str, dict]:
    """Build the 'request_role' available-action entry (roles mode).

    `open_roles` maps role name -> free slot count, listing only roles with at
    least one free slot. The request is granted by fixed rules: the role must
    have a free slot and the requester must hold no role. Nothing else decides.

    Returns `(name, spec)` so callers can do `actions.update([...])`.
    """
    return "request_role", {
        "description": (
            "Request to occupy an open role. Granted immediately when the "
            "role has a free slot and you hold no role. Occupying a role "
            "gives you its charter, its current mandate, and its permissions."
        ),
        "params": {
            "role": {
                "description": "Name of an open role to occupy.",
                "choices": sorted(open_roles),
            },
        },
    }


def build_leave_role_action(
    role_name: str, on_vacancy: str = "open"
) -> Tuple[str, dict]:
    """Build the 'leave_role' available-action entry (roles mode).

    The role itself, its charter, mandate and permissions persist. What happens
    to the freed slot depends on the role's `on_vacancy` policy: `open` reopens
    it, `succeed` passes it to the occupant's named successor when one exists.
    Roles with `on_vacancy: retire` never offer this action.
    """
    if on_vacancy == "succeed":
        slot_text = (
            "the slot passes to your named successor, or opens up if you named none."
        )
    else:
        slot_text = "the role and everything it owns stays and its slot opens up."
    return "leave_role", {
        "description": (
            f"Leave your current role ({role_name}). You lose its permissions; "
            f"{slot_text}"
        ),
        "params": {},
    }


def build_transfer_role_action(
    role_name: str, target_names: list
) -> Tuple[str, dict]:
    """Build the 'transfer_role' available-action entry (roles mode).

    `target_names` lists the roleless agents within the occupant's reach. The
    hand-over is immediate and needs no consent from the receiver.
    """
    return "transfer_role", {
        "description": (
            f"Hand your role ({role_name}) to a roleless agent within your reach. "
            "The transfer is immediate: they gain the role's charter, mandate and "
            "permissions, and you lose them."
        ),
        "params": {
            "target": {
                "description": "Name of the agent who receives the role.",
                "choices": sorted(target_names),
            },
        },
    }


def build_name_successor_action(
    role_name: str, target_names: list, current_successor: str = ""
) -> Tuple[str, dict]:
    """Build the 'name_successor' available-action entry (roles mode).

    Offered to occupants of roles with `on_vacancy: succeed`. `target_names`
    lists the agents within the occupant's reach. Naming replaces any earlier
    choice.
    """
    description = (
        f"Name the agent who takes your role ({role_name}) when you leave it or "
        "die. They must hold no role at that moment. Replaces any earlier choice."
    )
    if current_successor:
        description += f" Current successor: {current_successor}."
    return "name_successor", {
        "description": description,
        "params": {
            "target": {
                "description": "Name of the agent to name as successor.",
                "choices": sorted(target_names),
            },
        },
    }


def build_disconnect_action(
    current_subscriptions: list,
    pending_outgoing_targets: list,
) -> Tuple[str, dict]:
    """Build the 'disconnect' available-action entry (social-graph world only).

    A single action covers both modes: it drops an agent you follow (removing
    only your one-way link to them — any link they have to you is unaffected) OR
    cancels a pending outgoing connection request you sent.

    The caller is responsible for only exposing this when at least one of the
    two lists is non-empty.
    """
    return "disconnect", {
        "description": (
            "Drop an agent you follow (removes only your link to them; theirs to "
            "you, if any, stays), or cancel a pending connection request you sent."
        ),
        "params": {
            "target": {
                "description": (
                    "Name of an agent you follow, or a pending outgoing request "
                    "to cancel."
                ),
                "choices": current_subscriptions + pending_outgoing_targets,
            },
        },
    }


def build_spawn_action(
    reproduction_cost: int,
    init_agent_energy: float,
    genome_type: str,
    same_type_nearby_names: list,
    *,
    finite_energy: bool = True,
) -> Tuple[str, dict]:
    """Build the 'spawn' available-action entry.

    Two-parent spawning is offered iff same_type_nearby_names is non-empty
    (caller is responsible for filtering to same genome type only).
    """
    if same_type_nearby_names:
        description = (
            "Spawn a being. "
            "Provide a partner name for two-parent spawning (genome crossover), "
            "or leave partner parameter empty for solo spawning."
        )

    else:
        description = "Solo-spawn a being."

    if reproduction_cost > 0:
        if finite_energy:
            description += f" Spawning costs {reproduction_cost} energy from you."
        description += f" The new being starts with {reproduction_cost} energy."
        if finite_energy:
            description += " Failed attempts still consume this cost."
    else:
        if finite_energy:
            description += " Spawning is free."
        if init_agent_energy != float("inf"):
            description += f" The new being starts with {init_agent_energy} energy."

    gift_description = (
        "Optional additional energy from you to the new being. Default: 0. "
        "Use a nonnegative integer within your balance after the reproduction cost. "
        "This gift is deducted only when the birth succeeds."
        if finite_energy
        else "Optional additional energy for the new being. Default: 0. "
        "Use a nonnegative integer. Applied only when the birth succeeds."
    )
    params: dict = {"energy": gift_description}
    params["name"] = "Name of the new being (use **unique** names, max 20 characters)"

    if genome_type == "sentence_directed":
        params["offspring_genome"] = (
            "A short sentence defining the personality of the new being."
        )

    if same_type_nearby_names:
        params["offspring_genome"] = (
            "A short sentence defining the personality of the new being."
            " Only used for solo spawning — ignored if a partner is provided."
        )
        params["partner"] = {
            "description": 'Name of a nearby being to spawn with for two-parent spawning. Leave empty ("") for solo spawning.',
            "choices": [""] + same_type_nearby_names,
        }

    return "spawn", {"description": description, "params": params, "optional": ["energy"]}


def build_show_library_action(
    library_count: int, library_preview: bool
) -> Tuple[str, dict]:
    """Build the 'show_library_content' available-action entry (social-graph only).

    The library is a shared commons every being can read from and write to. This
    action lists what it currently holds so the agent can decide what to retrieve.
    The live count is embedded so the agent always sees the library size.
    """
    detail = "name and a short content preview" if library_preview else "name"
    return "show_library_content", {
        "description": (
            f"List the contents of the shared library, which currently holds "
            f"{library_count} artifact(s). Returns the {detail} of every artifact "
            "in it so you can decide what to retrieve."
        ),
        "params": {},
    }


def build_retrieve_artifact_action(library_names: list) -> Tuple[str, dict]:
    """Build the 'retrieve_artifact' available-action entry (social-graph only).

    Moves an artifact from the shared library onto the being's own node, where it
    becomes readable and interactable. The artifact leaves the library.
    """
    return "retrieve_artifact", {
        "description": (
            "Take an artifact from the shared library into your public artifacts. "
            "It leaves the library until deposited again."
        ),
        "params": {
            "name": {
                "description": "Name of an artifact currently in the library.",
                "choices": library_names,
            },
        },
    }


def build_read_artifact_action(library_names: list) -> Tuple[str, dict]:
    """Build the 'read_artifact' available-action entry (social-graph only).

    Reads the full content of a library artifact in place, WITHOUT removing it
    from the library — so it stays available for every other agent to read too.
    """
    return "read_artifact", {
        "description": (
            "Read the full content of an artifact in the shared library WITHOUT "
            "removing it — it stays in the library for everyone else."
        ),
        "params": {
            "name": {
                "description": "Name of an artifact currently in the library.",
                "choices": library_names,
            },
        },
    }


def build_deposit_artifact_action(depositable_names: list) -> Tuple[str, dict]:
    """Build the 'deposit_artifact' available-action entry (social-graph only).

    Moves a movable artifact the being holds (on its node or in its inventory)
    into the shared library, where any being can later retrieve it. Artifacts in
    the library remain available until they expire or are retrieved.
    """
    return "deposit_artifact", {
        "description": (
            "Put one of your movable public or private artifacts into the shared library. "
            "Other agents can read or retrieve it. "
            "It remains available until the artifact expires or is retrieved."
        ),
        "params": {
            "name": {
                "description": (
                    "Name of one of your movable public or private artifacts."
                ),
                "choices": depositable_names,
            },
        },
    }
