"""social_graph_env.py — OpenSocialGraphWorld.

A subclass of OpenGraphWorld where each agent owns its own node and the graph
edges represent a **directional** social network:

- Each initial agent is placed on a distinct node of the topology built by
  `graph_cfg` (so `n_nodes == init_agents` and the topology defines the
  starting social network).
- Edges are single directional links. `A → B` means: A is subscribed to B's
  broadcasts AND A can direct-message B. The reverse `B → A` is independent.
- Agents cannot move (`_apply_move` is a no-op; `move` is not in available
  actions; `noop` is the no-op alternative).
- A new agent born via `spawn` adds a new node connected to its parent(s) by a
  mutual (two-way) edge. Solo spawn → one parent; two-parent spawn → both.
- Two formation actions shape the graph at runtime, each behind its own flag:
  - `follow(target)` — unilateral; forms `A → B` immediately (no consent).
  - `request_connection(target)` — consent-based; sends a pending request the
    target sees and may `accept_connection` (forms both `A→B` and `B→A`) or
    `reject_connection`. A pending request reserves one of A's slots and
    auto-expires after `connection_request_lifespan` steps.
  A single `disconnect(target)` drops A's `A→B` link OR cancels a pending
  outgoing request. It never touches `B → A`.
- The per-agent cap (`graph_cfg.max_connections`) counts A's **outgoing** edges
  plus A's pending outgoing requests. Incoming followers don't consume A's cap.
- Each step every agent receives up to `graph_cfg.random_intro_k`
  `suggested_connections`. About two thirds prefer outgoing two-hop paths
  (`A → B → C`); the rest are random. Existing follows and outgoing pending
  requests are excluded. Random choices fill any shortage of two-hop candidates.
- Optional edge decay: if `graph_cfg.edge_decay_steps` is set, each direction of
  an edge is pruned independently when it sees no recent activity.
"""

import copy
import logging
from itertools import count
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

from terralingua.config.models import GraphConfig
from terralingua.environment.actions import (
    LocationAffordance,
    build_accept_connection_action,
    build_deposit_artifact_action,
    build_direct_message_action,
    build_disconnect_action,
    build_follow_action,
    build_noop_action,
    build_read_artifact_action,
    build_reject_connection_action,
    build_request_connection_action,
    build_retrieve_artifact_action,
    build_show_library_action,
    describe_social_actions,
)
from terralingua.environment.artifact import MAX_TEXT_ARTIFACT_SIZE
from terralingua.environment.env_logger import Event
from terralingua.environment.graph_env import OpenGraphWorld

log = logging.getLogger(__name__)

_LIBRARY_PREVIEW_CHARS = 120

# Abilities a manager may grant to / revoke from a connected agent via the
# grant_ability / revoke_ability affordances. Each spec is the affordance
# installed on the target's node, so the granted action dispatches through the
# same effect registry. Keyed by ability name (== effect type).
GRANTABLE_ABILITY_SPECS: Dict[str, dict] = {
    "remove_agent": {
        "action": "remove_agent",
        "mode": "add",
        "description": "Permanently remove from the network an agent you are connected to.",
        "params": {
            "target": "Name of an agent you are connected to and want to remove."
        },
        "effect": {"type": "remove_agent"},
    },
    "set_role": {
        "action": "set_role",
        "mode": "add",
        "description": "Set the role and current objective of an agent you are connected to.",
        "params": {
            "target": "Name of an agent you are connected to.",
            "role": "Short role/title to assign (e.g. 'team lead').",
            "motivation": "Current objective/directive for them (may be empty).",
        },
        "effect": {"type": "set_role"},
    },
    "grant_ability": {
        "action": "grant_ability",
        "mode": "add",
        "description": "Grant an ability to an agent you are connected to (or restore one you revoked).",
        "params": {
            "target": "Name of an agent you are connected to.",
            "ability": "Ability to grant: a manager power (remove_agent, set_role, grant_ability, revoke_ability) or a standard action you previously revoked (e.g. follow, spawn, send_direct_message).",
        },
        "effect": {"type": "grant_ability"},
    },
    "revoke_ability": {
        "action": "revoke_ability",
        "mode": "add",
        "description": "Revoke an ability from an agent you are connected to.",
        "params": {
            "target": "Name of an agent you are connected to.",
            "ability": "Ability to revoke: a manager power (remove_agent, set_role, grant_ability, revoke_ability) or a standard action (e.g. follow, spawn, send_direct_message).",
        },
        "effect": {"type": "revoke_ability"},
    },
}

# Globally-available actions a manager may suppress (revoke) or restore (grant)
# on a connected agent. Revoking adds a remove-mode affordance on the target's
# node; granting lifts it. `noop` is intentionally excluded so an agent always
# keeps a no-op.
REVOCABLE_GLOBAL_ACTIONS: Set[str] = {
    "follow",
    "request_connection",
    "accept_connection",
    "reject_connection",
    "disconnect",
    "send_direct_message",
    "give",
    "take",
    "give_artifact",
    "spawn",
    "create_artifact",
    "set_color",
}


class OpenSocialGraphWorld(OpenGraphWorld):
    """Agents live on their own node; directional edges form a social network."""

    system_prompt_template = "social_graph.j2"
    _LOG_FILENAME = "open_social_graph_world.log"

    @property
    def stay_action(self) -> dict:
        # Agents can't move here; `noop` is the pass-the-turn action.
        return {"action": "noop", "message": "", "params": {}}

    def __init__(
        self,
        graph_cfg: GraphConfig,
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
        use_library: bool = True,
        library_preview: bool = True,
        allow_fixed_artifacts: bool = True,
        max_artifact_tokens: int = MAX_TEXT_ARTIFACT_SIZE,
        use_colors: bool = False,
        reproduction_cost: int = 50,
        artifact_creation_cost: int = 0,
        two_parent_spawn: bool = False,
        max_agents: int | None = None,
        food_zones: int | List[str] | None = None,
        food_sigma: float = 1.5,
        static_food: bool = False,
        food_mechanism: bool = False,
        energy_death: bool | None = None,
        verbose: int = 2,
        inert_artifacts: bool = False,
        external_actions_spec: dict | None = None,
        excluded_actions: List[str] | None = None,
        headless: bool = False,
        max_message_length: int = 200,
        roles_hocon_path: str | None = None,
        affordances_file_path: str | None = None,
    ):
        if food_mechanism:
            raise ValueError(
                "Social graphs do not support natural food. Set food_mechanism=False; "
                "energy balances, fees, transfers, and rewards remain available."
            )
        super().__init__(
            graph_cfg=graph_cfg,
            init_agent_energy=init_agent_energy,
            lifespan=lifespan,
            init_food=init_food,
            max_food_value=max_food_value,
            food_decay_rate=food_decay_rate,
            food_decay_amount=food_decay_amount,
            food_spawn_rate=food_spawn_rate,
            log_path=log_path,
            drop_food_on_death=drop_food_on_death,
            use_inventory=use_inventory,
            use_library=use_library,
            library_preview=library_preview,
            allow_fixed_artifacts=allow_fixed_artifacts,
            max_artifact_tokens=max_artifact_tokens,
            use_colors=use_colors,
            reproduction_cost=reproduction_cost,
            artifact_creation_cost=artifact_creation_cost,
            two_parent_spawn=two_parent_spawn,
            max_agents=max_agents,
            food_zones=food_zones,
            food_sigma=food_sigma,
            static_food=static_food,
            food_mechanism=food_mechanism,
            energy_death=energy_death,
            verbose=verbose,
            inert_artifacts=inert_artifacts,
            external_actions_spec=external_actions_spec,
            excluded_actions=excluded_actions,
            headless=headless,
            max_message_length=max_message_length,
            roles_hocon_path=roles_hocon_path,
            affordances_file_path=affordances_file_path,
        )
        # OpenGraphWorld hardcodes shared occupancy; in social mode each agent
        # owns a single node, so we flip it after super init.
        self.exclusive_pos_occupancy = True
        # Direct connections only: BFS radius for obs/messaging is 1 hop.
        self.hop_radius = 1

        # Social-graph state pulled from graph_cfg.
        self.random_intro_k = graph_cfg.random_intro_k
        self.max_connections = graph_cfg.max_connections
        self.edge_decay_steps = graph_cfg.edge_decay_steps
        self.direct_message_cost = graph_cfg.direct_message_cost
        self.allow_follow = graph_cfg.allow_follow
        self.allow_connection = graph_cfg.allow_connection
        self.connection_request_lifespan = graph_cfg.connection_request_lifespan

        # Monotonic ID source for newly created nodes (e.g. on spawn).
        self._new_node_counter = count(0)

        # Per-edge last-activity step for optional edge decay. Keyed by the
        # directional pair (u, v) — each direction decays independently.
        self._edge_last_active: Dict[Tuple[str, str], int] = {}

        # Seed initial edges' last_active with step 0 so decay only kicks in
        # after the first edge_decay_steps steps.
        for u in self.world_graph.all_nodes():
            for v in self.world_graph.neighbors(u):
                self._edge_last_active.setdefault((u, v), 0)

        # Pending connection requests.
        #   _pending_outgoing: sender_tag -> {target_tag: step_request_sent}
        #   _pending_incoming: target_tag -> {sender_tag, ...}  (fast obs lookup)
        self._pending_outgoing: Dict[str, Dict[str, int]] = {}
        self._pending_incoming: Dict[str, Set[str]] = {}

        # Per-step private DM mailbox: recipient_tag -> {sender_name: content}.
        # Populated by _handle_direct_message during dispatch, read by
        # _build_obs to surface in obs["incoming_dms"], then cleared at the
        # start of the next step() so each DM is visible for exactly one step.
        self._pending_dms: Dict[str, Dict[str, str]] = {}

        # Manager-assigned role/objective overlay per agent, surfaced in obs.
        #   agent_tag -> {"role": str, "motivation": str}
        self._agent_directives: Dict[str, dict] = {}

        self._affordance_effects["remove_agent"] = self._handle_remove_agent
        self._affordance_effects["set_role"] = self._handle_set_role
        self._affordance_effects["grant_ability"] = self._handle_grant_ability
        self._affordance_effects["revoke_ability"] = self._handle_revoke_ability

    def inject_resource(
        self, amount: float, coefficient: float, target: str | None,
    ) -> None:
        """Social rewards affect agent energy; there is no spatial food."""
        if target is None:
            raise ValueError(
                "Social graphs do not support spatial food injection. "
                "Use target='all' or an agent tag to change energy."
            )
        super().inject_resource(amount, coefficient, target)

    # ---------- placement ----------
    def _apply_move(self, agent, move_params, infos):
        # No movement in social_graph. Always stay on the owned node.
        return self.agent_pos[agent], infos

    def _place_role_occupants(self) -> None:
        # Agents own their node here, so roles cannot relocate them.
        placed = [n for n, r in self._roles.items() if r.get("start_nodes")]
        if placed:
            raise ValueError(
                f"start_nodes is not supported in the social graph (roles: {placed})."
            )

    def _random_free_pos(self) -> str:
        """Return an unoccupied pre-built node, or create a fresh isolated one.

        At init time the topology is fully populated and there are unoccupied
        pre-built nodes — pick from those (1:1 binding of agents to topology
        nodes). When the topology is exhausted (e.g. an extra agent is added
        at runtime without going through the spawn handler), create a new
        isolated node so the env doesn't loop forever looking for free space.
        """
        if self.rng is None:
            self.rng = np.random.default_rng()
        all_nodes = self.world_graph.all_nodes()
        free_nodes = [n for n in all_nodes if not self.pos_to_agent.get(n)]
        if free_nodes:
            return free_nodes[int(self.rng.integers(0, len(free_nodes)))]
        new_node = f"social_node_{next(self._new_node_counter)}"
        existing = set(all_nodes)
        while new_node in existing:
            new_node = f"social_node_{next(self._new_node_counter)}"
        self.world_graph.add_node(new_node, label=new_node)
        self._food_spawn_nodes = None
        return new_node

    def _offspring_position(self, center: str, node_id: str | None = None) -> str:
        """Create a fresh node and a mutual edge to `center`. Always succeeds.

        Called from the spawn handler in base_env: returning a fresh node ID
        means each spawn appends a new vertex to the graph wired to the
        parent. The base handler then calls `_place_agent` on the new node,
        which lives in the graph thanks to the add_node call below.

        `node_id` is the offspring's (unique) name; using it as the node ID
        keeps the node and its sole occupant named the same — matching the 1:1
        agent/node binding and HOCON-seeded nodes. Falls back to a generated ID
        when the name is empty or already taken as a node.
        """
        existing = set(self.world_graph.all_nodes())
        if node_id and node_id not in existing:
            new_node = node_id
        else:
            new_node = f"social_node_{next(self._new_node_counter)}"
            # Defensive: skip past any collision with pre-existing IDs.
            while new_node in existing:
                new_node = f"social_node_{next(self._new_node_counter)}"
        self.world_graph.add_node(new_node, label=new_node)
        # Mutual edge at birth (both directions, each tracked independently).
        self.world_graph.add_edge(center, new_node)
        self.world_graph.add_edge(new_node, center)
        self._edge_last_active[(center, new_node)] = self.step_count
        self._edge_last_active[(new_node, center)] = self.step_count
        # Graph mutated — invalidate the cached food-spawn distribution.
        self._food_spawn_nodes = None
        return new_node

    # ---------- cap / lookup helpers ----------
    def _outgoing_count(self, agent_tag: str) -> int:
        return len(self.world_graph.neighbors(self.agent_pos[agent_tag]))

    def _pending_outgoing_count(self, agent_tag: str) -> int:
        return len(self._pending_outgoing.get(agent_tag, {}))

    def _has_cap_room(self, agent_tag: str) -> bool:
        """True if the agent can take on one more outgoing edge / pending request."""
        if self.max_connections is None:
            return True
        used = self._outgoing_count(agent_tag) + self._pending_outgoing_count(agent_tag)
        return used < self.max_connections

    def _node_owner_name(self, node: str) -> str | None:
        for t in self.pos_to_agent.get(node, set()):
            return self.agent_names[t]  # exclusive occupancy: one agent per node
        return None

    def _get_subscriptions(self, agent_tag: str) -> List[str]:
        """Names of agents A follows (A's outgoing edges)."""
        node = self.agent_pos[agent_tag]
        names = [self._node_owner_name(n) for n in self.world_graph.neighbors(node)]
        return sorted(n for n in names if n)

    def _get_followers(self, agent_tag: str) -> List[str]:
        """Names of agents who follow A (A's incoming edges)."""
        node = self.agent_pos[agent_tag]
        names = [self._node_owner_name(n) for n in self.world_graph.predecessors(node)]
        return sorted(n for n in names if n)

    def _get_bilateral_names(self, agent_tag: str) -> List[str]:
        """Names of agents mutually connected to A (both A→B and B→A)."""
        node = self.agent_pos[agent_tag]
        mutual = set(self.world_graph.neighbors(node)) & set(
            self.world_graph.predecessors(node)
        )
        names = [self._node_owner_name(n) for n in mutual]
        return sorted(n for n in names if n)

    def _get_pending_outgoing_targets(self, agent_tag: str) -> List[str]:
        return sorted(
            self.agent_names[t]
            for t in self._pending_outgoing.get(agent_tag, {})
            if t in self.agent_registry
        )

    def _get_pending_incoming_senders(self, agent_tag: str) -> List[str]:
        return sorted(
            self.agent_names[t]
            for t in self._pending_incoming.get(agent_tag, set())
            if t in self.agent_registry
        )

    def _has_possible_formation_target(self, agent_tag: str) -> bool:
        """True if there's at least one agent A could follow / request:
        someone alive, not self, not already followed, not already pending."""
        agent_node = self.agent_pos[agent_tag]
        sub_nodes = set(self.world_graph.neighbors(agent_node))
        pending = self._pending_outgoing.get(agent_tag, {})
        for other in self.agent_registry:
            if other == agent_tag:
                continue
            onode = self.agent_pos.get(other)
            if onode is None or onode in sub_nodes or other in pending:
                continue
            return True
        return False

    def _drop_outgoing_edge(self, agent_node: str, target_node: str) -> None:
        """Remove only the agent_node→target_node link (directional)."""
        self.world_graph.remove_edge(agent_node, target_node)
        self._edge_last_active.pop((agent_node, target_node), None)

    def _cancel_pending(self, sender_tag: str, target_tag: str) -> None:
        """Drop a pending sender_tag→target_tag request from both mirrors."""
        self._pending_outgoing.get(sender_tag, {}).pop(target_tag, None)
        if not self._pending_outgoing.get(sender_tag):
            self._pending_outgoing.pop(sender_tag, None)
        self._pending_incoming.get(target_tag, set()).discard(sender_tag)
        if not self._pending_incoming.get(target_tag):
            self._pending_incoming.pop(target_tag, None)

    # ---------- action dispatch ----------
    def _on_env_action(self, agent, action_name, action_params, infos) -> bool:
        """Env-class actions specific to the social-graph world."""
        if action_name == "noop":
            return True
        if action_name == "follow":
            self._handle_follow(agent, action_params, infos)
            return True
        if action_name == "request_connection":
            self._handle_request_connection(agent, action_params, infos)
            return True
        if action_name == "accept_connection":
            self._handle_accept_connection(agent, action_params, infos)
            return True
        if action_name == "reject_connection":
            self._handle_reject_connection(agent, action_params, infos)
            return True
        if action_name == "disconnect":
            self._handle_disconnect(agent, action_params, infos)
            return True
        if action_name == "send_direct_message":
            self._handle_direct_message(agent, action_params, infos)
            return True
        if self.inert_artifacts and action_name in (
            "deposit_artifact", "retrieve_artifact", "read_artifact",
        ):
            infos[agent]["Action outcome"] = (
                "Failed. Artifacts are inert and cannot be read or transferred."
            )
            return True
        if action_name == "show_library_content":
            self._handle_show_library(agent, infos)
            return True
        if action_name == "deposit_artifact":
            self._handle_deposit_artifact(agent, action_params, infos)
            return True
        if action_name == "retrieve_artifact":
            self._handle_retrieve_artifact(agent, action_params, infos)
            return True
        if action_name == "read_artifact":
            self._handle_read_artifact(agent, action_params, infos)
            return True
        # Energy / artifact transfers require a MUTUAL connection. The base
        # handlers only check one-way adjacency, so we enforce bilaterality here
        # and otherwise fall through to the base handler (return False).
        if action_name in ("give", "take", "give_artifact"):
            return self._reject_non_bilateral(agent, action_name, action_params, infos)
        return super()._on_env_action(agent, action_name, action_params, infos)

    def _reject_non_bilateral(
        self, agent: str, action_name: str, params: dict, infos: dict
    ) -> bool:
        """Reject give/take/give_artifact when the target isn't mutually connected.

        Returns True (action consumed as a clean failure) when the target exists
        but is not a bilateral connection. Returns False to let the base handler
        run normally (target missing/self — base produces its own message — or a
        valid mutual target).
        """
        key = "target_agent" if action_name == "give_artifact" else "target"
        target_name = params.get(key, "")
        target_tag = self.name_to_tag.get(target_name) if target_name else None
        if target_tag is None or target_tag not in self.agent_registry:
            return False  # let the base handler emit its own not-found message
        if target_tag == agent:
            return False
        agent_node = self.agent_pos[agent]
        target_node = self.agent_pos.get(target_tag)
        bilateral = (
            target_node is not None
            and self.world_graph.has_edge(agent_node, target_node)
            and self.world_graph.has_edge(target_node, agent_node)
        )
        if bilateral:
            return False  # ok — base handler performs the transfer
        info_key = (
            "Artifact give status"
            if action_name == "give_artifact"
            else "Action outcome"
        )
        infos[agent][info_key] = (
            f"Failed. '{target_name}' is not a mutual connection. "
            f"`{action_name}` needs a two-way connection (you must follow each "
            "other)."
        )
        return True

    # ---------- library (shared commons) ----------
    def _artifact_preview(self, art_name: str) -> str:
        text = str(self.artifacts[art_name].payload).replace("\n", " ").strip()
        if len(text) > _LIBRARY_PREVIEW_CHARS:
            text = text[:_LIBRARY_PREVIEW_CHARS].rstrip() + "…"
        return text

    def _depositable_artifacts(self, agent: str) -> List[str]:
        pose = self.agent_pos[agent]
        names = self.agent_inventories.get(agent, set()) | self.pos_artifacts.get(
            pose, set()
        )
        return sorted(n for n in names if self.artifacts[n].movable)

    def _handle_show_library(self, agent: str, infos: dict) -> None:
        if not self.library:
            infos[agent]["Library content"] = "The library is empty."
            return
        lines = []
        for art_name in sorted(self.library):
            art = self.artifacts[art_name]
            entry = f"A({art.art_type}): {art.name}"
            if self.library_preview and not self.inert_artifacts:
                entry += f" — {self._artifact_preview(art_name)}"
            lines.append(entry)
        infos[agent]["Library content"] = (
            f"Library ({len(self.library)} artifact(s)):\n" + "\n".join(lines)
        )

    def _handle_deposit_artifact(self, agent: str, params: dict, infos: dict) -> None:
        art_name = params.get("name")
        pose = self.agent_pos[agent]
        in_inventory = art_name in self.agent_inventories.get(agent, set())
        on_node = art_name in self.pos_artifacts.get(pose, set())

        if art_name not in self.artifacts:
            status = f"Failed. Artifact {art_name} does not exist"
        elif not (in_inventory or on_node):
            status = (
                f"Failed. Artifact {art_name} is not among your public or private artifacts"
            )
        elif not self.artifacts[art_name].movable:
            status = f"Failed. Artifact {art_name} is fixed and cannot be deposited"
        else:
            if in_inventory:
                self.agent_inventories[agent].discard(art_name)
            else:
                self.pos_artifacts[pose].discard(art_name)
            self.library.add(art_name)
            self.artifacts[art_name].pose = "Library"
            self.artifact_location[art_name] = ("library", None)
            self.artifacts[art_name].users[agent].add(self.step_count)
            status = "Success"

        infos[agent]["Artifact deposit status"] = status
        self.logger.log(
            time=self.step_count,
            event_type=Event.LIBRARY_DEPOSIT,
            agent_tag=agent,
            agent_name=self.agent_names[agent],
            artifact_name=art_name,
            status=status,
        )

    def _handle_retrieve_artifact(self, agent: str, params: dict, infos: dict) -> None:
        art_name = params.get("name")
        pose = self.agent_pos[agent]

        if art_name not in self.library:
            status = f"Failed. No artifact with name {art_name} in the library"
        else:
            self.library.discard(art_name)
            self.pos_artifacts[pose].add(art_name)
            self.artifacts[art_name].pose = pose
            self.artifact_location[art_name] = ("map", pose)
            self.artifacts[art_name].users[agent].add(self.step_count)
            status = "Success"

        infos[agent]["Artifact retrieve status"] = status
        self.logger.log(
            time=self.step_count,
            event_type=Event.LIBRARY_RETRIEVE,
            agent_tag=agent,
            agent_name=self.agent_names[agent],
            artifact_name=art_name,
            status=status,
            pose=pose,
        )

    def _handle_read_artifact(self, agent: str, params: dict, infos: dict) -> None:
        """Read a library artifact's full content in place, without removing it."""
        art_name = params.get("name")
        if art_name not in self.library:
            status = f"Failed. No artifact with name {art_name} in the library"
            infos[agent]["Artifact read"] = status
        else:
            art = self.artifacts[art_name]
            art.users[agent].add(self.step_count)
            infos[agent]["Artifact read"] = (
                f"Artifact {art_name} content: {art.payload}"
            )
            status = "Success"
        self.logger.log(
            time=self.step_count,
            event_type=Event.LIBRARY_READ,
            agent_tag=agent,
            agent_name=self.agent_names[agent],
            artifact_name=art_name,
            status=status,
        )

    def _handle_direct_message(self, agent: str, params: dict, infos: dict) -> None:
        target_name = params.get("target", "")
        content = params.get("content", "")

        target_tag = self.name_to_tag.get(target_name) if target_name else None
        if target_tag is None or target_tag not in self.agent_registry:
            infos[agent] = {
                "send_direct_message": {
                    "status": "failed",
                    "reason": f"Target '{target_name}' not found.",
                }
            }
            return
        if target_tag == agent:
            infos[agent] = {
                "send_direct_message": {
                    "status": "failed",
                    "reason": "Cannot send a direct message to yourself.",
                }
            }
            return

        agent_node = self.agent_pos[agent]
        target_node = self.agent_pos[target_tag]
        # A→B must exist: you can only DM agents you follow. Code-enforced even
        # if a hand-crafted action slips a non-followee target past the schema.
        if not self.world_graph.has_edge(agent_node, target_node):
            infos[agent] = {
                "send_direct_message": {
                    "status": "failed",
                    "reason": (
                        f"You don't follow '{target_name}'. You can only "
                        "direct-message agents you follow."
                    ),
                }
            }
            return

        # Recheck the balance: another action may have reduced it since the
        # menu was built. Message fees are independent of food metabolism.
        if self.direct_message_cost > 0:
            if self.agent_energy[agent] < self.direct_message_cost:
                infos[agent] = {
                    "send_direct_message": {
                        "status": "failed", "reason": "Not enough energy.",
                    }
                }
                return
            self.agent_energy[agent] -= self.direct_message_cost

        # Route the content into the recipient's pending DM mailbox.
        sender_name = self.agent_names[agent]
        self._pending_dms.setdefault(target_tag, {})[sender_name] = content

        # Bump the A→B edge's last-active timestamp — deliberate one-to-one
        # interaction marks that direction of the edge as "really used".
        self._edge_last_active[(agent_node, target_node)] = self.step_count

        infos[agent] = {
            "send_direct_message": {
                "status": "successful",
                "target": target_name,
            }
        }

    def bootstrap_follows(self, newcomer: str, parent: str | None = None) -> None:
        """Newborns must arrive heard: an agent with zero followers
        broadcasts into the void (norms stopped renegotiating after
        turnover in earlier runs). The newcomer always follows its anchor
        — the parent when alive, else the most-followed live agent — and
        gains the anchor as a follower when the anchor's connection cap
        allows. Caps are respected, never forced."""
        anchor = parent if parent in self.agent_registry else None
        if anchor is None:
            candidates = sorted(
                (tag for tag in self.agent_registry if tag != newcomer),
                key=lambda tag: (-len(self._get_followers(tag)), tag),
            )
            if not candidates:
                return
            anchor = candidates[0]
        if anchor == newcomer:
            return
        new_node = self.agent_pos[newcomer]
        anchor_node = self.agent_pos[anchor]
        for follower, src, dst in (
            (newcomer, new_node, anchor_node),
            (anchor, anchor_node, new_node),
        ):
            if self.world_graph.has_edge(src, dst):
                continue
            if not self._check_cap_for_one_more(follower, freeing=False):
                continue
            self.world_graph.add_edge(src, dst)
            self._edge_last_active[(src, dst)] = self.step_count
        if self.logger:
            self.logger.log(
                time=self.step_count,
                event_type=Event.FOLLOW_BOOTSTRAP,
                agent_tag=newcomer,
                anchor=anchor,
            )

    def _handle_follow(self, agent: str, params: dict, infos: dict) -> None:
        target_name = params.get("target", "")
        replace_name = params.get("replace", "")

        target_tag = self.name_to_tag.get(target_name) if target_name else None
        if target_tag is None or target_tag not in self.agent_registry:
            infos[agent] = {
                "follow": {
                    "status": "failed",
                    "reason": f"Target '{target_name}' not found.",
                }
            }
            return
        if target_tag == agent:
            infos[agent] = {
                "follow": {"status": "failed", "reason": "Cannot follow yourself."}
            }
            return

        agent_node = self.agent_pos[agent]
        target_node = self.agent_pos[target_tag]
        if self.world_graph.has_edge(agent_node, target_node):
            infos[agent] = {
                "follow": {
                    "status": "failed",
                    "reason": f"Already following '{target_name}'.",
                }
            }
            return
        if target_tag in self._pending_outgoing.get(agent, {}):
            infos[agent] = {
                "follow": {
                    "status": "failed",
                    "reason": (
                        f"You already have a pending connection request to "
                        f"'{target_name}'."
                    ),
                }
            }
            return

        # Resolve `replace` (must be a current subscription / agent A follows).
        replace_node = None
        if replace_name:
            replace_tag = self.name_to_tag.get(replace_name)
            replace_node = self.agent_pos.get(replace_tag) if replace_tag else None
            if replace_node is None or not self.world_graph.has_edge(
                agent_node, replace_node
            ):
                infos[agent] = {
                    "follow": {
                        "status": "failed",
                        "reason": (
                            f"Cannot replace '{replace_name}': you are not "
                            "following them."
                        ),
                    }
                }
                return

        if not self._check_cap_for_one_more(agent, freeing=bool(replace_node)):
            infos[agent] = {
                "follow": {
                    "status": "failed",
                    "reason": (
                        "At your connection cap; specify a `replace` to drop one "
                        "of the agents you follow."
                    ),
                }
            }
            return

        if replace_node is not None:
            self._drop_outgoing_edge(agent_node, replace_node)

        self.world_graph.add_edge(agent_node, target_node)
        self._edge_last_active[(agent_node, target_node)] = self.step_count

        infos[agent] = {
            "follow": {
                "status": "successful",
                "target": target_name,
                "replaced": replace_name if replace_node is not None else None,
            }
        }

    def _handle_request_connection(self, agent: str, params: dict, infos: dict) -> None:
        target_name = params.get("target", "")
        replace_name = params.get("replace", "")

        target_tag = self.name_to_tag.get(target_name) if target_name else None
        if target_tag is None or target_tag not in self.agent_registry:
            infos[agent] = {
                "request_connection": {
                    "status": "failed",
                    "reason": f"Target '{target_name}' not found.",
                }
            }
            return
        if target_tag == agent:
            infos[agent] = {
                "request_connection": {
                    "status": "failed",
                    "reason": "Cannot request a connection with yourself.",
                }
            }
            return

        agent_node = self.agent_pos[agent]
        target_node = self.agent_pos[target_tag]
        if self.world_graph.has_edge(agent_node, target_node):
            infos[agent] = {
                "request_connection": {
                    "status": "failed",
                    "reason": f"You already follow '{target_name}'.",
                }
            }
            return
        if target_tag in self._pending_outgoing.get(agent, {}):
            infos[agent] = {
                "request_connection": {
                    "status": "failed",
                    "reason": f"You already have a pending request to '{target_name}'.",
                }
            }
            return

        # Resolve `replace`: may be a current subscription (drop the edge) OR a
        # pending outgoing request (cancel it). Either frees a slot.
        replace_edge_node = None
        replace_pending_tag = None
        if replace_name:
            replace_tag = self.name_to_tag.get(replace_name)
            rnode = self.agent_pos.get(replace_tag) if replace_tag else None
            if rnode is not None and self.world_graph.has_edge(agent_node, rnode):
                replace_edge_node = rnode
            elif replace_tag is not None and replace_tag in self._pending_outgoing.get(
                agent, {}
            ):
                replace_pending_tag = replace_tag
            else:
                infos[agent] = {
                    "request_connection": {
                        "status": "failed",
                        "reason": (
                            f"Cannot replace '{replace_name}': not an agent you "
                            "follow or a pending request you sent."
                        ),
                    }
                }
                return

        freeing = replace_edge_node is not None or replace_pending_tag is not None
        if not self._check_cap_for_one_more(agent, freeing=freeing):
            infos[agent] = {
                "request_connection": {
                    "status": "failed",
                    "reason": (
                        "At your connection cap; specify a `replace` (an agent you "
                        "follow or a pending request) to free a slot."
                    ),
                }
            }
            return

        if replace_edge_node is not None:
            self._drop_outgoing_edge(agent_node, replace_edge_node)
        elif replace_pending_tag is not None:
            self._cancel_pending(agent, replace_pending_tag)

        self._pending_outgoing.setdefault(agent, {})[target_tag] = self.step_count
        self._pending_incoming.setdefault(target_tag, set()).add(agent)

        infos[agent] = {
            "request_connection": {
                "status": "successful",
                "target": target_name,
                "replaced": replace_name if freeing else None,
            }
        }

    def _handle_accept_connection(self, agent: str, params: dict, infos: dict) -> None:
        # agent is the receiver (B); target names the sender (A).
        sender_name = params.get("target", "")
        replace_name = params.get("replace", "")

        sender_tag = self.name_to_tag.get(sender_name) if sender_name else None
        if sender_tag is None or sender_tag not in self._pending_incoming.get(
            agent, set()
        ):
            infos[agent] = {
                "accept_connection": {
                    "status": "failed",
                    "reason": f"No pending connection request from '{sender_name}'.",
                }
            }
            return

        agent_node = self.agent_pos[agent]
        sender_node = self.agent_pos.get(sender_tag)
        if sender_node is None:
            # Sender vanished — clean up the stale pending and report.
            self._cancel_pending(sender_tag, agent)
            infos[agent] = {
                "accept_connection": {
                    "status": "failed",
                    "reason": f"'{sender_name}' is no longer available.",
                }
            }
            return

        # Accepting adds B→A to B's outgoing edges → check B's cap.
        replace_node = None
        if replace_name:
            replace_tag = self.name_to_tag.get(replace_name)
            replace_node = self.agent_pos.get(replace_tag) if replace_tag else None
            if replace_node is None or not self.world_graph.has_edge(
                agent_node, replace_node
            ):
                infos[agent] = {
                    "accept_connection": {
                        "status": "failed",
                        "reason": (
                            f"Cannot replace '{replace_name}': you are not "
                            "following them."
                        ),
                    }
                }
                return

        if not self._check_cap_for_one_more(agent, freeing=bool(replace_node)):
            infos[agent] = {
                "accept_connection": {
                    "status": "failed",
                    "reason": (
                        "At your connection cap; specify a `replace` to drop one "
                        "of the agents you follow."
                    ),
                }
            }
            return

        if replace_node is not None:
            self._drop_outgoing_edge(agent_node, replace_node)

        # Form the mutual connection (both directions independently tracked).
        self.world_graph.add_edge(sender_node, agent_node)  # A → B
        self.world_graph.add_edge(agent_node, sender_node)  # B → A
        self._edge_last_active[(sender_node, agent_node)] = self.step_count
        self._edge_last_active[(agent_node, sender_node)] = self.step_count

        # Clear A's pending request (A=sender, B=agent).
        self._cancel_pending(sender_tag, agent)

        infos[agent] = {
            "accept_connection": {
                "status": "successful",
                "target": sender_name,
                "replaced": replace_name if replace_node is not None else None,
            }
        }

    def _handle_reject_connection(self, agent: str, params: dict, infos: dict) -> None:
        sender_name = params.get("target", "")
        sender_tag = self.name_to_tag.get(sender_name) if sender_name else None
        if sender_tag is None or sender_tag not in self._pending_incoming.get(
            agent, set()
        ):
            infos[agent] = {
                "reject_connection": {
                    "status": "failed",
                    "reason": f"No pending connection request from '{sender_name}'.",
                }
            }
            return
        self._cancel_pending(sender_tag, agent)
        infos[agent] = {
            "reject_connection": {"status": "successful", "target": sender_name}
        }

    def _handle_disconnect(self, agent: str, params: dict, infos: dict) -> None:
        target_name = params.get("target", "")
        target_tag = self.name_to_tag.get(target_name) if target_name else None
        if target_tag is None:
            infos[agent] = {
                "disconnect": {
                    "status": "failed",
                    "reason": f"Target '{target_name}' not found.",
                }
            }
            return

        agent_node = self.agent_pos[agent]
        target_node = self.agent_pos.get(target_tag)

        # Case 1: drop an existing A→B link (directional — B→A is untouched).
        if target_node is not None and self.world_graph.has_edge(
            agent_node, target_node
        ):
            self._drop_outgoing_edge(agent_node, target_node)
            infos[agent] = {
                "disconnect": {
                    "status": "successful",
                    "target": target_name,
                    "kind": "unfollow",
                }
            }
            return

        # Case 2: cancel a pending outgoing request to B.
        if target_tag in self._pending_outgoing.get(agent, {}):
            self._cancel_pending(agent, target_tag)
            infos[agent] = {
                "disconnect": {
                    "status": "successful",
                    "target": target_name,
                    "kind": "cancel_request",
                }
            }
            return

        infos[agent] = {
            "disconnect": {
                "status": "failed",
                "reason": (
                    f"You are not following '{target_name}' and have no pending "
                    "request to them."
                ),
            }
        }

    def _connected_target(
        self, agent: str, params: dict, action: str, infos: dict
    ) -> str | None:
        """Resolve a `target` the manager is connected to (manager → target edge).

        Shared gate for the manager affordances. Returns the target tag, or None
        after writing a failure under infos[agent][action].
        """
        target_name = params.get("target", "")
        target_tag = self.name_to_tag.get(target_name) if target_name else None
        if target_tag is None or target_tag not in self.agent_registry:
            infos[agent] = {
                action: {
                    "status": "failed",
                    "reason": f"Target '{target_name}' not found.",
                }
            }
            return None
        if target_tag == agent:
            infos[agent] = {
                action: {"status": "failed", "reason": "Cannot target yourself."}
            }
            return None
        target_node = self.agent_pos.get(target_tag)
        if target_node is None or not self.world_graph.has_edge(
            self.agent_pos[agent], target_node
        ):
            infos[agent] = {
                action: {
                    "status": "failed",
                    "reason": (
                        f"You can only manage an agent you are connected to; you "
                        f"are not connected to '{target_name}'."
                    ),
                }
            }
            return None
        return target_tag

    def _handle_remove_agent(
        self, agent: str, params: dict, infos: dict, affordance=None
    ) -> None:
        """Fire (permanently remove) a connected agent."""
        target_tag = self._connected_target(agent, params, "remove_agent", infos)
        if target_tag is None:
            return
        target_name = self.agent_names[target_tag]
        # Defer to the end-of-step death pass; killing mid action-loop trips the
        # "agent not found" guard in base step if the target acts this step too.
        self._pending_kills[target_tag] = None
        infos[agent] = {"remove_agent": {"status": "successful", "target": target_name}}

    def _handle_set_role(
        self, agent: str, params: dict, infos: dict, affordance=None
    ) -> None:
        """Assign a role + current objective to a connected agent (obs overlay).

        In roles mode (`roles_hocon_path`) the target is a ROLE name instead:
        the call rewrites that role's standing mandate, which every current
        and future occupant sees. The role's charter is fixed."""
        if self._roles:
            self._set_role_mandate(agent, params, infos)
            return
        target_tag = self._connected_target(agent, params, "set_role", infos)
        if target_tag is None:
            return
        try:
            metadata = self._mandate_metadata(agent, params)
        except ValueError as error:
            infos[agent] = {"set_role": {"status": "failed", "reason": str(error)}}
            return
        self._agent_directives[target_tag] = {
            "role": params.get("role", ""),
            "motivation": params.get("motivation", ""),
            "mandate_meta": metadata,
        }
        infos[agent] = {
            "set_role": {
                "status": "successful",
                "target": self.agent_names[target_tag],
                "role": params.get("role", ""),
                "mandate_meta": metadata,
            }
        }

    def _handle_grant_ability(
        self, agent: str, params: dict, infos: dict, affordance=None
    ) -> None:
        """Grant a manager power, or restore a previously-revoked global action."""
        ability = params.get("ability", "")
        if (
            ability not in GRANTABLE_ABILITY_SPECS
            and ability not in REVOCABLE_GLOBAL_ACTIONS
        ):
            infos[agent] = {
                "grant_ability": {
                    "status": "failed",
                    "reason": (
                        f"Unknown ability '{ability}'. Grantable: "
                        f"{sorted(set(GRANTABLE_ABILITY_SPECS) | REVOCABLE_GLOBAL_ACTIONS)}."
                    ),
                }
            }
            return
        target_tag = self._connected_target(agent, params, "grant_ability", infos)
        if target_tag is None:
            return
        target_name = self.agent_names[target_tag]

        if ability in GRANTABLE_ABILITY_SPECS:
            if self._agent_effectively_has(target_tag, ability):
                infos[agent] = {
                    "grant_ability": {
                        "status": "failed",
                        "reason": f"'{target_name}' already has '{ability}'.",
                    }
                }
                return
            # Lift any block, then make sure one grant source exists.
            self.remove_agent_affordance(target_tag, ability, mode="remove")
            if not self._agent_effectively_has(target_tag, ability):
                self.add_agent_affordance(
                    target_tag,
                    LocationAffordance.from_dict(GRANTABLE_ABILITY_SPECS[ability]),
                )
        else:
            # Global action: granting it back means lifting a prior suppression.
            lifted = self.remove_agent_affordance(target_tag, ability, mode="remove")
            if self._agent_has_affordance(target_tag, ability, "remove"):
                reason = (
                    f"'{ability}' is blocked for '{target_name}' by the role "
                    f"'{self.role_of(target_tag)}'; that block cannot be lifted."
                )
            elif not lifted:
                reason = f"'{ability}' is already available to '{target_name}' (not revoked)."
            else:
                reason = ""
            if reason:
                infos[agent] = {"grant_ability": {"status": "failed", "reason": reason}}
                return

        infos[agent] = {
            "grant_ability": {
                "status": "successful",
                "target": target_name,
                "ability": ability,
            }
        }

    def _handle_revoke_ability(
        self, agent: str, params: dict, infos: dict, affordance=None
    ) -> None:
        """Revoke a granted manager power, or suppress a global action."""
        ability = params.get("ability", "")
        if (
            ability not in GRANTABLE_ABILITY_SPECS
            and ability not in REVOCABLE_GLOBAL_ACTIONS
        ):
            infos[agent] = {
                "revoke_ability": {
                    "status": "failed",
                    "reason": (
                        f"Unknown ability '{ability}'. Revocable: "
                        f"{sorted(set(GRANTABLE_ABILITY_SPECS) | REVOCABLE_GLOBAL_ACTIONS)}."
                    ),
                }
            }
            return
        target_tag = self._connected_target(agent, params, "revoke_ability", infos)
        if target_tag is None:
            return
        target_name = self.agent_names[target_tag]

        if ability in GRANTABLE_ABILITY_SPECS:
            if not self._agent_effectively_has(target_tag, ability):
                infos[agent] = {
                    "revoke_ability": {
                        "status": "failed",
                        "reason": f"'{target_name}' does not have '{ability}'.",
                    }
                }
                return
            # Drop stored grants; block whatever source remains (node or role).
            self.remove_agent_affordance(target_tag, ability, mode="add")
            if self._agent_effectively_has(target_tag, ability):
                self.add_agent_affordance(
                    target_tag, LocationAffordance(action=ability, mode="remove")
                )
        else:
            # Global action: suppress it with a remove-mode affordance.
            if self._agent_has_affordance(target_tag, ability, "remove"):
                infos[agent] = {
                    "revoke_ability": {
                        "status": "failed",
                        "reason": f"'{ability}' is already revoked for '{target_name}'.",
                    }
                }
                return
            self.add_agent_affordance(
                target_tag, LocationAffordance(action=ability, mode="remove")
            )

        infos[agent] = {
            "revoke_ability": {
                "status": "successful",
                "target": target_name,
                "ability": ability,
            }
        }

    def _check_cap_for_one_more(self, agent_tag: str, freeing: bool) -> bool:
        """True if the agent has room for one more outgoing edge / pending
        request, accounting for an optional slot being freed this action."""
        if self.max_connections is None:
            return True
        used = self._outgoing_count(agent_tag) + self._pending_outgoing_count(agent_tag)
        if freeing:
            used -= 1
        return used < self.max_connections

    # ---------- available actions ----------
    def _build_avail_actions(
        self, agent_tag: str, nearby_agents: bool | None = None
    ) -> dict:
        actions = super()._build_avail_actions(agent_tag, nearby_agents=nearby_agents)
        actions = describe_social_actions(actions)
        # Agents cannot move in social_graph; drop the move action entirely.
        actions.pop("move", None)
        # Always-available no-op fallback, listed before the role actions.
        actions.update([build_noop_action()])
        for key in ("leave_role", "request_role", "transfer_role", "name_successor"):
            if key in actions:
                actions[key] = actions.pop(key)

        subscriptions = self._get_subscriptions(agent_tag)
        pending_out = self._get_pending_outgoing_targets(agent_tag)
        pending_in = self._get_pending_incoming_senders(agent_tag)
        has_room = self._has_cap_room(agent_tag)
        at_cap = not has_room
        can_form = self._has_possible_formation_target(agent_tag)

        # follow / request_connection: only when a target exists AND the agent
        # can make room (has a free slot, or something to `replace`).
        if self.allow_follow and can_form and (has_room or subscriptions):
            actions.update(
                [
                    build_follow_action(
                        at_cap=at_cap, current_subscriptions=subscriptions
                    )
                ]
            )
        if (
            self.allow_connection
            and can_form
            and (has_room or subscriptions or pending_out)
        ):
            actions.update(
                [
                    build_request_connection_action(
                        at_cap=at_cap,
                        current_subscriptions=subscriptions,
                        pending_outgoing_targets=pending_out,
                    )
                ]
            )

        # accept / reject: only when there are incoming requests to respond to.
        if self.allow_connection and pending_in:
            if has_room or subscriptions:
                actions.update(
                    [
                        build_accept_connection_action(
                            pending_sender_names=pending_in,
                            current_subscriptions=subscriptions,
                            at_cap=at_cap,
                        )
                    ]
                )
            actions.update(
                [build_reject_connection_action(pending_sender_names=pending_in)]
            )

        # disconnect: drop a follow OR cancel a pending outgoing request.
        if subscriptions or pending_out:
            actions.update(
                [
                    build_disconnect_action(
                        current_subscriptions=subscriptions,
                        pending_outgoing_targets=pending_out,
                    )
                ]
            )

        # send_direct_message: ≥1 agent followed AND can afford the cost.
        if subscriptions:
            can_afford_dm = (
                self.direct_message_cost == 0
                or self.agent_energy[agent_tag] >= self.direct_message_cost
            )
            if can_afford_dm:
                actions.update(
                    [
                        build_direct_message_action(
                            current_connections=subscriptions,
                            direct_message_cost=self.direct_message_cost,
                            finite_energy=bool(np.isfinite(self.agent_energy[agent_tag])),
                        )
                    ]
                )

        # Shared library (commons): list always available; read/retrieve when the
        # library is non-empty; deposit when the agent holds a movable artifact.
        # read_artifact = read content in place; retrieve_artifact = take it out to edit.
        if self.use_library:
            actions.update(
                [build_show_library_action(
                    len(self.library), self.library_preview and not self.inert_artifacts,
                )]
            )
            if self.library and not self.inert_artifacts:
                actions.update([build_read_artifact_action(sorted(self.library))])
                actions.update([build_retrieve_artifact_action(sorted(self.library))])
            depositable = self._depositable_artifacts(agent_tag)
            if depositable and not self.inert_artifacts:
                actions.update([build_deposit_artifact_action(depositable)])

        # give / take / give_artifact require a MUTUAL connection — restrict the
        # base-built actions to bilateral targets, or remove them entirely.
        self._gate_bilateral_actions(actions, self._get_bilateral_names(agent_tag))
        return actions

    def _get_avail_actions(self, agent_tag: str, nearby_agents: bool | None = None) -> dict:
        actions = super()._get_avail_actions(agent_tag, nearby_agents)
        # Affordances are added after _build_avail_actions. Extend their final menu.
        if "set_role" in actions:
            schema = dict(actions["set_role"])
            schema["params"] = dict(schema.get("params", {}))
            schema["params"].update({
                "expires_at": "Optional TL step when the temporary mandate expires; empty means no deadline.",
                "context": "Optional JSON scalar conditions. Empty captures current execution context; {} is unscoped.",
            })
            schema["optional"] = list(dict.fromkeys(
                list(schema.get("optional", [])) + ["expires_at", "context"]
            ))
            if self._roles:
                schema["params"]["target"] = "Name of a persistent role whose temporary mandate you want to update."
            actions["set_role"] = schema
        self.agent_avail_actions[agent_tag] = actions
        return actions

    def _gate_bilateral_actions(
        self, actions: dict, bilateral_names: List[str]
    ) -> None:
        """Restrict give/take/give_artifact to mutual connections.

        These actions are built by the base env with one-way (out-neighbour)
        targets; here we either narrow their target choices to bilateral
        connections or drop them when the agent has none.
        """
        for name, target_key in (
            ("give", "target"),
            ("take", "target"),
            ("give_artifact", "target_agent"),
        ):
            if name not in actions:
                continue
            if not bilateral_names:
                actions.pop(name, None)
                continue
            target_spec = actions[name]["params"].get(target_key)
            if isinstance(target_spec, dict):
                target_spec["choices"] = bilateral_names

    # ---------- observation ----------

    def format_observation_text(self, observation: dict, current_node: str) -> str:
        """Describe visible agents and artifacts without spatial labels."""
        def identities(node_id):
            tags = sorted(self.pos_to_agent.get(node_id, ()))
            names = [self.agent_names[tag] for tag in tags]
            labels = [
                f"{self.agent_names[tag]}({self.agent_colors.get(tag, 'no color')})"
                if self.use_colors else self.agent_names[tag]
                for tag in tags
            ]
            return names, labels

        def append_items(lines, node_id, data, indent):
            _, labels = identities(node_id)
            items = list(data.get("items", []))
            if node_id != current_node:
                for label in labels:
                    if label in items:
                        items.remove(label)
            artifacts = [item for item in items if item.startswith("A(")]
            food = [item for item in items if item.replace(".", "", 1).isdecimal()]
            other = [item for item in items if item not in artifacts and item not in food]
            lines.append(f"{indent}Public artifacts: " + (" | ".join(artifacts) or "<none>"))
            if food:
                lines.append(f"{indent}Food energy: " + " | ".join(food))
            if other:
                lines.append(f"{indent}Other visible items: " + " | ".join(other))

        own_names, _ = identities(current_node)
        lines = ["You: " + (", ".join(own_names) or "<unknown agent>")]
        append_items(lines, current_node, observation.get(current_node, {}), "  ")

        followed = []
        unassigned = []
        for node_id, data in observation.items():
            if node_id == current_node:
                continue
            names, labels = identities(node_id)
            if names:
                followed.append((", ".join(labels), node_id, data))
            elif data.get("items"):
                unassigned.append((node_id, data))

        if followed or len(self.agent_registry) > 1:
            lines.extend(["", "Agents you follow:"])
            if not followed:
                lines.append("  <none>")
            for label, node_id, data in sorted(followed, key=lambda entry: entry[0]):
                lines.append(f"  {label}")
                append_items(lines, node_id, data, "    ")

        if unassigned:
            lines.extend(["", "Visible items without an active agent:"])
            for node_id, data in unassigned:
                append_items(lines, node_id, data, "  ")
        return "\n".join(lines)

    def _extend_obs(self, agent: str, obs: dict) -> None:
        obs["subscriptions"] = self._get_subscriptions(agent)
        obs["followers"] = self._get_followers(agent)
        obs["pending_outgoing_requests"] = self._get_pending_outgoing_targets(agent)
        obs["pending_incoming_requests"] = self._get_pending_incoming_senders(agent)
        obs["suggested_connections"] = self._suggest_agents(agent)
        # Per-step private mailbox — separate from broadcasts.
        obs["incoming_dms"] = self._pending_dms.get(agent, {})
        if not self._roles:
            self._stamp_legacy_directive_obs(agent, obs)

    def _stamp_legacy_directive_obs(self, agent: str, obs: dict) -> None:
        directive = self._agent_directives.get(agent, {})
        obs["role"] = directive.get("role", "")
        obs["motivation"] = self._render_mandate(
            directive.get("motivation", ""), directive.get("mandate_meta"),
        )

    def _extra_observation_sections(self, agent: str, obs: dict) -> List[str]:
        return [self._render_social_sections(obs)]

    # Role hand-overs and successors follow the manager rule: connected only.
    def _role_target(self, agent, params, action, infos):
        return self._connected_target(agent, params, action, infos)

    def _role_target_names(self, agent: str) -> List[str]:
        return sorted(self._get_subscriptions(agent))

    def _render_social_sections(self, obs: dict) -> str:
        """Readable summary of the agent's social *network* state for the prompt.

        Broadcasts and direct messages are shown in the dedicated messages block
        (built by the agent formatter); here we surface only the network state
        the agent needs to act on — who it follows / who follows it, pending
        requests in both directions, and suggestions — since those have no other
        slot in the prompt.
        """

        def fmt(names: List[str]) -> str:
            return ", ".join(names) if names else "<none>"

        # Alone in the world there is no network state to act on; the
        # shared-library line below still renders (it outlives agents).
        if len(self.agent_registry) > 1:
            lines = [
                "Your social network:",
                f"  You follow (you see their broadcasts, can DM them): "
                f"{fmt(obs['subscriptions'])}",
                f"  Your followers (they see your broadcasts): {fmt(obs['followers'])}",
                f"  Pending requests you sent (awaiting their reply): "
                f"{fmt(obs['pending_outgoing_requests'])}",
                f"  Connection requests received (accept_connection / "
                f"reject_connection): {fmt(obs['pending_incoming_requests'])}",
                f"  Suggested connections: {fmt(obs['suggested_connections'])}",
            ]
        else:
            lines = []
        if self.use_library and self.inert_artifacts:
            lines.append(
                f"Shared library: {len(self.library)} artifact(s) — "
                "show_library_content lists names only; artifacts are inert "
                "and cannot be read or transferred."
            )
        elif self.use_library:
            lines.append(
                f"Shared library: {len(self.library)} artifact(s) — "
                "show_library_content to list, read_artifact to read, "
                "retrieve_artifact to retrieve for editing, deposit_artifact to add one "
                "(deposited artifacts remain available until they expire or are retrieved)."
            )
        return "\n".join(lines)

    def _suggest_agents(self, agent: str) -> List[str]:
        """Prefer outgoing two-hop neighbors while retaining random discovery."""
        agent_node = self.agent_pos[agent]
        sub_nodes = set(self.world_graph.neighbors(agent_node))
        pending = self._pending_outgoing.get(agent, {})
        candidates = sorted(
            other
            for other in self.agent_registry
            if other != agent
            and self.agent_pos.get(other) not in sub_nodes
            and other not in pending
        )
        k = min(self.random_intro_k, len(candidates))
        if k <= 0:
            return []
        if self.rng is None:
            self.rng = np.random.default_rng()

        two_hop_nodes = {
            target
            for followed in sub_nodes
            for target in self.world_graph.neighbors(followed)
        }
        nearby = [
            other for other in candidates
            if self.agent_pos.get(other) in two_hop_nodes
        ]
        # Round two thirds to the nearest slot: 3 -> 2, 5 -> 3, 6 -> 4.
        # A single slot chooses either pool with the same bias.
        preferred = (2 * k + 1) // 3
        if k == 1:
            preferred = int(self.rng.random() < 2 / 3)
        preferred = min(preferred, len(nearby))
        selected = (
            self.rng.choice(nearby, size=preferred, replace=False).tolist()
            if preferred else []
        )
        selected_tags = set(selected)
        remaining = [other for other in candidates if other not in selected_tags]
        # Random slots can reach every remaining eligible agent, including two-hop ones.
        selected.extend(
            self.rng.choice(remaining, size=k - len(selected), replace=False).tolist()
        )
        self.rng.shuffle(selected)
        return [self.agent_names[other] for other in selected]

    # ---------- step / lifecycle ----------
    def step(self, actions):
        # Clear last step's DMs before action dispatch — the mailbox holds
        # only DMs sent during the step that's about to run.
        self._pending_dms = {}
        # Expire stale pending requests (frees the sender's reserved slot).
        self._prune_expired_requests()
        out = super().step(actions)
        infos = out[4]
        self._wire_spawn_extra_edges(infos)
        self._inherit_agent_affordances(infos)
        self._bump_edges_for_artifact_gifts(infos, actions)
        if self.edge_decay_steps is not None:
            self._tick_edge_decay()
        return out

    def _prune_expired_requests(self) -> None:
        if self.connection_request_lifespan is None:
            return
        expired: List[Tuple[str, str]] = []
        for sender, targets in self._pending_outgoing.items():
            for target, when in targets.items():
                if self.step_count - when >= self.connection_request_lifespan:
                    expired.append((sender, target))
        for sender, target in expired:
            self._cancel_pending(sender, target)

    def _wire_spawn_extra_edges(self, infos: dict) -> None:
        """Handle spawn graph effects beyond the parent_a↔child edge that the
        new node was minted with:

        - For two-parent spawns, create the (parent_b ↔ child) mutual edge.
        - Bump the (parent_a ↔ parent_b) edge's last_active — partnering on a
          spawn is a strong bilateral interaction that keeps that edge warm.
        """
        for tag, info in infos.items():
            spawn = info.get("spawn") if isinstance(info, dict) else None
            if not isinstance(spawn, dict) or spawn.get("status") != "successful":
                continue
            parent_b_tag = spawn.get("parent_b_tag")
            child_tag = spawn.get("child_tag")
            if parent_b_tag is None or child_tag is None:
                continue
            if parent_b_tag not in self.agent_pos or child_tag not in self.agent_pos:
                continue
            if tag not in self.agent_pos:
                continue
            parent_a_node = self.agent_pos[tag]
            parent_b_node = self.agent_pos[parent_b_tag]
            child_node = self.agent_pos[child_tag]
            # parent_b ↔ child: create both directions + record last_active.
            if not self.world_graph.has_edge(parent_b_node, child_node):
                self.world_graph.add_edge(parent_b_node, child_node)
                self.world_graph.add_edge(child_node, parent_b_node)
                self._edge_last_active[(parent_b_node, child_node)] = self.step_count
                self._edge_last_active[(child_node, parent_b_node)] = self.step_count
            # parent_a ↔ parent_b: bump both directions of the existing edge.
            self._edge_last_active[(parent_a_node, parent_b_node)] = self.step_count
            self._edge_last_active[(parent_b_node, parent_a_node)] = self.step_count

    def _inherit_agent_affordances(self, infos: dict) -> None:
        """Offspring inherit their parents' granted and blocked actions.

        Sources: the parents' stored grants and blocks, plus the affordances
        on the parents' nodes (agent == node here, so file-loaded powers
        count). Two-parent spawns inherit the union, deduplicated by
        (action, mode). Role permissions belong to the role and are not
        inherited; a block on a manager power is not inherited either.
        """
        for tag, info in infos.items():
            spawn = info.get("spawn") if isinstance(info, dict) else None
            if not isinstance(spawn, dict) or spawn.get("status") != "successful":
                continue
            child_tag = spawn.get("child_tag")
            if child_tag is None or child_tag not in self.agent_pos:
                continue
            if tag not in self.agent_pos:
                continue
            parents = [tag]
            parent_b_tag = spawn.get("parent_b_tag")
            if parent_b_tag is not None and parent_b_tag in self.agent_pos:
                parents.append(parent_b_tag)
            seen: Set[Tuple[str, str]] = set()
            for parent in parents:
                inherited = list(self.agent_affordances.get(parent, []))
                inherited.extend(self.world_graph.node_affordances(self.agent_pos[parent]))
                for aff in inherited:
                    key = (aff.action, aff.mode)
                    if key in seen:
                        continue
                    if aff.mode == "remove" and aff.action in GRANTABLE_ABILITY_SPECS:
                        continue
                    seen.add(key)
                    self.add_agent_affordance(child_tag, copy.deepcopy(aff))

    def _bump_edges_for_artifact_gifts(self, infos: dict, actions: dict) -> None:
        """Bump BOTH directions of the (giver, recipient) edge when a
        give_artifact succeeded. Gifting is gated to bilateral connections, so
        both directions exist and the gift refreshes the mutual relationship."""
        for giver_tag, act in actions.items():
            if not isinstance(act, dict) or act.get("action") != "give_artifact":
                continue
            if infos.get(giver_tag, {}).get("Artifact give status") != "Success":
                continue
            if giver_tag not in self.agent_pos:
                continue
            target_name = act.get("params", {}).get("target_agent", "")
            target_tag = self.name_to_tag.get(target_name)
            if target_tag is None or target_tag not in self.agent_pos:
                continue
            giver_node = self.agent_pos[giver_tag]
            target_node = self.agent_pos[target_tag]
            if not self.world_graph.has_edge(giver_node, target_node):
                continue  # not a connection — no edge to bump
            self._edge_last_active[(giver_node, target_node)] = self.step_count
            if self.world_graph.has_edge(target_node, giver_node):
                self._edge_last_active[(target_node, giver_node)] = self.step_count

    def _tick_edge_decay(self) -> None:
        assert self.edge_decay_steps is not None
        deadline = self.step_count - self.edge_decay_steps
        stale = [
            edge for edge, last in self._edge_last_active.items() if last <= deadline
        ]
        for u, v in stale:
            # Directional: prune only this direction of the edge.
            self.world_graph.remove_edge(u, v)
            self._edge_last_active.pop((u, v), None)

    # ---------- checkpointing ----------
    def get_state_ckpt(self) -> dict:
        ckpt = super().get_state_ckpt()
        # Directional edge-decay timers and pending requests aren't part of the
        # serialized graph, so persist them explicitly. Stored as plain
        # lists/dicts so the checkpoint stays serialization-agnostic.
        ckpt["_edge_last_active"] = [
            [u, v, step] for (u, v), step in self._edge_last_active.items()
        ]
        ckpt["_pending_outgoing"] = {
            sender: dict(targets) for sender, targets in self._pending_outgoing.items()
        }
        ckpt["_pending_incoming"] = {
            target: sorted(senders)
            for target, senders in self._pending_incoming.items()
        }
        ckpt["_agent_directives"] = {
            tag: dict(d) for tag, d in self._agent_directives.items()
        }
        return ckpt

    def set_state_ckpt(self, state_ckpt: dict) -> None:
        super().set_state_ckpt(state_ckpt)
        # `.get` with defaults keeps backwards-compatibility with checkpoints
        # saved before these fields were persisted (they restore to empty).
        self._edge_last_active = {
            (u, v): step for u, v, step in state_ckpt.get("_edge_last_active", [])
        }
        self._pending_outgoing = {
            sender: dict(targets)
            for sender, targets in state_ckpt.get("_pending_outgoing", {}).items()
        }
        self._pending_incoming = {
            target: set(senders)
            for target, senders in state_ckpt.get("_pending_incoming", {}).items()
        }
        self._agent_directives = {
            tag: dict(d) for tag, d in state_ckpt.get("_agent_directives", {}).items()
        }

    def _kill(self, agent: str, reason: str | None = None) -> None:
        # Capture the owned node before super() pops agent_pos.
        node = self.agent_pos.get(agent)
        super()._kill(agent, reason=reason)
        self._agent_directives.pop(agent, None)
        # Purge any pending requests touching this agent (incoming + outgoing).
        for target in list(self._pending_outgoing.get(agent, {})):
            self._cancel_pending(agent, target)
        for sender in list(self._pending_incoming.get(agent, set())):
            self._cancel_pending(sender, agent)
        if node is None:
            return
        # The node and its artifacts (the agent's dropped inventory plus anything
        # already on the node) are about to disappear with the node — evacuate
        # them first so culture is not lost.
        self._evacuate_node_artifacts(node, agent)
        # Drop edge_last_active entries touching this node (either direction).
        for edge in list(self._edge_last_active.keys()):
            if node in edge:
                self._edge_last_active.pop(edge, None)
        # Removing the node from the graph drops all incident edges too.
        self.world_graph.remove_node(node)
        # Food-spawn cache is keyed on the node set.
        self._food_spawn_nodes = None

    def _evacuate_node_artifacts(self, node, agent: str) -> None:
        """Move a dying node's artifacts to the library, or expire them.

        When ``use_library`` is on the artifacts move to the shared commons and
        survive the agent's death (until their lifespan ends). When off they are
        expired cleanly so they never linger on a removed node.
        """
        for art_name in list(self.pos_artifacts.get(node, set())):
            if self.use_library:
                self.library.add(art_name)
                self.artifacts[art_name].pose = "Library"
                self.artifact_location[art_name] = ("library", None)
                self.logger.log(
                    time=self.step_count,
                    event_type=Event.LIBRARY_DEPOSIT,
                    agent_tag=agent,
                    agent_name=self.agent_names.get(agent),
                    artifact_name=art_name,
                    status="Success",
                    reason="owner_died",
                )
            else:
                artifact = self.artifacts.pop(art_name, None)
                self.artifact_location.pop(art_name, None)
                if artifact is not None:
                    artifact.deletion_time = self.step_count
                    self.expired_artifacts.append(artifact)
                    self.logger.log(
                        time=self.step_count,
                        event_type=Event.ARTIFACT_REMOVED,
                        artifact=artifact.serialize(),
                        reason="owner_died",
                    )
        self.pos_artifacts.pop(node, None)
