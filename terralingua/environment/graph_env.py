"""
graph_world.py — OpenGraphWorld: graph-based world environment.

Drop-in complement to OpenGridWorld. When EnvConfig.graph is set,
the runner instantiates this class instead.
"""

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pygame
import pygame.gfxdraw

log = logging.getLogger(__name__)

from terralingua.config.models import GraphConfig
from terralingua.environment.actions import ACTION_TEXT, LocationAffordance
from terralingua.environment.artifact import MAX_TEXT_ARTIFACT_SIZE
from terralingua.environment.base_env import BaseWorld
from terralingua.environment.env_logger import Event
from terralingua.environment.graph_builder import build_graph
from terralingua.environment.world_graph import WorldGraph


class OpenGraphWorld(BaseWorld):
    """Agents live on graph nodes and move via named exit labels."""

    system_prompt_template = "graph.j2"
    _LOG_FILENAME = "open_graph_world.log"

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
        use_library: bool = False,
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
        food_mechanism: bool = True,
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
        super().__init__(
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
            exclusive_pos_occupancy=False,
            roles_hocon_path=roles_hocon_path,
            affordances_file_path=affordances_file_path,
        )
        # graph-only state
        self.graph_cfg = graph_cfg
        self.hop_radius = graph_cfg.hop_radius
        # The affordances file holds node overlays and, in roles mode, role
        # permissions keyed by role name. Only the node keys go to the graph.
        role_names = set(self._roles)
        node_props = None
        if affordances_file_path:
            with open(affordances_file_path) as _f:
                node_props = {
                    k: v for k, v in json.load(_f).items() if k not in role_names
                }
        self.world_graph: WorldGraph = build_graph(
            topology=graph_cfg.topology,
            n_nodes=graph_cfg.n_nodes,
            node_props=node_props,
            **graph_cfg._topology_params,
        )
        clash = role_names & set(self.world_graph.all_nodes())
        if clash:
            raise ValueError(
                f"Role names must differ from node ids; both: {sorted(clash)}."
            )
        self._food_spawn_nodes: List[str] | None = None
        self._food_spawn_probs: np.ndarray = np.array([])
        self.empty_food: List[str] = []
        self._node_layout: Dict[str, Tuple[float, float]] = {}

    def _adjacent(self, node: str, other: str) -> bool:
        """True when an edge joins the two nodes in either direction."""
        return node != other and (
            self.world_graph.has_edge(node, other) or self.world_graph.has_edge(other, node)
        )

    def _place_role_occupants(self) -> None:
        """Move each role's occupants to the role's start_nodes, round-robin."""
        nodes_in_graph = set(self.world_graph.all_nodes())
        for role_name in sorted(self._roles):
            start_nodes = self._roles[role_name].get("start_nodes") or []
            if not start_nodes:
                continue
            missing = [n for n in start_nodes if n not in nodes_in_graph]
            if missing:
                raise ValueError(
                    f"Role '{role_name}': start_nodes not in the graph: {missing}."
                )
            for i, tag in enumerate(sorted(self._role_occupants(role_name))):
                self._update_agent_pos(tag, start_nodes[i % len(start_nodes)])

    # ---------- internal helpers ----------
    def _apply_move(self, agent, move_params, infos):
        if move_params is None:
            return self.agent_pos[agent], infos
        if "direction" not in move_params:
            self.logger.log(
                time=self.step_count, event_type=Event.ACTION_REFUSED,
                agent_tag=agent, agent_name=self.agent_names[agent],
                action="move", reason="missing_direction",
            )
        move_target = move_params.get("direction", "stay")
        current_node = self.agent_pos[agent]

        if move_target == "stay":
            new_node = current_node
        elif self.world_graph.has_edge(current_node, move_target):
            new_node = move_target
        else:
            new_node = current_node

        if move_target != "stay" and new_node == current_node:
            infos[agent]["Move outcome"] = (
                f"Failed to move to {move_target}. Not a neighbor of current node."
            )
            self.logger.log(
                time=self.step_count, event_type=Event.ACTION_REFUSED,
                agent_tag=agent, agent_name=self.agent_names[agent],
                action="move", reason="no_edge", direction=move_target,
            )
        if new_node != current_node:
            cost = self._move_cost(current_node, new_node)
            if cost > 0 and self.agent_energy[agent] < cost:
                infos[agent]["Move outcome"] = (
                    f"Failed to move to {move_target}. The crossing costs {cost} energy "
                    f"and you have {self.agent_energy[agent]:.0f}."
                )
                self.logger.log(
                    time=self.step_count, event_type=Event.ACTION_REFUSED,
                    agent_tag=agent, agent_name=self.agent_names[agent],
                    action="move", reason="insufficient_energy", direction=move_target, cost=cost,
                )
                new_node = current_node
            elif cost > 0:
                self.agent_energy[agent] -= cost
                infos[agent]["Move cost"] = f"Moving to {move_target} cost {cost} energy."
                self.logger.log(
                    time=self.step_count, event_type=Event.MOVE_COST,
                    agent_tag=agent, agent_name=self.agent_names[agent],
                    src=current_node, dst=move_target, cost=cost,
                )
        if new_node != current_node:
            self._update_agent_pos(agent=agent, new_pos=new_node)
        return new_node, infos

    def _move_cost(self, src: str, dst: str) -> int:
        """Energy charged for moving along src->dst.

        Zero when ``graph.move_cost_attr`` is unset. Otherwise the edge's attribute of
        that name, or ``graph.default_move_cost`` when the edge does not carry it.
        """
        attr = self.graph_cfg.move_cost_attr
        if not attr:
            return 0
        value = self.world_graph.edge_attrs(src, dst).get(attr, self.graph_cfg.default_move_cost)
        try:
            return max(0, int(round(float(value))))
        except (TypeError, ValueError):
            return self.graph_cfg.default_move_cost

    def _get_food_weights(self):
        """Compute per-node spawn weights.

        With no food_zones: uniform over all nodes.
        With food_zones (int or List[str]): Gaussian around zone centers
        using BFS hop distance, matching OpenGridWorld's zone semantics.
        Weights are normalised to sum to 1.
        """
        if self.rng is None:
            self.rng = np.random.default_rng()
        all_nodes = self.world_graph.all_nodes()
        n_nodes = len(all_nodes)

        if self.food_zones is None:
            probs = np.ones(n_nodes) / n_nodes
            self._food_spawn_nodes = all_nodes
            self._food_spawn_probs = probs
            self._food_zone_centers = None
            return

        # Determine zone centers (persisted so respawn uses the same ones)
        if self._food_zone_centers is None:
            if isinstance(self.food_zones, int):
                chosen = self.rng.choice(n_nodes, size=self.food_zones, replace=False)
                self._food_zone_centers = [all_nodes[int(i)] for i in chosen]
            else:
                self._food_zone_centers = []
                for node in self.food_zones:
                    if node in all_nodes:
                        self._food_zone_centers.append(node)
                    else:
                        log.warning(
                            "Node %s in food_zones is not a valid node in the world graph. Selecting a random node instead.",
                            node
                        )
                        self._food_zone_centers.append(
                            all_nodes[int(self.rng.integers(0, n_nodes))]
                        )

        sigma = float(self.food_sigma if self.food_sigma is not None else 1.5)
        if sigma <= 0:
            raise ValueError("food_sigma must be > 0.")
        weights: Dict[str, float] = {n: 0.0 for n in all_nodes}

        for center in self._food_zone_centers:
            hop_distances = self.world_graph.nodes_within_hops(
                center, hops=int(sigma * 4)
            )
            for node, dist in hop_distances.items():
                weights[node] += np.exp(-0.5 * (dist / sigma) ** 2)

        # Build and normalize the probability array once using numpy
        raw_weights = np.array([weights[n] for n in all_nodes], dtype=np.float64)
        total = raw_weights.sum()
        if total <= 0 or not np.isfinite(total):
            raise RuntimeError("Density normalization failed (sum <= 0 or non-finite).")
        probs = raw_weights / total
        self._food_spawn_nodes = all_nodes
        self._food_spawn_probs = probs

    def _offspring_position(self, center: str, node_id: str | None = None) -> str | None:
        """Return a random neighbor of center with no agents, or None.
        When shared occupancy is allowed, returns any random neighbor."""
        if self.rng is None:
            self.rng = np.random.default_rng()
        neighbors = list(self.world_graph.neighbors(center))
        if not neighbors:
            return None
        if not self.exclusive_pos_occupancy:
            return neighbors[int(self.rng.integers(0, len(neighbors)))]
        free = [nbr for nbr in neighbors if not self.pos_to_agent[nbr]]
        if free:
            return free[int(self.rng.integers(0, len(free)))]
        return None

    def _random_free_pos(self) -> str:
        """Return a random unoccupied graph node, or any node when shared occupancy is allowed."""
        if self.rng is None:
            self.rng = np.random.default_rng()
        all_nodes = self.world_graph.all_nodes()
        if not self.exclusive_pos_occupancy:
            return all_nodes[int(self.rng.integers(0, len(all_nodes)))]
        while True:
            node = all_nodes[int(self.rng.integers(0, len(all_nodes)))]
            if not self.pos_to_agent[node]:
                return node

    def _respawn_food_one(self, value: float | None = None) -> None:
        food_value = value if value is not None else self._max_food_value
        if self.rng is None:
            self.rng = np.random.default_rng()
        # Ensure spawn distribution is initialised
        if self._food_spawn_nodes is None:
            self._get_food_weights()
        if not self.static_food:
            n = len(self._food_spawn_nodes)  # type: ignore
            for _ in range(n):  # type: ignore
                idx = int(self.rng.choice(n, p=self._food_spawn_probs))
                node = self._food_spawn_nodes[idx]  # type: ignore
                if node not in self.food:
                    self.food[node] = food_value
                    return
            log.warning("No available node to respawn food")
        else:
            if len(self.empty_food):
                idx = int(self.rng.integers(0, len(self.empty_food)))
                node = self.empty_food.pop(idx)
                self.food[node] = food_value

    def _seed_initial_food(self) -> None:
        if self.rng is None:
            self.rng = np.random.default_rng()
        if self._food_spawn_nodes is None:
            self._get_food_weights()

        n_tiles = int(self._init_food_count / self._max_food_value)

        # Only sample from nodes with non-zero spawn probability.
        # A narrow sigma can make most nodes zero-weight; sampling replace=False
        # from a pool smaller than count would raise ValueError.
        nonzero_mask = self._food_spawn_probs > 0  # type: ignore
        nonzero_indices = np.where(nonzero_mask)[0]
        if len(nonzero_indices) == 0:
            return
        probs = self._food_spawn_probs[nonzero_mask]  # type: ignore
        probs = probs / probs.sum()

        count = min(n_tiles, len(nonzero_indices))
        chosen_local = self.rng.choice(
            len(nonzero_indices), size=count, replace=False, p=probs
        )
        chosen = nonzero_indices[chosen_local]

        for idx in chosen:
            self.food[self._food_spawn_nodes[int(idx)]] = self._max_food_value  # type: ignore
        self.empty_food = []

    # ---------- observation ----------
    def _build_obs(
        self,
        agent: str,
        food_snapshot: dict | None = None,
        artifact_snapshot: dict | None = None,
    ) -> Tuple[dict, bool]:
        """Builds the observation for a given agent.

        Returns (obs_dict, has_nearby_agents) so the caller can reuse the
        nearby-agent flag in _get_avail_actions without a second BFS scan.
        """
        current_node = self.agent_pos[agent]
        neighborhood = self.world_graph.nodes_within_hops(current_node, self.hop_radius)

        messages: Dict[str, str] = {}
        observation: Dict[str, dict] = {}
        has_nearby_agents = False

        for node_id, hop_dist in neighborhood.items():
            items: List[str] = []

            # Food
            if food_snapshot is not None:
                food_str = food_snapshot.get(node_id)
                if food_str is not None:
                    items.append(food_str)
            elif node_id in self.food:
                items.append(str(self.food[node_id]))

            # Agents
            for a2 in self.pos_to_agent.get(node_id, set()):
                if a2 != agent:
                    has_nearby_agents = True
                    if self.use_colors:
                        agent_descr = (
                            f"{self.agent_names[a2]}"
                            f"({self.agent_colors.get(a2, 'no color')})"
                        )
                    else:
                        agent_descr = self.agent_names[a2]
                    items.append(agent_descr)
                    msg = self.msg_raw.get(a2, "")
                    if len(msg):
                        messages[self.agent_names[a2]] = msg
                        self.chat_recipients.setdefault(
                            self.step_count, {}
                        ).setdefault(a2, set()).add(agent)

            # Artifacts
            if artifact_snapshot is not None:
                art_strs = artifact_snapshot.get(node_id)
                if art_strs:
                    items.extend(art_strs)
            elif not self.inert_artifacts:
                for art_name in self.pos_artifacts[node_id]:
                    art = self.artifacts[art_name]
                    items.append(
                        f"A({art.art_type},"
                        f"{'movable' if art.movable else 'fixed'}): {art.name}"
                    )

            # exits = {
            #     data["label"]: nbr for nbr, data in self.world_graph.edges_from(node_id)
            # }

            observation[node_id] = {
                "items": items,
                "hop_distance": hop_dist,
                "exits": self.world_graph.neighbors(node_id),
            }

        inventory_list = self._inventory_lines(agent)

        complete_obs = {
            "observation": observation,
            "observation_text": self.format_observation_text(observation, current_node),
            "current_node": current_node,
            "incoming_broadcasts": messages,
            "energy": self.agent_energy[agent],
            "time": self.agent_time[agent],
            "inventory": inventory_list,
            "hop_radius": self.hop_radius,
        }
        self._finish_obs(agent, complete_obs)
        return complete_obs, has_nearby_agents

    def format_observation_text(self, observation: dict, current_node: str) -> str:
        """Render the graph neighborhood observation as a readable location list."""
        lines = [f"You are at: {current_node}", ""]
        for node_id, data in sorted(
            observation.items(), key=lambda kv: kv[1]["hop_distance"]
        ):
            dist = data["hop_distance"]
            if dist == 0:
                if data["items"]:
                    items_str = "<yourself> | " + " | ".join(data["items"])
                else:
                    items_str = "<yourself>"
            else:
                items_str = " | ".join(data["items"]) if data["items"] else "(empty)"
            lines.append(f"  [{node_id}, dist={dist}]: {items_str}")
        return "\n".join(lines)

    def _get_move_description(self, agent_tag: str) -> dict:
        pos = self.agent_pos[agent_tag]
        neighbor_nodes = self.world_graph.neighbors(pos)
        description = ACTION_TEXT["move"]["description"]
        if self.graph_cfg.move_cost_attr and neighbor_nodes:
            costs = ", ".join(f"{n} ({self._move_cost(pos, n)})" for n in neighbor_nodes)
            description += f" Energy cost of each crossing from here: {costs}. Staying costs nothing extra."
        return {
            "description": description,
            "params": {
                "direction": {
                    "description": "Node ID of a directly connected neighbor to move to, or 'stay'.",
                    "choices": ["stay"] + neighbor_nodes,
                }
            },
        }

    def _get_nearby_agents(self, agent_tag: str) -> List[str]:
        current_node = self.agent_pos[agent_tag]
        neighborhood = self.world_graph.nodes_within_hops(current_node, self.hop_radius)
        result = []
        for node_id in neighborhood:
            for a in self.pos_to_agent.get(node_id, set()):
                if a != agent_tag:
                    result.append(a)
        return result

    def distance(self, pos_a, pos_b) -> float:
        """Edges on the shortest path from pos_a to pos_b, following edge direction."""
        return self.world_graph.hop_distance(pos_a, pos_b)

    def agents_within(self, tag: str, r: float) -> List[str]:
        hops = int(r) if math.isfinite(r) else None
        reach = self.world_graph.nodes_within_hops(self.agent_pos[tag], hops)
        return sorted(
            other
            for node_id in reach
            for other in self.pos_to_agent.get(node_id, set())
            if other != tag
        )

    # ---------- node affordances ----------
    #
    # Three hooks control location-based actions:
    #
    #   _get_location_affordance(location) -> dict
    #       Returns add-mode affordances as an agent-facing action dict.
    #       Called by _get_avail_actions; runs after global actions so entries
    #       overwrite any global action with the same name.
    #
    #   _get_location_removals(location) -> set[str]
    #       Returns a set of action names to suppress at this location even if
    #       they are globally available (e.g. remove spawn on spoke
    #       nodes to restrict it to the hub only).  Applied immediately after
    #       _get_location_affordance, before excluded_actions.
    #
    #   _on_location_action(agent, action_name, action_params, infos) -> bool
    #       Called at the top of the step dispatch chain for custom (non-built-in)
    #       actions only.  Built-in ACTION_TEXT actions are never intercepted here
    #       so energy deduction and logging work correctly.  A matched affordance
    #       whose effect["type"] is registered in self._affordance_effects invokes
    #       the registered handler.

    def _location_affordance_list(self, location: str) -> List[LocationAffordance]:
        return self.world_graph.node_affordances(location)

    # ---------- layout ----------

    def _compute_node_layout(self) -> None:
        """Compute and cache the node layout for non-grid topologies.

        For grid topologies the layout is implicit in the node IDs, so nothing
        is stored.  For all other topologies a networkx layout is computed once
        and cached in ``self._node_layout``.
        """
        if self.graph_cfg.topology == "grid":
            return
        if self._node_layout:
            return

        topo = self.graph_cfg.topology
        if topo in ("ring", "complete", "small_world"):
            all_nodes = self.world_graph.all_nodes()
            n = len(all_nodes)
            # Number of shells: outer shell capped at ~80 nodes (proportional distribution
            # gives n_outer = 2n/(k+1), so k = ceil(2n/80 - 1))
            k = max(1, math.ceil(2 * n / 80 - 1))
            # Distribute nodes proportionally to shell radius (shell i gets weight i)
            # so arc spacing is equal across all shells
            total_weight = k * (k + 1) // 2
            shells, idx = [], 0
            for i in range(1, k + 1):
                count = n - idx if i == k else round(n * i / total_weight)
                shells.append(all_nodes[idx : idx + count])
                idx += count
            raw_layout = nx.shell_layout(self.world_graph._G, nlist=shells)
        else:
            raw_layout = nx.kamada_kawai_layout(self.world_graph._G)
        self._node_layout = {
            nid: (float(pos[0]), float(pos[1])) for nid, pos in raw_layout.items()
        }

    def restart_env(self, seed=None, **options):
        observations, infos = super().restart_env(seed=seed, **options)
        self._compute_node_layout()
        return observations, infos

    # ---------- rendering ----------

    def render(self, mode: str = "human"):
        assert mode in ("ascii", "rgb_array", "human"), mode

        if mode == "ascii":
            log.debug(
                "Step %s | nodes=%d | agents=%d",
                self.step_count, len(self.world_graph.all_nodes()), len(self.agent_registry)
            )
            for node_id in self.world_graph.all_nodes():
                agents = [
                    self.agent_names[a] for a in self.pos_to_agent.get(node_id, set())
                ]
                food = f"food:{self.food[node_id]:.1f}" if node_id in self.food else ""
                arts = [a for a in self.pos_artifacts.get(node_id, set())]
                items = agents + ([food] if food else []) + arts
                if items:
                    log.debug("  [%s]: %s", node_id, " | ".join(items))
            return

        # --- Init pygame ---
        if not self._pygame_inited:
            self._sidebar_width = 300
            is_grid = self.graph_cfg.topology == "grid"
            if is_grid:
                k = max(2, int(math.isqrt(self.graph_cfg.n_nodes)))
                default_size = k * self._cell_size
            else:
                default_size = 600
            self._window_size = (default_size + self._sidebar_width, default_size)
            if self._headless:
                pygame.font.init()
                self._screen = pygame.Surface(self._window_size)
            else:
                pygame.display.init()
                pygame.font.init()
                self._screen = pygame.display.set_mode(
                    self._window_size, pygame.RESIZABLE
                )
                pygame.display.set_caption("OpenGraphWorld")
            self._font = pygame.font.SysFont(None, 15)
            self._font_hdr = pygame.font.SysFont(None, 17, bold=True)
            self._font_tag = pygame.font.SysFont(None, 13, bold=True)
            self._scroll_offset = 0
            self._msg_log = []
            self._seen_msgs: set = set()
            self._pygame_inited = True

        # --- Append messages for the step ---
        if self.step_count not in self._seen_msgs:
            messages = self.chat.get(self.step_count - 1)
            if messages:
                self._msg_log.append(f"Step {self.step_count - 1}")
                self._msg_log.extend(messages)
                self._msg_log.append("--")
            self._seen_msgs.add(self.step_count)

        # --- Common draw config ---
        max_width = self._sidebar_width - 20

        wrapped_lines = []
        for raw_line in self._msg_log:
            wrapped_lines.extend(self._wrap_text(raw_line or "", max_width))

        # --- Sidebar draw helper ---
        SB_BG = (252, 252, 252)
        SB_STRIP = (236, 236, 236)
        SB_ACCENT = (70, 130, 180)
        SB_TEXT = (35, 35, 35)
        SB_DIVIDER = (218, 218, 218)

        def draw_sidebar_surface(height: int) -> pygame.Surface:
            surface = pygame.Surface((self._sidebar_width, height))
            surface.fill(SB_BG)
            pygame.draw.rect(surface, SB_ACCENT, pygame.Rect(0, 0, 4, height))
            hdr_h = 38
            pygame.draw.rect(
                surface, SB_STRIP, pygame.Rect(4, 0, self._sidebar_width - 4, hdr_h)
            )
            step_surf = self._font_hdr.render(f"Step  {self.step_count}", True, SB_TEXT)
            surface.blit(step_surf, (12, (hdr_h - step_surf.get_height()) // 2))
            pygame.draw.line(
                surface, SB_DIVIDER, (4, hdr_h), (self._sidebar_width, hdr_h)
            )
            y = hdr_h + 1
            msg_lh = 17
            visible_height = height - y
            max_lines = visible_height // msg_lh
            start_idx = max(0, len(wrapped_lines) - max_lines - self._scroll_offset)
            end_idx = start_idx + max_lines
            visible_lines = wrapped_lines[start_idx:end_idx]
            for line in visible_lines:
                if y + msg_lh > height:
                    break
                if line == "--":
                    pygame.draw.line(
                        surface,
                        SB_DIVIDER,
                        (12, y + msg_lh // 2),
                        (self._sidebar_width - 12, y + msg_lh // 2),
                    )
                elif line.startswith("Step ") and line[5:].strip().isdigit():
                    text_surf = self._font_tag.render(line, True, SB_ACCENT)
                    surface.blit(
                        text_surf, (12, y + (msg_lh - text_surf.get_height()) // 2)
                    )
                else:
                    text_surf = self._font.render(line, True, SB_TEXT)
                    surface.blit(
                        text_surf, (12, y + (msg_lh - text_surf.get_height()) // 2)
                    )
                y += msg_lh
            return surface

        # --- Pixel dimensions ---
        if mode == "rgb_array":
            sidebar_w = self._sidebar_width
            grid_pixel_w = self._window_size[0] - sidebar_w
            grid_pixel_h = self._window_size[1]
        else:
            sidebar_w = self._sidebar_width
            grid_pixel_w = self._window_size[0] - sidebar_w
            grid_pixel_h = self._window_size[1]

        # --- Color palette ---
        BG_COLOR = (245, 245, 245)
        GRID_LINE_COLOR = (220, 220, 220)
        HOP_COLOR = (225, 225, 225)
        FOOD_LIGHT = (200, 235, 205)
        FOOD_DARK = (90, 180, 100)
        ARTIFACT_COLOR = (180, 60, 70)
        AGENT_COLOR = (40, 90, 140)
        EDGE_COLOR = (200, 200, 200)
        NODE_COLOR = (230, 230, 230)

        grid_surf = pygame.Surface((grid_pixel_w, grid_pixel_h))
        grid_surf.fill(BG_COLOR)

        is_grid = self.graph_cfg.topology == "grid"

        if is_grid:
            # ── Grid topology: cell-per-node rendering ──────────────────────
            k = max(2, int(math.isqrt(self.graph_cfg.n_nodes)))
            cell_w = grid_pixel_w / k
            cell_h = grid_pixel_h / k
            cw = max(1, int(cell_w))
            ch = max(1, int(cell_h))

            def _rc(node_id: str) -> Tuple[int, int]:
                inner = node_id.strip("()")
                parts = inner.split(",")
                return int(parts[0].strip()), int(parts[1].strip())

            # Hop-radius highlight
            for agent in self.agent_registry:
                r_pos, c_pos = _rc(self.agent_pos[agent])
                neighborhood = self.world_graph.nodes_within_hops(
                    self.agent_pos[agent], self.hop_radius
                )
                for nid in neighborhood:
                    nr, nc = _rc(nid)
                    pygame.draw.rect(
                        grid_surf,
                        HOP_COLOR,
                        pygame.Rect(int(nc * cell_w), int(nr * cell_h), cw, ch),
                    )

            # Food
            for node_id, val in self.food.items():
                nr, nc = _rc(node_id)
                ratio = max(0.0, min(1.0, float(val) / float(self._max_food_value)))
                color = tuple(
                    int(FOOD_LIGHT[i] + (FOOD_DARK[i] - FOOD_LIGHT[i]) * ratio)
                    for i in range(3)
                )
                pygame.draw.rect(
                    grid_surf,
                    color,
                    pygame.Rect(int(nc * cell_w), int(nr * cell_h), cw, ch),
                )

            # Artifacts
            for node_id, arts in self.pos_artifacts.items():
                if arts:
                    nr, nc = _rc(node_id)
                    pygame.draw.rect(
                        grid_surf,
                        ARTIFACT_COLOR,
                        pygame.Rect(int(nc * cell_w), int(nr * cell_h), cw, ch),
                    )

            # Agents
            for agent in self.agent_registry:
                color = AGENT_COLOR
                nr, nc = _rc(self.agent_pos[agent])
                pygame.draw.rect(
                    grid_surf,
                    color,
                    pygame.Rect(int(nc * cell_w), int(nr * cell_h), cw, ch),
                )

            # Grid lines
            for i in range(k + 1):
                pygame.draw.line(
                    grid_surf,
                    GRID_LINE_COLOR,
                    (int(i * cell_w), 0),
                    (int(i * cell_w), grid_pixel_h),
                )
                pygame.draw.line(
                    grid_surf,
                    GRID_LINE_COLOR,
                    (0, int(i * cell_h)),
                    (grid_pixel_w, int(i * cell_h)),
                )

        else:
            # ── General topology: force-directed graph rendering ─────────────
            # Build/cache topology-aware layout
            self._compute_node_layout()

            margin = 30
            eff_w = grid_pixel_w - 2 * margin
            eff_h = grid_pixel_h - 2 * margin

            xs = [v[0] for v in self._node_layout.values()]
            ys = [v[1] for v in self._node_layout.values()]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            span_x = max(max_x - min_x, 1e-6)
            span_y = max(max_y - min_y, 1e-6)

            def _px(node_id: str) -> Tuple[int, int]:
                lx, ly = self._node_layout[node_id]
                px = int(margin + (lx - min_x) / span_x * eff_w)
                py = int(margin + (ly - min_y) / span_y * eff_h)
                return px, py

            n_nodes = len(self._node_layout)
            node_radius = max(6, min(16, grid_pixel_w // (n_nodes // 2 + 1)))

            # Collect per-agent hop neighborhoods
            agent_hop: Dict[str, set] = {}
            for agent in self.agent_registry:
                neighborhood = self.world_graph.nodes_within_hops(
                    self.agent_pos[agent], self.hop_radius
                )
                agent_hop[agent] = set(neighborhood.keys())
            all_hop_nodes: set = (
                set().union(*agent_hop.values()) if agent_hop else set()
            )

            # Edges — drawn directionally. Each unordered pair is drawn once.
            # A mutual pair (both u→v and v→u) is a plain line; a one-way edge
            # additionally gets an arrowhead at its destination so asymmetric
            # links (as in the social-graph world) are visible. Hop-radius edges
            # (both endpoints reachable) use a stronger colour/width.
            HOP_EDGE_COLOR = (100, 120, 180)

            def _arrowhead(color, src, dst):
                """Draw an arrowhead on the line src→dst, just outside dst's node."""
                dx, dy = dst[0] - src[0], dst[1] - src[1]
                dist = math.hypot(dx, dy) or 1.0
                ux, uy = dx / dist, dy / dist
                tip = (dst[0] - ux * node_radius, dst[1] - uy * node_radius)
                ang = math.atan2(uy, ux)
                size = max(5, node_radius - 1)
                left = (
                    tip[0] - size * math.cos(ang - 0.5),
                    tip[1] - size * math.sin(ang - 0.5),
                )
                right = (
                    tip[0] - size * math.cos(ang + 0.5),
                    tip[1] - size * math.sin(ang + 0.5),
                )
                pygame.draw.polygon(grid_surf, color, [tip, left, right])

            seen_pairs: set = set()
            for u in self.world_graph.all_nodes():
                for v in self.world_graph.neighbors(u):
                    a, b = (u, v) if u < v else (v, u)
                    if (a, b) in seen_pairs:
                        continue
                    seen_pairs.add((a, b))
                    in_hop = a in all_hop_nodes and b in all_hop_nodes
                    color = HOP_EDGE_COLOR if in_hop else EDGE_COLOR
                    width = 2 if in_hop else 1
                    pa, pb = _px(a), _px(b)
                    pygame.draw.line(grid_surf, color, pa, pb, width)
                    fwd = self.world_graph.has_edge(a, b)
                    rev = self.world_graph.has_edge(b, a)
                    if fwd and not rev:
                        _arrowhead(color, pa, pb)
                    elif rev and not fwd:
                        _arrowhead(color, pb, pa)

            def _aa_circle(surf, color, pos, r):
                x, y = pos
                pygame.gfxdraw.filled_circle(surf, x, y, r, color)
                pygame.gfxdraw.aacircle(surf, x, y, r, color)

            def _aa_ring(surf, color, pos, r):
                pygame.gfxdraw.aacircle(surf, pos[0], pos[1], r, color)

            # Nodes: base circle — HOP_COLOR fill for reachable nodes
            for nid in self.world_graph.all_nodes():
                fill = HOP_COLOR if nid in all_hop_nodes else NODE_COLOR
                _aa_circle(grid_surf, fill, _px(nid), node_radius)

            # Food overlay
            for nid, val in self.food.items():
                ratio = max(0.0, min(1.0, float(val) / float(self._max_food_value)))
                color = tuple(
                    int(FOOD_LIGHT[i] + (FOOD_DARK[i] - FOOD_LIGHT[i]) * ratio)
                    for i in range(3)
                )
                _aa_circle(grid_surf, color, _px(nid), node_radius)

            # Artifact overlay
            for nid, arts in self.pos_artifacts.items():
                if arts:
                    _aa_circle(grid_surf, ARTIFACT_COLOR, _px(nid), node_radius)

            # Agent overlay
            for agent in self.agent_registry:
                _aa_circle(grid_surf, AGENT_COLOR, _px(self.agent_pos[agent]), node_radius)

            # Node outlines on top — blue border for reachable nodes
            for nid in self.world_graph.all_nodes():
                outline = HOP_EDGE_COLOR if nid in all_hop_nodes else GRID_LINE_COLOR
                _aa_ring(grid_surf, outline, _px(nid), node_radius)

        # --- RGB array output ---
        if mode == "rgb_array":
            sidebar_surf = draw_sidebar_surface(grid_pixel_h)
            combined = pygame.Surface((grid_pixel_w + sidebar_w, grid_pixel_h))
            combined.blit(grid_surf, (0, 0))
            combined.blit(sidebar_surf, (grid_pixel_w, 0))
            return pygame.surfarray.array3d(combined).transpose((1, 0, 2))

        # --- Human display ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self._pygame_inited = False
            elif event.type == pygame.VIDEORESIZE:
                self._window_size = (event.w, event.h)
                self._screen = pygame.display.set_mode(
                    self._window_size, pygame.RESIZABLE
                )
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self._scroll_offset = max(0, self._scroll_offset - 3)
                elif event.key == pygame.K_DOWN:
                    self._scroll_offset += 3
            elif event.type == pygame.MOUSEWHEEL:
                if event.y > 0:
                    self._scroll_offset = max(0, self._scroll_offset - 3)
                elif event.y < 0:
                    self._scroll_offset += 3

        self._screen.blit(grid_surf, (0, 0))  # type: ignore
        sidebar_surf = draw_sidebar_surface(self._window_size[1])
        self._screen.blit(sidebar_surf, (grid_pixel_w, 0))  # type: ignore
        pygame.display.flip()
        return pygame.surfarray.array3d(grid_surf).transpose((1, 0, 2))

    # ---------- checkpointing ----------

    def get_state_ckpt(self) -> dict:
        ckpt = super().get_state_ckpt()
        ckpt["world_graph"] = self.world_graph.serialize()
        return ckpt

    def set_state_ckpt(self, state_ckpt: dict) -> None:
        super().set_state_ckpt(state_ckpt)
        if "world_graph" in state_ckpt:
            self.world_graph = WorldGraph.deserialize(state_ckpt["world_graph"])
        if "agent_affordances" not in state_ckpt:
            self._strip_role_node_affordances()

    def _strip_role_node_affordances(self) -> None:
        """Older checkpoints stored role permissions on the occupant's node.
        Drop those copies; the assignment now provides them."""
        for tag, role_name in self._role_assignment.items():
            node = self.agent_pos.get(tag)
            if node is None or role_name not in self._roles:
                continue
            for aff in self._roles[role_name]["affordances"]:
                self.world_graph.remove_node_affordance(node, aff["action"])


if __name__ == "__main__":
    log_path = Path("logs/test_env")
    image_path = log_path / "images"
    image_path.mkdir(parents=True, exist_ok=True)
    # Example usage
    env = OpenGraphWorld(
        graph_cfg=GraphConfig(
            topology="small_world",
            hop_radius=2,
            n_nodes=500,
            small_world_k=4,
            small_world_p=0.1,
        ),
        use_colors=True,
        food_mechanism=True,
        use_inventory=True,
        food_zones=["100"],
        food_sigma=0.5,
        headless=True,
        log_path=log_path,
    )

    from PIL import Image
    from tqdm import tqdm

    env.add_agent(agent_tag="being0", agent_name="aa", position="5", genome_type="no_traits")

    image_path.mkdir(parents=True, exist_ok=True)
    env.restart_env()
    for i in tqdm(range(80)):
        response = env.step(
            {"being0": {"action": "move", "params": {"direction": "stay"}}}
        )
        if i % 10 == 0:
            rgb = env.render(mode="rgb_array")
            img = Image.fromarray(rgb)  # type: ignore
            img.save(image_path / f"step_{i:04d}.png")
