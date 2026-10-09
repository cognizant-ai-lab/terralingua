# survival_parallel_env.py

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pygame

from terralingua.environment.actions import LocationAffordance
from terralingua.environment.artifact import MAX_TEXT_ARTIFACT_SIZE
from terralingua.environment.base_env import BaseWorld
from terralingua.environment.env_logger import Event

log = logging.getLogger(__name__)

MOVE_DICT = {
    "stay": (0, 0),
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
# The prompt explains the axes with compass points; beings sometimes answer with them.
MOVE_ALIASES = {"north": "up", "south": "down", "east": "right", "west": "left"}


class OpenGridWorld(BaseWorld):
    system_prompt_template = "grid.j2"

    """Same grid rules as before, plus agent registry & coded messages."""

    _LOG_FILENAME = "open_gridworld.log"

    def __init__(
        self,
        grid_size: int = 100,
        vision_radius: int = 2,
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
        food_zones: int | List[Tuple[int, int]] | None = None,
        food_sigma: float = 2.0,
        static_food: bool = False,
        food_mechanism: bool = True,
        energy_death: bool | None = None,
        energy_upkeep: int | None = None,
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
            energy_upkeep=energy_upkeep,
            verbose=verbose,
            inert_artifacts=inert_artifacts,
            external_actions_spec=external_actions_spec,
            excluded_actions=excluded_actions,
            headless=headless,
            max_message_length=max_message_length,
            exclusive_pos_occupancy=True,
            roles_hocon_path=roles_hocon_path,
            affordances_file_path=affordances_file_path,
        )
        # grid-only state
        self.grid_size = grid_size
        self.vision_radius = vision_radius
        self._spawn_dist_flat = None
        self._initial_food_zone_count: int = 0
        self.empty_food: List[Tuple[int, int]] = []
        self.food_distribution = None
        self._window_size = (grid_size * self._cell_size, grid_size * self._cell_size)
        # Cell affordances come from the same JSON overlay as the graph worlds.
        # Keys that name roles are role permissions; the rest are "row,col" cells.
        self.cell_affordances: Dict[Tuple[int, int], List[LocationAffordance]] = {}
        if affordances_file_path:
            with open(affordances_file_path) as _f:
                overlay = json.load(_f)
            for key, props in overlay.items():
                if key in self._roles:
                    continue
                cell = self._parse_location(key)
                if cell is None:
                    raise ValueError(
                        f"Affordance key '{key}' is neither a role name nor a 'row,col' cell inside the grid."
                    )
                self.cell_affordances[cell] = [
                    LocationAffordance.from_dict(d) for d in props.get("affordances", [])
                ]

    def _location_affordance_list(self, location) -> List[LocationAffordance]:
        return self.cell_affordances.get(tuple(location), []) if location is not None else []

    def _destination_free(self, destination) -> bool:
        return not self.pos_to_agent[destination]

    def _place_role_occupants(self) -> None:
        # Roles have start places only in the graph world.
        placed = [n for n, r in self._roles.items() if r.get("start_nodes")]
        if placed:
            raise ValueError(f"start_nodes is not supported on the grid (roles: {placed}).")

    def _parse_location(self, text):
        """Read a 'row,col' cell inside the grid. Returns None otherwise."""
        parts = str(text).split(",")
        if len(parts) != 2:
            return None
        try:
            cell = (int(parts[0].strip()), int(parts[1].strip()))
        except ValueError:
            return None
        if not (0 <= cell[0] < self.grid_size and 0 <= cell[1] < self.grid_size):
            return None
        return cell

    def _adjacent(self, pos_a, pos_b) -> bool:
        """Cells that touch by a single up, down, left, or right move, with wrap-around."""
        dx = abs(int(pos_a[0]) - int(pos_b[0]))
        dy = abs(int(pos_a[1]) - int(pos_b[1]))
        dx, dy = min(dx, self.grid_size - dx), min(dy, self.grid_size - dy)
        return dx + dy == 1

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
        raw = move_params.get("direction", "stay")
        direction = raw.strip().lower() if isinstance(raw, str) else raw
        direction = MOVE_ALIASES.get(direction, direction)
        if direction not in MOVE_DICT:
            self.logger.log(
                time=self.step_count, event_type=Event.ACTION_REFUSED,
                agent_tag=agent, agent_name=self.agent_names[agent],
                action="move", reason="invalid_direction", direction=raw,
            )
            infos[agent]["Move outcome"] = (
                f"Unknown direction '{raw}'. Use one of: up, down, left, right, stay."
            )
        move = MOVE_DICT.get(direction, (0, 0))
        new_pose = self.wrap_xy(
            x=self.agent_pos[agent][0] + move[0],
            y=self.agent_pos[agent][1] + move[1],
        )
        if move != (0, 0):
            if not self.pos_to_agent[new_pose] or not self.exclusive_pos_occupancy:
                self._update_agent_pos(agent=agent, new_pos=new_pose)
            else:
                new_pose = self.agent_pos[agent]
                infos[agent]["Move outcome"] = (
                    f"Failed to move {direction}. Cell {new_pose} is occupied."
                )
                self.logger.log(
                    time=self.step_count, event_type=Event.ACTION_REFUSED,
                    agent_tag=agent, agent_name=self.agent_names[agent],
                    action="move", reason="cell_occupied", direction=direction,
                )
        return new_pose, infos

    def _get_food_distribution(self):
        if self.rng is None:
            self.rng = np.random.default_rng()

        if self.food_zones is None:
            density = np.full((self.grid_size, self.grid_size), 1 / self.grid_size**2)
            centers = []
        else:
            if isinstance(self.food_zones, int):
                if self._food_zone_centers is None:
                    ys = self.rng.integers(0, self.grid_size, size=self.food_zones)
                    xs = self.rng.integers(0, self.grid_size, size=self.food_zones)
                    self._food_zone_centers = [(int(x), int(y)) for x, y in zip(xs, ys)]
                    self._initial_food_zone_count = self.food_zones
                centers = self._food_zone_centers
            else:
                if self._food_zone_centers is None:
                    self._food_zone_centers = list(self.food_zones)
                    self._initial_food_zone_count = len(self._food_zone_centers)
                centers = self._food_zone_centers

            sigma = float(self.food_sigma if self.food_sigma is not None else 2.0)
            if sigma <= 0:
                raise ValueError("food_sigma must be > 0.")

            w = np.full(len(centers), 1.0 / len(centers), dtype=np.float64)
            yy, xx = np.mgrid[0 : self.grid_size, 0 : self.grid_size]
            density = np.zeros((self.grid_size, self.grid_size), dtype=np.float64)

            for (cx, cy), wt in zip(centers, w):
                dx = np.abs(xx - cx)
                dy = np.abs(yy - cy)
                dx = np.minimum(dx, self.grid_size - dx)
                dy = np.minimum(dy, self.grid_size - dy)

                r2 = dx * dx + dy * dy  # squared toroidal distance
                g = np.exp(-0.5 * r2 / (sigma * sigma))
                density += wt * g

            total = density.sum()
            if total <= 0 or not np.isfinite(total):
                raise RuntimeError(
                    "Density normalization failed (sum <= 0 or non-finite)."
                )
            density /= total

        self.food_distribution = density
        flat = density.ravel().astype(np.float64)
        self._spawn_dist_flat = flat / flat.sum()
        return density

    def _offspring_position(self, center, node_id=None):
        neighbours = [
            (center[0] + c[0], center[1] + c[1])
            for c in [
                [1, 0],
                [-1, 0],
                [0, 1],
                [0, -1],
                [1, 1],
                [1, -1],
                [-1, 1],
                [-1, -1],
            ]
        ]
        free_cells = [
            p
            for p in neighbours
            if (0 <= p[0] < self.grid_size)  # inside grid
            and (0 <= p[1] < self.grid_size)
            and not self.pos_to_agent[p]
        ]
        if free_cells:
            if self.rng is None:
                self.rng = np.random.default_rng()
            p = free_cells[int(self.rng.integers(0, len(free_cells)))]
            return (p[0], p[1])
        else:
            return None

    def _random_free_pos(self):
        """Find a random position in the grid that is not occupied by any agent."""
        if self.rng is None:
            self.rng = np.random.default_rng()
        while True:
            p = tuple(int(v) for v in self.rng.integers(1, self.grid_size - 1, size=2))
            if not self.pos_to_agent[p]:
                return p

    def _respawn_food_one(self, value: float | None = None):
        food_value = value if value is not None else self._max_food_value
        if self.rng is None:
            self.rng = np.random.default_rng()

        if self._spawn_dist_flat is None:
            self._get_food_distribution()

        if not self.static_food:
            # Rejection sampling: draw candidates from the base distribution
            # and accept the first cell that is unoccupied. This avoids
            # rebuilding an O(G²) masked array on every spawn.
            n = self.grid_size * self.grid_size
            for _ in range(max(100, n)):
                idx = int(self.rng.choice(n, p=self._spawn_dist_flat))
                y, x = divmod(idx, self.grid_size)
                p = (int(x), int(y))
                occupied = (
                    bool(self.pos_to_agent[p])
                    or p in self.food
                    or bool(self.pos_artifacts[p])
                )
                if not occupied:
                    self.food[p] = food_value
                    return
            log.warning("No available cell with nonzero probability to respawn food")
        else:
            if len(self.empty_food):
                idx = self.rng.integers(0, len(self.empty_food))
                p = self.empty_food.pop(idx)
                self.food[p] = food_value

    def _seed_initial_food(self):
        if self.rng is None:
            self.rng = np.random.default_rng()

        density = self._get_food_distribution()

        flat_p = density.ravel()
        # init_food is energy, not a cell count. Restrict placement to the
        # configured distribution's support, as in the graph world.
        candidates = np.flatnonzero(flat_p > 0)
        count = min(int(self._init_food_count / self._max_food_value), len(candidates))
        probabilities = flat_p[candidates]
        probabilities = probabilities / probabilities.sum()
        idx = self.rng.choice(candidates, size=count, replace=False, p=probabilities)
        ys = idx // self.grid_size
        xs = idx % self.grid_size
        spots = np.stack([xs, ys], axis=1).astype(int)
        self.food = {tuple(pos): self._max_food_value for pos in spots}
        self.empty_food = []

    def wrap_xy(self, x: int, y: int) -> tuple[int, int]:
        return x % self.grid_size, y % self.grid_size

    def distance(self, pos_a, pos_b) -> int:
        """Largest of the row and column offsets, with wrap-around."""
        dx = abs(int(pos_a[0]) - int(pos_b[0]))
        dy = abs(int(pos_a[1]) - int(pos_b[1]))
        return max(min(dx, self.grid_size - dx), min(dy, self.grid_size - dy))

    # ---------- observation ----------
    def _build_obs(
        self,
        agent: str,
        food_snapshot: dict | None = None,
        artifact_snapshot: dict | None = None,
    ) -> tuple[dict, bool]:
        """Builds the observation for a given agent.

        Returns (obs_dict, has_nearby_agents) so the caller can reuse the
        nearby-agent flag in _get_avail_actions without a second vision scan.

        When food_snapshot and artifact_snapshot are provided (built once per step
        via _build_step_snapshot), per-cell lookups drop from 3 dicts to 1.
        """
        x, y = self.agent_pos[agent]
        r = self.vision_radius
        messages = {}
        observation = defaultdict(list)
        has_nearby_agents = False

        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                gx, gy = self.wrap_xy(x + dx, y + dy)
                rel_pos = (dx, dy)

                if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                    # FOOD
                    if food_snapshot is not None:
                        food_str = food_snapshot.get((gx, gy))
                        if food_str is not None:
                            observation[rel_pos].append(food_str)
                    elif (gx, gy) in self.food:
                        observation[rel_pos].append(str(self.food[(gx, gy)]))

                    # AGENT
                    for a2 in self.pos_to_agent[(gx, gy)]:
                        if a2 != agent:
                            has_nearby_agents = True
                            if self.use_colors:
                                agent_descr = f"{self.agent_names[a2]}({self.agent_colors.get(a2, 'no color')})"
                            else:
                                agent_descr = self.agent_names[a2]
                            observation[rel_pos].append(agent_descr)  # type: ignore

                            # Add message if it exists
                            msg = self.msg_raw.get(a2, "")
                            if len(msg):
                                messages[self.agent_names[a2]] = msg
                                self.chat_recipients.setdefault(
                                    self.step_count, {}
                                ).setdefault(a2, set()).add(agent)

                    # ARTIFACT
                    if artifact_snapshot is not None:
                        art_strs = artifact_snapshot.get((gx, gy))
                        if art_strs:
                            observation[rel_pos].extend(art_strs)
                    elif not self.inert_artifacts:
                        for art_name in self.pos_artifacts[(gx, gy)]:
                            art = self.artifacts[art_name]
                            observation[rel_pos].append(
                                f"A({art.art_type},{'movable' if art.movable else 'fixed'}): {art.name}"
                            )
                # CELLS OUTSIDE OF MAP
                else:
                    observation[rel_pos].append("X")

        inventory_list = self._inventory_lines(agent)

        complete_obs = {
            "observation": observation,
            "observation_text": self.format_observation_text(observation),
            "incoming_broadcasts": messages,
            "energy": self.agent_energy[agent],
            "time": self.agent_time[agent],
            "inventory": inventory_list,
            "vision_radius": self.vision_radius,  # Passing it here as this can change
        }
        self._finish_obs(agent, complete_obs)
        return complete_obs, has_nearby_agents

    def format_observation_text(self, observation: dict) -> str:
        """Render the spatial observation as a list of non-empty cells."""
        lines = []
        for coords, content in observation.items():
            rx, ry = coords
            if (rx, ry) == (0, 0):
                content = ["yourself"] + content
            list_coords = f"({ry}, {-rx})"
            list_content = " | ".join(map(str, content))
            lines.append(f"{list_coords}: {list_content}")
        return "\n".join(lines)

    def _get_move_description(self, agent_tag: str) -> dict:
        return {
            "description": "Move of one cell in the specified direction, or stay in the current position",
            "params": {"direction": "One among [right, left, up, down, stay]."},
        }

    def _get_nearby_agents(self, agent_tag: str) -> List[str]:
        agent_position = self.agent_pos[agent_tag]
        x, y = agent_position
        r = self.vision_radius
        nearby_agents = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                gx, gy = self.wrap_xy(x + dx, y + dy)
                if self.pos_to_agent[(gx, gy)] and (gx, gy) != agent_position:
                    nearby_agents.extend(self.pos_to_agent[(gx, gy)])
        return nearby_agents

    # ---------- rendering (optional) ----------

    def render(self, mode="human"):
        assert mode in ("ascii", "rgb_array", "human"), mode

        if mode == "ascii":
            grid = np.full((self.grid_size, self.grid_size), ".", dtype=str)
            for fx, fy in self.food:
                grid[fx, fy] = "F"
            for a, (x, y) in self.agent_pos.items():
                grid[x, y] = a[0].upper()
            log.debug("\n".join(" ".join(row) for row in grid))
            return

        # --- Init pygame ---
        if not self._pygame_inited:
            self._sidebar_width = 300
            default_size = self.grid_size * self._cell_size
            self._window_size = (default_size + self._sidebar_width, default_size)
            if self._headless:
                pygame.font.init()
                self._screen = pygame.Surface(self._window_size)
            else:
                pygame.init()
                self._screen = pygame.display.set_mode(
                    self._window_size, pygame.RESIZABLE
                )
                pygame.display.set_caption("OpenGridWorld")
            self._font = pygame.font.SysFont(None, 15)  # body / wrapping
            self._font_hdr = pygame.font.SysFont(None, 17, bold=True)  # section headers
            self._font_tag = pygame.font.SysFont(
                None, 13, bold=True
            )  # small caps labels
            self._scroll_offset = 0
            self._msg_log = []
            self._seen_msgs = set()
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
        x_margin = 10
        max_width = self._sidebar_width - 2 * x_margin

        wrapped_lines = []
        for raw_line in self._msg_log:
            wrapped_lines.extend(self._wrap_text(raw_line or "", max_width))

        # --- Draw sidebar to a surface ---
        SB_BG = (252, 252, 252)
        SB_STRIP = (236, 236, 236)
        SB_ACCENT = (70, 130, 180)
        SB_TEXT = (35, 35, 35)
        SB_DIVIDER = (218, 218, 218)

        def draw_sidebar_surface(height):
            surface = pygame.Surface((self._sidebar_width, height))
            surface.fill(SB_BG)

            # Left accent bar
            pygame.draw.rect(surface, SB_ACCENT, pygame.Rect(0, 0, 4, height))

            # ── Header ──────────────────────────────────────
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

            # ── Chat log ────────────────────────────────────
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

        # --- Determine grid pixel dimensions ---
        if mode == "rgb_array":
            grid_pixel_w = self.grid_size * self._cell_size
            grid_pixel_h = self.grid_size * self._cell_size
        else:
            grid_pixel_w = self._window_size[0] - self._sidebar_width
            grid_pixel_h = self._window_size[1]

        cell_w = grid_pixel_w / self.grid_size
        cell_h = grid_pixel_h / self.grid_size

        # --- Draw grid surface ---
        # Colors from publication-ready palette
        BG_COLOR = (245, 245, 245)
        GRID_LINE_COLOR = (220, 220, 220)  # ~10% black blend on BG
        VISION_COLOR = (225, 225, 225)
        FOOD_LIGHT = (200, 235, 205)
        FOOD_DARK = (90, 180, 100)
        ARTIFACT_COLOR = (180, 60, 70)
        AGENT_COLOR = (40, 90, 140)

        cw = max(1, int(cell_w))
        ch = max(1, int(cell_h))

        grid_surf = pygame.Surface((grid_pixel_w, grid_pixel_h))
        grid_surf.fill(BG_COLOR)

        # Vision radius — solid gray cells
        r = self.vision_radius
        vision_cells = set()
        for agent in self.agent_registry:
            cx, cy = self.agent_pos[agent]
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    vision_cells.add(self.wrap_xy(cx + dx, cy + dy))
        for gx, gy in vision_cells:
            pygame.draw.rect(
                grid_surf,
                VISION_COLOR,
                pygame.Rect(int(gy * cell_w), int(gx * cell_h), cw, ch),
            )

        # Food — full cell, varying green
        for (x, y), val in self.food.items():
            if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
                continue
            ratio = max(0.0, min(1.0, float(val) / float(self._max_food_value)))
            color = tuple(
                int(FOOD_LIGHT[i] + (FOOD_DARK[i] - FOOD_LIGHT[i]) * ratio)
                for i in range(3)
            )
            pygame.draw.rect(
                grid_surf,
                color,
                pygame.Rect(int(y * cell_w), int(x * cell_h), cw, ch),
            )

        # Artifacts — full cell, amber
        for (x, y), arts in self.pos_artifacts.items():
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size and len(arts):
                pygame.draw.rect(
                    grid_surf,
                    ARTIFACT_COLOR,
                    pygame.Rect(int(y * cell_w), int(x * cell_h), cw, ch),
                )

        # Agents — full cell, colored
        for agent in self.agent_registry:
            color = AGENT_COLOR
            x, y = self.agent_pos[agent]
            pygame.draw.rect(
                grid_surf,
                color,
                pygame.Rect(int(y * cell_w), int(x * cell_h), cw, ch),
            )

        # Grid lines (drawn last, on top of everything)
        for i in range(self.grid_size + 1):
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

        # --- RGB array output ---
        if mode == "rgb_array":
            sidebar_surf = draw_sidebar_surface(grid_pixel_h)
            combined = pygame.Surface(
                (grid_pixel_w + self._sidebar_width, grid_pixel_h)
            )
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
        if self.food_distribution is None:
            self._get_food_distribution()
        ckpt.update(
            {
                "grid_size": self.grid_size,
                "food_distribution": self._serialize(self.food_distribution),
                "_initial_food_zone_count": self._initial_food_zone_count,
                "empty_food": self.empty_food,
            }
        )
        return ckpt

    def set_state_ckpt(self, state_ckpt: dict) -> None:
        super().set_state_ckpt(state_ckpt)
        saved_grid_size = state_ckpt.get("grid_size", self.grid_size)
        if saved_grid_size != self.grid_size:
            self.grid_size = saved_grid_size
        self.food_distribution = self._deserialize(state_ckpt["food_distribution"])
        self._initial_food_zone_count = state_ckpt.get("_initial_food_zone_count", 0)
        self.empty_food = state_ckpt["empty_food"]

    def resize_grid(self, new_size: int) -> None:
        """Expand the grid to new_size. Only grows — never shrinks.
        Adds food zone centers proportionally into the newly expanded area,
        then resets the food distribution so it rebuilds on the next spawn."""
        if new_size <= self.grid_size:
            return
        old_size = self.grid_size
        self.grid_size = new_size

        if self.food_zones is not None and self._food_zone_centers is not None:
            n_target = round(self._initial_food_zone_count * new_size**2 / old_size**2)
            n_add = max(0, n_target - len(self._food_zone_centers))
            new_centers = [
                (
                    int(self.rng.integers(0, new_size)),  # type: ignore
                    int(self.rng.integers(0, new_size)),  # type: ignore
                )
                for _ in range(n_add)
            ]
            self._food_zone_centers.extend(new_centers)
            log.info(
                "[Env] Food zones: %d (+%d new) at %s",
                len(self._food_zone_centers), n_add, self._food_zone_centers
            )

        self._spawn_dist_flat = None
        self.food_distribution = None


if __name__ == "__main__":
    from PIL import Image

    log_path = Path("logs/test_env")

    env = OpenGridWorld(food_zones=[(10, 10)], grid_size=50, log_path=log_path)
    image_path = log_path / "images"
    image_path.mkdir(parents=True, exist_ok=True)
    env.restart_env()
    for i in range(100):
        env.step({})
        rgb = env.render(mode="rgb_array")
        img = Image.fromarray(rgb)  # type: ignore
        img.save(image_path / f"step_{i:04d}.png")

    env.close()
    env.load_state()
    for i in range(100, 150):
        env.step({})
        rgb = env.render(mode="rgb_array")
        img = Image.fromarray(rgb)  # type: ignore
        img.save(image_path / f"step_{i:04d}.png")
    env.close()
