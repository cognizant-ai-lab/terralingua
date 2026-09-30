"""Grayscale, symbol-coded version of the food-rich vs food-scarce grid figure.

Regenerates the two-panel world snapshot used in the paper without relying on
color: each entity is drawn with a distinct *shape* so the figure survives a
black-and-white print. Data is reconstructed by instantiating a real
``OpenGridWorld`` for each food regime (uniform vs single central zone), so the
food layouts are the genuine output of the simulation's food-distribution
mechanics rather than hand-drawn decoration.

Run: ``python -m analysis_scripts.plot_food_settings``
"""

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from terralingua.environment.grid_env import OpenGridWorld
from terralingua.utils.generic import get_render_state

# --- figure configuration -------------------------------------------------
GRID_SIZE = 40
VISION_RADIUS = 5
MAX_FOOD_VALUE = 10.0
N_AGENTS = 10
N_ARTIFACTS = 18

# Food-rich: uniform density, plenty of cells. Food-scarce: a single central
# zone with a tight spread and fewer cells.
RICH = dict(food_zones=None, food_sigma=2.0, init_food=1500, seed=1)
SCARCE = dict(
    food_zones=[(GRID_SIZE // 2, GRID_SIZE // 2)],
    food_sigma=3.5,
    init_food=800,
    seed=2,
)

OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_STEM = "food_settings_grayscale"

# --- grayscale / symbol palette -------------------------------------------
# Entities are distinguished by SHAPE (not hue) so the figure reads in B&W:
#   agent    = solid black circle
#   artifact = hollow (white-filled, black-outlined) triangle
#   food     = solid mid-gray cell (shade darkens with the stored food value)
#   field of view = light-gray cell
VISION_GRAY = 0.90        # light background patch for field-of-view
FOOD_LIGHT_GRAY = 0.72    # low-value food (lighter)
FOOD_DARK_GRAY = 0.42     # high-value food (darker) — stays well above black markers
GRID_LINE_GRAY = 0.80
ENTITY_BLACK = "0.0"

AGENT_MARKER_SIZE = 55    # scatter s (points^2)
ARTIFACT_MARKER_SIZE = 85


def build_state(food_zones, food_sigma, init_food, seed):
    """Instantiate one grid world for a food regime and return its render state."""
    log_dir = Path("/tmp") / f"food_fig_{seed}"
    log_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    env = OpenGridWorld(
        grid_size=GRID_SIZE,
        vision_radius=VISION_RADIUS,
        init_food=init_food,
        max_food_value=MAX_FOOD_VALUE,
        food_zones=food_zones,
        food_sigma=food_sigma,
        food_mechanism=True,
        headless=True,
        log_path=log_dir,
    )
    env.rng = np.random.default_rng(seed)

    for i in range(N_AGENTS):
        env.add_agent(f"a{i}", f"being_{i}", genome_type="no_traits")
    env.restart_env()

    occupied = set(env.agent_pos.values()) | set(env.food.keys())
    art_rng = np.random.default_rng(seed + 100)
    placed = 0
    while placed < N_ARTIFACTS:
        pos = (int(art_rng.integers(0, GRID_SIZE)), int(art_rng.integers(0, GRID_SIZE)))
        if pos in occupied:
            continue
        env.seed_artifact(pos, "text", f"art_{placed}", "seed", lifespan=100)
        occupied.add(pos)
        placed += 1

    return get_render_state(env)


def render_panel(ax, state):
    """Draw one grid world in grayscale with shape-coded entities."""
    g = GRID_SIZE
    ax.set_xlim(0, g)
    ax.set_ylim(0, g)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # row 0 at the top, like an image
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # env coords are (row, col); plot col on x, row on y. Cell centers sit at +0.5.
    # (Scatter markers are used for point entities so they stay upright regardless
    # of the y-axis inversion and always match the legend glyphs.)

    # field-of-view patches (drawn first, underneath everything)
    for row, col in state["vision_cells"]:
        ax.add_patch(Rectangle((col, row), 1, 1, facecolor=str(VISION_GRAY), edgecolor="none"))

    # food: solid squares, darker == more food
    for f in state["food"]:
        shade = FOOD_LIGHT_GRAY + (FOOD_DARK_GRAY - FOOD_LIGHT_GRAY) * f["ratio"]
        ax.add_patch(Rectangle((f["y"], f["x"]), 1, 1, facecolor=str(shade), edgecolor="none"))

    # faint grid lines, drawn on top of the cells (like the original render)
    for i in range(g + 1):
        ax.axhline(i, color=str(GRID_LINE_GRAY), lw=0.3, alpha=0.55, zorder=2.5)
        ax.axvline(i, color=str(GRID_LINE_GRAY), lw=0.3, alpha=0.55, zorder=2.5)

    # artifacts: hollow (white-filled, black-outlined) triangles — stay legible on dark food
    if state["artifacts"]:
        ax.scatter(
            [a["y"] + 0.5 for a in state["artifacts"]],
            [a["x"] + 0.5 for a in state["artifacts"]],
            marker="^", s=ARTIFACT_MARKER_SIZE,
            facecolor="white", edgecolor=ENTITY_BLACK, linewidths=1.4, zorder=4,
        )

    # agents: solid black circles with a thin white halo
    if state["agents"]:
        ax.scatter(
            [a["y"] + 0.5 for a in state["agents"]],
            [a["x"] + 0.5 for a in state["agents"]],
            marker="o", s=AGENT_MARKER_SIZE,
            facecolor=ENTITY_BLACK, edgecolor="white", linewidths=0.7, zorder=5,
        )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rich_state = build_state(**RICH)
    scarce_state = build_state(**SCARCE)

    fig, axes = plt.subplots(1, 2, figsize=(9, 5.2))
    render_panel(axes[0], rich_state)
    render_panel(axes[1], scarce_state)

    for ax, letter, title in zip(axes, "ab", ("Food-rich setting", "Food-scarce setting")):
        ax.text(-0.02, 1.04, letter, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="bottom", ha="left")
        ax.text(0.04, 1.04, title, transform=ax.transAxes,
                fontsize=12, va="bottom", ha="left")

    legend_handles = [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=ENTITY_BLACK,
               markeredgecolor="white", markeredgewidth=0.7, markersize=10, label="Agent"),
        Line2D([], [], marker="^", linestyle="none", markerfacecolor="white",
               markeredgecolor=ENTITY_BLACK, markeredgewidth=1.4, markersize=11, label="Artifact"),
        Patch(facecolor=str(FOOD_DARK_GRAY), edgecolor="none", label="Food"),
        Patch(facecolor=str(VISION_GRAY), edgecolor="none", label="Field of view"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.01))

    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.09, wspace=0.08)

    for ext in ("pdf", "png"):
        out = OUT_DIR / f"{OUT_STEM}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
