import itertools

from matplotlib.colors import to_rgb  # noqa: F401 (re-exported for callers)

EXP_BASE_NAMES = [
    "no_hist",
    "scarcity_new",
    "no_personality",
    "no_motivation",
    "creative",
    "artifact_cost",
    "inert_artifacts",
    "abundant",
]
PLOT_NAMES = {
    "abundant": "ABUNDANCE",
    "scarcity_new": "LONG MEMORY",
    "no_hist": "CORE",
    "no_personality": "NO PERSONALITY",
    "no_motivation": "NO MOTIVATION",
    "artifact_cost": "ARTIFACT COST",
    "creative": "CREATIVE",
    "inert_artifacts": "INERT",
}

plot_params = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.frameon": False,
}

colorblind_cm = {
    "blue": ([55, 126, 184], "#377eb8"),
    "orange": ([255, 127, 0], "#ff7f00"),
    "green": ([77, 175, 74], "#4daf4a"),
    "pink": ([247, 129, 191], "#f781bf"),
    "brown": ([166, 86, 40], "#a65628"),
    "purple": ([152, 78, 163], "#984ea3"),
    "gray": ([153, 153, 153], "#999999"),
    "red": ([228, 26, 28], "#e41a1c"),
    "yellow": ([222, 222, 0], "#dede00"),
}


class AutoColorMap:
    """Assigns colorblind-safe colors to experiment names on first access.

    Colors are drawn in order from ``colorblind_cm`` and remembered for the
    session, so the same name always maps to the same color.  Supports the
    same dict-like interface as a plain dict (``[]``, ``.get()``, ``in``).
    """

    def __init__(self):
        self._assigned: dict[str, str] = {}
        self._pool = itertools.cycle(
            hex_color for _, hex_color in colorblind_cm.values()
        )

    def __getitem__(self, key: str) -> str:
        if key not in self._assigned:
            self._assigned[key] = next(self._pool)
        return self._assigned[key]

    def __contains__(self, key: object) -> bool:
        return key in self._assigned

    def get(self, key: str, *_) -> str:
        return self[key]


color_map = AutoColorMap()
