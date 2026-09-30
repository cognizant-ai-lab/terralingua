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
    "red": ([228, 26, 28], "#e41a1c"),
    "pink": ([247, 129, 191], "#f781bf"),
    "brown": ([166, 86, 40], "#a65628"),
    "gray": ([153, 153, 153], "#999999"),
    "green": ([77, 175, 74], "#4daf4a"),
    "purple": ([152, 78, 163], "#984ea3"),
    "blue": ([55, 126, 184], "#377eb8"),
    "orange": ([255, 127, 0], "#ff7f00"),
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


# --- symbol / marker map (for black & white figures) ----------------------
# Mirrors color_map: a consistent matplotlib marker per experiment condition so
# that grayscale figures encode condition by SHAPE instead of hue. Reuse this
# across every plot you convert to B&W so the same condition always gets the
# same symbol. Markers are chosen to stay distinguishable in a grayscale print.
marker_cm = {
    "no_hist": "o",          # CORE            — circle
    "scarcity_new": "s",     # LONG MEMORY     — square
    "no_personality": "^",   # NO PERSONALITY  — triangle up
    "no_motivation": "v",    # NO MOTIVATION   — triangle down
    "creative": "D",         # CREATIVE        — diamond
    "artifact_cost": "P",    # ARTIFACT COST   — filled plus
    "inert_artifacts": "X",  # INERT           — filled x
    "abundant": "*",         # ABUNDANCE       — star
}

# Extra markers handed out (in order) to any condition not in marker_cm.
_MARKER_POOL = ["p", "h", "<", ">", "d", "H", "8", "o", "s", "^", "v", "D", "P", "X", "*"]


class AutoMarkerMap:
    """Assigns a distinct marker to each experiment name, consistently.

    Known names use the fixed ``marker_cm`` assignment; any unseen name draws
    the next unused marker from a pool and remembers it for the session. Same
    dict-like interface as ``color_map`` (``[]``, ``.get()``, ``in``).
    """

    def __init__(self):
        self._assigned: dict[str, str] = dict(marker_cm)
        used = set(self._assigned.values())
        self._pool = itertools.cycle(
            [m for m in _MARKER_POOL if m not in used] or _MARKER_POOL
        )

    def __getitem__(self, key: str) -> str:
        if key not in self._assigned:
            self._assigned[key] = next(self._pool)
        return self._assigned[key]

    def __contains__(self, key: object) -> bool:
        return key in self._assigned

    def get(self, key: str, *_) -> str:
        return self[key]


marker_map = AutoMarkerMap()


# --- hatch / texture map (for black & white bar & area figures) -----------
# Point markers can't encode condition on bars/areas, so the B&W analog of
# marker_map is a per-condition HATCH pattern (drawn on a light-gray fill with a
# dark edge). Same keys as color_map / marker_map, so a condition keeps a stable
# identity across scatter (marker) and bar (hatch) figures. "" means a solid
# (unhatched) bar — reserved for the CORE baseline.
hatch_cm = {
    "no_hist": "",           # CORE            — solid
    "scarcity_new": "//",    # LONG MEMORY
    "no_personality": "\\\\",  # NO PERSONALITY
    "no_motivation": "||",   # NO MOTIVATION
    "creative": "--",        # CREATIVE
    "artifact_cost": "xx",   # ARTIFACT COST
    "inert_artifacts": "..",  # INERT
    "abundant": "++",        # ABUNDANCE
}

# Extra hatches handed out (in order) to any condition not in hatch_cm.
_HATCH_POOL = ["//", "\\\\", "||", "--", "xx", "..", "++", "oo", "OO", "**", "/o", "x."]


class AutoHatchMap:
    """Assigns a distinct hatch pattern to each experiment name, consistently.

    Known names use the fixed ``hatch_cm`` assignment; any unseen name draws the
    next unused hatch from a pool and remembers it. Dict-like interface like
    ``color_map`` / ``marker_map``.
    """

    def __init__(self):
        self._assigned: dict[str, str] = dict(hatch_cm)
        used = set(self._assigned.values())
        self._pool = itertools.cycle(
            [h for h in _HATCH_POOL if h not in used] or _HATCH_POOL
        )

    def __getitem__(self, key: str) -> str:
        if key not in self._assigned:
            self._assigned[key] = next(self._pool)
        return self._assigned[key]

    def __contains__(self, key: object) -> bool:
        return key in self._assigned

    def get(self, key: str, *_) -> str:
        return self[key]


hatch_map = AutoHatchMap()
