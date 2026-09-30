"""
graph_builder.py — factory functions for WorldGraph topologies.

Each builder creates a nx.DiGraph with string node IDs and directional edge labels,
then wraps it in a WorldGraph.

Node naming by topology
-----------------------
grid        : "(r,c)" coordinate strings, e.g. "(0,0)", "(2,3)"
ring        : stringified integers "0" … "n-1"
tree        : stringified integers from nx.balanced_tree (root = "0")
complete    : stringified integers "0" … "n-1"
bipartite   : "A0", "A1", … for partition A; "B0", "B1", … for partition B
small_world : stringified integers from Watts-Strogatz
scale_free  : stringified integers from Barabási-Albert
random      : stringified integers from Erdős-Rényi
star        : "hub" for the central node; "0" … "n-2" for spoke nodes
wheel       : "hub" for the central node; "0" … "n-2" for ring nodes
file        : the "id" field of each node entry in the JSON file
custom      : defined by the user-supplied factory function
"""

import importlib
import json
from typing import Dict

import networkx as nx

from terralingua.environment.world_graph import WorldGraph

# Label pairs for opposite directions
_FLIP_LABEL: Dict[str, str] = {
    "north": "south",
    "south": "north",
    "east": "west",
    "west": "east",
    "clockwise": "counterclockwise",
    "counterclockwise": "clockwise",
    "up": "down",
    "down": "up",
}


def _add_bidirectional(G: nx.DiGraph, u: str, v: str, label: str, **attrs) -> None:
    """Add both u→v (label) and v→u (flipped label)."""
    G.add_edge(u, v, label=label, **attrs)
    G.add_edge(v, u, label=_FLIP_LABEL.get(label, label), **attrs)


# ------------------------------------------------------------------
# Topology builders
# ------------------------------------------------------------------


def _build_grid(n_nodes: int, wrap: bool = False, **kwargs) -> WorldGraph:
    """2D square lattice. n_nodes is treated as the per-side count (k×k grid).
    If n_nodes is not a perfect square, k = floor(sqrt(n_nodes))."""
    import math

    k = max(2, int(math.isqrt(n_nodes)))
    G = nx.DiGraph()

    for r in range(k):
        for c in range(k):
            G.add_node(f"({r},{c})", x=r, y=c, label=f"({r},{c})")

    for r in range(k):
        for c in range(k):
            nid = f"({r},{c})"
            for direction, (dr, dc) in [
                ("north", (-1, 0)),
                ("south", (1, 0)),
                ("east", (0, 1)),
                ("west", (0, -1)),
            ]:
                nr, nc = r + dr, c + dc
                if wrap:
                    nr, nc = nr % k, nc % k
                if 0 <= nr < k and 0 <= nc < k:
                    nbr = f"({nr},{nc})"
                    if not G.has_edge(nid, nbr):
                        G.add_edge(nid, nbr, label=direction, traversable=True)

    return WorldGraph(G)


def _build_ring(n_nodes: int, **kwargs) -> WorldGraph:
    """Circular ring. Nodes are "0", "1", ..., "n-1"."""
    G = nx.DiGraph()
    n = max(3, n_nodes)

    for i in range(n):
        G.add_node(str(i), label=str(i))

    for i in range(n):
        _add_bidirectional(G, str(i), str((i + 1) % n), "clockwise", traversable=True)

    return WorldGraph(G)


def _build_tree(
    n_nodes: int, branching: int = 3, height: int | None = None, **kwargs
) -> WorldGraph:
    """Balanced tree. Children are labeled with direction 'down', parent with 'up'.
    height is inferred from n_nodes and branching when not explicitly provided."""
    import math

    if height is None:
        if branching > 1:
            # smallest h s.t. (branching^(h+1) - 1) / (branching - 1) >= n_nodes
            h = (
                math.ceil(math.log(n_nodes * (branching - 1) + 1) / math.log(branching))
                - 1
            )
            height = max(1, h)
        else:
            height = n_nodes - 1  # linear chain
    raw = nx.balanced_tree(branching, height)
    G = nx.DiGraph()

    # nx.balanced_tree uses integer node IDs; stringify them
    for node in raw.nodes:
        G.add_node(str(node), label=str(node))

    # BFS from root=0 to assign up/down labels
    for parent, child in nx.bfs_edges(raw, source=0):
        p, c = str(parent), str(child)
        G.add_edge(p, c, label="down", traversable=True)
        G.add_edge(c, p, label="up", traversable=True)

    return WorldGraph(G)


def _build_complete(n_nodes: int, **kwargs) -> WorldGraph:
    """All nodes connected to all others. Edge label = target node name."""
    n = max(2, n_nodes)
    G = nx.DiGraph()

    for i in range(n):
        G.add_node(str(i), label=str(i))

    for i in range(n):
        for j in range(n):
            if i != j:
                G.add_edge(str(i), str(j), label=str(j), traversable=True)

    return WorldGraph(G)


def _build_bipartite(n_nodes: int, n1: int = 0, n2: int = 0, **kwargs) -> WorldGraph:
    """Complete bipartite graph. If n1/n2 not given, split n_nodes in half."""
    if n1 == 0 or n2 == 0:
        n1 = max(1, n_nodes // 2)
        n2 = max(1, n_nodes - n1)

    G = nx.DiGraph()

    for i in range(n1):
        G.add_node(f"A{i}", label=f"A{i}", partition="A")
    for j in range(n2):
        G.add_node(f"B{j}", label=f"B{j}", partition="B")

    for i in range(n1):
        for j in range(n2):
            _add_bidirectional(G, f"A{i}", f"B{j}", f"B{j}", traversable=True)
            # Set reverse label explicitly
            G[f"B{j}"][f"A{i}"]["label"] = f"A{i}"

    return WorldGraph(G)


def _build_small_world(
    n_nodes: int, k: int = 4, p: float = 0.1, seed: int | None = None, **kwargs
) -> WorldGraph:
    """Watts-Strogatz small-world graph. Edge label = target node name."""
    raw = nx.watts_strogatz_graph(max(k + 1, n_nodes), k, p, seed=seed)
    G = nx.DiGraph()

    for node in raw.nodes:
        G.add_node(str(node), label=str(node))

    for u, v in raw.edges:
        _add_bidirectional(G, str(u), str(v), str(v), traversable=True)
        G[str(v)][str(u)]["label"] = str(u)

    return WorldGraph(G)


def _build_scale_free(
    n_nodes: int, m: int = 2, seed: int | None = None, **kwargs
) -> WorldGraph:
    """Barabási-Albert scale-free graph. Edge label = target node name."""
    raw = nx.barabasi_albert_graph(max(m + 1, n_nodes), m, seed=seed)
    G = nx.DiGraph()

    for node in raw.nodes:
        G.add_node(str(node), label=str(node))

    for u, v in raw.edges:
        _add_bidirectional(G, str(u), str(v), str(v), traversable=True)
        G[str(v)][str(u)]["label"] = str(u)

    return WorldGraph(G)


def _build_random(
    n_nodes: int, p: float = 0.15, seed: int | None = None, **kwargs
) -> WorldGraph:
    """Erdős-Rényi random graph. Edge label = target node name."""
    raw = nx.erdos_renyi_graph(max(2, n_nodes), p, seed=seed)
    G = nx.DiGraph()

    for node in raw.nodes:
        G.add_node(str(node), label=str(node))

    for u, v in raw.edges:
        _add_bidirectional(G, str(u), str(v), str(v), traversable=True)
        G[str(v)][str(u)]["label"] = str(u)

    return WorldGraph(G)


def _build_star(n_nodes: int, **kwargs) -> WorldGraph:
    """Star topology. One central "hub" node connected to all spokes "0"…"n-2".
    Spoke nodes are only connected to the hub; hub label on return edge is "hub"."""
    G = nx.DiGraph()

    G.add_node("hub", label="hub")
    for i in range(n_nodes - 1):
        G.add_node(str(i), label=str(i))

    for i in range(n_nodes - 1):
        G.add_edge("hub", str(i), label=str(i), traversable=True)
        G.add_edge(str(i), "hub", label="hub", traversable=True)

    return WorldGraph(G)


def _build_wheel(n_nodes: int, **kwargs) -> WorldGraph:
    """Wheel topology. Ring of "0"…"n-2" nodes plus a central "hub" connected to all.
    Ring edges use clockwise/counterclockwise; hub↔spoke edges use spoke name / "hub"."""
    if n_nodes < 4:
        raise ValueError("Wheel topology requires at least 4 nodes (1 hub + 3 ring)")
    n = max(4, n_nodes)  # need at least 3 ring nodes + 1 hub
    ring_count = n - 1

    G = nx.DiGraph()

    G.add_node("hub", label="hub")
    for i in range(ring_count):
        G.add_node(str(i), label=str(i))

    for i in range(ring_count):
        _add_bidirectional(
            G, str(i), str((i + 1) % ring_count), "clockwise", traversable=True
        )

    for i in range(ring_count):
        G.add_edge("hub", str(i), label=str(i), traversable=True)
        G.add_edge(str(i), "hub", label="hub", traversable=True)

    return WorldGraph(G)


def _build_from_file(path: str, **kwargs) -> WorldGraph:
    """Load graph from a JSON file.

    Expected format:
    {
        "nodes": [{"id": "node_a", ...attrs}, ...],
        "edges": [{"from": "node_a", "to": "node_b", "label": "east"}, ...]
    }

    Nodes may include an "affordances" list of dicts. Each is validated and
    converted to a LocationAffordance at load time — missing required fields
    will raise a ValueError immediately with a description of what is missing.
    """
    from terralingua.environment.actions import LocationAffordance

    with open(path) as f:
        data = json.load(f)

    G = nx.DiGraph()
    wg = WorldGraph(G)

    for node in data.get("nodes", []):
        node = dict(node)  # don't mutate the parsed JSON
        node_id = node.pop("id")
        affordances = node.pop("affordances", [])
        node.setdefault("label", node_id)
        G.add_node(node_id, **node)
        for aff_dict in affordances:
            try:
                wg.add_node_affordance(node_id, LocationAffordance.from_dict(aff_dict))
            except Exception as e:
                raise ValueError(f"Node '{node_id}': {e}") from e

    for edge in data.get("edges", []):
        u = edge["from"]
        v = edge["to"]
        label = edge.get("label", v)
        attrs = {k: edge[k] for k in edge if k not in ("from", "to", "label")}
        attrs.setdefault("traversable", True)
        G.add_edge(u, v, label=label, **attrs)

    return wg


def _build_agent_network(
    path: str, bidirectional: bool = False, **kwargs
) -> WorldGraph:
    """Build a graph from a Neuro-SAN agent-network HOCON file.

    Each local agent/tool becomes a node. Each local name in an agent's HOCON
    ``tools`` list becomes a directed edge from the declaring agent to the
    downstream agent.
    """
    from terralingua.experiment.neuro_san_hocon import load_neuro_san_agent_network

    data = load_neuro_san_agent_network(path).graph_data(bidirectional=bidirectional)
    G = nx.DiGraph()
    wg = WorldGraph(G)

    for node in data.get("nodes", []):
        node = dict(node)
        node_id = node.pop("id")
        affordances = node.pop("affordances", [])
        node.setdefault("label", node_id)
        G.add_node(node_id, **node)
        for aff_dict in affordances:
            try:
                from terralingua.environment.actions import LocationAffordance

                wg.add_node_affordance(node_id, LocationAffordance.from_dict(aff_dict))
            except Exception as e:
                raise ValueError(f"Node '{node_id}': {e}") from e

    for edge in data.get("edges", []):
        u = edge["from"]
        v = edge["to"]
        label = edge.get("label", v)
        attrs = {k: edge[k] for k in edge if k not in ("from", "to", "label")}
        attrs.setdefault("traversable", True)
        G.add_edge(u, v, label=label, **attrs)

    return wg


def _build_custom(factory: str, **kwargs) -> WorldGraph:
    """Call a user-defined factory function (dotted import path).
    The function must return a WorldGraph or nx.DiGraph."""
    module_path, _, fn_name = factory.rpartition(".")
    module = importlib.import_module(module_path)
    fn = getattr(module, fn_name)
    result = fn(**kwargs)
    if isinstance(result, WorldGraph):
        return result
    if isinstance(result, nx.DiGraph):
        return WorldGraph(result)
    raise TypeError(
        f"Custom factory must return WorldGraph or nx.DiGraph, got {type(result)}"
    )


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------

_BUILDERS = {
    "grid": _build_grid,
    "ring": _build_ring,
    "tree": _build_tree,
    "complete": _build_complete,
    "bipartite": _build_bipartite,
    "small_world": _build_small_world,
    "scale_free": _build_scale_free,
    "random": _build_random,
    "star": _build_star,
    "wheel": _build_wheel,
    "file": _build_from_file,
    "agent_network": _build_agent_network,
    "custom": _build_custom,
}

AVAILABLE_TOPOLOGIES = list(_BUILDERS.keys())


def build_graph(
    topology: str,
    n_nodes: int = 25,
    node_props: dict | None = None,
    **params,
) -> WorldGraph:
    """Build a WorldGraph from a named topology and parameters.

    Args:
        topology:   One of AVAILABLE_TOPOLOGIES.
        n_nodes:    Primary size parameter (meaning varies per topology).
        node_props: Optional mapping of ``{node_id: {attr: value, ...}}`` applied
                    after the graph is built.  Use this to attach affordances or
                    any other metadata to specific nodes without editing the
                    builder, e.g.::

                        build_graph("star", n_nodes=10, node_props={
                            "hub": {"affordances": [
                                {"action": "energy_boost",
                                 "description": "Absorb energy from the hub.",
                                 "params": {},
                                 "effect": {"type": "add_energy", "amount": 20}},
                            ]},
                        })

        **params:   Topology-specific parameters (e.g. wrap=True for grid,
                    k=4 p=0.1 for small_world, path="..." for file).

    Returns:
        A WorldGraph ready for use in OpenGraphWorld.
    """
    if topology not in _BUILDERS:
        raise ValueError(
            f"Unknown topology '{topology}'. Choose from: {AVAILABLE_TOPOLOGIES}"
        )
    wg = _BUILDERS[topology](n_nodes=n_nodes, **params)
    if node_props:
        for node_id, attrs in node_props.items():
            wg.set_node_attr(node_id, **attrs)
    return wg
