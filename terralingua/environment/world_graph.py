import logging
from typing import Dict, List, Tuple

import networkx as nx

from terralingua.environment.actions import LocationAffordance

log = logging.getLogger(__name__)


class WorldGraph:
    """Directed graph of world locations.

    Uses nx.DiGraph so each traversal direction carries its own label
    (e.g. "north" for A→B, "south" for B→A on a grid).
    """

    def __init__(self, G: nx.DiGraph) -> None:
        self._G = G

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def neighbors(self, node: str) -> List[str]:
        """Nodes reachable in one step from *node* (outgoing edges)."""
        return list(self._G.successors(node))

    def predecessors(self, node: str) -> List[str]:
        """Nodes with an edge pointing into *node* (incoming edges)."""
        return list(self._G.predecessors(node))

    def nodes_within_hops(self, node: str, hops: int | None) -> Dict[str, int]:
        """BFS from *node*. Returns {node_id: hop_distance} within *hops* steps (None: no limit)."""
        return dict(nx.single_source_shortest_path_length(self._G, node, cutoff=hops))

    def hop_distance(self, u: str, v: str) -> float:
        """Edges on the shortest directed path from u to v. inf when v is unreachable."""
        try:
            return float(nx.shortest_path_length(self._G, u, v))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return float("inf")

    def edges_from(self, node: str) -> List[Tuple[str, dict]]:
        """Returns [(neighbor_id, edge_attrs)] for all outgoing edges from *node*."""
        return [(nbr, dict(data)) for nbr, data in self._G[node].items()]

    def has_edge(self, u: str, v: str) -> bool:
        return self._G.has_edge(u, v)

    def all_nodes(self) -> List[str]:
        return list(self._G.nodes)

    def node_count(self) -> int:
        return len(self._G)

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def add_node(self, node_id: str, **attrs) -> None:
        self._G.add_node(node_id, **attrs)

    def remove_node(self, node_id: str) -> None:
        if node_id in self._G:
            self._G.remove_node(node_id)

    def add_edge(self, u: str, v: str, **attrs) -> None:
        """Add a single directed edge u→v with given attributes."""
        self._G.add_edge(u, v, **attrs)

    def remove_edge(self, u: str, v: str) -> None:
        """Remove directed edge u→v (one direction only)."""
        if self._G.has_edge(u, v):
            self._G.remove_edge(u, v)

    # ------------------------------------------------------------------
    # Node metadata
    # ------------------------------------------------------------------

    def node_metadata(self, node: str) -> dict:
        return dict(self._G.nodes[node])

    def node_affordances(self, node: str) -> list:
        """Return the affordances for *node* as a list of LocationAffordance.

        Raw entries stored on the node can be either LocationAffordance instances
        (set programmatically) or plain dicts (loaded from JSON); both are
        normalised to LocationAffordance here so callers never need to handle
        both cases.

        Set affordances via ``set_node_attr(node, affordances=[...])``,
        the JSON file loader (add an ``"affordances"`` key to the node dict),
        or ``build_graph(..., node_props={"node_id": {"affordances": [...]}})``
        """
        raw = self._G.nodes[node].get("affordances", [])
        return [
            entry
            if isinstance(entry, LocationAffordance)
            else LocationAffordance.from_dict(entry)
            for entry in raw
        ]

    def set_node_attr(self, node: str, **attrs) -> None:
        if not self._G.has_node(node):
            log.warning(
                "WARNING: Node %s does not exist in graph. Cannot set attributes.", node
            )
            return
        for k, v in attrs.items():
            self._G.nodes[node][k] = v

    def add_node_affordance(self, node: str, affordance: LocationAffordance):
        if not self._G.has_node(node):
            log.warning(
                "WARNING: Node %s does not exist in graph. Cannot add affordance.", node
            )
            return
        if "affordances" not in self._G.nodes[node]:
            self._G.nodes[node]["affordances"] = []
        self._G.nodes[node]["affordances"].append(affordance)

    def remove_node_affordance(self, node: str, action: str) -> bool:
        """Drop every affordance for *action* on *node*. Returns True if any was removed.

        Raw entries may be LocationAffordance instances or plain dicts (JSON
        loaded); both are matched on their action name.
        """
        if not self._G.has_node(node):
            return False
        raw = self._G.nodes[node].get("affordances", [])
        kept = [
            entry
            for entry in raw
            if (entry.action if isinstance(entry, LocationAffordance) else entry.get("action"))
            != action
        ]
        self._G.nodes[node]["affordances"] = kept
        return len(kept) != len(raw)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def serialize(self) -> dict:
        return nx.node_link_data(self._G, edges="edges")

    @classmethod
    def deserialize(cls, data: dict) -> "WorldGraph":
        G = nx.node_link_graph(data, directed=True, edges="edges")
        return cls(G)

    def __len__(self) -> int:
        return len(self._G)

    def __repr__(self) -> str:
        return f"WorldGraph(nodes={len(self._G)}, edges={self._G.number_of_edges()})"
