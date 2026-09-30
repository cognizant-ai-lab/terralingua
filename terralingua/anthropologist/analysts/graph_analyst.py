"""
GraphAnalyst: expert in social interaction network analysis.

Wraps graph_utils.py functions from 002_make_graph.py.
"""

import logging
import os
import pickle as pkl
from collections import defaultdict, namedtuple
from pathlib import Path

log = logging.getLogger(__name__)

import networkx as nx
import numpy as np
import pandas as pd

from terralingua.anthropologist.analysis_utils import get_last_ts
from terralingua.anthropologist.analysts.base import BaseAnalyst
from terralingua.anthropologist.graph_utils import (
    build_graph as _build_graph,
)
from terralingua.anthropologist.graph_utils import (
    comm_weight_matrix,
    compute_modularity,
    compute_participation_stats,
    compute_partition_entropy,
    edge_df,
    get_slpa_communities,
    graph_complexity_metrics,
    graph_window,
    negative_cut_edges,
    negative_cuts_stats,
    negative_subgraph,
    positive_subgraph,
    reciprocity,
    sliding_windows,
)
from terralingua.utils import ROOT

GraphData = namedtuple("GraphData", ["graph", "contrib_df", "graph_data"])


class GraphAnalyst(BaseAnalyst):
    """Expert in social interaction network analysis."""

    def build_graph(self, exp_path: Path, save: bool = True) -> GraphData:
        """
        Build interaction graph, compute all metrics, optionally save graph.pkl.

        Args:
            exp_path: Experiment directory.
            save: If True, saves results to exp_path/graph.pkl.

        Returns:
            GraphData namedtuple with (graph, contrib_df, graph_data).
        """
        exp_path = Path(exp_path)
        log.info("[GraphAnalyst] Building graph for %s", exp_path.name)

        G, contrib_df = _build_graph(exp_path)
        E = edge_df(G)

        graph_data = {}
        graph_data["nodes"] = list(G.nodes())
        graph_data["edges"] = G.number_of_edges()
        graph_data["positive_edges"] = int((E["weight"] > 0).sum())
        graph_data["negative_edges"] = int((E["weight"] < 0).sum())
        graph_data["interaction_rate"] = nx.density(G)
        graph_data["reciprocity"] = reciprocity(G)
        graph_data["positive_reciprocity"] = reciprocity(positive_subgraph(G))
        graph_data["negative_reciprocity"] = reciprocity(negative_subgraph(G))
        graph_data["cliques"] = list(nx.find_cliques(G.to_undirected()))
        graph_data["num_cliques"] = len(graph_data["cliques"])
        comms, node2com = get_slpa_communities(G)
        log.info(
            "[GraphAnalyst] Found %d communities, sizes: %s",
            len(comms), [len(c) for c in comms],
        )
        graph_data["communities"] = comms
        graph_data["node2com"] = node2com
        graph_data["negative_cuts"] = negative_cut_edges(G, comms)

        M = comm_weight_matrix(G, node2com)
        com_sizes = pd.DataFrame(
            {"community_id": range(len(comms)), "size": [len(c) for c in comms]}
        )
        intra = pd.DataFrame(
            {"community_id": np.arange(M.shape[0]), "intra_weight": np.diag(M)}
        )
        inter = pd.DataFrame(
            {
                "community_id": np.arange(M.shape[0]),
                "out_weight": (M.sum(axis=1) - np.diag(M)),
            }
        )
        community_summary = com_sizes.merge(intra, on="community_id", how="left").merge(
            inter, on="community_id", how="left"
        )
        community_summary["intra_share"] = community_summary["intra_weight"] / (
            community_summary["intra_weight"] + community_summary["out_weight"]
        ).replace({0: np.nan})
        graph_data["community_summary"] = community_summary

        log.info("[GraphAnalyst] Computing time-based metrics...")
        time_metrics = defaultdict(list)
        exp_len = get_last_ts(exp_path / "open_gridworld.log") + 1
        for t0, t1 in sliding_windows(exp_len, 100, 50):
            GT = graph_window(G, contrib_df=contrib_df, t0=t0, t1=t1)
            UGT = positive_subgraph(GT, directed=False, keep_isolates=False)
            avg_degree, avg_clustering, entropy = graph_complexity_metrics(GT)
            commsT, node2comT = get_slpa_communities(GT)
            try:
                mod = compute_modularity(UGT, comms=commsT)
            except Exception:
                mod = 0
            H, Hn, K = compute_partition_entropy(UGT, comms=commsT)
            mean_P, std_P = compute_participation_stats(UGT, commsT, node2comT)
            tot_neg, pol_neg = negative_cuts_stats(GT, node2comT)

            time_metrics["T"].append(t0)
            time_metrics["modularity"].append(mod)
            time_metrics["partition_entropy"].append(H)
            time_metrics["nomalized_partition_entropy"].append(Hn)
            time_metrics["communities"].append(K)
            time_metrics["mean_participation"].append(mean_P)
            time_metrics["std_participation"].append(std_P)
            time_metrics["neg_cuts_total"].append(tot_neg)
            time_metrics["neg_polarization"].append(pol_neg)
            time_metrics["avg_degree"].append(avg_degree)
            time_metrics["avg_clustering"].append(avg_clustering)
            time_metrics["entropy"].append(entropy)

        graph_data["time_metrics"] = (
            pd.DataFrame.from_dict(time_metrics).sort_values("T").reset_index(drop=True)
        )

        if save:
            log.info("[GraphAnalyst] Saving graph.pkl...")
            os.makedirs(exp_path, exist_ok=True)
            with open(exp_path / "graph.pkl", "wb") as f:
                pkl.dump(
                    {"graph": G, "contrib_df": contrib_df, "graph_data": graph_data}, f
                )

        return GraphData(graph=G, contrib_df=contrib_df, graph_data=graph_data)

    def build_graph_from_edge_weights(
        self,
        edge_weights: dict[tuple[str, str], float],
    ) -> GraphData:
        """
        Build a graph snapshot from a pre-aggregated edge weight dict.
        Used by the live detection accumulator in the orchestrator.

        Args:
            edge_weights: {(source, target): cumulative_weight} dict.

        Returns:
            GraphData with graph, contrib_df=None, and graph_data metrics.
        """
        G = nx.DiGraph()
        for (u, v), w in edge_weights.items():
            G.add_edge(u, v, weight=w, sign=1 if w >= 0 else -1)

        empty_gd: dict = {
            "nodes": [],
            "edges": 0,
            "positive_edges": 0,
            "negative_edges": 0,
            "interaction_rate": 0.0,
            "reciprocity": 0.0,
            "communities": [],
            "node2com": {},
            "neg_polarization": 0.0,
        }
        if G.number_of_nodes() == 0:
            return GraphData(graph=G, contrib_df=None, graph_data=empty_gd)

        E = edge_df(G)
        comms, node2com = get_slpa_communities(G)
        _, pol_neg = negative_cuts_stats(G, node2com)

        graph_data = {
            "nodes": list(G.nodes()),
            "edges": G.number_of_edges(),
            "positive_edges": int((E["weight"] > 0).sum()),
            "negative_edges": int((E["weight"] < 0).sum()),
            "interaction_rate": nx.density(G),
            "reciprocity": reciprocity(G),
            "communities": comms,
            "node2com": node2com,
            "neg_polarization": pol_neg,
        }
        return GraphData(graph=G, contrib_df=None, graph_data=graph_data)

    def detect_communities(self, graph) -> dict[int, set]:
        """
        Run SLPA community detection on a graph.

        Returns:
            Dict mapping community_idx → set of agent tags.
        """
        comms, _ = get_slpa_communities(graph)
        return {i: set(c) for i, c in enumerate(comms)}
