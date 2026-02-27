import os
import pickle as pkl
import traceback
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from graph_utils import (
    build_graph,
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

from core.utils import ROOT
from core.utils.analysis_utils import get_last_ts

"""
This script processes the logs of experiments to build interaction graphs between agents, 
compute various graph and community metrics, and save the results for further analysis. 

For each experiment, it performs the following steps:
1. Load the interaction graph and contribution data from the logs.
2. Compute basic graph statistics (number of nodes, edges, density, reciprocity, 
    cliques, communities).
3. Compute community-level metrics (intra-community weight, inter-community weight, 
    intra-community share).
4. Compute time-based metrics by sliding a window over the experiment timeline 
    and analyzing the graph within each window (modularity, partition entropy, 
    participation stats, negative cuts, average degree, clustering, entropy).
5. Save the computed graph and metrics to a pickle file for later use.
"""

EXPERIMENTS_NAMES = []  # e.g. ['core_run', 'scarcity_run', ...]


def main(exp_name):
    EXP_PATH = ROOT / "logs" / exp_name
    out_dir = EXP_PATH
    if not out_dir.exists():
        os.makedirs(out_dir, exist_ok=True)

    G, contrib_df = build_graph(EXP_PATH)

    E = edge_df(G)

    # Get basic graph stats
    # ---------------------------
    print("Getting basic graph stats")
    graph_data = {}
    graph_data["nodes"] = list(G.nodes())
    graph_data["edges"] = G.number_of_edges()
    graph_data["positive_edges"] = (E["weight"] > 0).sum()
    graph_data["negative_edges"] = (E["weight"] < 0).sum()
    graph_data["density"] = nx.density(G)
    graph_data["reciprocity"] = reciprocity(G)
    graph_data["positive_reciprocity"] = reciprocity(positive_subgraph(G))
    graph_data["negative_reciprocity"] = reciprocity(negative_subgraph(G))
    graph_data["cliques"] = list(nx.find_cliques(G.to_undirected()))
    graph_data["num_cliques"] = len(graph_data["cliques"])
    comms, node2com = get_slpa_communities(G)
    print(f"Found {len(comms)} communities")
    print(f"Sizes: {[len(c) for c in comms]}")
    graph_data["communities"] = comms
    graph_data["node2com"] = node2com
    graph_data["negative_cuts"] = negative_cut_edges(G, comms)
    # ---------------------------

    # Get community metrics
    # ---------------------------
    print("Getting community metrics")
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
    # ---------------------------

    # Get metrics with time
    # ---------------------------
    print("Getting time-based metrics")
    time_metrics = defaultdict(list)
    exp_len = get_last_ts(EXP_PATH / "open_gridworld.log") + 1
    for t0, t1 in sliding_windows(exp_len, 100, 50):
        GT = graph_window(G, contrib_df=contrib_df, t0=t0, t1=t1)

        UGT = positive_subgraph(GT, directed=False, keep_isolates=False)
        avg_degree, avg_clustering, entropy = graph_complexity_metrics(GT)
        commsT, node2comT = get_slpa_communities(GT)
        try:
            mod = compute_modularity(UGT, comms=commsT)
        except:
            mod = 0
        H, Hn, K = compute_partition_entropy(UGT, comms=commsT)
        mean_P, std_P = compute_participation_stats(UGT, commsT, node2comT)
        tot_neg, pol_neg = negative_cuts_stats(
            GT, node2comT
        )  # (total |neg| across communities, polarization share)

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
    time_metrics = (
        pd.DataFrame.from_dict(time_metrics).sort_values("T").reset_index(drop=True)
    )
    graph_data["time_metrics"] = time_metrics
    # ---------------------------

    # Save outputs
    print("Saving outputs")
    with open(out_dir / "graph.pkl", "wb") as f:
        all_data = {
            "graph": G,
            "contrib_df": contrib_df,
            "graph_data": graph_data,
        }
        pkl.dump(all_data, f)


if __name__ == "__main__":
    failed = []
    for exp_name in EXPERIMENTS_NAMES:
        try:
            print("Processing", exp_name)
            main(exp_name)
            print(f"Done with {exp_name}")
            print()
        except Exception as e:
            failed.append(exp_name)
            print(f"Experiment {exp_name} failed: {e}")
            print("Stack trace:")
            traceback.print_exc()
            print()

    if len(failed) > 0:
        print("The following experiments failed:")
        for exp in failed:
            print(f"- {exp}")
