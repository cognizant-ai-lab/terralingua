#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interaction graph for broadcast messaging:
- Messages: ONLY from observation.message (sender → current agent)
- Energy give/take and parenthood from actions

Outputs:
- out/edges.csv    (source,target,weight,types_json,sign,t_min,t_max)
- out/graph.graphml
- out/graph.gexf
"""

import json
import math
import os
import pickle as pkl
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import networkx.algorithms.community as nxcom
import numpy as np
import pandas as pd
from cdlib import algorithms

from core.utils import ROOT
from core.utils.analysis_utils import load_agent_log, load_worldlog

# TODO add sent analysis of messages (to differentiate between positive and negative interactions)

DEFAULT_ALPHA_MSG = 0.5  # per observed message (broadcast)
DEFAULT_ALPHA_SEEN = 0.1  # per observed agent
DEFAULT_ALPHA_GIVE = 1.0  # per energy unit
DEFAULT_ALPHA_STEAL = -1.0  # per energy unit (negative)
DEFAULT_ALPHA_PARENT = 10.0  # per parent→child link
DEFAULT_ALPHA_GIVE_ARTIFACT = 5.0


def find_agents_heard(
    observation: Dict[str, List[str]], name_to_tags: Dict[str, List[str]]
):
    senders = set()
    for sender in observation.keys():
        senders.update(name_to_tags.get(sender, ()))
    return senders


def find_agents_seen(
    observation: Dict[str, List[str]], name_to_tags: Dict[str, List[str]]
):
    """Find all the agents in the observation"""
    seen = set()
    for objects in observation.values():
        for obj in objects:
            # Remove the color parenthesis
            # It takes into account the presence of other color parenthesis
            if "(" in obj:
                name = "(".join(obj.split("(")[:-1])
            else:
                name = obj
            tags = name_to_tags.get(name, ())
            seen.update(tags)
    return seen


def _touch_time(edge_times, key, step):
    if key not in edge_times:
        edge_times[key] = [step, step]
    else:
        edge_times[key][0] = min(edge_times[key][0], step)
        edge_times[key][1] = max(edge_times[key][1], step)


def build_graph(
    run_dir: Path,
    alpha_msg=DEFAULT_ALPHA_MSG,
    alpha_seen=DEFAULT_ALPHA_SEEN,
    alpha_give=DEFAULT_ALPHA_GIVE,
    alpha_steal=DEFAULT_ALPHA_STEAL,
    alpha_parent=DEFAULT_ALPHA_PARENT,
    alpha_give_art=DEFAULT_ALPHA_GIVE_ARTIFACT,
):
    "Returns the graph and the contribution over time. Useful for sliding window subgraph calculations"
    agent_names = json.load(open(run_dir / "agent_names.json", "r"))
    agent_logs_dir = run_dir / "agent_logs"

    edge_weights = defaultdict(float)
    breakdown = defaultdict(lambda: defaultdict(float))
    edge_times = {}  # (u,v) -> [t_min, t_max]
    contribs = defaultdict(list)

    # Handles multiple agents with the same name, as agents do not know other agents names.
    name_to_tags = defaultdict(set)
    for tag, name in agent_names.items():
        name_to_tags[name].add(tag)

    for agent_tag, agent_name in agent_names.items():
        agent_log_path = agent_logs_dir / f"{agent_tag}.jsonl"
        agent_log = load_agent_log(agent_log_path, reduce=False)
        world_log = load_worldlog(
            run_dir / "open_gridworld.log", agent_name=agent_name, agent_tag=agent_tag
        )

        # Get messages from agent_log
        for step, data_point in agent_log.items():
            # If it's a list, just take the first dict for observation message/seens.
            if isinstance(data_point, list):
                # pick the first that has observation
                rec = next(
                    (
                        r
                        for r in data_point
                        if isinstance(r, dict) and "observation" in r
                    ),
                    data_point[0],
                )
                obs = rec.get("observation") or {}
            else:
                obs = data_point.get("observation") or {}

            # For each sender listed in observation.message, add sender -> current agent
            senders = find_agents_heard(
                observation=obs.get("message", {}),
                name_to_tags=name_to_tags,  # type: ignore
            )

            seen = find_agents_seen(
                observation=obs.get("observation", {}),
                name_to_tags=name_to_tags,  # type: ignore
            )

            assert senders <= seen, (
                f"Error agent {agent_name}({agent_tag}) at TS: {step}: \n\t Message senders {senders} not seen {seen}"
            )

            for u in senders:
                if u == agent_tag:
                    continue
                edge_weights[(u, agent_tag)] += alpha_msg
                breakdown[(u, agent_tag)]["message_observed"] += 1.0
                _touch_time(edge_times, (u, agent_tag), int(step))
                contribs[(u, agent_tag)].append(
                    (int(step), float(alpha_msg), "message_observed")
                )

            for u in seen:
                if u == agent_tag:
                    continue
                edge_weights[(u, agent_tag)] += alpha_seen
                breakdown[(u, agent_tag)]["agent_seen"] += 1.0
                _touch_time(edge_times, (u, agent_tag), int(step))
                contribs[(u, agent_tag)].append(
                    (int(step), float(alpha_seen), "agent_seen")
                )

        # World events where THIS agent is the actor
        for data_point in world_log:
            evt_step = int(data_point.get("timestamp", data_point.get("step", 0)))

            if data_point.get("event") == "GIFT_ENERGY" and agent_tag == data_point.get(
                "agent_tag"
            ):
                target_tag = data_point.get("target_tag")
                amount = data_point.get("amount")
                if target_tag and amount:
                    u, v = agent_tag, str(target_tag).lower()
                    w = alpha_give * float(amount)
                    edge_weights[(u, v)] += w
                    breakdown[(u, v)]["give_energy"] += float(amount)
                    _touch_time(edge_times, (u, v), evt_step)
                    contribs[(u, v)].append((evt_step, float(w), "give_energy"))

            if data_point.get("event") == "TAKE_ENERGY" and agent_tag == data_point.get(
                "agent_tag"
            ):
                target_tag = data_point.get("target_tag")
                amount = data_point.get("amount")
                if target_tag and amount:
                    u, v = agent_tag, str(target_tag).lower()
                    w = alpha_steal * float(amount)
                    edge_weights[(u, v)] += w
                    breakdown[(u, v)]["take_energy"] += float(amount)
                    _touch_time(edge_times, (u, v), evt_step)
                    contribs[(u, v)].append((evt_step, float(w), "take_energy"))

            if data_point.get(
                "event"
            ) == "AGENT_REPRODUCED" and agent_tag == data_point.get("agent_tag"):
                child_tag = data_point.get("child_tag")
                if child_tag:
                    u, v = agent_tag, str(child_tag).lower()
                    edge_weights[(u, v)] += alpha_parent
                    breakdown[(u, v)]["parent_of"] += 1.0
                    _touch_time(edge_times, (u, v), evt_step)
                    contribs[(u, v)].append(
                        (evt_step, float(alpha_parent), "parent_of")
                    )

            if data_point.get(
                "event"
            ) == "GIVE_ARTIFACT" and agent_tag == data_point.get("agent_tag"):
                target_tag = data_point.get("target_tag")
                status = data_point.get("status")
                if target_tag and status == "Success":
                    u, v = agent_tag, str(target_tag).lower()
                    edge_weights[(u, v)] += alpha_give_art
                    breakdown[(u, v)]["give_artifact"] += 1.0
                    _touch_time(edge_times, (u, v), evt_step)
                    contribs[(u, v)].append(
                        (evt_step, float(alpha_give_art), "give_artifact")
                    )

    G = nx.DiGraph()
    # attach per-edge events
    for (u, v), w in edge_weights.items():
        tmin, tmax = edge_times.get((u, v), (None, None))
        sign = 1 if w > 0 else (-1 if w < 0 else 0)
        events = contribs.get((u, v), [])
        G.add_edge(
            u,
            v,
            weight=float(w),
            sign=sign,
            t_min=tmin,
            t_max=tmax,
            types=dict(breakdown[(u, v)]),
            events=events,  # <— new
        )

    # Flat DataFrame for fast windowing
    Eev = (
        pd.DataFrame(
            [
                (u, v, t, dw, typ)
                for (u, v), lst in contribs.items()
                for (t, dw, typ) in lst
            ],
            columns=["source", "target", "t", "dw", "typ"],
        )
        .sort_values("t")
        .reset_index(drop=True)
    )
    return G, Eev


def graph_window(G: nx.DiGraph, contrib_df: pd.DataFrame, t0, t1):
    """Returns the graph between two time intervals

    Args:
        nodes (_type_): Graph
        Eev (_type_): Dataframe of contributivies returned by build_graph
        t0 (_type_): _description_
        t1 (_type_): _description_

    Returns:
        _type_: _description_
    """
    sel = contrib_df[(contrib_df.t >= t0) & (contrib_df.t <= t1)]
    DG = nx.DiGraph()
    DG.add_nodes_from(G)
    for (u, v), grp in sel.groupby(["source", "target"], sort=False):
        w = grp.dw.sum()
        s = 1 if w > 0 else (-1 if w < 0 else 0)
        DG.add_edge(
            u,
            v,
            weight=float(w),
            sign=s,
            t_min=int(grp.t.min()),
            t_max=int(grp.t.max()),
        )
    return DG


def get_leuven_communities(G: nx.DiGraph):
    """Finds group of agents that interacted positively among themselves"""
    Gpos = positive_subgraph(G)
    UG = Gpos.to_undirected()
    comms = list(nxcom.louvain_communities(UG, weight="weight", seed=0))
    # map node -> communities id
    node2com = {n: [] for n in G.nodes()}
    for cid, members in enumerate(comms):
        for n in members:
            node2com[n].append(cid)
    # include isolated nodes that didn't appear in Gpos
    for n in G.nodes():
        node2com.setdefault(n, [-1])
    return comms, node2com


def get_slpa_communities(G, r=0.1):
    """
    This function groups the nodes of the graph using SLPA algorithm.
    Contrary to Leuven communities, this considers also negative edges.
    Params:
        G (nx.DiGraph): directed graph
        r (float, optional): threshold for SLPA. Defaults to 0.1.
    """
    # Build undirected absolute-weight graph
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    for u, v, d in G.edges(data=True):
        w = abs(d.get("weight", 1.0))
        if w == 0:
            continue
        if H.has_edge(u, v):
            H[u][v]["w"] += w
        else:
            H.add_edge(u, v, w=w)
    for u, v in list(H.edges()):
        w = H[u][v]["w"]
        w_sym = w + abs(G.get_edge_data(v, u, {}).get("weight", 0))
        H[u][v]["w"] = np.log1p(w_sym)

    # SLPA with fixed iterations T=100, only r exposed
    coms = algorithms.slpa(H, t=100, r=r)
    communities = [set(c) for c in coms.communities]

    node2com = {n: [] for n in G.nodes()}
    for cid, members in enumerate(communities):
        for n in members:
            node2com[n].append(cid)
    for n in G.nodes():
        if not node2com[n]:
            node2com[n] = [-1]
    return communities, node2com


def positive_subgraph(G: nx.DiGraph, directed=True, keep_isolates=False):
    """Returns positive subgraph of G

    Args:
        G (nx.DiGraph): directed graph
        directed (bool, optional): if True returns the directed version, otherwise the undirected one
        keep_isolates (bool, optional): if True it also returns nodes that have no edges

    Returns:
        _type_: _description_
    """
    H = G.edge_subgraph(
        [(u, v) for u, v, d in G.edges(data=True) if d.get("weight", 0) > 0]
    ).copy()
    if keep_isolates:
        H.add_nodes_from(G.nodes())
    if not directed:
        H = H.to_undirected()
    return H


def negative_subgraph(G: nx.DiGraph, directed=True, keep_isolates=False):
    """Returns negative subgraph of G

    Args:
        G (nx.DiGraph): directed graph
        directed (bool, optional): if True returns the directed version, otherwise the undirected one
        keep_isolates (bool, optional): if True it also returns nodes that have no edges

    Returns:
        _type_: _description_
    """
    H = G.edge_subgraph(
        [(u, v) for u, v, d in G.edges(data=True) if d.get("weight", 0) < 0]
    ).copy()
    if keep_isolates:
        H.add_nodes_from(G.nodes())
    if not directed:
        H = H.to_undirected()
    return H


def reciprocity(G):
    """Returns the reciprocity of the graph

    Args:
        G (nx.DiGraph): directed graph

    Returns:
        float: reciprocity of the graph
    """
    if G.number_of_edges() == 0:
        return 0.0
    return nx.reciprocity(G)


def negative_cut_edges(G, communities):
    """
    communities: list[set[str]] of nodes; return edges with negative weight crossing communities.

    These are the negative interaction edges between agents of different communities.
    can be used to identify conflicting communities
    """
    com_index = {}
    for i, c in enumerate(communities):
        for n in c:
            com_index[n] = i
    cut = []
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 0)
        if w < 0 and com_index.get(u) != com_index.get(v):
            cut.append((u, v, w))
    return cut


def edge_df(G: nx.DiGraph) -> pd.DataFrame:
    rows = []
    for u, v, d in G.edges(data=True):
        rows.append(
            {
                "source": u,
                "target": v,
                "weight": float(d.get("weight", 0.0)),
                "sign": int(d.get("sign", 0)),
                "t_min": d.get("t_min", None),
                "t_max": d.get("t_max", None),
                "types": json.dumps(d.get("types", {}), ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows)


def comm_weight_matrix(G: nx.DiGraph, node2com: Dict[str, List[int]]) -> np.ndarray:
    """
    Build CxC matrix of total weights from community i -> community j for overlapping memberships.
    For each edge u->v with weight w:
      - Let C(u) and C(v) be u and v's community lists (excluding -1).
      - Distribute w evenly over all ordered pairs (cu, cv).
    """
    # collect valid community ids
    all_coms = set()
    for cs in node2com.values():
        for c in cs:
            if c != -1:
                all_coms.add(c)
    if not all_coms:
        return np.zeros((0, 0), dtype=float)

    C = max(all_coms) + 1
    M = np.zeros((C, C), dtype=float)

    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", 0.0))
        Cu = [c for c in node2com.get(u, [-1]) if c != -1]
        Cv = [c for c in node2com.get(v, [-1]) if c != -1]
        if not Cu or not Cv:
            continue

        pairs = [(cu, cv) for cu in Cu for cv in Cv]
        share = w / len(pairs)
        for cu, cv in pairs:
            M[cu, cv] += share

    return M


def compute_modularity(UG: nx.Graph, comms) -> float:
    """
    Modularity Q measures how strongly the graph decomposes into communities
    relative to a null model with the same node strengths (weighted degrees).

    Definition (Newman-Girvan, weighted):
        Q = (1 / (2m)) * sum_{ij} [ A_ij - (k_i k_j / (2m)) ] * delta(g_i, g_j)
      where A_ij is edge weight, k_i is node i strength, m is total edge weight,
      and delta is 1 if i,j in same community else 0.

    Interpretation:
      High Q (≈ 0.3–0.8 in practice): strong within-community concentration.
      Low/near 0: structure close to random given degrees.
      Negative: fewer within-community edges than expected.

    Args:
        UG: undirected graph with nonnegative 'weight'.
        comms: optional list of node-sets. If None, Louvain is run.

    Returns:
        modularity score (float).
    """
    return nx.algorithms.community.modularity(UG, comms, weight="weight")


def compute_partition_entropy(
    UG: nx.Graph,
    comms,
    normalize_by: str = "logK",
) -> Tuple[float, float, int]:
    """
    It measures how evenly community sizes are distributed in the partition:
    high values mean many similarly sized communities, low values mean one or few dominant ones.
    Shannon entropy of the community size distribution:
        H = - sum_c p_c * ln p_c,  with p_c = |c| / N

    Interpretation:
      H reflects *diversity/balance of community sizes* (not their separation).
      - Higher H: many communities with balanced sizes (more differentiated roles).
      - Lower H: one dominant block or very skewed sizes.

    Normalization:
      - 'logK' (default): H_norm = H / ln(K)  in [0,1] when K>1
      - 'logN': H_norm = H / ln(N)
      - None: return raw H only (H_norm = H)

    Args:
        UG: undirected graph.
        comms: communities
        normalize_by: 'logK' | 'logN' | None

    Returns:
        (H_raw, H_norm, K) where K is #communities.
    """
    if UG.number_of_nodes() == 0:
        return math.nan, math.nan, 0
    K = len(comms)
    if K == 0:
        return math.nan, math.nan, 0
    N = UG.number_of_nodes()
    p = np.array([len(c) / float(N) for c in comms], dtype=float)
    p = p[p > 0]
    H = float(-(p * np.log(p)).sum())
    if normalize_by == "logK":
        Z = math.log(K) if K > 1 else 1.0
    elif normalize_by == "logN":
        Z = math.log(N) if N > 1 else 1.0
    else:
        Z = 1.0
    Hn = H / Z if Z > 0 else math.nan
    return H, float(Hn), K


def compute_participation_stats(
    UG: nx.Graph,
    comms,
    node2com,
) -> Tuple[float, float]:
    """
    It measures how evenly each node distributes its links across communities:
    low mean P → most nodes are intra-community specialists, high mean P → many nodes act as inter-community brokers.

    Participation coefficient (Guimerà & Amaral):
        For node i with strength s_i and community-strengths s_ic (sum of weights
        from i to nodes in community c):
            P_i = 1 - sum_c (s_ic / s_i)^2,  defined for s_i > 0

    Interpretation:
      - P_i ~ 0: i connects almost exclusively within its own community (specialist).
      - P_i high (approaches 1): i spreads links across many communities (broker).
      We report mean and std across nodes with s_i > 0.

    Args:
        UG: undirected graph with 'weight'.
        comms: optional communities; if None, Louvain is run.
        node2c: optional node->community mapping to avoid recomputing.

    Returns:
        (mean_P, std_P)
    """
    C = len(comms)
    if C == 0 or UG.number_of_nodes() == 0:
        return math.nan, math.nan

    P_vals: List[float] = []
    for i in UG.nodes():
        s_i = 0.0
        s_by_c = np.zeros(C, dtype=float)

        for j, d in UG[i].items():
            w = float(d.get("weight", 1.0))
            s_i += w
            Cj = [c for c in node2com.get(j, [-1]) if 0 <= c < C]
            if not Cj:
                continue
            share = w / len(Cj)
            for c in Cj:
                s_by_c[c] += share

        if s_i <= 0.0:
            continue
        frac2 = float(((s_by_c / s_i) ** 2).sum())
        P_i = 1.0 - frac2
        P_vals.append(P_i)

    if not P_vals:
        return math.nan, math.nan

    P_vals_array = np.array(P_vals, dtype=float)
    return float(P_vals_array.mean()), float(P_vals_array.std())


def negative_cuts_stats(
    GT: nx.DiGraph, node2com: Dict[str, List[int]]
) -> Tuple[float, float]:
    """
    Compute stats of negative-weight edges that cross communities with overlapping memberships.

    For each negative edge u->v, let C(u) and C(v) be the community lists for u and v.
    Distribute |w| evenly across all ordered pairs (cu, cv) with cu ∈ C(u), cv ∈ C(v), cu != cv.
    Nodes with community [-1] are treated as unassigned and ignored.

    Returns:
        total: Sum of absolute weights of negative edges that run between communities.
        polarization: Share of 'total' held by the single most hostile ordered community pair.
    """
    total = 0.0
    by_pair: Dict[Tuple[int, int], float] = {}

    for u, v, d in GT.edges(data=True):
        w = float(d.get("weight", 0.0))
        if w >= 0:
            continue

        Cu = [c for c in node2com.get(u, [-1]) if c != -1]
        Cv = [c for c in node2com.get(v, [-1]) if c != -1]
        if not Cu or not Cv:
            continue

        # all ordered inter-community pairs
        pairs = [(cu, cv) for cu in Cu for cv in Cv if cu != cv]
        if not pairs:
            # all memberships overlap exactly -> intra-community, skip
            continue

        amt = abs(w)
        total += amt
        share = amt / len(pairs)
        for p in pairs:
            by_pair[p] = by_pair.get(p, 0.0) + share

    if total == 0.0:
        return 0.0, 0.0

    polarization = max(by_pair.values()) / total
    return total, polarization


def graph_complexity_metrics(G: nx.DiGraph):
    """
    measures:
    - Basic measure of connection density. Higher average degree indicates more locally complex connectivity.
    - Measures the prevalence of triangles. Captures local cohesiveness; high values indicate clustered structure.
    - Captures heterogeneity of node connectivity; higher entropy → more irregular structure
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()

    avg_degree = 2 * m / n if n > 0 else 0
    avg_clustering = nx.average_clustering(G.to_undirected())

    degrees = np.array([d for _, d in G.degree()])  # type: ignore
    if len(degrees) == 0:
        entropy = 0.0
    else:
        _, counts = np.unique(degrees, return_counts=True)
        p = counts / counts.sum()
        entropy = -np.sum(p * np.log2(p))

    return (
        avg_degree,
        avg_clustering,
        entropy,
    )


def sliding_windows(N, w, overlap):
    step = w - overlap
    windows = [(i, i + w) for i in range(0, N - w + 1, step)]
    if windows:
        last = windows[-1][1]
        if last < N:
            windows.append((last, N))
        return windows
    else:
        return [(0, N)]
