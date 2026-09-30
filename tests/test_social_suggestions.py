"""Directed two-hop suggestions preserve global discovery and eligibility."""

import numpy as np
import pytest

from tests.test_social_graph_env import make_social_env, seed_n_agents


@pytest.fixture
def suggestion_world(tmp_path):
    names = ["Self", "Bridge"] + [f"Near{i}" for i in range(8)] + [
        f"Far{i}" for i in range(20)
    ]
    env = make_social_env(tmp_path, n_nodes=len(names))
    seed_n_agents(env, names)
    clear_edges(env)
    connect(env, "a0", "a1")
    for i in range(2, 10):
        connect(env, "a1", f"a{i}")
    original_rng = env.rng
    yield env
    env.rng = original_rng
    env.close()


def clear_edges(env):
    for node in env.world_graph.all_nodes():
        for other in list(env.world_graph.neighbors(node)):
            env._drop_outgoing_edge(node, other)


def connect(env, source, target):
    env.world_graph.add_edge(env.agent_pos[source], env.agent_pos[target])


def suggest(env, *, count=3, seed=0):
    env.random_intro_k = count
    env.rng = np.random.default_rng(seed)
    return env._suggest_agents("a0")


def test_default_has_two_two_hop_choices_and_can_explore(suggestion_world):
    env = suggestion_world
    # Exactly two eligible two-hop neighbors leaves one slot for discovery.
    for i in range(4, 10):
        env.world_graph.remove_edge(env.agent_pos["a1"], env.agent_pos[f"a{i}"])
    preferred = {"Near0", "Near1"}
    results = [suggest(env, seed=seed) for seed in range(20)]
    for result in results:
        assert len(result) == len(set(result)) == 3
        assert preferred <= set(result)
        assert len(set(result) - preferred) == 1
    # The preferred selections do not permanently occupy the first two ranks.
    assert any(result[0] not in preferred for result in results)


@pytest.mark.parametrize("count,quota", [(2, 1), (5, 3), (12, 8)])
def test_bias_scales_with_total_suggestion_count(suggestion_world, count, quota):
    env = suggestion_world
    preferred = {f"Near{i}" for i in range(8)}
    local_counts = []
    for seed in range(30):
        result = suggest(env, count=count, seed=seed)
        assert len(result) == len(set(result)) == count
        local_counts.append(len(set(result) & preferred))
    assert min(local_counts) >= quota
    # With plenty of distant candidates, the unreserved slots really are global
    # discovery slots, rather than a larger fixed local quota.
    assert min(local_counts) == quota


def test_two_hop_paths_follow_outgoing_direction(suggestion_world):
    env = suggestion_world
    clear_edges(env)
    connect(env, "a0", "a1")
    connect(env, "a1", "a2")
    connect(env, "a1", "a3")
    # Self <- Far0 -> Far1 has undirected distance two, but no outgoing path.
    connect(env, "a10", "a0")
    connect(env, "a10", "a11")
    # Near2 -> Bridge -> Self is a reverse-only two-hop path.
    connect(env, "a4", "a1")
    connect(env, "a1", "a0")
    preferred = {"Near0", "Near1"}
    results = [set(suggest(env, seed=seed)) for seed in range(20)]
    assert all(preferred <= result for result in results)
    assert any("Far1" not in result and "Near2" not in result for result in results)


def test_eligibility_excludes_self_followed_outgoing_pending_and_dead(suggestion_world):
    env = suggestion_world
    # A direct edge excludes a candidate even when a two-hop path also exists.
    connect(env, "a0", "a2")
    env._pending_outgoing["a0"] = {"a3": 0}
    env._pending_incoming["a3"] = {"a0"}
    # Incoming-only followers and requests remain eligible, as before.
    connect(env, "a10", "a0")
    env._pending_outgoing["a11"] = {"a0": 0}
    env._pending_incoming["a0"] = {"a11"}
    env._kill("a4")
    env.world_graph.add_node("unoccupied")
    env.world_graph.add_edge(env.agent_pos["a1"], "unoccupied")
    result = suggest(env, count=100)
    excluded = {"Self", "Bridge", "Near0", "Near1", "Near2"}
    expected = set(env.agent_names.values()) - excluded
    assert set(result) == expected
    assert len(result) == len(expected)
    assert {"Far0", "Far1"} <= set(result)
    assert "unoccupied" not in result


@pytest.mark.parametrize("remaining", [0, 1])
def test_short_two_hop_pool_fills_all_slots_from_global_pool(suggestion_world, remaining):
    env = suggestion_world
    for i in range(2 + remaining, 10):
        env.world_graph.remove_edge(env.agent_pos["a1"], env.agent_pos[f"a{i}"])
    for seed in range(10):
        result = suggest(env, count=6, seed=seed)
        assert len(result) == len(set(result)) == 6
        assert "Self" not in result and "Bridge" not in result
        if remaining:
            assert "Near0" in result


def test_zero_count_and_no_eligible_candidates_return_empty(suggestion_world):
    env = suggestion_world
    assert suggest(env, count=0) == []
    for tag in env.agent_registry - {"a0"}:
        connect(env, "a0", tag)
    assert suggest(env, count=3) == []


class ControlledRNG:
    """Control the single-slot mixture without probabilistic assertions."""

    def __init__(self, probability):
        self.probability = probability

    def random(self):
        return self.probability

    def choice(self, values, size=None, replace=True):
        values = list(range(values)) if isinstance(values, (int, np.integer)) else list(values)
        if size is None:
            return values[0]
        return np.asarray(values[:size])

    def shuffle(self, values):
        values.reverse()


@pytest.mark.parametrize(
    "probability,expected",
    [(2 / 3 - 0.0001, "Near0"), (2 / 3, "Far0")],
)
def test_one_slot_mixes_two_hop_preference_and_global_discovery(suggestion_world, probability, expected):
    env = suggestion_world
    env.random_intro_k = 1
    env.rng = ControlledRNG(probability)
    # Sorted tags place a10 (Far0) first globally, a2 (Near0) first locally.
    assert env._suggest_agents("a0") == [expected]


def test_one_slot_local_branch_falls_back_when_no_two_hop_neighbor(suggestion_world):
    env = suggestion_world
    for i in range(2, 10):
        env.world_graph.remove_edge(env.agent_pos["a1"], env.agent_pos[f"a{i}"])
    env.random_intro_k = 1
    env.rng = ControlledRNG(0.0)
    assert env._suggest_agents("a0") == ["Far0"]


class ReverseIterationSet(set):
    def __iter__(self):
        return iter(sorted(super().__iter__(), reverse=True))


def test_seeded_results_do_not_depend_on_registry_iteration_order(suggestion_world):
    env = suggestion_world
    expected = suggest(env, count=6, seed=123)
    env.agent_registry = ReverseIterationSet(env.agent_registry)
    assert suggest(env, count=6, seed=123) == expected
