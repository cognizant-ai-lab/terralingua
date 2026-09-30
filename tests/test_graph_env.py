"""
Tests for terralingua/environment/graph_env.py (OpenGraphWorld).
Covers all public and internal methods.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from terralingua.config.models import GraphConfig
from terralingua.environment.actions import LocationAffordance
from terralingua.environment.graph_builder import build_graph
from terralingua.environment.graph_env import OpenGraphWorld

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_graph_env(tmp_path, topology="grid", n_nodes=25, hop_radius=2, **kwargs):
    cfg = GraphConfig(topology=topology, n_nodes=n_nodes, hop_radius=hop_radius)
    defaults = dict(
        graph_cfg=cfg,
        init_agent_energy=100,
        lifespan=200,
        init_food=25,
        max_food_value=5.0,
        food_decay_rate=0.0,
        food_spawn_rate=0,
        log_path=tmp_path,
        drop_food_on_death=False,
        use_inventory=True,
        use_colors=True,
        reproduction_cost=20,
        artifact_creation_cost=0,
        headless=True,
    )
    defaults.update(kwargs)
    return OpenGraphWorld(**defaults) # type: ignore


@pytest.fixture
def tmp_log(tmp_path):
    return tmp_path / "logs"


@pytest.fixture
def env(tmp_log):
    e = make_graph_env(tmp_log)
    all_nodes = e.world_graph.all_nodes()
    start_node = all_nodes[0]
    e.add_agent("a0", "Alice", "text", position=start_node)
    e.restart_env(agent_poses={"a0": start_node})
    return e


@pytest.fixture
def two_agent_env(tmp_log):
    e = make_graph_env(tmp_log)
    all_nodes = e.world_graph.all_nodes()
    n0, n1 = all_nodes[0], e.world_graph.neighbors(all_nodes[0])[0]
    e.add_agent("a0", "Alice", "text", position=n0)
    e.add_agent("a1", "Bob", "text", position=n1)
    e.restart_env(agent_poses={"a0": n0, "a1": n1})
    return e


# ---------------------------------------------------------------------------
# _apply_move
# ---------------------------------------------------------------------------

class TestApplyMove:
    def test_stay(self, env):
        node = env.agent_pos["a0"]
        new_node, _ = env._apply_move("a0", {"direction": "stay"}, {"a0": {}})
        assert new_node == node

    def test_move_to_invalid_target(self, env):
        node = env.agent_pos["a0"]
        new_node, infos = env._apply_move("a0", {"direction": "nonexistent_node"}, {"a0": {}})
        assert new_node == node
        assert "Move outcome" in infos["a0"]

    def test_move_none_params(self, env):
        node = env.agent_pos["a0"]
        new_node, _ = env._apply_move("a0", None, {"a0": {}})
        assert new_node == node

    def test_move_updates_pos_to_agent(self, env):
        node = env.agent_pos["a0"]
        nbr = env.world_graph.neighbors(node)[0]
        new_node, _ = env._apply_move("a0", {"direction": nbr}, {"a0": {}})
        assert new_node == nbr
        assert env.agent_pos["a0"] == nbr
        assert "a0" in env.pos_to_agent[nbr]
        assert "a0" not in env.pos_to_agent.get(node, set())


# ---------------------------------------------------------------------------
# _get_food_weights
# ---------------------------------------------------------------------------

class TestGetFoodWeights:
    def test_no_zones_uniform(self, env):
        env._food_spawn_nodes = None
        env.food_zones = None
        env._get_food_weights()
        probs = env._food_spawn_probs
        assert abs(probs.sum() - 1.0) < 1e-6
        assert np.all(probs > 0)

    def test_int_food_zones(self, tmp_log):
        e = make_graph_env(tmp_log, food_zones=2)
        e.rng = np.random.default_rng(42)
        e._get_food_weights()
        assert abs(e._food_spawn_probs.sum() - 1.0) < 1e-6
        assert len(e._food_zone_centers) == 2 # type: ignore

    def test_list_food_zones(self, tmp_log):
        e = make_graph_env(tmp_log)
        zone_node = e.world_graph.all_nodes()[0]
        e.food_zones = [zone_node]
        e.rng = np.random.default_rng(0)
        e._get_food_weights()
        assert abs(e._food_spawn_probs.sum() - 1.0) < 1e-6

    def test_zone_centers_persisted(self, tmp_log):
        e = make_graph_env(tmp_log, food_zones=2)
        e.rng = np.random.default_rng(0)
        e._get_food_weights()
        centers = list(e._food_zone_centers) # type: ignore
        e._get_food_weights()  # should reuse
        assert e._food_zone_centers == centers

    def test_invalid_zone_node_replaced(self, tmp_log):
        e = make_graph_env(tmp_log)
        e.food_zones = ["invalid_node_xyz"]
        e.rng = np.random.default_rng(0)
        e._get_food_weights()  # should not raise; replaced with random node
        assert len(e._food_zone_centers) == 1 # type: ignore

    def test_food_spawn_nodes_set(self, env):
        env._food_spawn_nodes = None
        env.food_zones = None
        env._get_food_weights()
        assert env._food_spawn_nodes is not None
        assert len(env._food_spawn_nodes) > 0


# ---------------------------------------------------------------------------
# _offspring_position
# ---------------------------------------------------------------------------

class TestOffspringPosition:
    def test_returns_neighbor_when_all_occupied(self, env):
        # Graph env has exclusive_pos_occupancy=False, so occupancy is ignored —
        # a neighbor is still returned even when all are occupied.
        node = env.agent_pos["a0"]
        for n in env.world_graph.neighbors(node):
            env.pos_to_agent[n].add("dummy")
        nbr = env._offspring_position(node)
        assert nbr in env.world_graph.neighbors(node)

    def test_returns_none_for_isolated_node(self, env):
        # The only case returning None is a node with no neighbors at all.
        env.world_graph._G.remove_edges_from(
            list(env.world_graph._G.edges(env.agent_pos["a0"]))
        )
        nbr = env._offspring_position(env.agent_pos["a0"])
        assert nbr is None


# ---------------------------------------------------------------------------
# _random_free_pos
# ---------------------------------------------------------------------------

class TestRandomFreePos:
    def test_allows_occupied_node_with_shared_occupancy(self, env):
        # fill every node — shared occupancy must still return a valid node
        for node in env.world_graph.all_nodes():
            env.pos_to_agent[node].add("dummy")
        pos = env._random_free_pos()
        assert pos in env.world_graph.all_nodes()


# ---------------------------------------------------------------------------
# _respawn_food_one
# ---------------------------------------------------------------------------

class TestRespawnFoodOne:
    def test_adds_food(self, env):
        env.food.clear()
        env._respawn_food_one()
        assert len(env.food) == 1

    def test_food_value_default(self, env):
        env.food.clear()
        env._respawn_food_one()
        val = list(env.food.values())[0]
        assert val == env._max_food_value

    def test_food_value_custom(self, env):
        env.food.clear()
        env._respawn_food_one(value=2.5)
        val = list(env.food.values())[0]
        assert val == 2.5

    def test_static_food_uses_empty_list(self, tmp_log):
        e = make_graph_env(tmp_log, static_food=True)
        e.rng = np.random.default_rng(0)
        nodes = e.world_graph.all_nodes()
        e.empty_food = [nodes[0], nodes[1]]
        e._food_spawn_nodes = nodes
        e._food_spawn_probs = np.ones(len(nodes)) / len(nodes)
        e._respawn_food_one()
        assert len(e.food) == 1
        assert len(e.empty_food) == 1

    def test_no_respawn_when_all_occupied(self, env):
        # Fill all nodes with agents or food so nothing is free
        nodes = env.world_graph.all_nodes()
        for n in nodes:
            env.food[n] = 1.0
        count_before = len(env.food)
        env._respawn_food_one()
        assert len(env.food) == count_before


# ---------------------------------------------------------------------------
# _seed_initial_food
# ---------------------------------------------------------------------------

class TestSeedInitialFood:
    def test_food_seeded(self, env):
        env._seed_initial_food()
        assert len(env.food) > 0

    def test_food_values_are_max(self, env):
        env._seed_initial_food()
        for val in env.food.values():
            assert val == env._max_food_value


# ---------------------------------------------------------------------------
# _build_obs
# ---------------------------------------------------------------------------

class TestBuildObs:
    def test_obs_keys(self, env):
        obs, _ = env._build_obs("a0")
        for key in ("observation", "current_node", "incoming_broadcasts", "energy",
                    "time", "inventory", "hop_radius"):
            assert key in obs

    def test_no_nearby_when_alone(self, env):
        _, has_nearby = env._build_obs("a0")
        assert not has_nearby

    def test_has_nearby_when_adjacent(self, two_agent_env):
        e = two_agent_env
        _, has_nearby = e._build_obs("a0")
        assert has_nearby

    def test_food_visible(self, env):
        node = env.agent_pos["a0"]
        env.food.clear()
        env.food[node] = 5.0
        obs, _ = env._build_obs("a0")
        assert node in obs["observation"]
        assert any("5.0" in s for s in obs["observation"][node]["items"])

    def test_neighbor_nodes_in_obs(self, env):
        node = env.agent_pos["a0"]
        neighborhood = env.world_graph.nodes_within_hops(node, env.hop_radius)
        obs, _ = env._build_obs("a0")
        for n in neighborhood:
            assert n in obs["observation"]

    def test_exits_in_obs_entry(self, env):
        node = env.agent_pos["a0"]
        obs, _ = env._build_obs("a0")
        assert "exits" in obs["observation"][node]

    def test_artifact_in_obs(self, env):
        node = env.agent_pos["a0"]
        env.add_artifact(
            pose=node, art_type="text", art_name="graph_sign",
            payload="hello", creator="a0", lifespan=10,
        )
        obs, _ = env._build_obs("a0")
        items = obs["observation"][node]["items"]
        assert any("graph_sign" in s for s in items)

    def test_inventory_in_obs(self, env):
        node = env.agent_pos["a0"]
        env.add_artifact(
            pose=node, art_type="text", art_name="inv_art",
            payload="data", creator="a0", lifespan=10,
        )
        env.pos_artifacts[node].discard("inv_art")
        env.agent_inventories["a0"].add("inv_art")
        obs, _ = env._build_obs("a0")
        assert any("inv_art" in s for s in obs["inventory"])


# ---------------------------------------------------------------------------
# _get_move_description
# ---------------------------------------------------------------------------

class TestGetMoveDescription:
    def test_neighbor_choices_correct(self, env):
        node = env.agent_pos["a0"]
        expected_neighbors = env.world_graph.neighbors(node)
        desc = env._get_move_description("a0")
        choices = desc["params"]["direction"]["choices"]
        assert "stay" in choices
        for nbr in expected_neighbors:
            assert nbr in choices


# ---------------------------------------------------------------------------
# _get_nearby_agents
# ---------------------------------------------------------------------------

class TestGetNearbyAgents:
    def test_no_nearby_when_alone(self, env):
        assert env._get_nearby_agents("a0") == []

    def test_nearby_within_hop_radius(self, two_agent_env):
        e = two_agent_env
        nearby = e._get_nearby_agents("a0")
        assert "a1" in nearby

    def test_not_nearby_beyond_hop_radius(self, tmp_log):
        # Build a chain graph long enough to put a1 outside hop_radius=1 of a0
        e = make_graph_env(tmp_log, topology="ring", n_nodes=20, hop_radius=1)
        all_nodes = e.world_graph.all_nodes()
        e.add_agent("a0", "Alice", "text", position=all_nodes[0])
        # Place a1 at hop distance > 1
        far_node = all_nodes[5]
        e.add_agent("a1", "Bob", "text", position=far_node)
        e.restart_env(agent_poses={"a0": all_nodes[0], "a1": far_node})
        nearby = e._get_nearby_agents("a0")
        assert "a1" not in nearby


# ---------------------------------------------------------------------------
# get_state_ckpt / set_state_ckpt (graph-specific parts)
# ---------------------------------------------------------------------------

class TestGraphCheckpointing:
    def test_ckpt_round_trip(self, env):
        env.step({"a0": {"action": "move", "params": {"direction": "stay"}}})
        ckpt = env.get_state_ckpt()
        env2 = make_graph_env(Path(str(env.log_path) + "_2"))
        all_nodes = env2.world_graph.all_nodes()
        env2.add_agent("a0", "Alice", "text", position=all_nodes[0])
        env2.restart_env(agent_poses={"a0": all_nodes[0]})
        env2.set_state_ckpt(ckpt)
        assert env2.step_count == env.step_count
        assert set(env2.world_graph.all_nodes()) == set(env.world_graph.all_nodes())


# ---------------------------------------------------------------------------
# Full step loop integration
# ---------------------------------------------------------------------------

class TestStepIntegration:
    def test_agent_moves_to_neighbor_via_step(self, env):
        node = env.agent_pos["a0"]
        nbr = env.world_graph.neighbors(node)[0]
        env.step({"a0": {"action": "move", "params": {"direction": nbr}}})
        assert env.agent_pos["a0"] == nbr

    def test_agent_eats_food_on_node(self, env):
        node = env.agent_pos["a0"]
        env.food[node] = 5.0
        energy_before = env.agent_energy["a0"]
        env.step({"a0": {"action": "move", "params": {"direction": "stay"}}})
        assert env.agent_energy["a0"] == energy_before + 5.0 - 1

    def test_give_energy_via_step(self, two_agent_env):
        e = two_agent_env
        e.food.pop(e.agent_pos["a0"], None)
        e.food.pop(e.agent_pos["a1"], None)
        alice_e = e.agent_energy["a0"]
        bob_e = e.agent_energy["a1"]
        e.step({"a0": {"action": "give", "params": {"target": "Bob", "amount": 10}}})
        assert e.agent_energy["a0"] == alice_e - 10 - 1
        assert e.agent_energy["a1"] == bob_e + 10 - 1

    def test_spawn_via_step(self, env):
        env.agent_energy["a0"] = 200
        env._get_avail_actions("a0")
        env.step({
            "a0": {"action": "spawn", "params": {"name": "GraphChild", "energy": 0}}
        })
        assert len(env.agent_registry) == 2


# ---------------------------------------------------------------------------
# Non-grid topology smoke tests
# ---------------------------------------------------------------------------

class TestNonGridTopologies:
    def test_ring_env_runs(self, tmp_log):
        e = make_graph_env(tmp_log, topology="ring", n_nodes=10)
        all_nodes = e.world_graph.all_nodes()
        e.add_agent("a0", "Alice", "text", position=all_nodes[0])
        e.restart_env(agent_poses={"a0": all_nodes[0]})
        e.step({"a0": {"action": "move", "params": {"direction": "stay"}}})
        assert e.step_count == 1

    def test_scale_free_env_runs(self, tmp_log):
        e = make_graph_env(tmp_log, topology="scale_free", n_nodes=15)
        all_nodes = e.world_graph.all_nodes()
        e.add_agent("a0", "Alice", "text", position=all_nodes[0])
        e.restart_env(agent_poses={"a0": all_nodes[0]})
        e.step({"a0": {"action": "move", "params": {"direction": "stay"}}})
        assert e.step_count == 1


# ---------------------------------------------------------------------------
# Star topology
# ---------------------------------------------------------------------------

class TestStarTopology:
    def test_hub_connects_to_all_spokes(self):
        wg = build_graph("star", n_nodes=6)
        assert len(wg.all_nodes()) == 6
        hub_neighbors = set(wg.neighbors("hub"))
        spokes = {n for n in wg.all_nodes() if n != "hub"}
        assert hub_neighbors == spokes

    def test_spokes_only_connect_to_hub(self):
        wg = build_graph("star", n_nodes=6)
        for node in wg.all_nodes():
            if node != "hub":
                assert wg.neighbors(node) == ["hub"]

    def test_minimum_size(self):
        # n_nodes=1 is valid — a single hub node with no spokes
        wg = build_graph("star", n_nodes=1)
        assert len(wg.all_nodes()) == 1


# ---------------------------------------------------------------------------
# Wheel topology
# ---------------------------------------------------------------------------

class TestWheelTopology:
    def test_hub_connects_to_all_ring_nodes(self):
        wg = build_graph("wheel", n_nodes=6)
        assert len(wg.all_nodes()) == 6
        ring_nodes = {n for n in wg.all_nodes() if n != "hub"}
        hub_neighbors = set(wg.neighbors("hub"))
        assert hub_neighbors == ring_nodes

    def test_ring_nodes_have_clockwise_edges(self):
        wg = build_graph("wheel", n_nodes=6)
        for node in wg.all_nodes():
            if node == "hub":
                continue
            edge_labels = {data["label"] for _, data in wg.edges_from(node)}
            assert "clockwise" in edge_labels or "counterclockwise" in edge_labels

    def test_ring_nodes_also_connect_to_hub(self):
        wg = build_graph("wheel", n_nodes=6)
        for node in wg.all_nodes():
            if node != "hub":
                assert "hub" in wg.neighbors(node)

    def test_minimum_size(self):
        # Fewer than 4 nodes is invalid — wheel requires at least 1 hub + 3 ring nodes
        with pytest.raises(ValueError):
            build_graph("wheel", n_nodes=2)


# ---------------------------------------------------------------------------
# build_graph node_props
# ---------------------------------------------------------------------------

class TestBuildGraphNodeProps:
    def test_node_props_applies_affordances(self):
        aff = LocationAffordance(action="recharge", description="Gain energy.")
        wg = build_graph("star", n_nodes=5, node_props={"hub": {"affordances": [aff]}})
        affs = wg.node_affordances("hub")
        assert len(affs) == 1
        assert affs[0].action == "recharge"


# ---------------------------------------------------------------------------
# LocationAffordance
# ---------------------------------------------------------------------------

class TestLocationAffordance:
    def test_from_dict_full(self):
        d = {"action": "boost", "description": "Gain energy.", "params": {"x": "y"}, "effect": {"type": "add_energy", "amount": 10}}
        aff = LocationAffordance.from_dict(d)
        assert aff.action == "boost"
        assert aff.description == "Gain energy."
        assert aff.params == {"x": "y"}
        assert aff.effect == {"type": "add_energy", "amount": 10}
        aff = LocationAffordance.from_dict({"action": "ping", "description": "A ping."})
        assert aff.params == {}
        assert aff.effect == {}

    def test_from_dict_missing_action_raises(self):
        with pytest.raises(ValueError, match="action"):
            LocationAffordance.from_dict({"description": "No action field."})

    def test_from_dict_missing_description_raises(self):
        with pytest.raises(ValueError, match="description"):
            LocationAffordance.from_dict({"action": "ping"})


# ---------------------------------------------------------------------------
# WorldGraph.node_affordances + add_node_affordance
# ---------------------------------------------------------------------------

class TestWorldGraphAffordances:
    def test_empty_node_returns_empty_list(self):
        wg = build_graph("star", n_nodes=5)
        assert wg.node_affordances("hub") == []

    def test_add_multiple_affordances(self):
        wg = build_graph("star", n_nodes=5)
        wg.add_node_affordance("hub", LocationAffordance(action="a", description="A."))
        wg.add_node_affordance("hub", LocationAffordance(action="b", description="B."))
        assert len(wg.node_affordances("hub")) == 2

    def test_mixed_list_all_converted(self):
        wg = build_graph("star", n_nodes=5)
        aff = LocationAffordance(action="a", description="A.")
        raw = {"action": "b", "description": "B."}
        wg.set_node_attr("hub", affordances=[aff, raw])
        result = wg.node_affordances("hub")
        assert all(isinstance(r, LocationAffordance) for r in result)
        assert [r.action for r in result] == ["a", "b"]


# ---------------------------------------------------------------------------
# _build_from_file with affordances
# ---------------------------------------------------------------------------

class TestBuildFromFileAffordances:
    def _write_graph(self, path, nodes, edges=None):
        data = {"nodes": nodes, "edges": edges or []}
        path.write_text(json.dumps(data))

    def test_affordances_loaded_as_location_affordance(self, tmp_path):
        f = tmp_path / "g.json"
        self._write_graph(f, [
            {"id": "hub", "affordances": [
                {"action": "recharge", "description": "Recharge energy."}
            ]},
            {"id": "spoke"},
        ], [{"from": "hub", "to": "spoke", "label": "out"}])
        wg = build_graph("file", path=str(f))
        affs = wg.node_affordances("hub")
        assert len(affs) == 1
        assert isinstance(affs[0], LocationAffordance)
        assert affs[0].action == "recharge"

    def test_missing_action_raises_with_node_info(self, tmp_path):
        f = tmp_path / "g.json"
        self._write_graph(f, [
            {"id": "hub", "affordances": [{"description": "No action."}]},
        ])
        with pytest.raises(ValueError, match="hub"):
            build_graph("file", path=str(f))


# ---------------------------------------------------------------------------
# graph_env._get_location_affordance
# ---------------------------------------------------------------------------

class TestGetLocationAffordance:
    def test_returns_action_description_params(self, env):
        node = env.agent_pos["a0"]
        env.world_graph.add_node_affordance(
            node, LocationAffordance(action="boost", description="Gain energy.", params={"x": "y"}, effect={"type": "add_energy", "amount": 10})
        )
        result = env._get_location_affordance(node)
        assert "boost" in result
        assert result["boost"]["description"] == "Gain energy."
        assert result["boost"]["params"] == {"x": "y"}
        assert "effect" not in result["boost"]


# ---------------------------------------------------------------------------
# graph_env._on_location_action
# ---------------------------------------------------------------------------

class TestOnLocationAction:
    def test_returns_false_when_no_affordance(self, env):
        infos = {"a0": {}}
        assert env._on_location_action("a0", "nonexistent", {}, infos) is False

    def test_returns_true_when_affordance_matches(self, env):
        node = env.agent_pos["a0"]
        env.world_graph.add_node_affordance(node, LocationAffordance(action="boost", description="Gain."))
        infos = {"a0": {}}
        assert env._on_location_action("a0", "boost", {}, infos) is True

    def test_returns_false_for_wrong_action_name(self, env):
        node = env.agent_pos["a0"]
        env.world_graph.add_node_affordance(node, LocationAffordance(action="boost", description="Gain."))
        infos = {"a0": {}}
        assert env._on_location_action("a0", "other_action", {}, infos) is False


# ---------------------------------------------------------------------------
# Affordance step integration
# ---------------------------------------------------------------------------

class TestAffordanceStepIntegration:
    def test_affordance_appears_in_available_actions(self, env):
        node = env.agent_pos["a0"]
        env.world_graph.add_node_affordance(
            node, LocationAffordance(action="boost", description="Gain energy.")
        )
        env._get_avail_actions("a0")
        assert "boost" in env.agent_avail_actions["a0"]

    def test_affordance_supersedes_global_action_description(self, env):
        # An affordance with the same name as a global action overwrites its description
        node = env.agent_pos["a0"]
        env.world_graph.add_node_affordance(
            node, LocationAffordance(action="set_color", description="Custom color action.")
        )
        env._get_avail_actions("a0")
        assert env.agent_avail_actions["a0"]["set_color"]["description"] == "Custom color action."

    def test_affordance_action_consumed_in_step(self, env):
        node = env.agent_pos["a0"]
        env.world_graph.add_node_affordance(
            node, LocationAffordance(action="boost", description="Gain energy.")
        )
        env._get_avail_actions("a0")
        _, _, _, _, infos = env.step({"a0": {"action": "boost", "params": {}}})
        # Action was consumed — no artifact error
        assert "Artifact interaction result" not in infos["a0"]
