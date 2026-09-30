"""Configured artifact options agree with menu choices and executed actions."""

import json

import pytest

from terralingua.config.models import GraphConfig
from terralingua.environment.graph_builder import build_graph
from terralingua.environment.graph_env import OpenGraphWorld
from terralingua.environment.grid_env import OpenGridWorld
from terralingua.environment.social_graph_env import OpenSocialGraphWorld


def make_world(kind, tmp_path):
    options = dict(
        log_path=tmp_path, headless=True, food_mechanism=False,
        init_agent_energy=100, lifespan=200, reproduction_cost=-1,
        init_food=5, food_spawn_rate=0, food_decay_rate=0,
        use_inventory=True, artifact_creation_cost=7,
    )
    if kind == "grid":
        env = OpenGridWorld(grid_size=6, **options)
        position = (2, 2)
    else:
        cls = OpenSocialGraphWorld if kind == "social" else OpenGraphWorld
        env = cls(graph_cfg=GraphConfig.complete(n_nodes=3), **options)
        position = "0"
    env.add_agent("a", "Alice", "no_traits", position=position)
    env.restart_env(seed=10, agent_poses={"a": position})
    return env


@pytest.fixture(params=["grid", "social"])
def world(request, tmp_path):
    return make_world(request.param, tmp_path)

def create(world, **params):
    world._get_avail_actions("a")
    return world.step({"a": {
        "action": "create_artifact", "message": "",
        "params": {"name": "note", "type": "text", "payload": "hello",
                   "lifespan": -1, **params},
    }})[4]["a"]


@pytest.mark.parametrize("value,expected", [("true", True), ("false", False), (True, True), (False, False)])
def test_artifact_movable_choice_is_respected(world, value, expected):
    assert world.agent_avail_actions["a"]["create_artifact"]["params"]["movable"]["choices"] == ["true", "false"]
    info = create(world, movable=value)
    assert "Created artifact" in info["Artifact creation status"]
    assert world.artifacts["note"].movable is expected
    assert world.agent_energy["a"] == 93
    actions = world._get_avail_actions("a")
    assert ("pickup_artifact" in actions) is expected
    if isinstance(world, OpenSocialGraphWorld):
        assert ("deposit_artifact" in actions) is expected


@pytest.mark.parametrize("value", [1, "not a boolean"])
def test_invalid_movable_choice_fails_without_creating_or_charging(world, value):
    info = create(world, movable=value)
    assert "must be true or false" in info["Artifact creation status"]
    assert not world.artifacts
    assert world.agent_energy["a"] == 100


def test_disallow_fixed_artifacts_removes_choice_and_creates_movable(world):
    world.allow_fixed_artifacts = False
    assert "movable" not in world._get_avail_actions("a")["create_artifact"]["params"]
    create(world)
    assert world.artifacts["note"].movable is True
    assert world.agent_energy["a"] == 93


@pytest.mark.parametrize("kind,food", [("grid", False), ("grid", True), ("social", False)])
@pytest.mark.parametrize("action", ["give", "take"])
def test_energy_transfer_does_not_depend_on_food(tmp_path, kind, food, action):
    world = make_world(kind, tmp_path)
    world.food_mechanism = food
    position = (2, 3) if isinstance(world, OpenGridWorld) else "1"
    world.add_agent("b", "Bob", "no_traits", position=position)
    actions = world._get_avail_actions("a")
    assert action in actions
    info = world.step({"a": {"action": action, "message": "",
                             "params": {"target": "Bob", "amount": 7}}})[4]
    delta = -7 if action == "give" else 7
    assert world.agent_energy["a"] == 100 + delta - int(food)
    assert world.agent_energy["b"] == 100 - delta - int(food)
    assert "Action error" not in info["a"]


def social(tmp_path, cost=7):
    env = OpenSocialGraphWorld(
        graph_cfg=GraphConfig.complete(n_nodes=3, direct_message_cost=cost),
        food_mechanism=False, energy_death=False, init_food=0,
        food_spawn_rate=0, food_decay_rate=0, init_agent_energy=100,
        lifespan=200, headless=True, log_path=tmp_path,
        use_library=True, library_preview=True, use_inventory=True,
    )
    for tag, name, pos in [("a", "Alice", "0"), ("b", "Bob", "1")]:
        env.add_agent(tag, name, "no_traits", position=pos)
    env.restart_env(seed=3, agent_poses={"a": "0", "b": "1"})
    return env


@pytest.mark.parametrize("cost", [0, 7])
def test_direct_message_fee_and_delivery_work_without_natural_food(tmp_path, cost):
    env = social(tmp_path, cost)
    action = env.agent_avail_actions["a"]["send_direct_message"]
    assert ("Costs" in action["description"]) is bool(cost)
    obs, _, _, _, infos = env.step({"a": {"action": "send_direct_message", "message": "",
        "params": {"target": "Bob", "content": "private"}}})
    assert infos["a"]["send_direct_message"]["status"] == "successful"
    assert env.agent_energy["a"] == 100 - cost
    assert obs["b"]["incoming_dms"] == {"Alice": "private"}


def test_direct_message_rechecks_affordability_without_charging_or_delivering(tmp_path):
    env = social(tmp_path)
    assert "send_direct_message" in env.agent_avail_actions["a"]
    env.agent_energy["a"] = 6
    obs, _, _, _, infos = env.step({"a": {"action": "send_direct_message", "message": "",
        "params": {"target": "Bob", "content": "private"}}})
    assert infos["a"]["send_direct_message"]["status"] == "failed"
    assert env.agent_energy["a"] == 6
    assert not obs["b"]["incoming_dms"]
    assert "send_direct_message" not in env.agent_avail_actions["a"]
    env.direct_message_cost = 0
    env.agent_energy["a"] = -1
    assert "send_direct_message" in env._get_avail_actions("a")


def test_inert_library_lists_names_without_content_or_transfer_actions(tmp_path):
    env = social(tmp_path)
    env.seed_artifact("0", "text", "private_recipe", "secret recipe text", -1)
    infos = {"a": {}}
    env._handle_deposit_artifact("a", {"name": "private_recipe"}, infos)
    env.seed_artifact("0", "text", "public_note", "another secret", -1)
    env.inert_artifacts = True
    actions = env._get_avail_actions("a")
    assert "show_library_content" in actions
    for action in ("read_artifact", "deposit_artifact", "retrieve_artifact", "pickup_artifact"):
        assert action not in actions
    info = env.step({"a": {"action": "show_library_content", "message": "", "params": {}}})[4]["a"]
    assert "private_recipe" in info["Library content"]
    assert "secret recipe" not in info["Library content"]
    assert "preview" not in actions["show_library_content"]["description"].lower()
    assert "cannot be read or transferred" in env._build_obs("a")[0]["observation_text"]
    # Inert mode must not discard the independent library death-persistence rule.
    env._kill("a")
    assert "public_note" in env.library


@pytest.mark.parametrize("action", ["read_artifact", "deposit_artifact", "retrieve_artifact"])
def test_inert_library_rejects_a_previously_cached_action(tmp_path, action):
    env = social(tmp_path)
    env.seed_artifact("0", "text", "note", "secret", -1)
    if action != "deposit_artifact":
        env._handle_deposit_artifact("a", {"name": "note"}, {"a": {}})
    env._get_avail_actions("a")
    assert action in env.agent_avail_actions["a"]
    before = env.artifact_location["note"]
    env.inert_artifacts = True
    info = env.step({"a": {"action": action, "message": "", "params": {"name": "note"}}})[4]["a"]
    assert "inert" in info["Action outcome"]
    assert "secret" not in str(info)
    assert env.artifact_location["note"] == before


@pytest.mark.parametrize("kind", ["grid", "graph"])
def test_food_can_respawn_after_all_existing_food_is_gone(tmp_path, kind):
    world = make_world(kind, tmp_path)
    world.food_mechanism = True
    world._food_spawn_rate = 100
    world.food.clear()
    world.step({})
    assert world.food


@pytest.mark.parametrize("kind", ["grid", "graph"])
def test_consumed_static_food_site_can_respawn(tmp_path, kind):
    world = make_world(kind, tmp_path)
    world.food_mechanism = True
    world.static_food = True
    world._food_spawn_rate = 100
    pos = world.agent_pos["a"]
    world.food = {pos: 5}
    world.step({"a": world.stay_action})
    assert world.food == {pos: world._max_food_value}
    assert not world.empty_food
    assert world.agent_energy["a"] == 104


@pytest.mark.parametrize("kind", ["grid", "graph"])
def test_withdrawn_static_food_site_can_respawn(tmp_path, kind):
    world = make_world(kind, tmp_path)
    world.static_food = True
    pos = world.agent_pos["a"]
    world.food = {pos: 5}
    world.inject_resource(-5, 1, None)
    assert world.empty_food == [pos]
    world._respawn_food_one()
    assert world.food == {pos: world._max_food_value}
    assert not world.empty_food


@pytest.mark.parametrize("initial,expected", [(0, 0), (60, 6), (1000, 36)])
def test_grid_initial_food_uses_energy_units_and_available_space(tmp_path, initial, expected):
    env = OpenGridWorld(grid_size=6, init_food=initial, max_food_value=10,
                        food_zones=None, headless=True, log_path=tmp_path)
    env.restart_env(seed=2)
    assert len(env.food) == expected
    assert sum(env.food.values()) == expected * 10


def test_graph_file_preserves_explicit_node_and_edge_attributes(tmp_path):
    source = tmp_path / "topology.json"
    source.write_text(json.dumps({"nodes": [{"id": "a", "label": "Alpha"}, {"id": "b"}],
        "edges": [{"from": "a", "to": "b", "label": "path", "traversable": False}]}))
    graph = build_graph("file", path=str(source))
    assert graph.node_metadata("a")["label"] == "Alpha"
    assert graph.node_metadata("b")["label"] == "b"
    assert graph.edges_from("a") == [("b", {"label": "path", "traversable": False})]



def test_social_rejects_natural_food_at_construction(tmp_path):
    with pytest.raises(ValueError, match="do not support natural food"):
        OpenSocialGraphWorld(graph_cfg=GraphConfig.complete(n_nodes=3),
                             food_mechanism=True, headless=True, log_path=tmp_path)


def test_social_defaults_to_no_food_or_upkeep_despite_spatial_food_options(tmp_path):
    env = OpenSocialGraphWorld(
        graph_cfg=GraphConfig.complete(n_nodes=3), headless=True, log_path=tmp_path,
        init_food=1000, food_spawn_rate=1000, food_decay_rate=1,
        food_zones=["missing-node"], drop_food_on_death=True,
        init_agent_energy=100, lifespan=200,
    )
    env.add_agent("a", "Alice", "no_traits", position="0")
    env.restart_env(seed=3, agent_poses={"a": "0"})
    assert env.food_mechanism is False
    assert env.energy_death is False
    assert not env.food
    for _ in range(3):
        env.step({"a": env.stay_action})
    assert env.agent_energy["a"] == 100
    assert not env.food
    env._kill("a")
    assert not env.food


def test_social_rejects_spatial_food_injection(tmp_path):
    env = social(tmp_path)
    before = dict(env.agent_energy)
    with pytest.raises(ValueError, match="spatial food injection"):
        env.inject_resource(10, 2, None)
    assert not env.food
    assert env.agent_energy == before


def test_social_agent_energy_injection_and_explicit_energy_death_remain_available(tmp_path):
    env = social(tmp_path)
    env.inject_resource(20, 2, "all")
    assert env.agent_energy == {"a": 120, "b": 120}
    env.inject_resource(10, 2, "a")
    assert env.agent_energy == {"a": 140, "b": 120}
    env.inject_resource(-10, 2, "a")
    assert env.agent_energy == {"a": 120, "b": 120}
    env.energy_death = True
    env.inject_resource(-120, 1, "a")
    env.step({})
    assert "a" not in env.agent_registry
    assert env.agent_energy["b"] == 120
    assert not env.food


@pytest.mark.parametrize("cost", [-1, 0, 7])
def test_artifact_cost_controls_creation_without_food(world, cost):
    world.artifact_creation_cost = cost
    assert ("create_artifact" in world._get_avail_actions("a")) is (cost >= 0)
    if cost >= 0:
        create(world, movable=True)
        assert "note" in world.artifacts
        assert world.agent_energy["a"] == 100 - cost
    else:
        result = world.add_artifact(world.agent_pos["a"], "text", "note", "hello", "a", 100)
        assert "disabled" in result
        create(world)
        assert not world.artifacts
        assert world.agent_energy["a"] == 100
        _, name = world.seed_artifact(world.agent_pos["a"], "text", "seeded", "hello", 100)
        assert name == "seeded"
        assert "pickup_artifact" in world._get_avail_actions("a")


def test_stale_artifact_menu_cannot_bypass_disabled_creation(world):
    assert "create_artifact" in world.agent_avail_actions["a"]
    world.artifact_creation_cost = -1
    world.step({"a": {"action": "create_artifact", "message": "", "params": {
        "name": "stale", "type": "text", "payload": "hello", "lifespan": 100, "movable": True,
    }}})
    assert not world.artifacts
    assert world.agent_energy["a"] == 100
