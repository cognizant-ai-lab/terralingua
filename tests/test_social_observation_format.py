"""Social observation formatting must preserve visibility and snapshot data."""

import copy

import pytest

from terralingua.config.models import GraphConfig
from terralingua.environment.social_graph_env import OpenSocialGraphWorld


@pytest.fixture
def world(tmp_path):
    env = OpenSocialGraphWorld(
        graph_cfg=GraphConfig(topology="ring", n_nodes=4, hop_radius=2),
        init_agent_energy=200, lifespan=500, init_food=0, food_spawn_rate=0,
        food_mechanism=False, log_path=tmp_path, drop_food_on_death=False,
        use_inventory=True, use_colors=True, reproduction_cost=20,
        headless=True,
    )
    poses = {}
    for index, name in enumerate(("Alice", "Bob", "Carol", "Dana")):
        tag, node = f"a{index}", env.world_graph.all_nodes()[index]
        env.add_agent(tag, name, "no_traits", position=node)
        poses[tag] = node
    env.restart_env(agent_poses=poses)
    for source in env.world_graph.all_nodes():
        for target in list(env.world_graph.neighbors(source)):
            env.world_graph.remove_edge(source, target)
    for source, target in (("a0", "a1"), ("a2", "a0"), ("a1", "a3")):
        env.world_graph.add_edge(poses[source], poses[target])
    env.agent_colors["a1"] = "blue"
    yield env
    env.close()


def _seed(env, tag, name, *, private=False, movable=True):
    node = env.agent_pos[tag]
    _, created = env.seed_artifact(
        pose=node, art_type="text", art_name=name,
        payload=f"Contents of {name}", lifespan=-1, movable=movable,
    )
    assert created == name
    if private:
        env.pos_artifacts[node].remove(name)
        env.agent_inventories[tag].add(name)
        env.artifact_location[name] = ("inv", tag)
        env.artifacts[name].pose = "Inventory"


def test_agent_labels_preserve_directed_visibility_and_artifact_privacy(world):
    _seed(world, "a0", "alice-public", movable=False)
    _seed(world, "a0", "alice-private", private=True)
    _seed(world, "a1", "bob-public")
    _seed(world, "a1", "bob-private", private=True)
    _seed(world, "a2", "incoming-only-artifact")
    _seed(world, "a3", "two-hop-artifact")

    obs, has_followed_agents = world._build_obs("a0")
    detailed = obs["observation_text"].split("Your social network:", 1)[0]

    assert has_followed_agents
    assert set(obs["observation"]) == {world.agent_pos["a0"], world.agent_pos["a1"]}
    assert obs["current_node"] == world.agent_pos["a0"]
    assert obs["observation"][world.agent_pos["a1"]]["hop_distance"] == 1
    assert obs["subscriptions"] == ["Bob"]
    assert obs["followers"] == ["Carol"]
    assert "You: Alice" in detailed
    assert "Agents you follow:" in detailed
    assert "Bob(blue)" in detailed
    assert "A(text,fixed): alice-public" in detailed
    assert "A(text,movable): bob-public" in detailed
    assert obs["inventory"] == ["A(text,movable): alice-private"]
    for hidden in ("alice-private", "bob-private", "incoming-only-artifact",
                   "two-hop-artifact", "Contents of", "Carol", "Dana"):
        assert hidden not in detailed
    assert "bob-private" not in str(obs)
    assert "You are at:" not in detailed
    assert "dist=" not in detailed


def test_text_uses_supplied_snapshots_without_mutating_them(world):
    own, followed = world.agent_pos["a0"], world.agent_pos["a1"]
    _seed(world, "a1", "live-artifact")
    food = {own: "12.5", followed: "8"}
    artifacts = {
        own: ["A(text,fixed): snapshot-own"],
        followed: ["A(text,movable): snapshot-followed", "custom visible item"],
    }
    original = copy.deepcopy((food, artifacts))
    obs, _ = world._build_obs("a0", food_snapshot=food, artifact_snapshot=artifacts)
    raw = copy.deepcopy(obs["observation"])
    text = world.format_observation_text(obs["observation"], own)

    assert "Food energy: 12.5" in text
    assert "Food energy: 8" in text
    assert "snapshot-own" in text and "snapshot-followed" in text
    assert "custom visible item" in text
    assert "live-artifact" not in text
    assert (food, artifacts) == original
    assert obs["observation"] == raw


def test_visible_items_without_an_agent_are_not_discarded(world):
    own, vacant = world.agent_pos["a0"], "vacant-node"
    raw = {
        own: {"items": [], "hop_distance": 0, "exits": [vacant]},
        vacant: {"items": ["A(text,movable): abandoned-notes"],
                 "hop_distance": 1, "exits": []},
    }
    original = copy.deepcopy(raw)
    text = world.format_observation_text(raw, own)

    assert "Visible items without an active agent:" in text
    assert "A(text,movable): abandoned-notes" in text
    assert vacant not in text
    assert raw == original


def test_numeric_agent_names_do_not_hide_matching_food_values(world):
    world.use_colors = False
    world.agent_names["a0"] = "12.5"
    world.agent_names["a1"] = "8"
    own, followed = world.agent_pos["a0"], world.agent_pos["a1"]
    obs, _ = world._build_obs("a0", food_snapshot={own: "12.5", followed: "8"})

    assert "Food energy: 12.5" in obs["observation_text"]
    assert "Food energy: 8" in obs["observation_text"]
    assert obs["observation"][followed]["items"].count("8") == 2
