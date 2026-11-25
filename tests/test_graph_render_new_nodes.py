"""Rendering keeps working after nodes join the graph mid-run, as spawned
beings do in the social graph: each new node gets a position near its
neighbour, and isolated ones go outside the layout."""

import math

from terralingua.config.models import GraphConfig
from terralingua.environment.social_graph_env import OpenSocialGraphWorld


def make_world(tmp_path):
    env = OpenSocialGraphWorld(
        graph_cfg=GraphConfig(topology="ring", n_nodes=5, hop_radius=1, max_connections=10),
        food_mechanism=False, init_food=0, food_spawn_rate=0,
        headless=True, log_path=str(tmp_path / "world.log"),
    )
    nodes = env.world_graph.all_nodes()
    for i, name in enumerate(["Alice", "Bob", "Carol"]):
        env.add_agent(f"a{i}", name, "no_traits", position=nodes[i])
    env.restart_env(agent_poses={f"a{i}": nodes[i] for i in range(3)})
    return env, nodes


def test_new_nodes_are_placed_and_rendered(tmp_path):
    env, nodes = make_world(tmp_path)
    frame = env.render(mode="rgb_array")
    assert frame is not None and len(env._node_layout) == 5
    parent = nodes[0]
    env.world_graph.add_node("child", label="child")
    env.world_graph.add_edge(parent, "child")
    env.world_graph.add_edge("child", parent)
    env.world_graph.add_node("child2", label="child2")
    env.world_graph.add_edge(parent, "child2")
    env.world_graph.add_edge("child2", parent)
    env.world_graph.add_node("loner", label="loner")
    env.render(mode="rgb_array")
    layout = env._node_layout
    assert {"child", "child2", "loner"} <= set(layout)
    px, py = layout[parent]
    spacing = 2 * math.sin(math.pi / 5)  # neighbour distance on the 5-node ring layout
    for child in ("child", "child2"):
        assert abs(math.hypot(layout[child][0] - px, layout[child][1] - py) - spacing) < 0.05
    assert layout["child"] != layout["child2"]
    others = [p for n, p in layout.items() if n != "loner"]
    cx = sum(x for x, _ in others) / len(others)
    cy = sum(y for _, y in others) / len(others)
    assert math.hypot(layout["loner"][0] - cx, layout["loner"][1] - cy) > max(
        math.hypot(x - cx, y - cy) for x, y in others
    )
    assert len(set(layout.values())) == len(layout)
