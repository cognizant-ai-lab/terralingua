"""
Checkpoint contract tests for every env.

Two guards, so save and load can't silently drift apart:

- ``test_*_ckpt_keys`` pins the exact set of keys an env's ``get_state_ckpt``
  emits. Add/remove a saved field and the test fails until the locked set is
  updated.
- ``test_*_roundtrip`` saves, restores into a fresh env, then saves again and
  asserts every field round-trips — so a field that ``get_state_ckpt`` writes
  but ``set_state_ckpt`` never reads is caught.
"""

from terralingua.config.models import GraphConfig
from terralingua.environment.graph_env import OpenGraphWorld
from terralingua.environment.grid_env import OpenGridWorld
from terralingua.environment.social_graph_env import OpenSocialGraphWorld

# Keys produced by BaseWorld.get_state_ckpt (shared by every env).
BASE_KEYS = {
    "rng",
    "food",
    "food_count",
    "_food_zone_centers",
    "pos_artifacts",
    "artifacts",
    "agent_inventories",
    "library",
    "artifact_location",
    "expired_artifacts",
    "_all_artifact_names",
    "agent_pos",
    "agent_trajectories",
    "agent_avail_actions",
    "pos_to_agent",
    "agent_energy",
    "agent_time",
    "agent_spawn",
    "agent_names",
    "agent_colors",
    "msg_raw",
    "chat",
    "agent_registry",
    "agent_excluded_actions",
    "step_count",
    "mechanics",
    "pending_deaths",
    "no_appetite",
    "agent_affordances",
    "_roles",
    "_role_assignment",
    "_role_successor",
    "_initial_roles_assigned",
    "execution_context",
    "logger_save_path",
    "logger_data",
}
GRID_KEYS = BASE_KEYS | {
    "grid_size",
    "food_distribution",
    "_initial_food_zone_count",
    "empty_food",
}
GRAPH_KEYS = BASE_KEYS | {
    "world_graph",
}
SOCIAL_KEYS = GRAPH_KEYS | {
    "_edge_last_active",
    "_pending_outgoing",
    "_pending_incoming",
    "_agent_directives",
}

# Env-local logger paths/data and the opaque rng blob aren't meaningfully
# comparable across instances.
_ROUNDTRIP_IGNORE = {"logger_save_path", "logger_data", "rng"}


def _grid_env(log_path):
    env = OpenGridWorld(
        grid_size=8,
        vision_radius=2,
        init_food=20,
        food_spawn_rate=1,
        headless=True,
        log_path=log_path,
    )
    env.add_agent("a0", "Alice", "no_traits")
    env.restart_env()
    return env


def _graph_env(log_path):
    env = OpenGraphWorld(
        graph_cfg=GraphConfig(topology="ring", n_nodes=4, hop_radius=1),
        init_food=0,
        food_spawn_rate=0,
        headless=True,
        log_path=log_path,
    )
    env.add_agent("a0", "Alice", "no_traits")
    env.restart_env()
    return env


def _social_env(log_path, n_agents=1):
    env = OpenSocialGraphWorld(
        graph_cfg=GraphConfig(
            topology="ring", n_nodes=5, hop_radius=1, max_connections=10
        ),
        food_mechanism=False,
        init_food=0,
        food_spawn_rate=0,
        headless=True,
        log_path=log_path,
    )
    nodes = env.world_graph.all_nodes()
    names = ["Alice", "Bob", "Carol", "Dave", "Eve"][:n_agents]
    for i, name in enumerate(names):
        env.add_agent(f"a{i}", name, "no_traits", position=nodes[i])
    env.restart_env(agent_poses={f"a{i}": nodes[i] for i in range(n_agents)})
    return env


def _assert_roundtrip(source_env, fresh_env):
    ckpt = source_env.get_state_ckpt()
    fresh_env.set_state_ckpt(ckpt)
    restored = fresh_env.get_state_ckpt()
    for key, value in ckpt.items():
        if key in _ROUNDTRIP_IGNORE:
            continue
        assert restored[key] == value, (
            f"set_state_ckpt did not restore '{key}' (save/load drift)"
        )


# ---------------------------------------------------------------------------
# Schema locks (save side)
# ---------------------------------------------------------------------------


def test_grid_ckpt_keys(tmp_path):
    assert set(_grid_env(tmp_path).get_state_ckpt()) == GRID_KEYS


def test_graph_ckpt_keys(tmp_path):
    assert set(_graph_env(tmp_path).get_state_ckpt()) == GRAPH_KEYS


def test_social_ckpt_keys(tmp_path):
    assert set(_social_env(tmp_path).get_state_ckpt()) == SOCIAL_KEYS


# ---------------------------------------------------------------------------
# Round-trips (load side)
# ---------------------------------------------------------------------------


def test_grid_roundtrip(tmp_path):
    _assert_roundtrip(_grid_env(tmp_path / "a"), _grid_env(tmp_path / "b"))


def test_graph_roundtrip(tmp_path):
    _assert_roundtrip(_graph_env(tmp_path / "a"), _graph_env(tmp_path / "b"))


def test_social_roundtrip(tmp_path):
    env = _social_env(tmp_path / "a", n_agents=4)
    # Populate the social-specific state so a load that drops it is caught.
    env.step(
        {"a0": {"action": "follow", "message": "", "params": {"target": "Carol", "replace": ""}}}
    )
    env.step(
        {"a0": {"action": "request_connection", "message": "", "params": {"target": "Dave", "replace": ""}}}
    )
    assert env._pending_outgoing and env._edge_last_active  # sanity: non-empty
    _assert_roundtrip(env, _social_env(tmp_path / "b", n_agents=1))
