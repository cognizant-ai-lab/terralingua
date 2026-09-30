"""
Tests for terralingua/environment/env.py (OpenGridWorld).
Covers all public and internal methods.
"""

import numpy as np
import pytest

from terralingua.environment.grid_env import OpenGridWorld

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_log(tmp_path):
    return tmp_path / "logs"


@pytest.fixture
def env(tmp_log):
    e = OpenGridWorld(
        grid_size=10,
        vision_radius=2,
        init_agent_energy=100,
        lifespan=200,
        init_food=50,
        max_food_value=5.0,
        food_decay_rate=0.0,         # no decay by default
        food_spawn_rate=0,           # no respawn by default
        log_path=tmp_log,
        drop_food_on_death=False,
        use_inventory=True,
        use_colors=True,
        reproduction_cost=20,
        artifact_creation_cost=0,
        headless=True,
    )
    e.add_agent("a0", "Alice", "text", position=(2, 2))
    e.restart_env()
    return e


@pytest.fixture
def two_agent_env(tmp_log):
    e = OpenGridWorld(
        grid_size=10,
        vision_radius=2,
        init_agent_energy=100,
        lifespan=200,
        init_food=50,
        max_food_value=5.0,
        food_decay_rate=0.0,
        food_spawn_rate=0,
        log_path=tmp_log,
        drop_food_on_death=False,
        use_inventory=True,
        use_colors=True,
        reproduction_cost=20,
        artifact_creation_cost=0,
        headless=True,
    )
    e.add_agent("a0", "Alice", "text", position=(2, 2))
    e.add_agent("a1", "Bob", "text", position=(2, 3))
    e.restart_env(agent_poses={"a0": (2, 2), "a1": (2, 3)})
    return e


# ---------------------------------------------------------------------------
# _wrap_xy
# ---------------------------------------------------------------------------

class TestWrapXY:
    def test_wrap_both(self, env):
        assert env.wrap_xy(-1, 10) == (9, 0)


# ---------------------------------------------------------------------------
# _apply_move
# ---------------------------------------------------------------------------

class TestApplyMove:
    def test_stay(self, env):
        pos = env.agent_pos["a0"]
        new_pos, infos = env._apply_move("a0", {"direction": "stay"}, {"a0": {}})
        assert new_pos == pos

    @pytest.mark.parametrize("start,direction,expected", [
        ((2, 2), "right", (2, 3)),
        ((2, 5), "left", (2, 4)),
        ((5, 5), "up", (4, 5)),
        ((5, 5), "down", (6, 5)),
    ])
    def test_move_direction(self, env, start, direction, expected):
        env.restart_env(agent_poses={"a0": start})
        new_pos, infos = env._apply_move("a0", {"direction": direction}, {"a0": {}})
        assert new_pos == expected

    def test_move_blocked_by_occupancy(self, two_agent_env):
        two_agent_env.restart_env(agent_poses={"a0": (2, 2), "a1": (2, 3)})
        new_pos, infos = two_agent_env._apply_move("a0", {"direction": "right"}, {"a0": {}})
        assert new_pos == (2, 2)
        assert "Move outcome" in infos["a0"]

    def test_move_none_params(self, env):
        pos = env.agent_pos["a0"]
        new_pos, infos = env._apply_move("a0", None, {"a0": {}})
        assert new_pos == pos

    def test_unknown_direction_stays(self, env):
        pos = env.agent_pos["a0"]
        new_pos, infos = env._apply_move("a0", {"direction": "diagonal"}, {"a0": {}})
        assert new_pos == pos

    def test_move_wraps_grid(self, env):
        env.restart_env(agent_poses={"a0": (5, 0)})
        new_pos, infos = env._apply_move("a0", {"direction": "left"}, {"a0": {}})
        assert new_pos == (5, 9)


# ---------------------------------------------------------------------------
# _get_food_distribution
# ---------------------------------------------------------------------------

class TestGetFoodDistribution:
    def test_uniform_values(self, env):
        density = env._get_food_distribution()
        assert density.shape == (env.grid_size, env.grid_size)
        assert np.allclose(density, 1.0 / env.grid_size ** 2)

    def test_with_int_food_zones(self, tmp_log):
        e = OpenGridWorld(
            grid_size=10,
            init_food=50,
            food_zones=2,
            log_path=tmp_log,
            headless=True,
        )
        e.rng = np.random.default_rng(42)
        density = e._get_food_distribution()
        assert density.shape == (10, 10)
        assert abs(density.sum() - 1.0) < 1e-6

    def test_with_list_food_zones(self, tmp_log):
        e = OpenGridWorld(
            grid_size=10,
            init_food=50,
            food_zones=[(3, 3), (7, 7)],
            log_path=tmp_log,
            headless=True,
        )
        e.rng = np.random.default_rng(0)
        density = e._get_food_distribution()
        assert abs(density.sum() - 1.0) < 1e-6

    def test_zone_centers_persisted(self, tmp_log):
        e = OpenGridWorld(grid_size=10, init_food=50, food_zones=3, log_path=tmp_log, headless=True)
        e.rng = np.random.default_rng(0)
        e._get_food_distribution()
        centers_first = list(e._food_zone_centers)
        e._get_food_distribution()  # second call should reuse
        assert e._food_zone_centers == centers_first


# ---------------------------------------------------------------------------
# _offspring_position
# ---------------------------------------------------------------------------

class TestOffspringPosition:
    def test_returns_neighbor(self, env):
        center = (5, 5)
        nbr = env._offspring_position(center)
        assert nbr is not None
        assert abs(nbr[0] - center[0]) <= 1 and abs(nbr[1] - center[1]) <= 1
        assert nbr != center

    def test_returns_none_when_all_blocked(self, env):
        center = (5, 5)
        # Occupy all 8 neighbours
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx_, ny = center[0] + dx, center[1] + dy
                if 0 <= nx_ < env.grid_size and 0 <= ny < env.grid_size:
                    env.pos_to_agent[(nx_, ny)].add("dummy")
        nbr = env._offspring_position(center)
        assert nbr is None

    def test_out_of_bounds_neighbours_excluded(self, env):
        # Corner cell — some neighbours are out of bounds
        center = (0, 0)
        for _ in range(20):
            nbr = env._offspring_position(center)
            if nbr is not None:
                assert 0 <= nbr[0] < env.grid_size
                assert 0 <= nbr[1] < env.grid_size


# ---------------------------------------------------------------------------
# _random_free_pos
# ---------------------------------------------------------------------------

class TestRandomFreePos:
    def test_returns_unoccupied(self, env):
        pos = env._random_free_pos()
        assert not env.pos_to_agent[pos]


# ---------------------------------------------------------------------------
# _respawn_food_one
# ---------------------------------------------------------------------------

class TestRespawnFoodOne:
    def test_food_value_default(self, env):
        env.food.clear()
        env._respawn_food_one()
        assert len(env.food) == 1
        assert list(env.food.values())[0] == env._max_food_value

    def test_food_value_custom(self, env):
        env.food.clear()
        env._respawn_food_one(value=2.5)
        val = list(env.food.values())[0]
        assert val == 2.5

    def test_does_not_overwrite_existing(self, env):
        env.food.clear()
        env._respawn_food_one()
        pos = list(env.food.keys())[0]
        # fill all other positions too so the same position would be re-chosen
        for r in range(env.grid_size):
            for c in range(env.grid_size):
                if (r, c) != pos:
                    env.food[(r, c)] = 1.0
        # Now all cells have food; respawn should not add a new entry
        count_before = len(env.food)
        env._respawn_food_one()
        assert len(env.food) == count_before


# ---------------------------------------------------------------------------
# _build_obs
# ---------------------------------------------------------------------------

class TestBuildObs:
    def test_obs_keys(self, env):
        obs, _ = env._build_obs("a0")
        for key in ("observation", "incoming_broadcasts", "energy", "time", "inventory", "vision_radius"):
            assert key in obs
        assert obs["energy"] == env.agent_energy["a0"]
        assert obs["time"] == env.agent_time["a0"]

    def test_no_nearby_when_alone(self, env):
        _, has_nearby = env._build_obs("a0")
        assert not has_nearby

    def test_has_nearby_when_adjacent(self, two_agent_env):
        e = two_agent_env
        # a0 at (2,2), a1 at (2,3) — within vision_radius=2
        _, has_nearby = e._build_obs("a0")
        assert has_nearby

    def test_food_visible_in_radius(self, env):
        x, y = env.agent_pos["a0"]
        env.food.clear()
        env.food[(x, y)] = 5.0  # food on same cell
        obs, _ = env._build_obs("a0")
        assert (0, 0) in obs["observation"]
        assert "5.0" in obs["observation"][(0, 0)]

    def test_artifact_in_obs(self, env):
        pos = env.agent_pos["a0"]
        env.add_artifact(
            pose=pos,
            art_type="text",
            art_name="sign1",
            payload="hello",
            creator="a0",
            lifespan=10,
        )
        obs, _ = env._build_obs("a0")
        items_at_pos = obs["observation"].get((0, 0), [])
        assert any("sign1" in s for s in items_at_pos)

    def test_inventory_in_obs(self, env):
        pos = env.agent_pos["a0"]
        env.add_artifact(
            pose=pos,
            art_type="text",
            art_name="inv_art",
            payload="data",
            creator="a0",
            lifespan=10,
        )
        # pickup manually
        env.pos_artifacts[pos].discard("inv_art")
        env.agent_inventories["a0"].add("inv_art")
        obs, _ = env._build_obs("a0")
        assert any("inv_art" in s for s in obs["inventory"])


# ---------------------------------------------------------------------------
# _get_nearby_agents
# ---------------------------------------------------------------------------

class TestGetNearbyAgents:
    def test_no_nearby_when_alone(self, env):
        assert env._get_nearby_agents("a0") == []

    def test_nearby_within_radius(self, two_agent_env):
        e = two_agent_env
        nearby = e._get_nearby_agents("a0")
        assert "a1" in nearby

    def test_not_nearby_outside_radius(self, tmp_log):
        e = OpenGridWorld(
            grid_size=20,
            init_food=200,
            vision_radius=1,
            log_path=tmp_log,
            headless=True,
        )
        e.add_agent("a0", "Alice", "text", position=(2, 2))
        e.add_agent("a1", "Bob", "text", position=(9, 9))
        e.restart_env(agent_poses={"a0": (2, 2), "a1": (9, 9)})
        nearby = e._get_nearby_agents("a0")
        assert "a1" not in nearby


# ---------------------------------------------------------------------------
# get_state_ckpt / set_state_ckpt (grid-specific parts)
# ---------------------------------------------------------------------------

class TestGridCheckpointing:
    def test_restore_grid_size(self, env):
        ckpt = env.get_state_ckpt()
        env.grid_size = 99  # corrupt
        env.set_state_ckpt(ckpt)
        assert env.grid_size == 10

    def test_restore_food_distribution(self, env):
        env._seed_initial_food()
        ckpt = env.get_state_ckpt()
        dist_before = env.food_distribution.copy()
        env.food_distribution = None
        env.set_state_ckpt(ckpt)
        assert np.allclose(env.food_distribution, dist_before) # type: ignore


# ---------------------------------------------------------------------------
# resize_grid
# ---------------------------------------------------------------------------

class TestResizeGrid:
    def test_shrink_is_noop(self, env):
        env.resize_grid(5)
        assert env.grid_size == 10

    def test_spawn_dist_reset_after_grow(self, env):
        env._seed_initial_food()
        env.resize_grid(20)
        assert env.grid_size == 20
        assert env._spawn_dist_flat is None
        assert env.food_distribution is None

    def test_food_zone_centers_extended(self, tmp_log):
        e = OpenGridWorld(
            grid_size=10,
            init_food=50,
            food_zones=2,
            log_path=tmp_log,
            headless=True,
        )
        e.rng = np.random.default_rng(0)
        e._seed_initial_food()
        e.resize_grid(20)
        assert len(e._food_zone_centers) == round(2 * 20**2 / 10**2)
