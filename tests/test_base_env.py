"""
Tests for terralingua/environment/base_env.py (BaseWorld), exercised through
the concrete OpenGridWorld subclass so all abstract methods are available.
"""

import numpy as np
import pytest

from terralingua.environment.grid_env import OpenGridWorld

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def make_env(tmp_path, **kwargs):
    defaults = dict(
        grid_size=10,
        vision_radius=2,
        init_agent_energy=100,
        lifespan=200,
        init_food=50,
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
    return OpenGridWorld(**defaults)  # type: ignore


@pytest.fixture
def tmp_log(tmp_path):
    return tmp_path / "logs"


@pytest.fixture
def env(tmp_log):
    e = make_env(tmp_log)
    e.add_agent("a0", "Alice", "text", position=(2, 2))
    e.restart_env(agent_poses={"a0": (2, 2)})
    return e


@pytest.fixture
def two_agent_env(tmp_log):
    e = make_env(tmp_log)
    e.add_agent("a0", "Alice", "text", position=(2, 2))
    e.add_agent("a1", "Bob", "text", position=(2, 3))
    e.restart_env(agent_poses={"a0": (2, 2), "a1": (2, 3)})
    return e


# ---------------------------------------------------------------------------
# add_agent
# ---------------------------------------------------------------------------


class TestAddAgent:
    def test_registers_agent(self, tmp_log):
        e = make_env(tmp_log)
        e.add_agent(
            "a0", "Alice", "text", position=(3, 3), excluded_actions=["spawn"]
        )
        assert "a0" in e.agent_registry
        assert e.agent_names["a0"] == "Alice"
        assert e.name_to_tag["Alice"] == "a0"
        assert e.agent_pos["a0"] == (3, 3)
        assert e.agent_energy["a0"] == e.init_agent_energy
        assert "a0" in e.pos_to_agent[(3, 3)]
        assert "spawn" in e.agent_excluded_actions["a0"]

    def test_duplicate_agent_raises(self, tmp_log):
        e = make_env(tmp_log)
        e.add_agent("a0", "Alice", "text", position=(2, 2))
        with pytest.raises(ValueError):
            e.add_agent("a0", "Alice", "text", position=(3, 3))

    def test_returns_observation_and_info(self, tmp_log):
        e = make_env(tmp_log)
        obs, info = e.add_agent("a0", "Alice", "text", position=(3, 3))
        assert isinstance(obs, dict)
        assert "available_actions" in info


# ---------------------------------------------------------------------------
# add_artifact
# ---------------------------------------------------------------------------


class TestAddArtifact:
    def test_artifact_added(self, env):
        result = env.add_artifact(
            pose=(2, 2),
            art_type="text",
            art_name="sign",
            payload="hello",
            creator="a0",
            lifespan=10,
        )
        assert "sign" in env.artifacts
        assert "Created" in result
        assert "sign" in env.pos_artifacts[(2, 2)]
        assert env.artifact_location["sign"] == ("map", (2, 2))

    def test_insufficient_energy_fails(self, tmp_log):
        e = make_env(tmp_log, artifact_creation_cost=200)
        e.add_agent("a0", "Alice", "text", position=(2, 2))
        e.restart_env(agent_poses={"a0": (2, 2)})
        result = e.add_artifact(
            pose=(2, 2),
            art_type="text",
            art_name="sign",
            payload="hi",
            creator="a0",
            lifespan=10,
        )
        assert "Failed" in result

    def test_invalid_art_type_fails(self, env):
        result = env.add_artifact(
            pose=(2, 2),
            art_type="container",
            art_name="box",
            payload="stuff",
            creator="a0",
            lifespan=10,
        )
        assert "not a valid type" in result

    def test_name_deduplicated(self, env):
        env.add_artifact(
            pose=(2, 2),
            art_type="text",
            art_name="sign",
            payload="v1",
            creator="a0",
            lifespan=10,
        )
        env.add_artifact(
            pose=(2, 2),
            art_type="text",
            art_name="sign",
            payload="v2",
            creator="a0",
            lifespan=10,
        )
        assert "sign" in env.artifacts
        assert "sign_1" in env.artifacts


# ---------------------------------------------------------------------------
# seed_artifact
# ---------------------------------------------------------------------------


class TestSeedArtifact:
    def test_artifact_seeded_without_energy(self, env):
        env.agent_energy["a0"] = 0
        message, final_name = env.seed_artifact(
            pose=(3, 3),
            art_type="text",
            art_name="sys_sign",
            payload="system",
            lifespan=10,
        )
        assert "sys_sign" in env.artifacts
        assert "Seeded" in message
        assert final_name == "sys_sign"
        assert env.agent_energy["a0"] == 0

    def test_invalid_type_fails(self, env):
        message, final_name = env.seed_artifact(
            pose=(3, 3),
            art_type="unknown",
            art_name="bad",
            payload="x",
            lifespan=10,
        )
        assert "not a valid type" in message
        assert final_name is None


# ---------------------------------------------------------------------------
# restart_env
# ---------------------------------------------------------------------------


class TestRestartEnv:
    def test_restart_resets_state(self, env):
        env.add_artifact(
            pose=(2, 2),
            art_type="text",
            art_name="tmp",
            payload="x",
            creator="a0",
            lifespan=100,
        )
        env.step(
            {"a0": {"action": "move", "params": {"direction": "stay"}, "message": "hi"}}
        )
        obs, infos = env.restart_env(agent_poses={"a0": (5, 5)})
        assert env.step_count == 0
        assert env.agent_pos["a0"] == (5, 5)
        assert len(env.artifacts) == 0
        assert env.chat == {}
        assert "a0" in obs
        assert "a0" in infos


# ---------------------------------------------------------------------------
# step — basic mechanics
# ---------------------------------------------------------------------------


class TestStepBasic:
    def test_step_count_increments(self, env):
        env.step({"a0": {"action": "move", "params": {"direction": "stay"}}})
        assert env.step_count == 1

    def test_energy_decreases_per_step(self, env):
        env.food.pop(env.agent_pos["a0"], None)  # guard against food seeded at start pos
        energy_before = env.agent_energy["a0"]
        env.step({"a0": {"action": "move", "params": {"direction": "stay"}}})
        assert env.agent_energy["a0"] == energy_before - 1

    def test_agent_eats_food(self, env):
        pos = env.agent_pos["a0"]
        env.food[pos] = 5.0
        energy_before = env.agent_energy["a0"]
        env.step({"a0": {"action": "move", "params": {"direction": "stay"}}})
        # energy += 5 (food) - 1 (step cost)
        assert env.agent_energy["a0"] == energy_before + 5.0 - 1

    def test_unknown_agent_raises(self, env):
        with pytest.raises((ValueError, KeyError)):
            env.step({"ghost": {"action": "move", "params": {"direction": "stay"}}})

    def test_unknown_action_logged_in_infos(self, env):
        _, _, _, _, infos = env.step({"a0": {"action": "fly", "params": {}}})
        assert "Action error" in infos["a0"]

    def test_wrong_params_logged_in_infos(self, env):
        _, _, _, _, infos = env.step(
            {"a0": {"action": "move", "params": {"wrong_key": "val"}}}
        )
        assert "Action error" in infos["a0"]

    def test_message_recorded(self, env):
        env.step(
            {
                "a0": {
                    "action": "move",
                    "params": {"direction": "stay"},
                    "message": "hello",
                }
            }
        )
        assert env.msg_raw["a0"] == "hello"

    def test_message_truncated_if_too_long(self, tmp_log):
        e = make_env(tmp_log, max_message_length=5)
        e.add_agent("a0", "Alice", "text", position=(2, 2))
        e.restart_env(agent_poses={"a0": (2, 2)})
        long_msg = "a " * 200
        e.step(
            {
                "a0": {
                    "action": "move",
                    "params": {"direction": "stay"},
                    "message": long_msg,
                }
            }
        )
        # raw message must have been truncated
        assert len(e.msg_raw["a0"]) < len(long_msg)


# ---------------------------------------------------------------------------
# step — energy give / take
# ---------------------------------------------------------------------------


class TestStepEnergyTransfer:
    def test_give_capped_at_sender_energy(self, two_agent_env):
        e = two_agent_env
        e.food.pop(e.agent_pos["a0"], None)
        e.food.pop(e.agent_pos["a1"], None)
        e.agent_energy["a0"] = 5.0
        bob_before = e.agent_energy["a1"]
        e.step({"a0": {"action": "give", "params": {"target": "Bob", "amount": 100}}})
        # Can only give min(100, 5) = 5; Bob gains 5, Alice loses 5 then step cost kills her
        assert e.agent_energy["a1"] == bob_before + 5 - 1  # Bob got 5, lost 1 to step
        # a0 is dead — not in registry
        assert "a0" not in e.agent_registry

    def test_give_not_nearby_fails(self, tmp_log):
        # Use a large grid so agents are truly out of vision range (non-toroidal gap > radius)
        e = make_env(tmp_log, grid_size=20, vision_radius=2)
        e.add_agent("a0", "Alice", "text", position=(0, 0))
        e.add_agent("a1", "Bob", "text", position=(10, 10))
        e.restart_env(agent_poses={"a0": (0, 0), "a1": (10, 10)})
        # "give" should not be in available actions — check then step
        actions = e._get_avail_actions("a0")
        assert "give" not in actions
        _, _, _, _, infos = e.step(
            {"a0": {"action": "give", "params": {"target": "Bob", "amount": 10}}}
        )
        assert "Action error" in infos["a0"]


# ---------------------------------------------------------------------------
# step — set_color
# ---------------------------------------------------------------------------


class TestStepSetColor:
    def test_set_color(self, env):
        _, _, _, _, infos = env.step(
            {"a0": {"action": "set_color", "params": {"color": "red"}}}
        )
        assert env.agent_colors["a0"] == "red"
        assert "Color selection" in infos["a0"]


# ---------------------------------------------------------------------------
# step — artifact lifecycle (create, pickup, drop, give, interact)
# ---------------------------------------------------------------------------


class TestStepArtifacts:
    def test_create_artifact_empty_name_fails(self, env):
        _, _, _, _, infos = env.step(
            {
                "a0": {
                    "action": "create_artifact",
                    "params": {
                        "type": "text",
                        "name": "",
                        "payload": "memo",
                        "lifespan": "10",
                        "movable": True,
                    },
                }
            }
        )
        assert "Artifact creation status" in infos["a0"]
        assert "Failed" in infos["a0"]["Artifact creation status"]

    def test_create_artifact_bad_lifespan_fails(self, env):
        _, _, _, _, infos = env.step(
            {
                "a0": {
                    "action": "create_artifact",
                    "params": {
                        "type": "text",
                        "name": "note",
                        "payload": "memo",
                        "lifespan": "abc",
                        "movable": True,
                    },
                }
            }
        )
        assert "Failed" in infos["a0"]["Artifact creation status"]

    def test_pickup_artifact(self, env):
        pos = env.agent_pos["a0"]
        env.add_artifact(
            pose=pos,
            art_type="text",
            art_name="box",
            payload="stuff",
            creator="a0",
            lifespan=50,
        )
        env._get_avail_actions("a0")
        env.step({"a0": {"action": "pickup_artifact", "params": {"name": "box"}}})
        assert "box" in env.agent_inventories["a0"]
        assert "box" not in env.pos_artifacts[pos]
        assert env.artifacts["box"].pose == "Inv(a0)"

    def test_drop_artifact(self, env):
        pos = env.agent_pos["a0"]
        env.add_artifact(
            pose=pos,
            art_type="text",
            art_name="box",
            payload="stuff",
            creator="a0",
            lifespan=50,
        )
        env.pos_artifacts[pos].discard("box")
        env.agent_inventories["a0"].add("box")
        env.artifacts["box"].pose = "Inv(a0)"
        env.artifact_location["box"] = ("inv", "a0")
        env._get_avail_actions("a0")
        env.step({"a0": {"action": "drop_artifact", "params": {"name": "box"}}})
        assert "box" in env.pos_artifacts[pos]
        assert "box" not in env.agent_inventories["a0"]
        assert env.artifacts["box"].pose == pos

    def test_give_artifact(self, two_agent_env):
        e = two_agent_env
        pos = e.agent_pos["a0"]
        e.add_artifact(
            pose=pos,
            art_type="text",
            art_name="gift",
            payload="present",
            creator="a0",
            lifespan=50,
        )
        e.pos_artifacts[pos].discard("gift")
        e.agent_inventories["a0"].add("gift")
        e.artifacts["gift"].pose = "Inv(a0)"
        e.artifact_location["gift"] = ("inv", "a0")
        e._get_avail_actions("a0")
        e.step(
            {
                "a0": {
                    "action": "give_artifact",
                    "params": {"artifact_name": "gift", "target_agent": "Bob"},
                }
            }
        )
        assert "gift" in e.agent_inventories["a1"]
        assert "gift" not in e.agent_inventories["a0"]
        assert e.artifacts["gift"].pose == "Inv(a1)"

    def test_artifact_interaction_modify(self, env):
        pos = env.agent_pos["a0"]
        env.add_artifact(
            pose=pos,
            art_type="text",
            art_name="memo",
            payload="original",
            creator="a0",
            lifespan=50,
        )
        env._get_avail_actions("a0")
        env.step(
            {
                "a0": {
                    "action": "modify_artifact",
                    "params": {"name": "memo", "payload": "updated", "lifespan": "50"},
                }
            }
        )
        assert env.artifacts["memo"].payload == "updated"

    def test_artifact_interaction_destroy(self, env):
        pos = env.agent_pos["a0"]
        env.add_artifact(
            pose=pos,
            art_type="text",
            art_name="tmp",
            payload="temp",
            creator="a0",
            lifespan=50,
        )
        env._get_avail_actions("a0")
        env.step({"a0": {"action": "destroy_artifact", "params": {"name": "tmp"}}})
        # artifact should expire on next step after remaining_time hits 0
        env.step({"a0": {"action": "move", "params": {"direction": "stay"}}})
        assert "tmp" not in env.artifacts

    def test_artifact_expires_by_lifespan(self, env):
        pos = env.agent_pos["a0"]
        env.add_artifact(
            pose=pos,
            art_type="text",
            art_name="short",
            payload="brief",
            creator="a0",
            lifespan=1,
        )
        # Step twice to let it expire (lifespan=1 means it expires after 1 decrement)
        env.step({"a0": {"action": "move", "params": {"direction": "stay"}}})
        env.step({"a0": {"action": "move", "params": {"direction": "stay"}}})
        assert "short" not in env.artifacts

    def test_pose_reflects_map_position_after_carrier_moves(self, env):
        # Artifact is in inventory — pose shows Inv(a0), not the carrier's moving position
        pos = env.agent_pos["a0"]
        env.add_artifact(
            pose=pos,
            art_type="text",
            art_name="note",
            payload="hi",
            creator="a0",
            lifespan=50,
        )
        env._get_avail_actions("a0")
        env.step({"a0": {"action": "pickup_artifact", "params": {"name": "note"}}})
        assert env.artifacts["note"].pose == "Inv(a0)"
        # Agent moves — artifact_location still correctly points to the carrier
        env.step({"a0": {"action": "move", "params": {"direction": "right"}}})
        assert env.artifacts["note"].pose == "Inv(a0)"
        assert env.artifact_location["note"] == ("inv", "a0")
        # Real position is agent_pos["a0"] which has changed
        assert env.agent_pos["a0"] != pos


# ---------------------------------------------------------------------------
# step — passive artifact effects
# ---------------------------------------------------------------------------


class TestPassiveArtifacts:
    def test_passive_effect_logged_in_infos(self, env):
        pos = env.agent_pos["a0"]
        env.add_artifact(
            pose=pos,
            art_type="text",
            art_name="reader",
            payload="read me",
            creator="a0",
            lifespan=100,
        )
        _, _, _, _, infos = env.step(
            {"a0": {"action": "move", "params": {"direction": "stay"}}}
        )
        assert "Passive interaction result - Artifacts at position" in infos["a0"]

    def test_passive_effect_inventory(self, env):
        pos = env.agent_pos["a0"]
        env.add_artifact(
            pose=pos,
            art_type="text",
            art_name="carried",
            payload="carry me",
            creator="a0",
            lifespan=100,
        )
        env.pos_artifacts[pos].discard("carried")
        env.agent_inventories["a0"].add("carried")
        env.artifact_location["carried"] = ("inv", "a0")
        _, _, _, _, infos = env.step(
            {"a0": {"action": "move", "params": {"direction": "stay"}}}
        )
        assert "Passive interaction result - Artifacts in inventory" in infos["a0"]


# ---------------------------------------------------------------------------
# step — agent death
# ---------------------------------------------------------------------------


class TestAgentDeath:
    def test_energy_depletion_kills(self, tmp_log):
        e = make_env(
            tmp_log, init_agent_energy=1, lifespan=1000, drop_food_on_death=False
        )
        e.add_agent("a0", "Alice", "text", position=(2, 2))
        e.restart_env(agent_poses={"a0": (2, 2)})
        # Clear food so Alice can't eat and survive
        e.food.clear()
        _, _, done, _, _ = e.step(
            {"a0": {"action": "move", "params": {"direction": "stay"}}}
        )
        assert done["a0"] is True
        assert "a0" not in e.agent_registry

    def test_lifespan_depletion_kills(self, tmp_log):
        e = make_env(
            tmp_log, init_agent_energy=10000, lifespan=1, drop_food_on_death=False
        )
        e.add_agent("a0", "Alice", "text", position=(2, 2))
        e.restart_env(agent_poses={"a0": (2, 2)})
        _, _, done, _, _ = e.step(
            {"a0": {"action": "move", "params": {"direction": "stay"}}}
        )
        assert done["a0"] is True

    def test_drop_food_on_death(self, tmp_log):
        e = make_env(
            tmp_log, init_agent_energy=1, lifespan=1000, drop_food_on_death=True
        )
        e.add_agent("a0", "Alice", "text", position=(5, 5))
        e.restart_env(agent_poses={"a0": (5, 5)})
        # Clear all food so Alice can't eat and survive
        e.food.clear()
        e.step({"a0": {"action": "move", "params": {"direction": "stay"}}})
        assert (5, 5) in e.food

    def test_dead_agent_observation_returned(self, tmp_log):
        e = make_env(
            tmp_log, init_agent_energy=1, lifespan=1000, drop_food_on_death=False
        )
        e.add_agent("a0", "Alice", "text", position=(2, 2))
        e.restart_env(agent_poses={"a0": (2, 2)})
        obs, _, done, _, _ = e.step(
            {"a0": {"action": "move", "params": {"direction": "stay"}}}
        )
        assert "a0" in obs


# ---------------------------------------------------------------------------
# inject_resource
# ---------------------------------------------------------------------------


class TestInjectResource:
    def test_inject_food(self, env):
        env.food.clear()
        env.inject_resource(amount=10.0, coefficient=1.0, target=None)
        total = sum(env.food.values())
        assert total > 0

    def test_inject_all_agents(self, two_agent_env):
        e = two_agent_env
        before_a0 = e.agent_energy["a0"]
        before_a1 = e.agent_energy["a1"]
        e.inject_resource(amount=20.0, coefficient=1.0, target="all")
        assert e.agent_energy["a0"] == before_a0 + 10.0
        assert e.agent_energy["a1"] == before_a1 + 10.0

    def test_withdraw_food(self, env):
        env.food = {(1, 1): 5.0, (2, 2): 5.0}
        env.inject_resource(amount=-8.0, coefficient=1.0, target=None)
        total = sum(env.food.values())
        assert total <= 2.0

    def test_coefficient_applied(self, env):
        before = env.agent_energy["a0"]
        env.inject_resource(amount=5.0, coefficient=2.0, target="a0")
        assert env.agent_energy["a0"] == before + 10.0

    def test_unknown_target_skipped(self, env):
        # Should not raise
        env.inject_resource(amount=10.0, coefficient=1.0, target="ghost_tag")


# ---------------------------------------------------------------------------
# _decay_and_respawn_food
# ---------------------------------------------------------------------------


class TestDecayAndRespawnFood:
    def test_food_can_decay(self, tmp_log):
        e = make_env(
            tmp_log,
            food_decay_rate=1.0,  # always decay
            food_decay_amount=2.0,
            food_spawn_rate=0,
        )
        e.add_agent("a0", "Alice", "text", position=(2, 2))
        e.restart_env(agent_poses={"a0": (2, 2)})
        e.food = {(0, 0): 3.0, (1, 1): 1.0}
        e.rng = np.random.default_rng(0)
        e._decay_and_respawn_food()
        # (0,0): 3 - 2 = 1; (1,1): 1 - 2 = -1 → expired
        assert (1, 1) not in e.food
        assert e.food.get((0, 0), 0) == 1.0


# ---------------------------------------------------------------------------
# _kill
# ---------------------------------------------------------------------------


class TestKill:
    def test_agent_removed(self, env):
        pos = env.agent_pos["a0"]
        env._kill("a0")
        assert "a0" not in env.agent_registry
        assert "a0" not in env.agent_pos
        assert "a0" not in env.pos_to_agent[pos]

    def test_inventory_dropped_to_map(self, env):
        pos = env.agent_pos["a0"]
        env.add_artifact(
            pose=(0, 0),
            art_type="text",
            art_name="carried",
            payload="x",
            creator="a0",
            lifespan=100,
        )
        env.pos_artifacts[(0, 0)].discard("carried")
        env.agent_inventories["a0"].add("carried")
        env.artifacts["carried"].pose = "Inv(a0)"
        env._kill("a0")
        assert "carried" in env.pos_artifacts[pos]
        assert env.artifacts["carried"].pose == pos


# ---------------------------------------------------------------------------
# _build_step_snapshot
# ---------------------------------------------------------------------------


class TestBuildStepSnapshot:
    def test_art_snap_contains_artifact_name(self, env):
        pos = env.agent_pos["a0"]
        env.add_artifact(
            pose=pos,
            art_type="text",
            art_name="snap_test",
            payload="hi",
            creator="a0",
            lifespan=100,
        )
        _, art_snap = env._build_step_snapshot()
        assert any("snap_test" in s for strs in art_snap.values() for s in strs)


# ---------------------------------------------------------------------------
# _get_avail_actions
# ---------------------------------------------------------------------------


class TestGetAvailActions:
    def test_excluded_actions_removed(self, tmp_log):
        e = make_env(tmp_log, excluded_actions=["spawn"])
        e.add_agent("a0", "Alice", "text", position=(2, 2))
        e.restart_env(agent_poses={"a0": (2, 2)})
        e.agent_energy["a0"] = 200
        actions = e._get_avail_actions("a0")
        assert "spawn" not in actions

    def test_agent_excluded_actions_removed(self, tmp_log):
        e = make_env(tmp_log)
        e.add_agent(
            "a0", "Alice", "text", position=(2, 2), excluded_actions=["spawn"]
        )
        e.restart_env(agent_poses={"a0": (2, 2)})
        e.agent_energy["a0"] = 200
        actions = e._get_avail_actions("a0")
        assert "spawn" not in actions

    @pytest.mark.parametrize("use_colors", [True, False])
    def test_set_color_follows_use_colors(self, tmp_log, use_colors):
        e = make_env(tmp_log, use_colors=use_colors)
        e.add_agent("a0", "Alice", "text", position=(2, 2))
        e.restart_env(agent_poses={"a0": (2, 2)})
        assert ("set_color" in e._get_avail_actions("a0")) is use_colors

    def test_artifact_actions_present_at_position(self, env):
        pos = env.agent_pos["a0"]
        env.add_artifact(
            pose=pos,
            art_type="text",
            art_name="note",
            payload="read",
            creator="a0",
            lifespan=100,
        )
        actions = env._get_avail_actions("a0")
        assert "modify_artifact" in actions
        assert "destroy_artifact" in actions
        assert actions["modify_artifact"]["params"]["name"]["choices"] == ["note"]
        assert actions["destroy_artifact"]["params"]["name"]["choices"] == ["note"]

    def test_external_action_present_when_enabled(self, tmp_log):
        spec = {
            "market_world_server": {
                "server": "market",
                "tool": "world_server",
                "cost": 0,
                "mode": "independent",
                "description": "Trade on the market. Costs 0 energy.",
                "params": {"payload": "The content of the call in text form."},
            }
        }
        e = make_env(tmp_log, external_actions_spec=spec)
        e.add_agent("a0", "Alice", "text", position=(2, 2))
        e.restart_env(agent_poses={"a0": (2, 2)})
        actions = e._get_avail_actions("a0")
        assert "market_world_server" in actions


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


class TestSerialize:
    def test_deserialize_rng(self):
        rng = np.random.default_rng(42)
        data = OpenGridWorld._serialize(rng)
        assert data["__type__"] == "rng"
        rng2 = OpenGridWorld._deserialize(data)
        assert isinstance(rng2, np.random.Generator)
        # both rngs should produce identical numbers
        assert rng.random() == rng2.random()

    def test_deserialize_ndarray(self):
        arr = np.array([[1, 2], [3, 4]])
        data = OpenGridWorld._serialize(arr)
        assert data["__type__"] == "ndarray"
        assert data["shape"] == [2, 2]
        arr2 = OpenGridWorld._deserialize(data)
        assert np.array_equal(arr, arr2) # type: ignore

    def test_serialize_unsupported_raises(self):
        with pytest.raises(TypeError):
            OpenGridWorld._serialize(object())


# ---------------------------------------------------------------------------
# get_state_ckpt / set_state_ckpt (base parts)
# ---------------------------------------------------------------------------


class TestBaseCheckpointing:
    def test_restore_artifacts(self, env):
        pos = env.agent_pos["a0"]
        env.add_artifact(
            pose=pos,
            art_type="text",
            art_name="saved_art",
            payload="persist",
            creator="a0",
            lifespan=100,
        )
        ckpt = env.get_state_ckpt()
        env.artifacts.clear()
        env.set_state_ckpt(ckpt)
        assert "saved_art" in env.artifacts


# ---------------------------------------------------------------------------
# save_state / load_state
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_and_load_state(self, env, tmp_path):
        env.step({"a0": {"action": "move", "params": {"direction": "stay"}}})
        fp = tmp_path / "state.pkl"
        env.save_state(fp)
        assert fp.exists()
        env.step({"a0": {"action": "move", "params": {"direction": "stay"}}})
        env.load_state(fp)
        assert env.step_count == 1

    def test_load_default_path(self, env):
        env.step({"a0": {"action": "move", "params": {"direction": "stay"}}})
        env.close()
        # Now load from default path
        env2 = make_env(env.log_path)
        env2.add_agent("a0", "Alice", "text", position=(2, 2))
        env2.restart_env(agent_poses={"a0": (2, 2)})
        env2.load_state()
        assert env2.step_count == 1

    def test_load_nonexistent_raises(self, env, tmp_path):
        with pytest.raises(AssertionError):
            env.load_state(tmp_path / "ghost.pkl")
