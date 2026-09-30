"""
Tests for terralingua/environment/social_graph_env.py (OpenSocialGraphWorld).

Covers the directional social-graph model: single-direction edges, the two
formation actions (`follow` + `request_connection`/`accept`/`reject`), the
per-agent cap that counts outgoing edges + pending outgoing requests, pending
request lifespan, directional disconnect, directional broadcast routing,
code-enforced DM/give targets, action-visibility flags, spawn edge wiring,
and directional edge decay.
"""

import pytest

from terralingua.config.models import GraphConfig
from terralingua.environment.actions import LocationAffordance
from terralingua.environment.social_graph_env import (
    GRANTABLE_ABILITY_SPECS,
    OpenSocialGraphWorld,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def make_social_env(
    tmp_path,
    topology="ring",
    n_nodes=5,
    max_connections=None,
    random_intro_k=3,
    edge_decay_steps=None,
    direct_message_cost=0,
    allow_follow=True,
    allow_connection=True,
    connection_request_lifespan=3,
    **kwargs,
):
    cfg = GraphConfig(
        topology=topology,
        n_nodes=n_nodes,
        hop_radius=1,
        max_connections=max_connections,
        random_intro_k=random_intro_k,
        edge_decay_steps=edge_decay_steps,
        direct_message_cost=direct_message_cost,
        allow_follow=allow_follow,
        allow_connection=allow_connection,
        connection_request_lifespan=connection_request_lifespan,
    )
    defaults = dict(
        graph_cfg=cfg,
        init_agent_energy=200,
        lifespan=500,
        init_food=0,
        food_spawn_rate=0,
        food_mechanism=False,
        log_path=tmp_path,
        drop_food_on_death=False,
        use_inventory=False,
        use_colors=False,
        reproduction_cost=20,
        artifact_creation_cost=-1,
        headless=True,
    )
    defaults.update(kwargs)
    return OpenSocialGraphWorld(**defaults)  # type: ignore


def seed_n_agents(env, names):
    """Place one agent per node, in the order returned by `all_nodes()`.
    Returns the list of (tag, name, node) actually used."""
    nodes = env.world_graph.all_nodes()
    placed = []
    for i, name in enumerate(names):
        tag = f"a{i}"
        node = nodes[i]
        env.add_agent(tag, name, "no_traits", position=node)
        placed.append((tag, name, node))
    poses = {tag: node for tag, _, node in placed}
    env.restart_env(agent_poses=poses)
    return placed


def follow(env, actor, target, replace=""):
    return env.step(
        {
            actor: {
                "action": "follow",
                "message": "",
                "params": {"target": target, "replace": replace},
            }
        }
    )


def request_connection(env, actor, target, replace=""):
    return env.step(
        {
            actor: {
                "action": "request_connection",
                "message": "",
                "params": {"target": target, "replace": replace},
            }
        }
    )


def noop(env, actor="a0"):
    return env.step({actor: {"action": "noop", "message": "", "params": {}}})


def grant_remove_agent(env, node):
    """Seed the `remove_agent` affordance on a manager's node."""
    env.world_graph.set_node_attr(
        node,
        affordances=[
            LocationAffordance(
                action="remove_agent",
                mode="add",
                description="Fire an agent you are connected to.",
                params={"target": {"type": "string"}},
                effect={"type": "remove_agent"},
            )
        ],
    )


def remove_agent(env, actor, target):
    return env.step(
        {
            actor: {
                "action": "remove_agent",
                "message": "",
                "params": {"target": target},
            }
        }
    )


def grant_manager_powers(env, node, abilities):
    """Seed the given manager-power affordances on a node (before restart_env)."""
    env.world_graph.set_node_attr(
        node,
        affordances=[
            LocationAffordance.from_dict(GRANTABLE_ABILITY_SPECS[a]) for a in abilities
        ],
    )


def set_role(env, actor, target, role, motivation=""):
    return env.step(
        {
            actor: {
                "action": "set_role",
                "message": "",
                "params": {"target": target, "role": role, "motivation": motivation},
            }
        }
    )


def grant_ability(env, actor, target, ability):
    return env.step(
        {
            actor: {
                "action": "grant_ability",
                "message": "",
                "params": {"target": target, "ability": ability},
            }
        }
    )


def revoke_ability(env, actor, target, ability):
    return env.step(
        {
            actor: {
                "action": "revoke_ability",
                "message": "",
                "params": {"target": target, "ability": ability},
            }
        }
    )


def node_has_ability(env, tag, ability):
    return env._agent_effectively_has(tag, ability)


def node_suppresses(env, tag, action):
    return env._agent_has_affordance(tag, action, "remove")


@pytest.fixture
def tmp_log(tmp_path):
    return tmp_path / "logs"


# ---------------------------------------------------------------------------
# 1. Initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_ring_5_agents_each_own_a_node(self, tmp_log):
        env = make_social_env(tmp_log, topology="ring", n_nodes=5)
        placed = seed_n_agents(env, ["Alice", "Bob", "Carol", "Dave", "Eve"])
        assert env.world_graph.node_count() == 5
        positions = {tag: env.agent_pos[tag] for tag, _, _ in placed}
        assert len(set(positions.values())) == 5
        # Ring edges are mutual at init: 2 outgoing AND 2 incoming per node.
        for _, _, node in placed:
            assert len(env.world_graph.neighbors(node)) == 2
            assert len(env.world_graph.predecessors(node)) == 2

    def test_init_forces_hop_radius_1_and_exclusive_occupancy(self, tmp_log):
        cfg = GraphConfig(topology="ring", n_nodes=3, hop_radius=4)
        env = OpenSocialGraphWorld(
            graph_cfg=cfg,
            init_agent_energy=100,
            lifespan=100,
            init_food=0,
            food_spawn_rate=0,
            food_mechanism=False,
            log_path=tmp_log,
            headless=True,
        )
        assert env.hop_radius == 1
        assert env.exclusive_pos_occupancy is True


# ---------------------------------------------------------------------------
# 2. Action set
# ---------------------------------------------------------------------------


class TestActions:
    def test_move_absent_wait_present(self, tmp_log):
        env = make_social_env(tmp_log, topology="ring", n_nodes=3)
        seed_n_agents(env, ["Alice", "Bob", "Carol"])
        actions = env._get_avail_actions("a0")
        assert "move" not in actions
        assert "noop" in actions
        assert actions["noop"]["params"] == {}

    def test_formation_and_disconnect_actions_present(self, tmp_log):
        env = make_social_env(tmp_log, topology="ring", n_nodes=4, max_connections=10)
        seed_n_agents(env, ["Alice", "Bob", "Carol", "Dave"])
        actions = env._get_avail_actions("a0")
        # Has room + a non-followed target (Carol) → follow + request_connection.
        assert "follow" in actions
        assert "request_connection" in actions
        # Has subscriptions → disconnect available.
        assert "disconnect" in actions
        # No incoming requests → no accept/reject.
        assert "accept_connection" not in actions
        assert "reject_connection" not in actions


# ---------------------------------------------------------------------------
# 3. follow — unilateral, one-way
# ---------------------------------------------------------------------------


class TestFollow:
    def test_follow_creates_only_one_direction(self, tmp_log):
        env = make_social_env(tmp_log, topology="ring", n_nodes=4, max_connections=10)
        seed_n_agents(env, ["Alice", "Bob", "Carol", "Dave"])
        alice, carol = env.agent_pos["a0"], env.agent_pos["a2"]
        assert not env.world_graph.has_edge(alice, carol)
        _, _, _, _, infos = follow(env, "a0", "Carol")
        assert infos["a0"]["follow"]["status"] == "successful"
        # Only Alice→Carol; not the reverse.
        assert env.world_graph.has_edge(alice, carol)
        assert not env.world_graph.has_edge(carol, alice)

    def test_follow_consumes_only_followers_cap(self, tmp_log):
        env = make_social_env(tmp_log, topology="ring", n_nodes=4, max_connections=10)
        seed_n_agents(env, ["Alice", "Bob", "Carol", "Dave"])
        carol_out_before = env._outgoing_count("a2")
        follow(env, "a0", "Carol")
        # Carol's outgoing (her cap usage) is unchanged — she just gained a follower.
        assert env._outgoing_count("a2") == carol_out_before
        assert "Alice" in env._get_followers("a2")
        assert "Carol" in env._get_subscriptions("a0")

    def test_follow_fails_if_already_following(self, tmp_log):
        # Handler-level guard: ring means Alice already follows Bob.
        env = make_social_env(tmp_log, topology="ring", n_nodes=3, max_connections=10)
        seed_n_agents(env, ["Alice", "Bob", "Carol"])
        infos = {"a0": {}}
        env._handle_follow("a0", {"target": "Bob", "replace": ""}, infos)
        assert infos["a0"]["follow"]["status"] == "failed"
        assert "Already following" in infos["a0"]["follow"]["reason"]

    def test_follow_fails_if_pending_request_to_target(self, tmp_log):
        env = make_social_env(tmp_log, topology="ring", n_nodes=5, max_connections=10)
        seed_n_agents(env, ["Alice", "Bob", "Carol", "Dave", "Eve"])
        request_connection(env, "a0", "Carol")  # pending Alice→Carol (not followed)
        infos = {"a0": {}}
        env._handle_follow("a0", {"target": "Carol", "replace": ""}, infos)
        assert infos["a0"]["follow"]["status"] == "failed"
        assert "pending" in infos["a0"]["follow"]["reason"]

    def test_follow_rejects_self_and_unknown(self, tmp_log):
        env = make_social_env(tmp_log, topology="ring", n_nodes=3, max_connections=10)
        seed_n_agents(env, ["Alice", "Bob", "Carol"])
        infos = {"a0": {}}
        env._handle_follow("a0", {"target": "Alice", "replace": ""}, infos)
        assert infos["a0"]["follow"]["status"] == "failed"
        assert "yourself" in infos["a0"]["follow"]["reason"]
        infos = {"a0": {}}
        env._handle_follow("a0", {"target": "Nobody", "replace": ""}, infos)
        assert infos["a0"]["follow"]["status"] == "failed"


# ---------------------------------------------------------------------------
# 4. request_connection / accept / reject
# ---------------------------------------------------------------------------


class TestRequestConnection:
    def test_request_reserves_slot_and_shows_in_receiver_obs(self, tmp_log):
        env = make_social_env(tmp_log, topology="ring", n_nodes=4, max_connections=10)
        seed_n_agents(env, ["Alice", "Bob", "Carol", "Dave"])
        obs, _, _, _, infos = request_connection(env, "a0", "Carol")
        assert infos["a0"]["request_connection"]["status"] == "successful"
        # Sender sees their own pending; receiver sees the incoming request.
        assert "Carol" in obs["a0"]["pending_outgoing_requests"]
        assert "Alice" in obs["a2"]["pending_incoming_requests"]
        # No edge formed yet.
        assert not env.world_graph.has_edge(
            env.agent_pos["a0"], env.agent_pos["a2"]
        )

    def test_request_counts_against_cap(self, tmp_log):
        # Ring of 5 → 2 outgoing each. Cap 3 → room for exactly one more.
        env = make_social_env(tmp_log, topology="ring", n_nodes=5, max_connections=3)
        seed_n_agents(env, ["Alice", "Bob", "Carol", "Dave", "Eve"])
        assert env._has_cap_room("a0") is True
        request_connection(env, "a0", "Carol")  # Carol (node2) not followed by Alice
        # Pending now reserves the last slot → no more room.
        assert env._has_cap_room("a0") is False

    def test_accept_forms_both_edges_and_bumps_receiver_cap(self, tmp_log):
        env = make_social_env(tmp_log, topology="ring", n_nodes=4, max_connections=10)
        seed_n_agents(env, ["Alice", "Bob", "Carol", "Dave"])
        alice, carol = env.agent_pos["a0"], env.agent_pos["a2"]
        request_connection(env, "a0", "Carol")
        carol_out_before = env._outgoing_count("a2")
        # Carol accepts.
        _, _, _, _, infos = env.step(
            {
                "a2": {
                    "action": "accept_connection",
                    "message": "",
                    "params": {"target": "Alice", "replace": ""},
                }
            }
        )
        assert infos["a2"]["accept_connection"]["status"] == "successful"
        # Mutual edges now exist.
        assert env.world_graph.has_edge(alice, carol)
        assert env.world_graph.has_edge(carol, alice)
        # Carol gained an outgoing edge (her cap usage went up by 1).
        assert env._outgoing_count("a2") == carol_out_before + 1
        # Pending cleared on both sides.
        assert "Carol" not in env._get_pending_outgoing_targets("a0")
        assert "Alice" not in env._get_pending_incoming_senders("a2")

    def test_reject_clears_pending_and_frees_sender_slot(self, tmp_log):
        env = make_social_env(tmp_log, topology="ring", n_nodes=5, max_connections=3)
        seed_n_agents(env, ["Alice", "Bob", "Carol", "Dave", "Eve"])
        request_connection(env, "a0", "Carol")  # Carol (node2) not followed
        assert env._has_cap_room("a0") is False
        # Carol rejects.
        _, _, _, _, infos = env.step(
            {
                "a2": {
                    "action": "reject_connection",
                    "message": "",
                    "params": {"target": "Alice"},
                }
            }
        )
        assert infos["a2"]["reject_connection"]["status"] == "successful"
        assert "Carol" not in env._get_pending_outgoing_targets("a0")
        # Alice's reserved slot is freed.
        assert env._has_cap_room("a0") is True
        assert not env.world_graph.has_edge(
            env.agent_pos["a0"], env.agent_pos["a2"]
        )

    def test_request_fails_if_already_following(self, tmp_log):
        # Handler-level guard: ring means Alice already follows Bob.
        env = make_social_env(tmp_log, topology="ring", n_nodes=3, max_connections=10)
        seed_n_agents(env, ["Alice", "Bob", "Carol"])
        infos = {"a0": {}}
        env._handle_request_connection("a0", {"target": "Bob", "replace": ""}, infos)
        assert infos["a0"]["request_connection"]["status"] == "failed"
        assert "already follow" in infos["a0"]["request_connection"]["reason"].lower()


# ---------------------------------------------------------------------------
# 5. Pending request lifespan
# ---------------------------------------------------------------------------


class TestPendingLifespan:
    def test_pending_expires_and_frees_slot(self, tmp_log):
        LIFESPAN = 3
        env = make_social_env(
            tmp_log,
            topology="ring",
            n_nodes=5,
            max_connections=10,
            connection_request_lifespan=LIFESPAN,
        )
        seed_n_agents(env, ["Alice", "Bob", "Carol", "Dave", "Eve"])
        obs, _, _, _, _ = request_connection(env, "a0", "Dave")  # node3, not followed
        assert "Dave" in obs["a0"]["pending_outgoing_requests"]
        assert "Alice" in obs["a3"]["pending_incoming_requests"]
        # Survives for LIFESPAN-1 further steps...
        for _ in range(LIFESPAN - 1):
            obs, _, _, _, _ = noop(env, "a0")
            assert "Dave" in obs["a0"]["pending_outgoing_requests"]
        # ...and is pruned on the next.
        obs, _, _, _, _ = noop(env, "a0")
        assert "Dave" not in obs["a0"]["pending_outgoing_requests"]
        assert "Alice" not in obs["a3"]["pending_incoming_requests"]
        # Both mirrors cleared.
        assert "a3" not in env._pending_outgoing.get("a0", {})
        assert "a0" not in env._pending_incoming.get("a3", set())


# ---------------------------------------------------------------------------
# 6. Directional disconnect
# ---------------------------------------------------------------------------


class TestDisconnectDirectional:
    def test_disconnect_removes_only_one_direction(self, tmp_log):
        env = make_social_env(tmp_log, topology="ring", n_nodes=3)
        seed_n_agents(env, ["Alice", "Bob", "Carol"])
        alice, bob = env.agent_pos["a0"], env.agent_pos["a1"]
        # Ring: mutual Alice↔Bob.
        assert env.world_graph.has_edge(alice, bob)
        assert env.world_graph.has_edge(bob, alice)
        _, _, _, _, infos = env.step(
            {
                "a0": {
                    "action": "disconnect",
                    "message": "",
                    "params": {"target": "Bob"},
                }
            }
        )
        assert infos["a0"]["disconnect"]["status"] == "successful"
        # Only Alice→Bob removed; Bob→Alice stays.
        assert not env.world_graph.has_edge(alice, bob)
        assert env.world_graph.has_edge(bob, alice)

    def test_disconnect_cancels_pending_outgoing(self, tmp_log):
        env = make_social_env(tmp_log, topology="ring", n_nodes=4, max_connections=10)
        seed_n_agents(env, ["Alice", "Bob", "Carol", "Dave"])
        request_connection(env, "a0", "Carol")
        assert "Carol" in env._get_pending_outgoing_targets("a0")
        _, _, _, _, infos = env.step(
            {
                "a0": {
                    "action": "disconnect",
                    "message": "",
                    "params": {"target": "Carol"},
                }
            }
        )
        assert infos["a0"]["disconnect"]["status"] == "successful"
        assert infos["a0"]["disconnect"]["kind"] == "cancel_request"
        assert "Carol" not in env._get_pending_outgoing_targets("a0")

    def test_disconnect_rejects_unrelated(self, tmp_log):
        env = make_social_env(tmp_log, topology="ring", n_nodes=3)
        seed_n_agents(env, ["Alice", "Bob", "Carol"])
        env.add_agent("a3", "Dave", "no_traits", position=None)
        _, _, _, _, infos = env.step(
            {
                "a0": {
                    "action": "disconnect",
                    "message": "",
                    "params": {"target": "Dave"},
                }
            }
        )
        assert infos["a0"]["disconnect"]["status"] == "failed"


# ---------------------------------------------------------------------------
# 7. At-cap replace contract (follow / request / accept)
# ---------------------------------------------------------------------------


class TestAtCapReplace:
    def test_follow_at_cap_requires_replace(self, tmp_log):
        # Ring of 3 → 2 outgoing each. Cap 2 → at cap.
        env = make_social_env(tmp_log, topology="ring", n_nodes=3, max_connections=2)
        seed_n_agents(env, ["Alice", "Bob", "Carol"])
        env.add_agent("a3", "Dave", "no_traits", position=None)
        actions = env._get_avail_actions("a0")
        # follow still shown (Alice has subscriptions to replace).
        assert "follow" in actions
        # Without replace → fails.
        _, _, _, _, infos = follow(env, "a0", "Dave")
        assert infos["a0"]["follow"]["status"] == "failed"
        assert "cap" in infos["a0"]["follow"]["reason"].lower()
        # With a valid replace (drop Bob) → succeeds and swaps.
        alice, bob, dave = (
            env.agent_pos["a0"],
            env.agent_pos["a1"],
            env.agent_pos["a3"],
        )
        _, _, _, _, infos = follow(env, "a0", "Dave", replace="Bob")
        assert infos["a0"]["follow"]["status"] == "successful"
        assert not env.world_graph.has_edge(alice, bob)  # only Alice→Bob dropped
        assert env.world_graph.has_edge(bob, alice)  # Bob→Alice kept (directional)
        assert env.world_graph.has_edge(alice, dave)

    def test_request_at_cap_replace_can_target_pending(self, tmp_log):
        # Cap 3, ring of 5 (2 outgoing). One pending fills the last slot.
        env = make_social_env(tmp_log, topology="ring", n_nodes=5, max_connections=3)
        seed_n_agents(env, ["Alice", "Bob", "Carol", "Dave", "Eve"])
        # Carol (node2) and Dave (node3) are not followed by Alice (node0).
        request_connection(env, "a0", "Carol")  # pending fills slot → at cap
        assert env._has_cap_room("a0") is False
        # New request to Dave without replace → fails.
        _, _, _, _, infos = request_connection(env, "a0", "Dave")
        assert infos["a0"]["request_connection"]["status"] == "failed"
        # Replacing the pending Carol request frees the slot.
        _, _, _, _, infos = request_connection(env, "a0", "Dave", replace="Carol")
        assert infos["a0"]["request_connection"]["status"] == "successful"
        assert "Carol" not in env._get_pending_outgoing_targets("a0")
        assert "Dave" in env._get_pending_outgoing_targets("a0")

    def test_accept_at_cap_requires_replace(self, tmp_log):
        env = make_social_env(tmp_log, topology="ring", n_nodes=3, max_connections=2)
        seed_n_agents(env, ["Alice", "Bob", "Carol"])
        env.add_agent("a3", "Dave", "no_traits", position=None)
        # Dave requests Alice; Alice is at cap (2 outgoing).
        request_connection(env, "a3", "Alice")
        assert env._has_cap_room("a0") is False
        # Accept without replace → fails.
        _, _, _, _, infos = env.step(
            {
                "a0": {
                    "action": "accept_connection",
                    "message": "",
                    "params": {"target": "Dave", "replace": ""},
                }
            }
        )
        assert infos["a0"]["accept_connection"]["status"] == "failed"
        # Accept dropping Bob → succeeds, forms mutual Alice↔Dave.
        alice, dave = env.agent_pos["a0"], env.agent_pos["a3"]
        _, _, _, _, infos = env.step(
            {
                "a0": {
                    "action": "accept_connection",
                    "message": "",
                    "params": {"target": "Dave", "replace": "Bob"},
                }
            }
        )
        assert infos["a0"]["accept_connection"]["status"] == "successful"
        assert env.world_graph.has_edge(alice, dave)
        assert env.world_graph.has_edge(dave, alice)


# ---------------------------------------------------------------------------
# 8. Action-visibility flags
# ---------------------------------------------------------------------------


class TestActionFlags:
    def test_allow_follow_false_removes_follow(self, tmp_log):
        env = make_social_env(
            tmp_log, topology="ring", n_nodes=4, max_connections=10, allow_follow=False
        )
        seed_n_agents(env, ["Alice", "Bob", "Carol", "Dave"])
        actions = env._get_avail_actions("a0")
        assert "follow" not in actions
        assert "request_connection" in actions

    def test_allow_connection_false_removes_connection_actions(self, tmp_log):
        env = make_social_env(
            tmp_log,
            topology="ring",
            n_nodes=4,
            max_connections=10,
            allow_connection=False,
        )
        seed_n_agents(env, ["Alice", "Bob", "Carol", "Dave"])
        # Give a0 an incoming request to prove accept/reject are suppressed too.
        request_connection(env, "a2", "Alice")
        actions = env._get_avail_actions("a0")
        assert "request_connection" not in actions
        assert "accept_connection" not in actions
        assert "reject_connection" not in actions
        assert "follow" in actions


# ---------------------------------------------------------------------------
# 9. Directional broadcast routing
# ---------------------------------------------------------------------------


class TestBroadcastDirectional:
    def test_broadcast_reaches_only_followers(self, tmp_log):
        env = make_social_env(tmp_log, topology="ring", n_nodes=3, max_connections=10)
        seed_n_agents(env, ["Alice", "Bob", "Carol"])
        # Make Bob stop following Alice (drop Bob→Alice). Carol still follows Alice.
        env.step(
            {
                "a1": {
                    "action": "disconnect",
                    "message": "",
                    "params": {"target": "Alice"},
                }
            }
        )
        assert not env.world_graph.has_edge(env.agent_pos["a1"], env.agent_pos["a0"])
        assert env.world_graph.has_edge(env.agent_pos["a2"], env.agent_pos["a0"])
        # Alice broadcasts.
        obs, _, _, _, _ = env.step(
            {"a0": {"action": "noop", "message": "hello from A", "params": {}}}
        )
        # Carol follows Alice → hears her. Bob no longer follows Alice → does not.
        assert obs["a2"]["incoming_broadcasts"].get("Alice") == "hello from A"
        assert "Alice" not in obs["a1"]["incoming_broadcasts"]


# ---------------------------------------------------------------------------
# 10. Direct messages
# ---------------------------------------------------------------------------


class TestDirectMessages:
    def test_dm_available_only_with_subscriptions_and_affordable(self, tmp_log):
        env = make_social_env(tmp_log, topology="ring", n_nodes=3)
        seed_n_agents(env, ["Alice", "Bob", "Carol"])
        assert "send_direct_message" in env._get_avail_actions("a0")

        # Drop all of Alice's follows → no DM action.
        env.step({"a0": {"action": "disconnect", "message": "", "params": {"target": "Bob"}}})
        env.step({"a0": {"action": "disconnect", "message": "", "params": {"target": "Carol"}}})
        assert "send_direct_message" not in env._get_avail_actions("a0")

    def test_dm_routes_to_recipient_only(self, tmp_log):
        env = make_social_env(tmp_log, topology="ring", n_nodes=3)
        seed_n_agents(env, ["Alice", "Bob", "Carol"])
        obs, _, _, _, _ = env.step(
            {
                "a0": {
                    "action": "send_direct_message",
                    "message": "",
                    "params": {"target": "Bob", "content": "secret to Bob"},
                }
            }
        )
        assert obs["a1"]["incoming_dms"] == {"Alice": "secret to Bob"}
        assert "Alice" not in obs["a1"]["incoming_broadcasts"]
        assert obs["a2"]["incoming_dms"] == {}

    def test_dm_mailbox_clears_each_step(self, tmp_log):
        env = make_social_env(tmp_log, topology="ring", n_nodes=3)
        seed_n_agents(env, ["Alice", "Bob", "Carol"])
        env.step(
            {
                "a0": {
                    "action": "send_direct_message",
                    "message": "",
                    "params": {"target": "Bob", "content": "hi"},
                }
            }
        )
        obs, _, _, _, _ = noop(env, "a0")
        assert obs["a1"]["incoming_dms"] == {}

    def test_dm_to_non_followee_is_code_enforced(self, tmp_log):
        # Ring of 4: Alice follows Bob and Dave but NOT Carol. send_direct_message
        # is available (Alice has follows) but targeting the non-followee Carol —
        # a value the schema's choices don't offer — must still be rejected.
        env = make_social_env(tmp_log, topology="ring", n_nodes=4, max_connections=10)
        seed_n_agents(env, ["Alice", "Bob", "Carol", "Dave"])
        assert "send_direct_message" in env._get_avail_actions("a0")
        _, _, _, _, infos = env.step(
            {
                "a0": {
                    "action": "send_direct_message",
                    "message": "",
                    "params": {"target": "Carol", "content": "hi"},
                }
            }
        )
        assert infos["a0"]["send_direct_message"]["status"] == "failed"
        assert "follow" in infos["a0"]["send_direct_message"]["reason"].lower()


# ---------------------------------------------------------------------------
# 12. Spawn edge wiring
# ---------------------------------------------------------------------------


class TestSpawnEdges:
    def test_insufficient_energy_leaves_graph_unchanged(self, tmp_log):
        env = make_social_env(tmp_log, topology="ring", n_nodes=3)
        seed_n_agents(env, ["Alice", "Bob", "Carol"])
        # Energy can change after the action menu is built.
        env._get_avail_actions("a0")
        assert "spawn" in env.agent_avail_actions["a0"]
        env.agent_energy["a0"] = env.reproduction_cost - 1
        nodes_before = set(env.world_graph.all_nodes())
        edges_before = {
            (node, neighbor)
            for node in nodes_before
            for neighbor in env.world_graph.neighbors(node)
        }
        activity_before = dict(env._edge_last_active)
        agents_before = set(env.agent_registry)
        children_before = list(env.agent_spawn["a0"])

        _, _, _, _, infos = env.step(
            {"a0": {"action": "spawn", "message": "", "params": {"name": "Child"}}}
        )

        assert infos["a0"]["spawn"] == {
            "status": "failed",
            "reason": "Not enough energy",
        }
        assert set(env.world_graph.all_nodes()) == nodes_before
        assert {
            (node, neighbor)
            for node in env.world_graph.all_nodes()
            for neighbor in env.world_graph.neighbors(node)
        } == edges_before
        assert env._edge_last_active == activity_before
        assert set(env.agent_registry) == agents_before
        assert env.agent_spawn["a0"] == children_before
        assert "Child" not in env.name_to_tag
        assert env.agent_energy["a0"] == -1

    def test_exact_reproduction_cost_creates_child(self, tmp_log):
        env = make_social_env(tmp_log, topology="ring", n_nodes=3)
        seed_n_agents(env, ["Alice", "Bob", "Carol"])
        env.agent_energy["a0"] = env.reproduction_cost
        env._get_avail_actions("a0")
        nodes_before = set(env.world_graph.all_nodes())

        _, _, _, _, infos = env.step(
            {"a0": {"action": "spawn", "message": "", "params": {"name": "Child"}}}
        )

        spawn = infos["a0"]["spawn"]
        assert spawn["status"] == "successful"
        assert set(env.world_graph.all_nodes()) == nodes_before | {"Child"}
        assert env.agent_pos[spawn["child_tag"]] == "Child"
        assert env.agent_energy["a0"] == 0
        alice = env.agent_pos["a0"]
        assert env.world_graph.has_edge(alice, "Child")
        assert env.world_graph.has_edge("Child", alice)

    def test_paired_spawn_wires_both_parents_and_bumps_inter_parent(self, tmp_log):
        env = make_social_env(
            tmp_log, topology="ring", n_nodes=3, two_parent_spawn=True
        )
        seed_n_agents(env, ["Alice", "Bob", "Carol"])
        env.agent_energy["a0"] = 200
        env.agent_energy["a1"] = 200
        alice, bob, carol = (
            env.agent_pos["a0"],
            env.agent_pos["a1"],
            env.agent_pos["a2"],
        )
        for _ in range(3):
            noop(env, "a0")
        ab_before = env._edge_last_active[(alice, bob)]
        ac_before = env._edge_last_active[(alice, carol)]
        env._get_avail_actions("a0")
        _, _, _, _, infos = env.step(
            {
                "a0": {
                    "action": "spawn",
                    "message": "",
                    "params": {"name": "Child", "partner": "Bob"},
                }
            }
        )
        assert infos["a0"]["spawn"]["status"] == "successful"
        child = env.agent_pos[infos["a0"]["spawn"]["child_tag"]]
        # Mutual edges to BOTH parents.
        assert env.world_graph.has_edge(alice, child)
        assert env.world_graph.has_edge(child, alice)
        assert env.world_graph.has_edge(bob, child)
        assert env.world_graph.has_edge(child, bob)
        # Inter-parent edge bumped (both directions); unrelated edge untouched.
        assert env._edge_last_active[(alice, bob)] > ab_before
        assert env._edge_last_active[(bob, alice)] > ab_before
        assert env._edge_last_active[(alice, carol)] == ac_before


# ---------------------------------------------------------------------------
# 12b. Spawn affordance inheritance (per-agent granted/blocked actions;
#      excluded_actions is the generic per-agent gate). Offspring inherit
#      both; two parents → union.
# ---------------------------------------------------------------------------


def _aff_keys(env, tag):
    """(action, mode) pairs stored on an agent."""
    return [(a.action, a.mode) for a in env.agent_affordances.get(tag, [])]


class TestSpawnAffordanceInheritance:
    def _add_aff(self, env, tag, affordance):
        env.add_agent_affordance(tag, affordance)

    def _spawn(self, env, actor="a0", name="Child", partner=None):
        params = {"name": name}
        if partner is not None:
            params["partner"] = partner
        _, _, _, _, infos = env.step(
            {actor: {"action": "spawn", "message": "", "params": params}}
        )
        assert infos[actor]["spawn"]["status"] == "successful"
        return infos[actor]["spawn"]["child_tag"]

    def test_solo_spawn_inherits_excluded_actions(self, tmp_log):
        env = make_social_env(tmp_log, topology="ring", n_nodes=3)
        seed_n_agents(env, ["Alice", "Bob", "Carol"])
        env.agent_energy["a0"] = 200
        env.agent_excluded_actions["a0"] = {"follow"}
        child_tag = self._spawn(env)
        assert "follow" in env.agent_excluded_actions[child_tag]
        # The child's set is independent — mutating the parent doesn't leak.
        env.agent_excluded_actions["a0"].add("disconnect")
        assert "disconnect" not in env.agent_excluded_actions[child_tag]

    def test_paired_spawn_unions_agent_affordances_and_dedups(self, tmp_log):
        env = make_social_env(
            tmp_log, topology="ring", n_nodes=3, two_parent_spawn=True
        )
        seed_n_agents(env, ["Alice", "Bob", "Carol"])
        env.agent_energy["a0"] = 200
        env.agent_energy["a1"] = 200
        # Distinct grants on each parent + a blocked action both parents share.
        self._add_aff(
            env, "a0", LocationAffordance(action="recharge", mode="add", description="A")
        )
        self._add_aff(env, "a0", LocationAffordance(action="follow", mode="remove"))
        self._add_aff(
            env, "a1", LocationAffordance(action="teleport", mode="add", description="B")
        )
        self._add_aff(env, "a1", LocationAffordance(action="follow", mode="remove"))
        child_tag = self._spawn(env, partner="Bob")
        keys = _aff_keys(env, child_tag)
        assert ("recharge", "add") in keys  # from parent A
        assert ("teleport", "add") in keys  # from parent B
        # Shared blocked action appears exactly once (deduped by action+mode).
        assert keys.count(("follow", "remove")) == 1

    def test_paired_spawn_unions_excluded_actions(self, tmp_log):
        env = make_social_env(
            tmp_log, topology="ring", n_nodes=3, two_parent_spawn=True
        )
        seed_n_agents(env, ["Alice", "Bob", "Carol"])
        env.agent_energy["a0"] = 200
        env.agent_energy["a1"] = 200
        env.agent_excluded_actions["a0"] = {"follow"}
        env.agent_excluded_actions["a1"] = {"disconnect"}
        child_tag = self._spawn(env, partner="Bob")
        assert env.agent_excluded_actions[child_tag] == {"follow", "disconnect"}

    def test_inherited_agent_affordances_are_deep_copied(self, tmp_log):
        env = make_social_env(tmp_log, topology="ring", n_nodes=3)
        seed_n_agents(env, ["Alice", "Bob", "Carol"])
        env.agent_energy["a0"] = 200
        self._add_aff(
            env,
            "a0",
            LocationAffordance(action="recharge", mode="add", description="original"),
        )
        child_tag = self._spawn(env)
        # Mutating the child's affordance object must not bleed into the parent.
        child_aff = env.agent_affordances[child_tag][0]
        child_aff.description = "MUTATED"
        parent_aff = next(
            a for a in env.agent_affordances["a0"] if a.action == "recharge"
        )
        assert parent_aff.description == "original"


# ---------------------------------------------------------------------------
# 13. Directional edge decay
# ---------------------------------------------------------------------------


class TestEdgeDecayDirectional:
    def test_each_direction_decays_independently(self, tmp_log):
        DECAY = 5
        env = make_social_env(
            tmp_log,
            topology="ring",
            n_nodes=3,
            max_connections=10,
            edge_decay_steps=DECAY,
        )
        seed_n_agents(env, ["Alice", "Bob", "Carol"])
        alice, bob = env.agent_pos["a0"], env.agent_pos["a1"]
        # Alice DMs Bob every step → bumps Alice→Bob only. Bob never DMs Alice.
        for _ in range(DECAY + 2):
            env.step(
                {
                    "a0": {
                        "action": "send_direct_message",
                        "message": "",
                        "params": {"target": "Bob", "content": "hi"},
                    }
                }
            )
        # Alice→Bob kept (kept warm); Bob→Alice decayed away.
        assert env.world_graph.has_edge(alice, bob)
        assert not env.world_graph.has_edge(bob, alice)


# ---------------------------------------------------------------------------
# 14. give_artifact requires a mutual connection
# ---------------------------------------------------------------------------


class TestGiveArtifactBilateral:
    def _give_inventory_item(self, env, tag, node, name="gift"):
        env.seed_artifact(
            pose=node,
            art_type="text",
            art_name=name,
            payload="hello",
            lifespan=-1,
            movable=True,
        )
        env.agent_inventories[tag].add(name)
        env.artifacts[name].pose = f"Inv({tag})"
        env.artifact_location[name] = ("inv", tag)
        env.pos_artifacts[node].discard(name)

    def test_unavailable_when_only_one_way(self, tmp_log):
        env = make_social_env(
            tmp_log,
            topology="ring",
            n_nodes=3,
            max_connections=10,
            artifact_creation_cost=0,
            use_inventory=True,
        )
        seed_n_agents(env, ["Alice", "Bob", "Carol"])
        alice = env.agent_pos["a0"]
        self._give_inventory_item(env, "a0", alice)
        # Make every relationship one-way (Bob & Carol unfollow Alice).
        env.step({"a1": {"action": "disconnect", "message": "", "params": {"target": "Alice"}}})
        env.step({"a2": {"action": "disconnect", "message": "", "params": {"target": "Alice"}}})
        assert env._get_bilateral_names("a0") == []
        assert "give_artifact" not in env._get_avail_actions("a0")

    def test_available_and_bumps_both_directions_when_mutual(self, tmp_log):
        env = make_social_env(
            tmp_log,
            topology="ring",
            n_nodes=3,
            max_connections=10,
            artifact_creation_cost=0,
            use_inventory=True,
        )
        seed_n_agents(env, ["Alice", "Bob", "Carol"])
        alice, bob, carol = (
            env.agent_pos["a0"],
            env.agent_pos["a1"],
            env.agent_pos["a2"],
        )
        self._give_inventory_item(env, "a0", alice)
        actions = env._get_avail_actions("a0")
        assert "give_artifact" in actions
        # Choices restricted to mutual connections.
        assert set(actions["give_artifact"]["params"]["target_agent"]["choices"]) == {
            "Bob",
            "Carol",
        }
        for _ in range(3):
            noop(env, "a0")
        ab_before = env._edge_last_active[(alice, bob)]
        ba_before = env._edge_last_active[(bob, alice)]
        ac_before = env._edge_last_active[(alice, carol)]
        env.step(
            {
                "a0": {
                    "action": "give_artifact",
                    "message": "",
                    "params": {"artifact_name": "gift", "target_agent": "Bob"},
                }
            }
        )
        # BOTH directions of the Alice↔Bob edge bumped; unrelated edge untouched.
        assert env._edge_last_active[(alice, bob)] > ab_before
        assert env._edge_last_active[(bob, alice)] > ba_before
        assert env._edge_last_active[(alice, carol)] == ac_before

    def test_give_to_non_mutual_is_code_enforced(self, tmp_log):
        env = make_social_env(
            tmp_log,
            topology="ring",
            n_nodes=3,
            max_connections=10,
            artifact_creation_cost=0,
            use_inventory=True,
        )
        seed_n_agents(env, ["Alice", "Bob", "Carol"])
        alice = env.agent_pos["a0"]
        self._give_inventory_item(env, "a0", alice)
        # Bob unfollows Alice → Alice→Bob is one-way only.
        env.step({"a1": {"action": "disconnect", "message": "", "params": {"target": "Alice"}}})
        _, _, _, _, infos = env.step(
            {
                "a0": {
                    "action": "give_artifact",
                    "message": "",
                    "params": {"artifact_name": "gift", "target_agent": "Bob"},
                }
            }
        )
        assert "not a mutual connection" in infos["a0"]["Artifact give status"]
        # Artifact stayed with Alice.
        assert "gift" in env.agent_inventories["a0"]


# ---------------------------------------------------------------------------
# 15. Strict availability matrix
# ---------------------------------------------------------------------------


class TestAvailableActionsStrict:
    def test_no_disconnect_when_nothing_to_drop(self, tmp_log):
        # Ring of 4, then Alice unfollows everyone → no outgoing edges, no pending.
        env = make_social_env(tmp_log, topology="ring", n_nodes=4, max_connections=10)
        seed_n_agents(env, ["Alice", "Bob", "Carol", "Dave"])
        for target in ("Bob", "Dave"):  # Alice's ring follows
            env.step({"a0": {"action": "disconnect", "message": "", "params": {"target": target}}})
        assert env._get_subscriptions("a0") == []
        actions = env._get_avail_actions("a0")
        assert "disconnect" not in actions
        assert "send_direct_message" not in actions
        # But forming actions are available (targets Alice doesn't follow exist).
        assert "follow" in actions
        assert "request_connection" in actions


# ---------------------------------------------------------------------------
# 16. Death cleanup
# ---------------------------------------------------------------------------


class TestDeath:
    def test_kill_purges_node_edges_and_pending(self, tmp_log):
        env = make_social_env(tmp_log, topology="ring", n_nodes=4, max_connections=10)
        seed_n_agents(env, ["Alice", "Bob", "Carol", "Dave"])
        # Alice has a pending request out to Carol; Dave has one in to Alice.
        request_connection(env, "a0", "Carol")
        request_connection(env, "a3", "Alice")
        alice_node = env.agent_pos["a0"]
        env._kill("a0")
        assert alice_node not in set(env.world_graph.all_nodes())
        # Pending mirrors no longer reference Alice in either direction.
        assert "a0" not in env._pending_outgoing
        assert "a0" not in env._pending_incoming
        assert "a0" not in env._pending_incoming.get("a2", set())
        assert "a0" not in env._pending_outgoing.get("a3", {})


class TestRemoveAgent:
    def _env_with_manager_a0(self, tmp_log):
        """Env where a0's node carries the remove_agent affordance from init."""
        env = make_social_env(tmp_log, topology="ring", n_nodes=4, max_connections=10)
        # Seed before restart_env (a0 binds to the first node) so the affordance
        # is exposed from step 0, mirroring static config seeding.
        grant_remove_agent(env, env.world_graph.all_nodes()[0])
        seed_n_agents(env, ["Alice", "Bob", "Carol", "Dave"])
        return env

    def test_manager_fires_connected_agent(self, tmp_log):
        env = self._env_with_manager_a0(tmp_log)
        # Alice must be connected to Bob (manager → employee edge).
        follow(env, "a0", "Bob")
        bob_node = env.agent_pos["a1"]
        _, _, _, _, infos = remove_agent(env, "a0", "Bob")
        assert infos["a0"]["remove_agent"]["status"] == "successful"
        assert "a1" not in env.agent_registry
        assert bob_node not in set(env.world_graph.all_nodes())

    def test_remove_requires_connection(self, tmp_log):
        env = self._env_with_manager_a0(tmp_log)
        # No edge Alice → Carol.
        _, _, _, _, infos = remove_agent(env, "a0", "Carol")
        assert infos["a0"]["remove_agent"]["status"] == "failed"
        assert "a2" in env.agent_registry

    def test_remove_gated_by_affordance(self, tmp_log):
        env = make_social_env(tmp_log, topology="ring", n_nodes=4, max_connections=10)
        seed_n_agents(env, ["Alice", "Bob", "Carol", "Dave"])
        # Alice has NO remove_agent affordance, even though connected to Bob.
        follow(env, "a0", "Bob")
        remove_agent(env, "a0", "Bob")
        assert "a1" in env.agent_registry


class TestManagerAbilities:
    def _env_with_manager_a0(self, tmp_log, abilities):
        env = make_social_env(tmp_log, topology="ring", n_nodes=4, max_connections=10)
        grant_manager_powers(env, env.world_graph.all_nodes()[0], abilities)
        seed_n_agents(env, ["Alice", "Bob", "Carol", "Dave"])
        follow(env, "a0", "Bob")  # manager → employee edge
        return env

    # ---- set_role ----
    def test_set_role_surfaces_in_target_obs(self, tmp_log):
        env = self._env_with_manager_a0(tmp_log, ["set_role"])
        _, _, _, _, infos = set_role(env, "a0", "Bob", "team lead", "coordinate spawns")
        assert infos["a0"]["set_role"]["status"] == "successful"
        assert {key: env._agent_directives["a1"][key] for key in ("role", "motivation")} == {
            "role": "team lead",
            "motivation": "coordinate spawns",
        }
        assert env._agent_directives["a1"]["mandate_meta"]["context"] == {}
        obs, _ = env._build_obs("a1")
        assert obs["role"] == "team lead"
        assert "team lead" in obs["observation_text"]
        assert "coordinate spawns" in obs["observation_text"]

    # ---- grant_ability ----
    def test_grant_ability_adds_affordance(self, tmp_log):
        env = self._env_with_manager_a0(tmp_log, ["grant_ability"])
        assert not node_has_ability(env, "a1", "remove_agent")
        _, _, _, _, infos = grant_ability(env, "a0", "Bob", "remove_agent")
        assert infos["a0"]["grant_ability"]["status"] == "successful"
        assert node_has_ability(env, "a1", "remove_agent")

    def test_grant_unknown_ability_rejected(self, tmp_log):
        env = self._env_with_manager_a0(tmp_log, ["grant_ability"])
        _, _, _, _, infos = grant_ability(env, "a0", "Bob", "fly")
        assert infos["a0"]["grant_ability"]["status"] == "failed"
        assert not node_has_ability(env, "a1", "fly")

    def test_grant_duplicate_rejected(self, tmp_log):
        env = self._env_with_manager_a0(tmp_log, ["grant_ability"])
        grant_ability(env, "a0", "Bob", "remove_agent")
        _, _, _, _, infos = grant_ability(env, "a0", "Bob", "remove_agent")
        assert infos["a0"]["grant_ability"]["status"] == "failed"

    def test_promoted_agent_can_use_granted_ability(self, tmp_log):
        # Manager grants remove_agent to Bob; Bob then fires a connection of his.
        env = self._env_with_manager_a0(tmp_log, ["grant_ability"])
        grant_ability(env, "a0", "Bob", "remove_agent")
        follow(env, "a1", "Carol")  # Bob → Carol edge + refreshes Bob's avail
        _, _, _, _, infos = remove_agent(env, "a1", "Carol")
        assert infos["a1"]["remove_agent"]["status"] == "successful"
        assert "a2" not in env.agent_registry

    # ---- revoke_ability ----
    def test_revoke_ability_removes_affordance(self, tmp_log):
        env = self._env_with_manager_a0(tmp_log, ["grant_ability", "revoke_ability"])
        grant_ability(env, "a0", "Bob", "remove_agent")
        assert node_has_ability(env, "a1", "remove_agent")
        _, _, _, _, infos = revoke_ability(env, "a0", "Bob", "remove_agent")
        assert infos["a0"]["revoke_ability"]["status"] == "successful"
        assert not node_has_ability(env, "a1", "remove_agent")

    def test_revoke_absent_ability_rejected(self, tmp_log):
        env = self._env_with_manager_a0(tmp_log, ["revoke_ability"])
        _, _, _, _, infos = revoke_ability(env, "a0", "Bob", "remove_agent")
        assert infos["a0"]["revoke_ability"]["status"] == "failed"

    # ---- revoke / restore a global action ----
    def test_revoke_global_suppresses_action(self, tmp_log):
        env = self._env_with_manager_a0(tmp_log, ["revoke_ability"])
        _, _, _, _, infos = revoke_ability(env, "a0", "Bob", "follow")
        assert infos["a0"]["revoke_ability"]["status"] == "successful"
        assert node_suppresses(env, "a1", "follow")
        # After an avail refresh, follow is gone from Bob's available actions.
        noop(env, "a1")
        assert "follow" not in env.agent_avail_actions["a1"]

    def test_revoke_already_revoked_global_rejected(self, tmp_log):
        env = self._env_with_manager_a0(tmp_log, ["revoke_ability"])
        revoke_ability(env, "a0", "Bob", "spawn")
        _, _, _, _, infos = revoke_ability(env, "a0", "Bob", "spawn")
        assert infos["a0"]["revoke_ability"]["status"] == "failed"

    def test_grant_restores_revoked_global(self, tmp_log):
        env = self._env_with_manager_a0(tmp_log, ["grant_ability", "revoke_ability"])
        revoke_ability(env, "a0", "Bob", "follow")
        assert node_suppresses(env, "a1", "follow")
        _, _, _, _, infos = grant_ability(env, "a0", "Bob", "follow")
        assert infos["a0"]["grant_ability"]["status"] == "successful"
        assert not node_suppresses(env, "a1", "follow")

    def test_grant_unrevoked_global_rejected(self, tmp_log):
        env = self._env_with_manager_a0(tmp_log, ["grant_ability"])
        _, _, _, _, infos = grant_ability(env, "a0", "Bob", "follow")
        assert infos["a0"]["grant_ability"]["status"] == "failed"

    # ---- delegation ----
    def test_grant_ability_is_delegable(self, tmp_log):
        # Manager grants grant_ability to Bob; Bob then grants set_role to Carol.
        env = self._env_with_manager_a0(tmp_log, ["grant_ability"])
        grant_ability(env, "a0", "Bob", "grant_ability")
        follow(env, "a1", "Carol")  # Bob → Carol edge + refresh Bob's avail
        _, _, _, _, infos = grant_ability(env, "a1", "Carol", "set_role")
        assert infos["a1"]["grant_ability"]["status"] == "successful"
        assert node_has_ability(env, "a2", "set_role")

    # ---- self-targeting is rejected for every manager action ----
    def test_cannot_target_self(self, tmp_log):
        env = self._env_with_manager_a0(
            tmp_log, ["set_role", "grant_ability", "revoke_ability"]
        )
        _, _, _, _, i1 = set_role(env, "a0", "Alice", "boss")
        _, _, _, _, i2 = grant_ability(env, "a0", "Alice", "remove_agent")
        _, _, _, _, i3 = revoke_ability(env, "a0", "Alice", "follow")
        assert i1["a0"]["set_role"]["status"] == "failed"
        assert i2["a0"]["grant_ability"]["status"] == "failed"
        assert i3["a0"]["revoke_ability"]["status"] == "failed"
        assert "a0" not in env._agent_directives
        assert not node_has_ability(env, "a0", "remove_agent")
        assert not node_suppresses(env, "a0", "follow")

    # ---- persistence / cleanup ----
    def test_kill_purges_directive(self, tmp_log):
        env = self._env_with_manager_a0(tmp_log, ["set_role"])
        set_role(env, "a0", "Bob", "team lead")
        env._kill("a1")
        assert "a1" not in env._agent_directives


# ---------------------------------------------------------------------------
# 14. Shared artifact library (commons)
# ---------------------------------------------------------------------------


class TestLibrary:
    """deposit / retrieve / show, availability gating, death handoff, decay,
    the use_library ablation flag, and checkpoint persistence."""

    def _make(self, tmp_log, n_nodes=4, **kwargs):
        env = make_social_env(
            tmp_log,
            topology="ring",
            n_nodes=n_nodes,
            max_connections=10,
            artifact_creation_cost=0,
            use_inventory=True,
            **kwargs,
        )
        seed_n_agents(env, ["Alice", "Bob", "Carol", "Dave"][:n_nodes])
        return env

    def _seed_on_node(self, env, node, name, payload="some knowledge", movable=True, lifespan=-1):
        env.seed_artifact(
            pose=node,
            art_type="text",
            art_name=name,
            payload=payload,
            lifespan=lifespan,
            movable=movable,
        )

    def _step(self, env, tag, action, **params):
        # Refresh stored avail-actions to current state (the step gate reads the
        # stored set, not a fresh one) and drive the real dispatch path.
        env._get_avail_actions(tag)
        return env.step(
            {tag: {"action": action, "message": "", "params": params}}
        )

    # ---- deposit ----
    def test_deposit_moves_node_artifact_to_library(self, tmp_log):
        env = self._make(tmp_log)
        node = env.agent_pos["a0"]
        self._seed_on_node(env, node, "recipe")
        _, _, _, _, infos = self._step(env, "a0", "deposit_artifact", name="recipe")
        assert infos["a0"]["Artifact deposit status"] == "Success"
        assert env.library == {"recipe"}
        assert env.artifact_location["recipe"] == ("library", None)
        assert "recipe" not in env.pos_artifacts[node]
        assert env.artifacts["recipe"].pose == "Library"

    def test_deposit_from_inventory(self, tmp_log):
        env = self._make(tmp_log)
        node = env.agent_pos["a0"]
        self._seed_on_node(env, node, "tool")
        env.agent_inventories["a0"].add("tool")
        env.artifact_location["tool"] = ("inv", "a0")
        env.pos_artifacts[node].discard("tool")
        self._step(env, "a0", "deposit_artifact", name="tool")
        assert env.library == {"tool"}
        assert "tool" not in env.agent_inventories["a0"]

    def test_fixed_artifact_not_depositable(self, tmp_log):
        env = self._make(tmp_log)
        node = env.agent_pos["a0"]
        self._seed_on_node(env, node, "monument", movable=False)
        # a fixed artifact is never offered as depositable
        assert "deposit_artifact" not in env._get_avail_actions("a0")
        # and the handler rejects it defensively if invoked anyway
        infos = {"a0": {}}
        env._handle_deposit_artifact("a0", {"name": "monument"}, infos)
        assert "Failed" in infos["a0"]["Artifact deposit status"]
        assert "monument" not in env.library
        assert "monument" in env.pos_artifacts[node]

    def test_deposit_not_held_rejected(self, tmp_log):
        env = self._make(tmp_log)
        # artifact lives on Bob's node; Alice holds nothing depositable
        self._seed_on_node(env, env.agent_pos["a1"], "bobs_note")
        assert "deposit_artifact" not in env._get_avail_actions("a0")
        infos = {"a0": {}}
        env._handle_deposit_artifact("a0", {"name": "bobs_note"}, infos)
        assert "Failed" in infos["a0"]["Artifact deposit status"]
        assert "bobs_note" not in env.library

    # ---- retrieve ----
    def test_retrieve_moves_to_own_node(self, tmp_log):
        env = self._make(tmp_log)
        self._seed_on_node(env, env.agent_pos["a0"], "recipe")
        self._step(env, "a0", "deposit_artifact", name="recipe")
        bob_node = env.agent_pos["a1"]
        _, _, _, _, infos = self._step(env, "a1", "retrieve_artifact", name="recipe")
        assert infos["a1"]["Artifact retrieve status"] == "Success"
        assert "recipe" not in env.library
        assert "recipe" in env.pos_artifacts[bob_node]
        assert env.artifact_location["recipe"] == ("map", bob_node)

    def test_retrieve_unknown_name_fails(self, tmp_log):
        env = self._make(tmp_log)
        # populate the library so retrieve is offered, then ask for a bad name
        self._seed_on_node(env, env.agent_pos["a0"], "recipe")
        self._step(env, "a0", "deposit_artifact", name="recipe")
        _, _, _, _, infos = self._step(env, "a1", "retrieve_artifact", name="ghost")
        assert "Failed" in infos["a1"]["Artifact retrieve status"]
        assert env.library == {"recipe"}

    # ---- show ----
    def test_show_includes_preview_when_enabled(self, tmp_log):
        env = self._make(tmp_log, library_preview=True)
        self._seed_on_node(env, env.agent_pos["a0"], "recipe", payload="Bake bread at high heat")
        self._step(env, "a0", "deposit_artifact", name="recipe")
        _, _, _, _, infos = self._step(env, "a1", "show_library_content")
        content = infos["a1"]["Library content"]
        assert "recipe" in content and "Bake bread" in content

    def test_show_names_only_when_preview_disabled(self, tmp_log):
        env = self._make(tmp_log, library_preview=False)
        self._seed_on_node(env, env.agent_pos["a0"], "recipe", payload="Bake bread at high heat")
        self._step(env, "a0", "deposit_artifact", name="recipe")
        _, _, _, _, infos = self._step(env, "a1", "show_library_content")
        content = infos["a1"]["Library content"]
        assert "recipe" in content and "Bake bread" not in content

    # ---- availability gating ----
    def test_action_availability(self, tmp_log):
        env = self._make(tmp_log)
        # empty library, no holdings: only show is available
        acts = env._get_avail_actions("a0")
        assert "show_library_content" in acts
        assert "retrieve_artifact" not in acts
        assert "deposit_artifact" not in acts
        # holding a movable artifact exposes deposit
        self._seed_on_node(env, env.agent_pos["a0"], "recipe")
        assert "deposit_artifact" in env._get_avail_actions("a0")
        # a non-empty library exposes retrieve to everyone
        self._step(env, "a0", "deposit_artifact", name="recipe")
        assert "retrieve_artifact" in env._get_avail_actions("a1")

    def test_count_surfaced_in_observation(self, tmp_log):
        env = self._make(tmp_log)
        self._seed_on_node(env, env.agent_pos["a0"], "recipe")
        obs, _, _, _, _ = self._step(env, "a0", "deposit_artifact", name="recipe")
        assert "Shared library: 1 artifact" in obs["a0"]["observation_text"]

    # ---- death handoff ----
    def test_death_moves_node_and_inventory_artifacts_to_library(self, tmp_log):
        env = self._make(tmp_log)
        bob_node = env.agent_pos["a1"]
        # one in Bob's inventory, one already sitting on his node (not picked up)
        self._seed_on_node(env, bob_node, "held")
        env.agent_inventories["a1"].add("held")
        env.artifact_location["held"] = ("inv", "a1")
        env.pos_artifacts[bob_node].discard("held")
        self._seed_on_node(env, bob_node, "onnode")
        env._kill("a1")
        assert {"held", "onnode"} <= env.library
        assert env.artifact_location["held"] == ("library", None)
        assert env.artifact_location["onnode"] == ("library", None)
        assert bob_node not in set(env.world_graph.all_nodes())
        assert not env.pos_artifacts.get(bob_node)

    # ---- ablation flag ----
    def test_disabled_hides_actions_and_cleans_up_on_death(self, tmp_log):
        env = self._make(tmp_log, use_library=False)
        assert "show_library_content" not in env._get_avail_actions("a0")
        bob_node = env.agent_pos["a1"]
        self._seed_on_node(env, bob_node, "doomed")
        env._kill("a1")
        assert "doomed" not in env.artifacts
        assert "doomed" not in env.artifact_location
        assert not env.library
        assert bob_node not in set(env.world_graph.all_nodes())
        assert not env.pos_artifacts.get(bob_node)
        assert any(a.name == "doomed" for a in env.expired_artifacts)

    # ---- decay ----
    def test_library_artifact_decays(self, tmp_log):
        env = self._make(tmp_log)
        self._seed_on_node(env, env.agent_pos["a0"], "ephemeral", lifespan=2)
        self._step(env, "a0", "deposit_artifact", name="ephemeral")
        assert "ephemeral" in env.library
        for _ in range(4):
            noop(env, "a0")
        assert "ephemeral" not in env.library
        assert "ephemeral" not in env.artifacts
        assert any(a.name == "ephemeral" for a in env.expired_artifacts)

    # ---- persistence ----
    def test_library_survives_checkpoint(self, tmp_log):
        env = self._make(tmp_log)
        self._seed_on_node(env, env.agent_pos["a0"], "recipe")
        self._step(env, "a0", "deposit_artifact", name="recipe")
        ckpt = env.get_state_ckpt()
        env2 = self._make(tmp_log)
        env2.set_state_ckpt(ckpt)
        assert env2.library == {"recipe"}
        assert env2.artifact_location["recipe"] == ("library", None)


# ---------------------------------------------------------------------------
# Follow bootstrap (newborns must arrive heard)
# ---------------------------------------------------------------------------


class TestFollowBootstrap:
    def test_child_and_parent_are_wired(self, tmp_path):
        env = make_social_env(tmp_path, topology="complete", n_nodes=6)
        seed_n_agents(env, ["Alice", "Bob", "Carol"])
        env.add_agent("a9", "Kid", "no_traits", position=env.world_graph.all_nodes()[4])
        env.bootstrap_follows("a9", parent="a0")
        kid, parent = env.agent_pos["a9"], env.agent_pos["a0"]
        assert env.world_graph.has_edge(kid, parent)  # child follows parent
        assert env.world_graph.has_edge(parent, kid)  # parent hears the child
        assert "Kid" in env._get_followers("a0") or "a9" in env._get_followers("a0")

    def test_respawn_anchors_to_most_followed(self, tmp_path):
        env = make_social_env(tmp_path, topology="complete", n_nodes=6)
        placed = seed_n_agents(env, ["Alice", "Bob", "Carol"])
        # Make Bob the hub: Alice and Carol follow him.
        bob = placed[1][0]
        for tag, _, _ in (placed[0], placed[2]):
            env.world_graph.add_edge(env.agent_pos[tag], env.agent_pos[bob])
        env.add_agent("a9", "Kid", "no_traits", position=env.world_graph.all_nodes()[4])
        env.bootstrap_follows("a9")  # no parent: hub anchor
        assert env.world_graph.has_edge(env.agent_pos["a9"], env.agent_pos[bob])
