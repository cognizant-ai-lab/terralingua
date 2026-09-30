"""
Tests for role permissions as per-agent affordances, in both graph worlds.

Covers: manager powers that coexist with role powers (one revoke removes the
power, one grant restores it), a block that dies with the role, the reason
given when a role blocks a global action, children inheriting node powers but
not role powers or manager-power blocks, the migration of checkpoints that
stored role permissions on nodes, the social menu order (noop before the role
actions), retired roles leaving the roster, and out-of-reach refusals for
transfer_role and name_successor.
"""

import json

import pytest

from terralingua.config.models import GraphConfig
from terralingua.environment.actions import LocationAffordance
from terralingua.environment.graph_env import OpenGraphWorld
from terralingua.environment.social_graph_env import (
    GRANTABLE_ABILITY_SPECS,
    OpenSocialGraphWorld,
)

HOCON = """
{
  tools: [
    {
      name: "boss"
      capacity: 1
      on_vacancy: "succeed"
      transferable: true
      instructions: "Lead."
    }
    {
      name: "clerk"
      capacity: 1
      instructions: "File."
    }
    {
      name: "temp"
      capacity: 1
      on_vacancy: "retire"
      instructions: "Help out."
    }
    {
      name: "inmate"
      capacity: 1
      instructions: "Wait."
    }
  ]
}
"""

AFFORDANCES = {
    "boss": {
        "affordances": [
            GRANTABLE_ABILITY_SPECS["grant_ability"],
            GRANTABLE_ABILITY_SPECS["revoke_ability"],
            GRANTABLE_ABILITY_SPECS["remove_agent"],
        ]
    },
    "clerk": {"affordances": [GRANTABLE_ABILITY_SPECS["remove_agent"]]},
    "inmate": {"affordances": [{"action": "spawn", "mode": "remove"}]},
}

KINDS = ("graph", "social")
N_SLOTS = 4


def make_env(kind, tmp_path, n_agents=6, topology="complete", roles=True):
    tmp_path.mkdir(parents=True, exist_ok=True)
    hocon = tmp_path / "roles.hocon"
    hocon.write_text(HOCON)
    aff = tmp_path / "aff.json"
    aff.write_text(json.dumps(AFFORDANCES, indent=2))
    cfg = GraphConfig(topology=topology, n_nodes=n_agents, hop_radius=1)
    cls = OpenGraphWorld if kind == "graph" else OpenSocialGraphWorld
    env = cls(
        graph_cfg=cfg,
        roles_hocon_path=str(hocon) if roles else None,
        affordances_file_path=str(aff) if roles else None,
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
    nodes = env.world_graph.all_nodes()
    poses = {}
    for i in range(n_agents):
        tag = f"a{i}"
        env.add_agent(tag, f"agent{i}", "no_traits", position=nodes[i])
        poses[tag] = nodes[i]
    env.restart_env(seed=5, agent_poses=poses)
    return env


def step_one(env, actor, action, params=None):
    _, _, _, _, infos = env.step(
        {actor: {"action": action, "message": "", "params": params or {}}}
    )
    return infos


def occupant_of(env, role_name):
    occupants = env._role_occupants(role_name)
    return occupants[0] if occupants else None


def roleless(env):
    return [t for t in sorted(env.agent_registry) if env.role_of(t) is None]


def menu(env, tag):
    return env._get_avail_actions(tag)


def name(env, tag):
    return env.agent_names[tag]


def spawn(env, actor):
    infos = step_one(env, actor, "spawn", {"name": f"Kid_{actor}"})
    assert infos[actor]["spawn"]["status"] == "successful"
    return infos[actor]["spawn"]["child_tag"]


# ---------------------------------------------------------------------------
# Manager powers next to role powers (social graph)
# ---------------------------------------------------------------------------


def test_one_revoke_removes_a_power_with_two_sources(tmp_path):
    env = make_env("social", tmp_path)
    boss = occupant_of(env, "boss")
    clerk = occupant_of(env, "clerk")
    newcomer = roleless(env)[0]
    infos = step_one(env, boss, "grant_ability", {"target": name(env, newcomer), "ability": "remove_agent"})
    assert infos[boss]["grant_ability"]["status"] == "successful"
    step_one(env, clerk, "leave_role")
    step_one(env, newcomer, "request_role", {"role": "clerk"})
    assert env.role_of(newcomer) == "clerk"
    assert "remove_agent" in menu(env, newcomer)
    infos = step_one(env, boss, "revoke_ability", {"target": name(env, newcomer), "ability": "remove_agent"})
    assert infos[boss]["revoke_ability"]["status"] == "successful"
    assert "remove_agent" not in menu(env, newcomer)
    assert not env._agent_effectively_has(newcomer, "remove_agent")
    infos = step_one(env, boss, "grant_ability", {"target": name(env, newcomer), "ability": "remove_agent"})
    assert infos[boss]["grant_ability"]["status"] == "successful"
    assert "remove_agent" in menu(env, newcomer)
    adds = [a for a in env.agent_affordances.get(newcomer, []) if a.action == "remove_agent"]
    assert adds == []  # the role is the only source again


def test_block_on_a_role_power_dies_with_the_role(tmp_path):
    env = make_env("social", tmp_path)
    boss = occupant_of(env, "boss")
    clerk = occupant_of(env, "clerk")
    step_one(env, boss, "revoke_ability", {"target": name(env, clerk), "ability": "remove_agent"})
    assert env._agent_has_affordance(clerk, "remove_agent", "remove")
    assert "remove_agent" not in menu(env, clerk)
    step_one(env, clerk, "leave_role")
    assert not env._agent_has_affordance(clerk, "remove_agent", "remove")
    infos = step_one(env, boss, "grant_ability", {"target": name(env, clerk), "ability": "remove_agent"})
    assert infos[boss]["grant_ability"]["status"] == "successful"
    assert "remove_agent" in menu(env, clerk)


def test_grant_names_the_role_that_blocks_a_global_action(tmp_path):
    env = make_env("social", tmp_path)
    boss = occupant_of(env, "boss")
    inmate = occupant_of(env, "inmate")
    assert "spawn" not in menu(env, inmate)
    infos = step_one(env, boss, "grant_ability", {"target": name(env, inmate), "ability": "spawn"})
    assert infos[boss]["grant_ability"]["status"] == "failed"
    assert "inmate" in infos[boss]["grant_ability"]["reason"]


def test_dispatch_refuses_a_blocked_power(tmp_path):
    env = make_env("social", tmp_path)
    boss = occupant_of(env, "boss")
    clerk = occupant_of(env, "clerk")
    step_one(env, boss, "revoke_ability", {"target": name(env, clerk), "ability": "remove_agent"})
    infos = {clerk: {}}
    claimed = env._on_location_action(clerk, "remove_agent", {"target": name(env, boss)}, infos)
    assert claimed is True
    assert infos[clerk]["remove_agent"]["status"] == "failed"
    assert boss in env.agent_registry


# ---------------------------------------------------------------------------
# Inheritance
# ---------------------------------------------------------------------------


def test_children_inherit_node_powers(tmp_path):
    env = make_env("social", tmp_path, roles=False)
    env.world_graph.add_node_affordance(
        env.agent_pos["a0"],
        LocationAffordance.from_dict(GRANTABLE_ABILITY_SPECS["set_role"]),
    )
    env.agent_energy["a0"] = 300
    child = spawn(env, "a0")
    assert "set_role" in menu(env, child)
    assert env._agent_has_affordance(child, "set_role", "add")


def test_children_inherit_neither_role_powers_nor_manager_blocks(tmp_path):
    env = make_env("social", tmp_path)
    boss = occupant_of(env, "boss")
    clerk = occupant_of(env, "clerk")
    step_one(env, boss, "revoke_ability", {"target": name(env, clerk), "ability": "remove_agent"})
    step_one(env, boss, "revoke_ability", {"target": name(env, clerk), "ability": "follow"})
    env.agent_energy[clerk] = 300
    child = spawn(env, clerk)
    child_affs = {(a.action, a.mode) for a in env.agent_affordances.get(child, [])}
    assert ("follow", "remove") in child_affs
    assert ("remove_agent", "remove") not in child_affs
    assert "remove_agent" not in menu(env, child)
    assert env.role_of(child) is None


# ---------------------------------------------------------------------------
# Checkpoints and menu order
# ---------------------------------------------------------------------------


def test_old_checkpoint_node_copies_of_role_powers_are_stripped(tmp_path):
    env = make_env("graph", tmp_path)
    boss = occupant_of(env, "boss")
    for aff in env._roles["boss"]["affordances"]:
        env.world_graph.add_node_affordance(env.agent_pos[boss], LocationAffordance.from_dict(aff))
    ckpt = env.get_state_ckpt()
    ckpt.pop("agent_affordances")
    env2 = make_env("graph", tmp_path / "restored")
    env2.set_state_ckpt(ckpt)
    boss_node = env2.agent_pos[boss]
    assert [a.action for a in env2.world_graph.node_affordances(boss_node)] == []
    assert "remove_agent" in menu(env2, boss)
    infos = {}
    env2._handle_leave_role(boss, infos)
    assert infos[boss]["leave_role"]["status"] == "successful"
    assert "remove_agent" not in menu(env2, boss)


def test_social_menu_lists_noop_before_role_actions(tmp_path):
    env = make_env("social", tmp_path)
    boss = occupant_of(env, "boss")
    keys = list(env._build_avail_actions(boss))
    assert keys.index("noop") < keys.index("leave_role")


# ---------------------------------------------------------------------------
# Retired roles
# ---------------------------------------------------------------------------


def test_retired_role_leaves_the_roster(tmp_path):
    env = make_env("graph", tmp_path)
    temp = occupant_of(env, "temp")
    boss = occupant_of(env, "boss")
    outsider = roleless(env)[0]
    env._kill(temp)
    infos = {}
    env._handle_request_role(outsider, {"role": "temp"}, infos)
    assert "Unknown role" in infos[outsider]["request_role"]["reason"]
    infos = {}
    env._set_role_mandate(boss, {"target": "temp", "role": "", "motivation": "x"}, infos)
    assert infos[boss]["set_role"]["status"] == "failed"


# ---------------------------------------------------------------------------
# Reach
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_transfer_and_successor_refused_out_of_reach(kind, tmp_path):
    env = make_env(kind, tmp_path, n_agents=8, topology="ring")
    boss = occupant_of(env, "boss")
    if kind == "graph":
        in_reach = set(env._get_nearby_agents(boss))
    else:
        in_reach = {env.name_to_tag[n] for n in env._get_subscriptions(boss)}
    far = [t for t in roleless(env) if t not in in_reach]
    assert far, "the ring must leave some roleless agent out of reach"
    target = far[0]
    for action, handler in (
        ("transfer_role", env._handle_transfer_role),
        ("name_successor", env._handle_name_successor),
    ):
        infos = {}
        handler(boss, {"target": name(env, target)}, infos)
        assert infos[boss][action]["status"] == "failed"
        offered = menu(env, boss)
        if action in offered:
            assert name(env, target) not in offered[action]["params"]["target"]["choices"]
    assert env.role_of(boss) == "boss"
    assert boss not in env._role_successor
