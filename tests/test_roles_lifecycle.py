"""
Tests for the role lifecycle rules shared by both graph worlds
(terralingua/environment/roles.py, mixed into BaseWorld).

Covers: parsing and validation of `on_vacancy` and `transferable`; the
`retire` policy (leave_role hidden, a slot dies with its occupant, the role
leaves the board at zero slots); the `succeed` policy (the named successor
inherits on death or leave, fallback to an open slot, stale pointers dropped);
`transfer_role` (immediate, no consent, permissions move with the role, no
vacancy policy); permissions that follow a moving occupant in the plain graph
world; dispatch of role-granted actions; checkpoint roundtrip including older
checkpoints; the roles setting living at the world level of the config.
"""

import json

import pytest

from terralingua.config.models import EnvConfig, GraphConfig
from terralingua.environment.graph_env import OpenGraphWorld
from terralingua.environment.social_graph_env import OpenSocialGraphWorld

HOCON = """
{
  tools: [
    {
      name: "director"
      capacity: 1
      on_vacancy: "succeed"
      transferable: true
      instructions: "Run the place."
    }
    {
      name: "keeper"
      capacity: 2
      on_vacancy: "retire"
      instructions: "Keep order."
    }
    {
      name: "clerk"
      capacity: 1
      transferable: true
      instructions: "File things."
    }
  ]
}
"""

AFFORDANCES = {
    "director": {
        "affordances": [
            {
                "action": "inspect",
                "mode": "add",
                "description": "Inspect the premises.",
                "params": {},
                "effect": {"type": "inspect"},
            }
        ]
    },
    "keeper": {
        "affordances": [
            {
                "action": "patrol",
                "mode": "add",
                "description": "Walk the corridor.",
                "params": {},
                "effect": {"type": "patrol"},
            }
        ]
    },
}

N_AGENTS = 5  # 4 slots (director 1, keeper 2, clerk 1) -> exactly one roleless agent
KINDS = ("graph", "social")


def write_role_files(tmp_path, hocon=HOCON):
    tmp_path.mkdir(parents=True, exist_ok=True)
    hocon_path = tmp_path / "roles.hocon"
    hocon_path.write_text(hocon)
    aff_path = tmp_path / "role_affordances.json"
    aff_path.write_text(json.dumps(AFFORDANCES, indent=2))
    return str(hocon_path), str(aff_path)


def make_env(kind, tmp_path, topology="complete", n_agents=N_AGENTS, hocon=HOCON):
    hocon_path, aff_path = write_role_files(tmp_path, hocon)
    cfg = GraphConfig(topology=topology, n_nodes=n_agents, hop_radius=1)
    cls = OpenGraphWorld if kind == "graph" else OpenSocialGraphWorld
    env = cls(
        graph_cfg=cfg,
        roles_hocon_path=hocon_path,
        affordances_file_path=aff_path,
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
    env.restart_env(seed=3, agent_poses=poses)
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
    return set(env._get_avail_actions(tag))


# ---------------------------------------------------------------------------
# Parsing and validation
# ---------------------------------------------------------------------------


def test_lifecycle_fields_parsed_with_defaults(tmp_path):
    env = make_env("graph", tmp_path)
    assert env._roles["director"]["on_vacancy"] == "succeed"
    assert env._roles["director"]["transferable"] is True
    assert env._roles["keeper"]["on_vacancy"] == "retire"
    assert env._roles["keeper"]["transferable"] is False
    assert env._roles["clerk"]["on_vacancy"] == "open"
    assert env._roles["clerk"]["transferable"] is True


def test_invalid_on_vacancy_rejected(tmp_path):
    bad = HOCON.replace('on_vacancy: "retire"', 'on_vacancy: "vanish"')
    with pytest.raises(ValueError, match="on_vacancy"):
        make_env("graph", tmp_path, hocon=bad)


def test_invalid_transferable_rejected(tmp_path):
    bad = HOCON.replace("transferable: true", 'transferable: "yes"')
    with pytest.raises(ValueError, match="transferable"):
        make_env("graph", tmp_path, hocon=bad)


def test_zero_capacity_rejected(tmp_path):
    bad = HOCON.replace("capacity: 2", "capacity: 0")
    with pytest.raises(ValueError, match="capacity"):
        make_env("graph", tmp_path, hocon=bad)


def test_roles_config_is_a_world_level_setting(tmp_path):
    hocon_path, _ = write_role_files(tmp_path)
    cfg = EnvConfig(world_type="grid", init_agents=5, roles_hocon_path=hocon_path)
    assert cfg.roles_hocon_path == hocon_path
    with pytest.raises(ValueError, match="mutually"):
        EnvConfig(
            init_agents=5,
            roles_hocon_path=hocon_path,
            graph=GraphConfig(agent_network_hocon_path=hocon_path),
        )


# ---------------------------------------------------------------------------
# retire
# ---------------------------------------------------------------------------


def test_retire_role_cannot_be_left(tmp_path):
    env = make_env("graph", tmp_path)
    keeper = occupant_of(env, "keeper")
    assert "leave_role" not in menu(env, keeper)
    assert "patrol" in menu(env, keeper)
    infos = {}
    env._handle_leave_role(keeper, infos)
    assert infos[keeper]["leave_role"]["status"] == "failed"
    assert env.role_of(keeper) == "keeper"


def test_retire_slot_dies_with_occupant_and_role_leaves_board(tmp_path):
    env = make_env("graph", tmp_path)
    first, second = env._role_occupants("keeper")
    env._kill(first)
    assert env._roles["keeper"]["capacity"] == 1
    assert "keeper" not in env._open_roles()
    survivor = roleless(env)[0]
    obs, _ = env._build_obs(survivor)
    assert "- keeper (1/1" in obs["observation_text"]
    env._kill(second)
    assert env._roles["keeper"]["capacity"] == 0
    assert "keeper" not in env._open_roles()
    obs, _ = env._build_obs(survivor)
    assert "- keeper" not in obs["observation_text"]
    infos = {}
    env._handle_request_role(survivor, {"role": "keeper"}, infos)
    assert infos[survivor]["request_role"]["status"] == "failed"


# ---------------------------------------------------------------------------
# succeed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_named_successor_inherits_on_death(kind, tmp_path):
    env = make_env(kind, tmp_path)
    director = occupant_of(env, "director")
    heir = roleless(env)[0]
    heir_name = env.agent_names[heir]
    assert "name_successor" in menu(env, director)
    assert heir_name in menu_choices(env, director, "name_successor")
    infos = step_one(env, director, "name_successor", {"target": heir_name})
    assert infos[director]["name_successor"]["status"] == "successful"
    assert env._role_successor[director] == heir
    obs, _ = env._build_obs(director)
    assert f"Your named successor: {heir_name}" in obs["observation_text"]
    env._kill(director)
    assert env.role_of(heir) == "director"
    assert env._role_successor == {}
    assert "inspect" in menu(env, heir)


def test_named_successor_inherits_on_leave(tmp_path):
    env = make_env("graph", tmp_path)
    director = occupant_of(env, "director")
    heir = roleless(env)[0]
    step_one(env, director, "name_successor", {"target": env.agent_names[heir]})
    infos = step_one(env, director, "leave_role")
    assert infos[director]["leave_role"]["status"] == "successful"
    assert env.role_of(director) is None
    assert env.role_of(heir) == "director"


def test_succeed_falls_back_to_open_slot(tmp_path):
    env = make_env("graph", tmp_path)
    director = occupant_of(env, "director")
    # No successor named: the slot reopens.
    env._kill(director)
    assert env._role_occupants("director") == []
    assert env._open_roles().get("director") == 1
    # A successor who took another role meanwhile is skipped.
    newcomer = roleless(env)[0]
    infos = {}
    env._handle_request_role(newcomer, {"role": "director"}, infos)
    assert infos[newcomer]["request_role"]["status"] == "successful"
    clerk = occupant_of(env, "clerk")
    infos = {}
    env._handle_name_successor(newcomer, {"target": env.agent_names[clerk]}, infos)
    assert infos[newcomer]["name_successor"]["status"] == "successful"
    env._kill(newcomer)
    assert env.role_of(clerk) == "clerk"
    assert env._open_roles().get("director") == 1


def test_successor_pointer_dropped_when_successor_dies(tmp_path):
    env = make_env("graph", tmp_path)
    director = occupant_of(env, "director")
    heir = roleless(env)[0]
    step_one(env, director, "name_successor", {"target": env.agent_names[heir]})
    env._kill(heir)
    assert director not in env._role_successor
    env._kill(director)
    assert env._open_roles().get("director") == 1


def test_name_successor_refused_for_other_policies(tmp_path):
    env = make_env("graph", tmp_path)
    clerk = occupant_of(env, "clerk")
    assert "name_successor" not in menu(env, clerk)
    infos = {}
    env._handle_name_successor(clerk, {"target": env.agent_names[roleless(env)[0]]}, infos)
    assert infos[clerk]["name_successor"]["status"] == "failed"


# ---------------------------------------------------------------------------
# transfer_role
# ---------------------------------------------------------------------------


def menu_choices(env, tag, action):
    return env._get_avail_actions(tag)[action]["params"]["target"]["choices"]


@pytest.mark.parametrize("kind", KINDS)
def test_transfer_moves_role_and_permissions_without_consent(kind, tmp_path):
    env = make_env(kind, tmp_path)
    director = occupant_of(env, "director")
    receiver = roleless(env)[0]
    receiver_name = env.agent_names[receiver]
    assert "inspect" in menu(env, director)
    assert "inspect" not in menu(env, receiver)
    assert menu_choices(env, director, "transfer_role") == [receiver_name]
    infos = step_one(env, director, "transfer_role", {"target": receiver_name})
    assert infos[director]["transfer_role"]["status"] == "successful"
    assert env.role_of(director) is None
    assert env.role_of(receiver) == "director"
    assert "inspect" not in menu(env, director)
    assert "leave_role" not in menu(env, director)
    assert "inspect" in menu(env, receiver)
    assert env._roles["director"]["capacity"] == 1


def test_transfer_refused_to_role_holder_and_for_bound_roles(tmp_path):
    env = make_env("graph", tmp_path)
    director = occupant_of(env, "director")
    clerk = occupant_of(env, "clerk")
    infos = {}
    env._handle_transfer_role(director, {"target": env.agent_names[clerk]}, infos)
    assert infos[director]["transfer_role"]["status"] == "failed"
    assert env.role_of(director) == "director"
    keeper = occupant_of(env, "keeper")
    assert "transfer_role" not in menu(env, keeper)
    infos = {}
    env._handle_transfer_role(keeper, {"target": env.agent_names[roleless(env)[0]]}, infos)
    assert infos[keeper]["transfer_role"]["status"] == "failed"


def test_transfer_skips_vacancy_policy(tmp_path):
    env = make_env("graph", tmp_path)
    director = occupant_of(env, "director")
    clerk = occupant_of(env, "clerk")
    heir = roleless(env)[0]
    step_one(env, director, "name_successor", {"target": env.agent_names[clerk]})
    step_one(env, director, "transfer_role", {"target": env.agent_names[heir]})
    assert env.role_of(heir) == "director"
    assert env.role_of(clerk) == "clerk"
    assert director not in env._role_successor


# ---------------------------------------------------------------------------
# Permissions follow the occupant
# ---------------------------------------------------------------------------


def test_permissions_follow_moving_occupant_in_graph_world(tmp_path):
    env = make_env("graph", tmp_path, topology="ring")
    director = occupant_of(env, "director")
    start = env.agent_pos[director]
    target = env.world_graph.neighbors(start)[0]
    bystander = next(t for t in env.agent_registry if env.agent_pos[t] == target)
    step_one(env, director, "move", {"direction": target})
    assert env.agent_pos[director] == target
    assert "inspect" in menu(env, director)
    if env.role_of(bystander) != "director":
        assert "inspect" not in menu(env, bystander)
    newcomer = roleless(env)[0]
    env.agent_pos[newcomer] = start
    env.pos_to_agent[start].add(newcomer)
    assert "inspect" not in menu(env, newcomer)


@pytest.mark.parametrize("kind", KINDS)
def test_role_granted_action_dispatches_to_effect_handler(kind, tmp_path):
    env = make_env(kind, tmp_path)
    calls = []
    env._affordance_effects["inspect"] = lambda agent, params, infos, aff: calls.append(
        agent
    )
    director = occupant_of(env, "director")
    step_one(env, director, "inspect", {})
    assert calls == [director]
    intruder = roleless(env)[0]
    step_one(env, intruder, "inspect", {})
    assert calls == [director]


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def test_checkpoint_roundtrip_keeps_lifecycle_state(tmp_path):
    env = make_env("graph", tmp_path)
    director = occupant_of(env, "director")
    heir = roleless(env)[0]
    step_one(env, director, "name_successor", {"target": env.agent_names[heir]})
    env._kill(env._role_occupants("keeper")[0])
    env._set_role_mandate(director, {"target": "clerk", "role": "", "motivation": "File by date."}, {})
    ckpt = env.get_state_ckpt()
    assert json.dumps(ckpt["_roles"])
    env2 = make_env("graph", tmp_path / "restored")
    env2.set_state_ckpt(ckpt)
    assert env2._role_successor == {director: heir}
    assert env2._roles["keeper"]["capacity"] == 1
    assert env2._roles["director"]["on_vacancy"] == "succeed"
    assert env2._roles["clerk"]["mandate"] == "File by date."
    assert env2._role_assignment == env._role_assignment


def test_checkpoint_without_lifecycle_keys_loads_with_defaults(tmp_path):
    env = make_env("graph", tmp_path)
    ckpt = env.get_state_ckpt()
    for role in ckpt["_roles"].values():
        role.pop("on_vacancy")
        role.pop("transferable")
    ckpt.pop("_role_successor")
    ckpt.pop("agent_affordances")
    env2 = make_env("graph", tmp_path / "restored")
    env2.set_state_ckpt(ckpt)
    assert all(r["on_vacancy"] == "open" for r in env2._roles.values())
    assert all(r["transferable"] is False for r in env2._roles.values())
    assert env2._role_successor == {}
    assert env2.agent_affordances == {}
