"""
Tests for the general affordance effects of the graph world (drain_energy,
add_energy, move_agent), per-role start nodes, effect-less add-mode
affordances, and the world-log events they write.

The fixture is a small building layout with one keeper role and one resident
role, so the tests also cover role permissions in a world with movement.
"""

import json

import pytest

from terralingua.agents.prompt_templates import render_system_prompt
from terralingua.config.models import GraphConfig
from terralingua.environment.graph_env import OpenGraphWorld
from terralingua.environment.social_graph_env import OpenSocialGraphWorld

LAYOUT = {
    "nodes": [
        {"id": "room_1"},
        {"id": "room_2"},
        {"id": "hall"},
        {"id": "storeroom"},
        {"id": "office"},
    ],
    "edges": [
        {"from": "room_1", "to": "hall"},
        {"from": "hall", "to": "room_1"},
        {"from": "room_2", "to": "hall"},
        {"from": "hall", "to": "room_2"},
        {"from": "hall", "to": "office"},
        {"from": "office", "to": "hall"},
        {"from": "office", "to": "storeroom"},
        {"from": "storeroom", "to": "office"},
        {"from": "office", "to": "room_1"},
        {"from": "office", "to": "room_2"},
    ],
}

HOCON = """
{
  tools: [
    {
      name: "keeper"
      capacity: 1
      start_nodes: ["office"]
      instructions: "Keep order."
    }
    {
      name: "resident"
      capacity: 3
      start_nodes: ["room_1", "room_2"]
      instructions: "Obey."
    }
  ]
}
"""

AFFORDANCES = {
    "resident": {
        "affordances": [
            {"action": "move", "mode": "remove"},
            {
                "action": "wait",
                "mode": "add",
                "description": "Stay where you are this turn.",
                "params": {},
                "effect": {"type": "none"},
            },
        ]
    },
    "keeper": {
        "affordances": [
            {
                "action": "sanction",
                "mode": "add",
                "description": "Sanction a resident you can see.",
                "params": {"target": "Resident name.", "kind": "extra_chores or no_dessert"},
                "effect": {
                    "type": "drain_energy",
                    "amounts": {"extra_chores": 5, "no_dessert": 10},
                    "floor": 0,
                    "target_roles": ["resident"],
                },
            },
            {
                "action": "serve_meal",
                "mode": "add",
                "description": "Serve a meal to a resident you can see.",
                "params": {"target": "Resident name."},
                "effect": {
                    "type": "add_energy",
                    "amount": 15,
                    "cap": 210,
                    "target_roles": ["resident"],
                },
            },
            {
                "action": "escort",
                "mode": "add",
                "description": "Move a resident you can see.",
                "params": {"target": "Resident name.", "destination": "Node id."},
                "effect": {"type": "move_agent", "target_roles": ["resident"]},
            },
            {
                "action": "recharge",
                "mode": "add",
                "description": "Rest for a moment.",
                "params": {},
                "effect": {"type": "add_energy", "amount": 5},
            },
        ]
    },
}


def write_files(tmp_path, hocon_text=HOCON):
    tmp_path.mkdir(parents=True, exist_ok=True)
    layout = tmp_path / "layout.json"
    layout.write_text(json.dumps(LAYOUT, indent=2))
    hocon = tmp_path / "roles.hocon"
    hocon.write_text(hocon_text)
    aff = tmp_path / "aff.json"
    aff.write_text(json.dumps(AFFORDANCES, indent=2))
    return str(layout), str(hocon), str(aff)


def make_env(tmp_path, n_agents=4, cls=OpenGraphWorld, hocon_text=HOCON, **env_kwargs):
    layout, hocon, aff = write_files(tmp_path, hocon_text)
    cfg = GraphConfig(topology="file", file_path=layout, hop_radius=1)
    env = cls(
        graph_cfg=cfg,
        roles_hocon_path=hocon,
        affordances_file_path=aff,
        init_agent_energy=200,
        lifespan=500,
        init_food=0,
        food_spawn_rate=0,
        food_mechanism=False,
        log_path=tmp_path,
        drop_food_on_death=False,
        use_inventory=False,
        use_colors=False,
        artifact_creation_cost=-1,
        reproduction_cost=-1,
        headless=True,
        **env_kwargs,
    )
    # One agent per node at first; the role draw then moves occupants to
    # their start nodes. A roleless extra agent keeps its node.
    nodes = ["hall", "storeroom", "room_1", "room_2", "office"]
    poses = {}
    for i in range(n_agents):
        tag = f"a{i}"
        env.add_agent(tag, f"agent{i}", "no_traits", position=nodes[i])
        poses[tag] = nodes[i]
    env.restart_env(seed=1, agent_poses=poses)
    return env


def step_one(env, actor, action, params=None):
    _, _, _, _, infos = env.step(
        {actor: {"action": action, "message": "", "params": params or {}}}
    )
    return infos


def keeper_and_residents(env):
    keeper = env._role_occupants("keeper")[0]
    residents = sorted(env._role_occupants("resident"))
    return keeper, residents


def menu(env, tag):
    return set(env._get_avail_actions(tag))


def name(env, tag):
    return env.agent_names[tag]


def log_lines(tmp_path):
    return (tmp_path / "open_graph_world.log").read_text().splitlines()


# ---------------------------------------------------------------------------
# Placement and menus
# ---------------------------------------------------------------------------


def test_start_nodes_place_occupants_round_robin(tmp_path):
    env = make_env(tmp_path)
    keeper, residents = keeper_and_residents(env)
    assert env.agent_pos[keeper] == "office"
    assert [env.agent_pos[p] for p in residents] == ["room_1", "room_2", "room_1"]


def test_missing_start_node_rejected(tmp_path):
    bad = HOCON.replace('start_nodes: ["office"]', 'start_nodes: ["tower"]')
    with pytest.raises(ValueError, match="tower"):
        make_env(tmp_path, hocon_text=bad)


def test_social_graph_rejects_start_nodes(tmp_path):
    with pytest.raises(ValueError, match="start_nodes"):
        make_env(tmp_path, cls=OpenSocialGraphWorld)


def test_role_menus_in_a_world_with_movement(tmp_path):
    env = make_env(tmp_path)
    keeper, residents = keeper_and_residents(env)
    assert {"move", "sanction", "serve_meal", "escort", "recharge"} <= menu(env, keeper)
    for p in residents:
        assert "move" not in menu(env, p)
        assert "wait" in menu(env, p)
        assert "sanction" not in menu(env, p)


def test_wait_is_consumed_without_effect(tmp_path):
    env = make_env(tmp_path)
    _, residents = keeper_and_residents(env)
    infos = step_one(env, residents[0], "wait")
    assert "Artifact interaction result" not in infos[residents[0]]
    assert env.agent_energy[residents[0]] == 200
    assert env.agent_pos[residents[0]] == "room_1"


# ---------------------------------------------------------------------------
# drain_energy / add_energy
# ---------------------------------------------------------------------------


def test_punish_drains_by_kind_and_reports_both_sides(tmp_path):
    env = make_env(tmp_path)
    keeper, residents = keeper_and_residents(env)
    target = residents[0]
    infos = step_one(env, keeper, "sanction", {"target": name(env, target), "kind": "extra_chores"})
    assert env.agent_energy[target] == 195
    result = infos[keeper]["sanction"]
    assert result["status"] == "successful"
    assert result["kind"] == "extra_chores" and result["energy_change"] == -5
    note = infos[target]["sanction"]
    assert name(env, keeper) in note and "extra_chores" in note and "-5" in note


def test_punish_respects_floor(tmp_path):
    env = make_env(tmp_path)
    keeper, residents = keeper_and_residents(env)
    target = residents[0]
    env.agent_energy[target] = 3
    infos = step_one(env, keeper, "sanction", {"target": name(env, target), "kind": "no_dessert"})
    assert env.agent_energy[target] == 0
    assert infos[keeper]["sanction"]["energy_change"] == -3


def test_punish_refusals(tmp_path):
    env = make_env(tmp_path)
    keeper, residents = keeper_and_residents(env)
    target = residents[0]
    infos = step_one(env, keeper, "sanction", {"target": name(env, target), "kind": "yoga"})
    assert infos[keeper]["sanction"]["status"] == "failed"
    assert "Kinds" in infos[keeper]["sanction"]["reason"]
    infos = step_one(env, keeper, "sanction", {"target": name(env, keeper), "kind": "extra_chores"})
    assert infos[keeper]["sanction"]["status"] == "failed"
    infos = step_one(env, keeper, "sanction", {"target": "nobody", "kind": "extra_chores"})
    assert infos[keeper]["sanction"]["status"] == "failed"
    assert env.agent_energy[target] == 200


def test_punish_needs_a_visible_target(tmp_path):
    env = make_env(tmp_path)
    keeper, residents = keeper_and_residents(env)
    env._update_agent_pos(keeper, "storeroom")  # the storeroom sees only the keeper room
    infos = step_one(env, keeper, "sanction", {"target": name(env, residents[0]), "kind": "extra_chores"})
    assert infos[keeper]["sanction"]["status"] == "failed"
    assert "reach" in infos[keeper]["sanction"]["reason"]


def test_target_roles_enforced(tmp_path):
    env = make_env(tmp_path, n_agents=5)  # the fifth agent is roleless
    keeper, _ = keeper_and_residents(env)
    roleless = next(t for t in env.agent_registry if env.role_of(t) is None)
    infos = step_one(env, keeper, "sanction", {"target": name(env, roleless), "kind": "extra_chores"})
    assert infos[keeper]["sanction"]["status"] == "failed"
    assert "valid target" in infos[keeper]["sanction"]["reason"]
    assert env.agent_energy[roleless] == 200


def test_serve_meal_adds_and_caps(tmp_path):
    env = make_env(tmp_path)
    keeper, residents = keeper_and_residents(env)
    target = residents[0]
    step_one(env, keeper, "serve_meal", {"target": name(env, target)})
    assert env.agent_energy[target] == 210
    infos = step_one(env, keeper, "serve_meal", {"target": name(env, target)})
    assert env.agent_energy[target] == 210
    assert infos[keeper]["serve_meal"]["energy_change"] == 0


def test_add_energy_without_target_param_acts_on_self(tmp_path):
    env = make_env(tmp_path)
    keeper, _ = keeper_and_residents(env)
    infos = step_one(env, keeper, "recharge")
    assert env.agent_energy[keeper] == 205
    assert infos[keeper]["recharge"]["status"] == "successful"
    assert infos[keeper]["recharge"]["target"] == name(env, keeper)


# ---------------------------------------------------------------------------
# move_agent
# ---------------------------------------------------------------------------


def test_escort_next_to_target_or_actor(tmp_path):
    env = make_env(tmp_path)
    keeper, residents = keeper_and_residents(env)
    target = residents[0]  # in room_1; the keeper in office sees the cells
    infos = step_one(env, keeper, "escort", {"target": name(env, target), "destination": "hall"})
    assert infos[keeper]["escort"]["status"] == "successful"
    assert env.agent_pos[target] == "hall"
    assert "hall" in infos[target]["escort"]
    # The storeroom touches the keeper's place, not the resident's.
    step_one(env, keeper, "escort", {"target": name(env, target), "destination": "storeroom"})
    assert env.agent_pos[target] == "storeroom"
    # From the storeroom back to a cell: the cell touches the keeper room.
    step_one(env, keeper, "escort", {"target": name(env, target), "destination": "room_2"})
    assert env.agent_pos[target] == "room_2"


def test_escort_refusals(tmp_path):
    env = make_env(tmp_path)
    keeper, residents = keeper_and_residents(env)
    target = residents[0]
    infos = step_one(env, keeper, "escort", {"target": name(env, target), "destination": "room_1"})
    assert infos[keeper]["escort"]["status"] == "failed"
    infos = step_one(env, keeper, "escort", {"target": name(env, target), "destination": "tower"})
    assert infos[keeper]["escort"]["status"] == "failed"
    env._update_agent_pos(keeper, "hall")
    infos = step_one(env, keeper, "escort", {"target": name(env, target), "destination": "storeroom"})
    assert infos[keeper]["escort"]["status"] == "failed"
    assert env.agent_pos[target] == "room_1"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def test_effects_and_roles_are_logged(tmp_path):
    env = make_env(tmp_path)
    keeper, residents = keeper_and_residents(env)
    step_one(env, keeper, "sanction", {"target": name(env, residents[0]), "kind": "extra_chores"})
    lines = log_lines(tmp_path)
    assert any("ROLE_ASSIGNED" in line for line in lines)
    effect = next(json.loads(line) for line in lines if "AFFORDANCE_EFFECT" in line)
    assert effect["action"] == "sanction"
    assert effect["kind"] == "extra_chores"
    assert effect["energy_change"] == -5
    assert effect["target_tag"] == residents[0]


# ---------------------------------------------------------------------------
# Death at zero energy without food
# ---------------------------------------------------------------------------


def test_energy_zero_has_no_effect_by_default_without_food(tmp_path):
    env = make_env(tmp_path)
    assert env.energy_death is False
    keeper, residents = keeper_and_residents(env)
    target = residents[0]
    env.agent_energy[target] = 3
    step_one(env, keeper, "sanction", {"target": name(env, target), "kind": "no_dessert"})
    assert env.agent_energy[target] == 0
    assert target in env.agent_registry


def test_energy_death_kills_at_zero_without_food(tmp_path):
    env = make_env(tmp_path, energy_death=True)
    assert env.energy_death is True
    keeper, residents = keeper_and_residents(env)
    target = residents[0]
    env.agent_energy[target] = 3
    step_one(env, keeper, "sanction", {"target": name(env, target), "kind": "no_dessert"})
    assert target not in env.agent_registry
    assert env.role_of(target) is None
    died = next(json.loads(line) for line in log_lines(tmp_path) if "AGENT_DIED" in line)
    assert died["agent_tag"] == target
    assert died["reason"] == "energy"


def test_energy_death_prompt_text():
    with_death = render_system_prompt(
        "graph.j2",
        agent_name="A",
        food_mechanism=False,
        energy_death=True,
        scenario_specific_instructions="",
        genome_string="",
    )
    assert "When your energy reaches 0, you die." in with_death
    assert "refill your energy" not in with_death
    without = render_system_prompt(
        "graph.j2",
        agent_name="A",
        food_mechanism=False,
        energy_death=False,
        scenario_specific_instructions="",
        genome_string="",
    )
    assert "energy reaches 0" not in without
