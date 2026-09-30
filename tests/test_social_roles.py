"""
Tests for persistent roles in OpenSocialGraphWorld (env.roles_hocon_path).

Covers: the one-time random initial assignment (capacity respected, extra
agents roleless), grant-only request_role / leave_role with permissions that
move to the new occupant, and the roles-mode set_role (mandate on the ROLE,
persists across occupants). Lifecycle rules (on_vacancy, transferable) are
covered in test_roles_lifecycle.py.
"""

from pathlib import Path

from terralingua.config.models import GraphConfig
from terralingua.environment.social_graph_env import OpenSocialGraphWorld

ROLES_HOCON = str(Path(__file__).resolve().parent / "fixtures/team_roles.hocon")
ROLES_AFFORDANCES = str(
    Path(__file__).resolve().parent / "fixtures/team_roles_affordances.json"
)

ROLE_CAPACITIES = {
    "founder": 1,
    "planner": 2,
    "analist": 2,
    "manager": 1,
    "knowledge_manager": 1,
}
TOTAL_SLOTS = sum(ROLE_CAPACITIES.values())


def make_roles_env(tmp_path, n_agents=10, affordances=True):
    cfg = GraphConfig(topology="complete", n_nodes=n_agents, hop_radius=1)
    env = OpenSocialGraphWorld(
        graph_cfg=cfg,
        roles_hocon_path=ROLES_HOCON,
        affordances_file_path=ROLES_AFFORDANCES if affordances else None,
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
    env.restart_env(seed=7, agent_poses=poses)
    return env


def step_one(env, actor, action, params=None):
    _, _, _, _, infos = env.step(
        {actor: {"action": action, "message": "", "params": params or {}}}
    )
    return infos


def role_of(env, tag):
    return env._role_assignment.get(tag)


def occupant_of(env, role_name):
    occupants = env._role_occupants(role_name)
    return occupants[0] if occupants else None


def permitted_actions(env, tag):
    return set(env._get_avail_actions(tag))


# ---------------------------------------------------------------------------
# Initial assignment
# ---------------------------------------------------------------------------


def test_initial_assignment_fills_slots_and_leaves_extras_roleless(tmp_path):
    env = make_roles_env(tmp_path, n_agents=10)
    assigned = env._role_assignment
    assert len(assigned) == TOTAL_SLOTS
    for name, cap in ROLE_CAPACITIES.items():
        assert len(env._role_occupants(name)) == cap
    roleless = [f"a{i}" for i in range(10) if f"a{i}" not in assigned]
    assert len(roleless) == 10 - TOTAL_SLOTS


def test_initial_assignment_with_fewer_agents_than_slots(tmp_path):
    env = make_roles_env(tmp_path, n_agents=3)
    assert len(env._role_assignment) == 3
    for name in ROLE_CAPACITIES:
        assert len(env._role_occupants(name)) <= ROLE_CAPACITIES[name]


# ---------------------------------------------------------------------------
# request_role / leave_role
# ---------------------------------------------------------------------------


def test_request_role_grant_and_refusals(tmp_path):
    env = make_roles_env(tmp_path, n_agents=10)
    roleless = [f"a{i}" for i in range(10) if role_of(env, f"a{i}") is None]
    holder = occupant_of(env, "founder")

    # The refusal paths are normally unreachable through step() because the
    # availability layer never offers the action — exercise the handler
    # guards directly (defense in depth for hand-built params).
    infos = {}
    env._handle_request_role(holder, {"role": "planner"}, infos)
    assert infos[holder]["request_role"]["status"] == "failed"

    infos = {}
    env._handle_request_role(roleless[0], {"role": "founder"}, infos)
    assert infos[roleless[0]]["request_role"]["status"] == "failed"

    infos = {}
    env._handle_request_role(roleless[0], {"role": "wizard"}, infos)
    assert infos[roleless[0]]["request_role"]["status"] == "failed"

    # Free a slot, then the request succeeds and permissions transfer.
    infos = step_one(env, holder, "leave_role")
    assert infos[holder]["leave_role"]["status"] == "successful"
    assert "set_role" not in permitted_actions(env, holder)
    infos = step_one(env, roleless[0], "request_role", {"role": "founder"})
    assert infos[roleless[0]]["request_role"]["status"] == "successful"
    assert role_of(env, roleless[0]) == "founder"
    assert "set_role" in permitted_actions(env, roleless[0])


def test_leave_role_without_role_fails(tmp_path):
    env = make_roles_env(tmp_path, n_agents=10)
    roleless = [f"a{i}" for i in range(10) if role_of(env, f"a{i}") is None][0]
    infos = {}
    env._handle_leave_role(roleless, infos)
    assert infos[roleless]["leave_role"]["status"] == "failed"


# ---------------------------------------------------------------------------
# Mandates (roles-mode set_role)
# ---------------------------------------------------------------------------


def test_set_role_updates_role_mandate_and_survives_occupant_change(tmp_path):
    env = make_roles_env(tmp_path, n_agents=10)
    founder = occupant_of(env, "founder")
    infos = step_one(
        env,
        founder,
        "set_role",
        {"target": "planner", "role": "", "motivation": "Plan day by day."},
    )
    assert infos[founder]["set_role"]["status"] == "successful"
    assert env._roles["planner"]["mandate"] == "Plan day by day."

    # The occupant's observation carries charter + mandate.
    planner = occupant_of(env, "planner")
    obs, _ = env._build_obs(planner, {}, {})
    assert "Plan day by day." in obs["observation_text"]
    assert obs["role"] == "planner"

    # The mandate persists across an occupant change.
    step_one(env, planner, "leave_role")
    roleless = [f"a{i}" for i in range(10) if role_of(env, f"a{i}") is None][0]
    step_one(env, roleless, "request_role", {"role": "planner"})
    obs, _ = env._build_obs(roleless, {}, {})
    assert "Plan day by day." in obs["observation_text"]
