"""Tests for the flattened external-action mechanism (lives in base_env).

The runner discovers each MCP server's tools and hands the env an
`external_actions_spec` keyed by `<server>_<tool>`. The env surfaces one action
per spec entry — with that server's per-action description (tool doc + energy
cost) — but only when the agent can afford its per-server energy cost. It also
deducts the per-action cost on execution.
"""

from terralingua.environment.grid_env import OpenGridWorld

SERVER_DOC = 'Payload: {"price": 5.0} or {"observe": true}'


def _spec(cost=0, mode="independent"):
    return {
        "market_world_server": {
            "server": "market",
            "tool": "world_server",
            "cost": cost,
            "mode": mode,
            "description": f"{SERVER_DOC} Costs {cost} energy.",
            "params": {"payload": "The content of the call in text form."},
        }
    }


def _make_grid_env(tmp_path, **kwargs):
    env = OpenGridWorld(
        grid_size=10, log_path=tmp_path, headless=True, food_mechanism=False, **kwargs
    )
    env.add_agent("a0", "Alice", "no_traits", position=(2, 2))
    env.restart_env(agent_poses={"a0": (2, 2)})
    return env


def test_absent_when_energy_below_cost(tmp_path):
    env = _make_grid_env(tmp_path, external_actions_spec=_spec(cost=50))
    env.agent_energy["a0"] = 10  # below the 50-energy cost → not offered
    actions = env._get_avail_actions("a0", nearby_agents=False)
    assert "market_world_server" not in actions


def test_per_action_cost_deducted_on_step(tmp_path):
    env = _make_grid_env(tmp_path, external_actions_spec=_spec(cost=10))
    before = env.agent_energy["a0"]
    env.step(
        {"a0": {"action": "market_world_server", "params": {"payload": "hi"}}}
    )
    assert env.agent_energy["a0"] == before - 10
