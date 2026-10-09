"""Temporary guidance can lose applicability without erasing roles or private context."""

import copy

import pytest

from tests.test_social_graph_env import (
    follow,
    grant_manager_powers,
    make_social_env,
    seed_n_agents,
)


def make_env(tmp_path, *, persistent=False):
    env = make_social_env(tmp_path, topology="ring", n_nodes=4, max_connections=10)
    grant_manager_powers(env, env.world_graph.all_nodes()[0], ["set_role"])
    seed_n_agents(env, ["Alice", "Bob", "Carol", "Dave"])
    follow(env, "a0", "Bob")
    if persistent:
        env._roles = {"worker": {
            "charter": "Coordinate useful work.", "mandate": "", "capacity": 1,
            "affordances": [], "on_vacancy": "open",
            "transferable": False, "start_nodes": [],
        }}
        env._role_assignment["a1"] = "worker"
    env.execution_context = {"run_id": "run-a", "world.episode": 1}
    return env


def set_mandate(env, **extra):
    params = {
        "target": "worker" if env._roles else "Bob", "role": "coordinator",
        "motivation": "Concentrate on the current task.", **extra,
    }
    env._get_avail_actions("a0")
    return env.step({"a0": {"action": "set_role", "params": params, "message": ""}})[-1]


@pytest.mark.parametrize("persistent", [False, True])
def test_context_change_marks_old_guidance_without_changing_context(tmp_path, persistent):
    env = make_env(tmp_path, persistent=persistent)
    infos = set_mandate(env)
    assert infos["a0"]["set_role"]["status"] == "successful"
    obs, _ = env._build_obs("a1")
    obs["external_response"] = ['{"measurement": 9}']
    obs["incoming_dms"] = {"Alice": "Keep the old story."}
    messages = copy.deepcopy(obs["incoming_dms"])
    external = list(obs["external_response"])
    role = obs["role"]
    assert "Current mandate:" in obs["motivation"]
    env.execution_context["world.episode"] = 2
    env.refresh_role_context("a1", obs)
    assert "REVIEW NEEDED" in obs["motivation"]
    assert "not a current requirement" in obs["motivation"]
    assert obs["role"] == role
    assert obs["incoming_dms"] == messages
    assert obs["external_response"] == external
    assert obs["observation_text"].count("Your role title:") == 1
    assert "Current mandate:" not in obs["observation_text"]
    if persistent:
        assert "Coordinate useful work." in obs["motivation"]
    snapshot = copy.deepcopy(obs)
    env.refresh_role_context("a1", obs)
    assert obs == snapshot


def test_deadline_and_explicit_unscoped_context(tmp_path):
    env = make_env(tmp_path)
    expiry = env.step_count + 4
    assert set_mandate(env, expires_at=expiry, context="{}")["a0"]["set_role"]["status"] == "successful"
    env.execution_context["world.episode"] = 99
    obs, _ = env._build_obs("a1")
    assert "REVIEW NEEDED" not in obs["motivation"]
    env.step_count = expiry
    env.refresh_role_context("a1", obs)
    assert f"expired at step {expiry}" in obs["motivation"]
    assert "Concentrate on the current task." in obs["motivation"]


def test_unknown_context_does_not_reactivate_a_scoped_mandate(tmp_path):
    env = make_env(tmp_path)
    set_mandate(env, context='{"world.episode": 2}')
    env.execution_context = {}
    obs, _ = env._build_obs("a1")
    assert "is unknown" in obs["motivation"]
    assert "REVIEW NEEDED" in obs["motivation"]


@pytest.mark.parametrize("extra", [
    {"expires_at": True}, {"expires_at": 1.5}, {"context": '{"nested": {"x": 1}}'},
])
def test_invalid_scope_does_not_replace_existing_mandate(tmp_path, extra):
    env = make_env(tmp_path, persistent=True)
    set_mandate(env)
    before = copy.deepcopy(env._roles["worker"])
    infos = set_mandate(env, **extra)
    assert infos["a0"]["set_role"]["status"] == "failed"
    assert env._roles["worker"] == before


def test_context_and_mandate_scope_survive_checkpoint(tmp_path):
    env = make_env(tmp_path)
    set_mandate(env, expires_at=50)
    checkpoint = env.get_state_ckpt()
    restored = make_env(tmp_path / "restored")
    restored.set_state_ckpt(checkpoint)
    assert restored.execution_context == env.execution_context
    assert restored._agent_directives == env._agent_directives
    restored.execution_context["world.episode"] = 2
    obs, _ = restored._build_obs("a1")
    assert "REVIEW NEEDED" in obs["motivation"]


def test_legacy_mandate_scope_is_explicitly_unknown(tmp_path):
    env = make_env(tmp_path, persistent=True)
    env._roles["worker"]["mandate"] = "Use the old policy."
    obs, _ = env._build_obs("a1")
    assert "context was not recorded" in obs["motivation"]
    assert "Use the old policy." in obs["motivation"]
    assert "Coordinate useful work." in obs["motivation"]
