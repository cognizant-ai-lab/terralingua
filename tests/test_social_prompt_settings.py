"""Agent prompts receive the world's configured social settings."""

from types import SimpleNamespace

import pytest

from terralingua.agents import llm_agent, prompt_templates, remote_agent
from terralingua.config.models import ExperimentConfig
from terralingua.experiment.runner import SimulationRunner
from terralingua.genome.no_traits import Genome as NoTraitsGenome


def capture_system_render(monkeypatch, kind):
    calls = []
    module = llm_agent if kind == "llm" else remote_agent

    def render(template_name, **kwargs):
        calls.append((template_name, kwargs))
        return "Rendered system instructions"

    monkeypatch.setattr(module, "render_system_prompt", render)
    return calls


def make_agent(kind, directory, **kwargs):
    cls = llm_agent.LLMAgent if kind == "llm" else remote_agent.RemoteAgent
    arguments = {
        "agent_name": "Alice",
        "agent_tag": "a0",
        "log_dir": directory,
        "genome": NoTraitsGenome(),
        "system_prompt_template": "social_graph.j2",
        **kwargs,
    }
    if kind == "remote":
        arguments["motivation_prompt"] = "Scenario instructions"
    return cls(**arguments)


@pytest.mark.parametrize("kind", ["llm", "remote"])
@pytest.mark.parametrize(
    "spawn_allowed,max_connections", [(False, None), (True, 0)]
)
def test_runner_uses_actual_world_settings(
    tmp_path, monkeypatch, kind, spawn_allowed, max_connections
):
    calls = capture_system_render(monkeypatch, kind)
    runner = SimulationRunner.__new__(SimulationRunner)
    runner.params = ExperimentConfig(
        agent={"scenario_specific_instructions": "none"},
        env={
            "world_type": "social_graph",
            "reproduction_cost": -1 if spawn_allowed else 20,
            "graph": {"max_connections": 17},
            "max_message_length": 341,
            "artifact_creation_cost": -1,
        },
    )
    runner.env = SimpleNamespace(
        system_prompt_template="social_graph.j2",
        energy_death=False,
        energy_upkeep=0,
        agent_energy={},
        init_agent_energy=100,
        reproduction_cost=20 if spawn_allowed else -1,
        spawn_allowed=spawn_allowed,
        max_connections=max_connections,
        stay_action={"action": "noop", "params": {}, "message": ""},
    )
    runner.exp_logdir = tmp_path
    runner.external_managers = []
    runner.hidden_external_keys = []
    runner.connection_manager = None
    if kind == "llm":
        agent = runner._make_llm_agent("a0", "Alice", NoTraitsGenome())
    else:
        agent = runner._make_remote_agent(
            "a0", "Alice", "Scenario instructions", NoTraitsGenome()
        )
    try:
        assert agent.spawn_allowed is spawn_allowed
        assert agent.max_connections == max_connections
        assert agent.max_message_length == 341
        template, settings = calls[-1]
        assert template == "social_graph.j2"
        assert settings["spawn_allowed"] is spawn_allowed
        assert settings["max_connections"] == max_connections
        assert settings["max_message_length"] == 341
        assert settings["artifact_creation"] is False
        assert agent.energy_death is False
        assert settings["energy_death"] is False
    finally:
        agent.close()


@pytest.mark.parametrize("kind", ["llm", "remote"])
def test_checkpoint_preserves_settings_and_saved_system_text(
    tmp_path, monkeypatch, kind
):
    calls = capture_system_render(monkeypatch, kind)
    original = make_agent(
        kind,
        tmp_path / "original",
        spawn_allowed=False,
        max_connections=0,
        max_message_length=347,
        food_mechanism=False,
        energy_death=True,
    )
    original.system_prompt = "Saved scenario, genome, and network instructions"
    saved = original.get_state_ckpt()
    original.close()
    assert saved["spawn_allowed"] is False
    assert saved["max_connections"] == 0
    assert saved["energy_death"] is True
    if kind == "remote":
        assert saved["max_message_length"] == 347

    restored = make_agent(
        kind,
        tmp_path / "restored",
        spawn_allowed=True,
        max_connections=9,
        max_message_length=22,
        energy_death=False,
    )
    restored.close()
    restored.set_state_ckpt(saved)
    try:
        assert restored.spawn_allowed is False
        assert restored.max_connections == 0
        assert restored.energy_death is True
        assert restored.system_prompt == saved["system_prompt"]
        assert (
            len(calls) == 2
        )  # Restore keeps the saved text without rendering it again.
        if kind == "remote":
            assert restored.max_message_length == 347
            restored.update_system_prompt()
            settings = calls[-1][1]
            assert settings["spawn_allowed"] is False
            assert settings["max_connections"] == 0
            assert settings["max_message_length"] == 347
            assert settings["energy_death"] is True
    finally:
        restored.close()


def test_remote_step_prompt_receives_configured_message_limit(tmp_path, monkeypatch):
    capture_system_render(monkeypatch, "remote")
    rendered = {}

    def render_step(**kwargs):
        rendered.update(kwargs)
        return "Rendered step instructions"

    monkeypatch.setattr(
        prompt_templates, "AGENT_PROMPT", SimpleNamespace(render=render_step)
    )
    agent = make_agent("remote", tmp_path, max_message_length=341)
    try:
        agent._make_prompt(
            formatted_obs={
                "observation": "Social state",
                "message": "Incoming messages",
                "energy": 50,
                "time": 20,
                "inventory": "<empty>",
            },
            available_actions={"noop": {"params": {}}},
            internal_memory="Remember this",
            info={},
        )
        assert rendered["max_message_length"] == 341
    finally:
        agent.close()
