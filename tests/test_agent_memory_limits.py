"""Memory limits apply to incoming text and restored private memory."""

import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from terralingua.agents.human_agent import HumanAgent
from terralingua.agents.llm_agent import LLMAgent
from terralingua.agents.remote_agent import RemoteAgent
from terralingua.config.models import ExperimentConfig
from terralingua.experiment import runner as runner_module
from terralingua.genome.no_traits import Genome

OBS = {"observation_text": "current scene", "incoming_broadcasts": {},
       "inventory": [], "energy": 50, "time": 100}
ACTION = {"action": "noop", "params": {}, "message": ""}
ACTIONS = {"noop": {"description": "Wait", "params": {}}}


def agent(kind, limit, directory):
    cls = {"llm": LLMAgent, "remote": RemoteAgent, "human": HumanAgent}[kind]
    return cls(
        agent_name="Alice", agent_tag="a", genome=Genome(), internal_memory_size=limit,
        logger=SimpleNamespace(save_genome=lambda **_: None, log=lambda **_: None,
                               close=lambda: None, log_dir=directory),
        **({"motivation_prompt": ""} if kind == "remote" else {}),
    )


@pytest.mark.parametrize("kind", ["llm", "remote"])
@pytest.mark.parametrize("limit", [0, 1, 500])
@pytest.mark.parametrize("text", ["remember this lesson " * 1000, "研究 é🙂 café " * 1000])
def test_response_memory_obeys_exact_token_cap(tmp_path, kind, limit, text):
    owner = agent(kind, limit, tmp_path)
    response = {**ACTION, "internal_memory": text}
    if kind == "llm":
        memory = owner._parse_response(json.dumps(response), ACTIONS)[3]
    else:
        owner.record_action(OBS, response, {})
        memory = owner.internal_memory
    assert len(owner.internal_memory_encoder.encode(memory)) <= limit
    if limit == 0:
        assert memory == ""
    else:
        assert memory


@pytest.mark.parametrize("kind", ["llm", "remote"])
def test_memory_within_limit_is_unchanged(tmp_path, kind):
    owner = agent(kind, 50, tmp_path)
    text = "Keep this exact memory, including punctuation."
    assert owner.validate_internal_memory(text) == text


@pytest.mark.parametrize("kind", ["llm", "remote"])
@pytest.mark.parametrize("limit", [0, 5])
def test_checkpoint_memory_obeys_current_limit_without_replacing_system_text(tmp_path, kind, limit):
    original = agent(kind, limit, tmp_path / "original")
    original.internal_memory = "old memory " * 100
    original.system_prompt = "Saved system text"
    saved = original.get_state_ckpt()
    restored = agent(kind, limit, tmp_path / "restored")
    restored.set_state_ckpt(saved)
    try:
        assert len(restored.internal_memory_encoder.encode(restored.internal_memory)) <= limit
        assert restored.system_prompt == "Saved system text"
    finally:
        restored.close()


def test_remote_legacy_checkpoint_retains_resolved_energy_setting(tmp_path):
    original = agent("remote", 50, tmp_path / "original")
    saved = original.get_state_ckpt()
    saved.pop("energy_death")
    restored = agent("remote", 50, tmp_path / "restored")
    restored.energy_death = False
    restored.set_state_ckpt(saved)
    try:
        assert restored.energy_death is False
        restored.update_system_prompt()
        assert restored.energy_death is False
    finally:
        restored.close()


@pytest.mark.parametrize("limit", [0, 5, 500])
def test_human_generated_summary_obeys_configured_cap(tmp_path, monkeypatch, limit):
    owner = agent("human", limit, tmp_path)
    monkeypatch.setattr(owner, "_prompt_for_action", lambda *_: "noop")
    monkeypatch.setattr(owner, "_prompt_for_params", lambda *_: {})
    monkeypatch.setattr(owner, "_prompt_for_message", lambda: "")
    owner._select_action(OBS, ACTIONS, 0, {}, 0)
    assert len(owner.internal_memory_encoder.encode(owner.internal_memory)) <= limit
    assert bool(owner.internal_memory) is (limit > 0)
    if limit == 500:
        assert "steps_recorded" in owner.internal_memory


@pytest.mark.parametrize("legacy", [False, True])
def test_human_checkpoint_restores_memory_size_with_legacy_default(tmp_path, legacy):
    original = agent("human", 7, tmp_path / "original")
    original.internal_memory = "saved memory " * 100
    saved = original.get_state_ckpt()
    assert saved["internal_memory_size"] == 7
    if legacy:
        saved.pop("internal_memory_size")
    restored = agent("human", 3, tmp_path / "restored")
    restored.set_state_ckpt(saved)
    try:
        assert restored.internal_memory_size == (3 if legacy else 7)
        assert len(restored.internal_memory_encoder.encode(restored.internal_memory)) <= restored.internal_memory_size
    finally:
        restored.close()


@pytest.mark.parametrize("limit", [17])
def test_runner_passes_memory_limit_to_human_agent(tmp_path, monkeypatch, limit):
    runner = runner_module.SimulationRunner.__new__(runner_module.SimulationRunner)
    runner.params = ExperimentConfig(agent={"internal_memory_size": limit})
    runner.exp_logdir = tmp_path
    monkeypatch.setattr(runner_module, "HumanAgent", lambda **kwargs: SimpleNamespace(**kwargs))
    assert runner._make_human_agent("a").internal_memory_size == limit


@pytest.mark.parametrize("kind", ["llm", "remote"])
@pytest.mark.parametrize("limit", [0, 7])
def test_zero_budget_omits_memory_from_prompts_and_checkpoint_flag(tmp_path, kind, limit):
    owner = agent(kind, limit, tmp_path)
    owner.internal_memory = "private sentinel"
    _, prompt = owner._build_prompt_sync(OBS, ACTIONS, {})
    assert ('"internal_memory"' in prompt) is (limit > 0)
    assert ("Previous INTERNAL MEMORY" in prompt) is (limit > 0)
    assert ("private sentinel" in prompt) is (limit > 0)
    assert ("INTERNAL MEMORY" in owner.system_prompt) is (limit > 0)
    state = owner.get_state_ckpt()
    assert state["internal_memory_size"] == limit
    restored = agent(kind, 500, tmp_path)
    restored.set_state_ckpt(state)
    assert restored.use_internal_memory is (limit > 0)
    assert restored.internal_memory_size == limit
    if limit == 0:
        assert restored.internal_memory == ""


@pytest.mark.parametrize("limit", [0, 7])
def test_retry_reply_schema_uses_budget_before_any_memory_exists(tmp_path, limit):
    owner = agent("llm", limit, tmp_path)
    calls = []

    class Client:
        async def get_response_async(self, *, messages, chat_params):
            calls.append(deepcopy(messages))
            content = "bad reply" if len(calls) == 1 else json.dumps({
                **ACTION, "internal_memory": "new private summary",
            })
            return SimpleNamespace(content=content, input_tokens=0, output_tokens=0, cost=0)

    result = asyncio.run(owner.select_action_async(
        OBS, ACTIONS, 0, {}, 1, {}, Client(), max_attempts=2,
    ))
    assert result["action"] == "noop"
    assert len(calls) == 2
    retry_prompt = calls[1][-1]["content"]
    assert ('"internal_memory"' in retry_prompt) is (limit > 0)
    assert bool(owner.internal_memory) is (limit > 0)
