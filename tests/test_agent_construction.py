"""Tests for LLMAgent and RemoteAgent construction with per-env templates.

Verifies that:
  - Both agent classes construct successfully with an env's template name.
  - The rendered `system_prompt` contains env-specific phrases.
  - Checkpoint round-trip preserves `system_prompt_template` (the field that
    replaced the old `obs_style` knob).
"""

import asyncio
import json as _json

import pytest

from terralingua.agents.llm_agent import LLMAgent
from terralingua.agents.remote_agent import RemoteAgent
from terralingua.environment.graph_env import OpenGraphWorld
from terralingua.environment.grid_env import OpenGridWorld
from terralingua.environment.social_graph_env import OpenSocialGraphWorld
from terralingua.genome.no_traits import Genome as NoTraitsGenome
from terralingua.genome.ocean_5 import Genome as Ocean5Genome
from terralingua.utils.llm_client import Response

# ---------------------------------------------------------------------------
# LLMAgent — constructor + system prompt content
# ---------------------------------------------------------------------------


class TestLLMAgentConstruction:
    @pytest.mark.parametrize(
        "env_cls, expected_phrase",
        [
            (OpenGridWorld, "2D grid world"),
            (OpenGraphWorld, "world of connected locations"),
            (OpenSocialGraphWorld, "network of other agents"),
        ],
    )
    def test_constructs_with_each_env_template(self, tmp_path, env_cls, expected_phrase):
        agent = LLMAgent(
            agent_name="Alice",
            agent_tag="a0",
            log_dir=tmp_path,
            system_prompt_template=env_cls.system_prompt_template,
            genome=NoTraitsGenome(),
        )
        assert agent.system_prompt_template == env_cls.system_prompt_template
        assert expected_phrase in agent.system_prompt


# ---------------------------------------------------------------------------
# LLMAgent — checkpoint round-trip
# ---------------------------------------------------------------------------


class TestLLMAgentCheckpoint:
    def test_checkpoint_preserves_template_name(self, tmp_path):
        original = LLMAgent(
            agent_name="Alice",
            agent_tag="a0",
            log_dir=tmp_path,
            system_prompt_template="social_graph.j2",
            genome=NoTraitsGenome(),
        )
        ckpt = original.get_state_ckpt()
        assert ckpt["system_prompt_template"] == "social_graph.j2"

        # Restore into a fresh agent built with a DIFFERENT template.
        restored = LLMAgent(
            agent_name="Placeholder",
            agent_tag="placeholder",
            log_dir=tmp_path,
            system_prompt_template="grid.j2",
            genome=NoTraitsGenome(),
        )
        restored.set_state_ckpt(ckpt)
        assert restored.system_prompt_template == "social_graph.j2"
        # The actual rendered prompt should also be restored (it was saved in the ckpt).
        assert "network of other agents" in restored.system_prompt

    def test_solo_flag_drops_social_prompt_and_survives_checkpoint(
        self, tmp_path
    ):
        solo = LLMAgent(
            agent_name="Alice",
            agent_tag="a0",
            log_dir=tmp_path,
            system_prompt_template="social_graph.j2",
            genome=NoTraitsGenome(),
            solo=True,
        )
        assert "broadcast" not in solo.system_prompt.lower()
        assert "follow" not in solo.system_prompt.lower()
        assert "autonomous agent" in solo.system_prompt
        assert "node" not in solo.system_prompt.lower()
        assert "cannot move" not in solo.system_prompt.lower()

        restored = LLMAgent(
            agent_name="Placeholder",
            agent_tag="placeholder",
            log_dir=tmp_path,
            system_prompt_template="grid.j2",
            genome=NoTraitsGenome(),
        )
        restored.set_state_ckpt(solo.get_state_ckpt())
        assert restored.solo is True
        # Old checkpoints without the key restore as not-solo.
        legacy = solo.get_state_ckpt()
        del legacy["solo"]
        restored.set_state_ckpt(legacy)
        assert restored.solo is False

    def test_parse_response_fills_optional_params(self, tmp_path):
        """Params the spec marks optional may be omitted (filled with "");
        required params stay strict."""
        agent = LLMAgent(
            agent_name="Alice",
            agent_tag="a0",
            log_dir=tmp_path,
            system_prompt_template="grid.j2",
            genome=NoTraitsGenome(),
        )
        actions = {
            "deposit": {
                "params": {"claim": "", "evidence": "", "status": ""},
                "optional": ["evidence", "status"],
            }
        }
        _, _, params, _, _ = agent._parse_response(
            '{"action": "deposit", "message": "", '
            '"params": {"claim": "it pays"}}',
            available_actions=actions,
        )
        assert params == {"claim": "it pays", "evidence": "", "status": ""}
        with pytest.raises(ValueError, match="MISSING"):
            agent._parse_response(
                '{"action": "deposit", "message": "", "params": {}}',
                available_actions=actions,
            )

    def test_parse_response_accepts_missing_message_field(self, tmp_path):
        agent = LLMAgent(
            agent_name="Alice",
            agent_tag="a0",
            log_dir=tmp_path,
            system_prompt_template="social_graph.j2",
            genome=NoTraitsGenome(),
            solo=True,
        )
        act, message, params, _, _ = agent._parse_response(
            '{"action": "noop", "params": {}}',
            available_actions={"noop": {"params": {}}},
        )
        assert act == "noop"
        assert message == ""
        assert params == {}


# ---------------------------------------------------------------------------
# RemoteAgent — constructor + system prompt content
# ---------------------------------------------------------------------------


class TestRemoteAgentConstruction:
    @pytest.mark.parametrize(
        "env_cls, expected_phrase",
        [
            (OpenSocialGraphWorld, "network of other agents"),
        ],
    )
    def test_constructs_with_each_env_template(self, tmp_path, env_cls, expected_phrase):
        agent = RemoteAgent(
            agent_name="Alice",
            agent_tag="a0",
            motivation_prompt="Be helpful.",
            log_dir=tmp_path,
            system_prompt_template=env_cls.system_prompt_template,
            genome=NoTraitsGenome(),
        )
        assert agent.system_prompt_template == env_cls.system_prompt_template
        assert expected_phrase in agent.system_prompt


# ---------------------------------------------------------------------------
# RemoteAgent — checkpoint round-trip
# ---------------------------------------------------------------------------


class TestRemoteAgentUpdateSystemPrompt:
    """Covers the dashboard's runtime update path:
    motivation_prompt and/or genome change → agent.update_system_prompt() →
    the rendered prompt reflects all current attribute values.
    """

    def test_update_after_motivation_change(self, tmp_path):
        agent = RemoteAgent(
            agent_name="Alice",
            agent_tag="a0",
            motivation_prompt="Initial motivation MARKER-A",
            log_dir=tmp_path,
            system_prompt_template="grid.j2",
            genome=NoTraitsGenome(),
        )
        assert "MARKER-A" in agent.system_prompt
        agent.motivation_prompt = "Updated motivation MARKER-B"
        agent.update_system_prompt()
        assert "MARKER-B" in agent.system_prompt
        assert "MARKER-A" not in agent.system_prompt

    def test_update_after_genome_change(self, tmp_path):
        agent = RemoteAgent(
            agent_name="Alice",
            agent_tag="a0",
            motivation_prompt="Be helpful.",
            log_dir=tmp_path,
            system_prompt_template="grid.j2",
            genome=NoTraitsGenome(),
        )
        # NoTraits → no trait block in prompt initially.
        assert "Your Traits" not in agent.system_prompt
        agent.genome = Ocean5Genome()
        agent.update_system_prompt()
        # Now Ocean5 traits should appear.
        assert "Your Traits" in agent.system_prompt or "honesty" in agent.system_prompt

    def test_update_after_both_motivation_and_genome_change(self, tmp_path):
        agent = RemoteAgent(
            agent_name="Alice",
            agent_tag="a0",
            motivation_prompt="Initial motivation MARKER-A",
            log_dir=tmp_path,
            system_prompt_template="grid.j2",
            genome=NoTraitsGenome(),
        )
        # Both change at once — single update_system_prompt call must reflect both.
        agent.motivation_prompt = "New motivation MARKER-Z"
        agent.genome = Ocean5Genome()
        agent.update_system_prompt()
        assert "MARKER-Z" in agent.system_prompt
        assert "MARKER-A" not in agent.system_prompt
        assert "Your Traits" in agent.system_prompt or "honesty" in agent.system_prompt


class TestRemoteAgentCheckpoint:
    def test_checkpoint_preserves_template_name(self, tmp_path):
        original = RemoteAgent(
            agent_name="Alice",
            agent_tag="a0",
            motivation_prompt="Be helpful.",
            log_dir=tmp_path,
            system_prompt_template="graph.j2",
            genome=NoTraitsGenome(),
        )
        ckpt = original.get_state_ckpt()
        assert ckpt["system_prompt_template"] == "graph.j2"

        restored = RemoteAgent(
            agent_name="Placeholder",
            agent_tag="placeholder",
            motivation_prompt="Be helpful.",
            log_dir=tmp_path,
            system_prompt_template="grid.j2",
            genome=NoTraitsGenome(),
        )
        restored.set_state_ckpt(ckpt)
        assert restored.system_prompt_template == "graph.j2"
        assert "world of connected locations" in restored.system_prompt


# ---------------------------------------------------------------------------
# Empty LLM replies are retried, never a crash (founder-run AttributeError)
# ---------------------------------------------------------------------------


class _FlakyClient:
    """First reply has content=None (a real provider failure mode), the
    second is a valid action."""

    def __init__(self):
        self.calls = 0

    async def get_response_async(self, messages, chat_params):
        self.calls += 1
        if self.calls == 1:
            return Response(content=None, input_tokens=1, output_tokens=0)
        return Response(
            content=_json.dumps(
                {"action": "wait", "message": "", "params": {}}
            ),
            input_tokens=1,
            output_tokens=1,
        )


def test_empty_llm_reply_is_retried_not_fatal(tmp_path):
    agent = LLMAgent(
        agent_name="Alice",
        agent_tag="a0",
        log_dir=tmp_path,
        system_prompt_template=OpenGridWorld.system_prompt_template,
        genome=NoTraitsGenome(),
    )
    obs = {
        "observation": {},
        "observation_text": "nothing",
        "incoming_broadcasts": {},
        "inventory": [],
        "energy": 10,
        "time": 1,
    }
    client = _FlakyClient()
    action = asyncio.run(
        agent.select_action_async(
            obs=obs,
            available_actions={"wait": {"description": "wait", "params": {}}},
            reward=0,
            info=None,
            time=1,
            chat_params={"model": "test"},
            client=client,
        )
    )
    assert client.calls == 2
    assert action["action"] == "wait"
    assert action["source"] == "llm"
