"""Tests for system-prompt rendering and env-side obs text formatting.

Covers the per-env Jinja templates (`terralingua/agents/prompts/*.j2`), the
`render_system_prompt` helper, and each env's `format_observation_text` method.
"""

import pytest

from terralingua.agents.prompt_templates import (
    AGENT_PROMPT,
    ERROR_MSG,
    render_system_prompt,
)
from terralingua.config.models import GraphConfig
from terralingua.environment.graph_env import OpenGraphWorld
from terralingua.environment.grid_env import OpenGridWorld
from terralingua.environment.social_graph_env import OpenSocialGraphWorld


def _default_kwargs(**overrides):
    """Minimal sensible defaults for render_system_prompt; tests override as needed."""
    kw = dict(
        agent_name="Alice",
        use_internal_memory=True,
        use_inventory=True,
        artifact_creation=True,
        food_mechanism=True,
        external_actions=False,
        scenario_specific_instructions="[motivation text]",
        internal_memory_size=150,
        max_message_length=200,
        max_connections=None,
        spawn_allowed=True,
        genome_string="",
        debug=False,
    )
    kw.update(overrides)
    return kw


# ---------------------------------------------------------------------------
# Social-graph template renders without spatial framing
# ---------------------------------------------------------------------------


class TestEnvSpecificContent:
    @pytest.mark.parametrize(
        "solo, food_mechanism, use_inventory",
        [(False, True, True), (True, False, False)],
    )
    def test_social_graph_template_has_no_spatial_framing(
        self, solo, food_mechanism, use_inventory
    ):
        text = render_system_prompt(
            "social_graph.j2",
            **_default_kwargs(
                solo=solo,
                food_mechanism=food_mechanism,
                use_inventory=use_inventory,
            ),
        )
        assert "autonomous agent" in text
        for phrase in ("node", "move", "location", "stay still"):
            assert phrase not in text.lower()


# ---------------------------------------------------------------------------
# Conditional blocks driven by flags (covered by the shared _base.j2)
# ---------------------------------------------------------------------------


class TestConditionalBlocks:
    def test_food_section_present_when_enabled(self):
        text = render_system_prompt("grid.j2", **_default_kwargs(food_mechanism=True))
        assert "Energy" in text
        assert "0 energy" in text or "lose 1 energy" in text

    def test_food_section_absent_when_disabled(self):
        text = render_system_prompt("grid.j2", **_default_kwargs(food_mechanism=False))
        # The phrase "You lose 1 energy at each turn" only appears in the
        # food/energy block; absent when food_mechanism is False.
        assert "lose 1 energy" not in text

    def test_inventory_section_present_when_enabled(self):
        text = render_system_prompt("grid.j2", **_default_kwargs(use_inventory=True))
        assert "Inventory" in text

    def test_inventory_section_absent_when_disabled(self):
        text = render_system_prompt(
            "grid.j2", **_default_kwargs(use_inventory=False, artifact_creation=False)
        )
        assert "Inventory" not in text

    def test_artifact_section_present_when_enabled(self):
        text = render_system_prompt(
            "grid.j2", **_default_kwargs(artifact_creation=True)
        )
        assert "Artifacts" in text

    def test_artifact_section_absent_when_disabled(self):
        text = render_system_prompt(
            "grid.j2", **_default_kwargs(artifact_creation=False)
        )
        assert "Artifacts" not in text

    def test_external_actions_section_present_when_enabled(self):
        text = render_system_prompt(
            "grid.j2", **_default_kwargs(external_actions=True)
        )
        assert "external systems" in text

    def test_external_actions_section_absent_when_disabled(self):
        text = render_system_prompt(
            "grid.j2", **_default_kwargs(external_actions=False)
        )
        assert "external systems" not in text


class TestSocialGraphConfiguration:
    @pytest.mark.parametrize("max_connections", [None, 6])
    def test_connection_limit_matches_configuration(self, max_connections):
        text = render_system_prompt(
            "social_graph.j2",
            **_default_kwargs(max_connections=max_connections),
        )
        if max_connections is None:
            assert "connection slots" not in text
            assert "connection limit" not in text
        else:
            assert f"{max_connections} connection slots" in text
            assert "Being followed does not use your connection slots" in text

    @pytest.mark.parametrize("spawn_allowed", [False, True])
    @pytest.mark.parametrize("solo", [False, True])
    def test_spawning_requires_enabled_social_population(self, spawn_allowed, solo):
        text = render_system_prompt(
            "social_graph.j2",
            **_default_kwargs(spawn_allowed=spawn_allowed, solo=solo),
        )
        assert ("- Spawning" in text) == (spawn_allowed and not solo)
        if spawn_allowed and not solo:
            assert "available in your action list" in text

    def test_scenario_and_limits_are_supplied_by_the_caller(self):
        for marker, message_limit, memory_limit in (
            ("First scenario instructions", 123, 456),
            ("Second scenario instructions", 789, 321),
        ):
            text = render_system_prompt(
                "social_graph.j2",
                **_default_kwargs(
                    scenario_specific_instructions=marker,
                    max_message_length=message_limit,
                    internal_memory_size=memory_limit,
                ),
            )
            assert text.count(marker) == 1
            assert f"limited to {message_limit} tokens" in text
            assert f"within {memory_limit} tokens" in text
            other_marker = (
                "Second scenario instructions"
                if marker.startswith("First")
                else "First scenario instructions"
            )
            assert other_marker not in text


# ---------------------------------------------------------------------------
# genome_string + debug bottom-of-template blocks
# ---------------------------------------------------------------------------


class TestGenomeAndDebugBlocks:
    def test_genome_string_included_when_provided(self):
        text = render_system_prompt(
            "grid.j2",
            **_default_kwargs(genome_string="Alice is curious and brave."),
        )
        assert "Alice is curious and brave." in text

    def test_debug_block_present_when_enabled(self):
        text = render_system_prompt("grid.j2", **_default_kwargs(debug=True))
        assert "[DEBUG]" in text
        assert "debug mode" in text

    def test_debug_block_absent_when_disabled(self):
        text = render_system_prompt("grid.j2", **_default_kwargs(debug=False))
        assert "[DEBUG]" not in text


# ---------------------------------------------------------------------------
# Env-owned obs-text formatting
# ---------------------------------------------------------------------------


class TestObsTextFormatting:
    def test_grid_format_observation_text_renders_cells(self, tmp_path):
        env = OpenGridWorld(
            grid_size=10,
            vision_radius=2,
            init_food=0,
            food_spawn_rate=0,
            food_mechanism=False,
            log_path=tmp_path,
            headless=True,
        )
        # Hand-crafted obs dict matching the (rel_x, rel_y) → items format.
        obs = {(0, 0): ["Alice"], (1, 0): ["food", "5"], (-1, 2): ["X"]}
        text = env.format_observation_text(obs)
        # Each non-empty cell becomes a line. Self is marked as <yourself>.
        assert "yourself" in text
        assert "food" in text
        # Lines are separated by newlines.
        assert "\n" in text

    def test_graph_format_observation_text_renders_locations(self, tmp_path):
        env = OpenGraphWorld(
            graph_cfg=GraphConfig(topology="ring", n_nodes=3, hop_radius=1),
            init_food=0,
            food_spawn_rate=0,
            food_mechanism=False,
            log_path=tmp_path,
            headless=True,
        )
        observation = {
            "0": {"items": [], "hop_distance": 0, "exits": ["1", "2"]},
            "1": {"items": ["Bob"], "hop_distance": 1, "exits": ["0", "2"]},
        }
        text = env.format_observation_text(observation, current_node="0")
        assert "You are at: 0" in text
        assert "<yourself>" in text
        assert "Bob" in text
        assert "dist=0" in text
        assert "dist=1" in text

    def test_build_obs_populates_observation_text(self, tmp_path):
        env = OpenSocialGraphWorld(
            graph_cfg=GraphConfig(topology="ring", n_nodes=3, hop_radius=1),
            init_food=0,
            food_spawn_rate=0,
            food_mechanism=False,
            log_path=tmp_path,
            headless=True,
        )
        env.add_agent("a0", "Alice", "no_traits", position="0")
        env.add_agent("a1", "Bob", "no_traits", position="1")
        env.add_agent("a2", "Carol", "no_traits", position="2")
        env.restart_env(agent_poses={"a0": "0", "a1": "1", "a2": "2"})
        obs, _ = env._build_obs("a0")
        assert "observation_text" in obs
        assert isinstance(obs["observation_text"], str)
        assert "You: Alice" in obs["observation_text"]
        assert "Agents you follow:" in obs["observation_text"]
        assert "You are at:" not in obs["observation_text"]
        assert "dist=" not in obs["observation_text"]


# ---------------------------------------------------------------------------
# Solo runs (env.max_agents == 1): social/broadcast content leaves every
# prompt surface; unpassed the flag is falsy, so other scenarios render
# byte-identical output.
# ---------------------------------------------------------------------------


class TestSoloPrompt:
    def test_solo_social_graph_drops_social_content(self):
        text = render_system_prompt(
            "social_graph.j2", **_default_kwargs(solo=True)
        )
        for needle in (
            "follow",
            "broadcast",
            "connection request",
            "Spawning",
            "Communication",
            "Your network",
        ):
            assert needle not in text, needle
        assert "You are Alice, an autonomous agent." in text

    def test_solo_omitted_matches_solo_false(self):
        explicit = render_system_prompt(
            "social_graph.j2", **_default_kwargs(solo=False)
        )
        omitted = render_system_prompt("social_graph.j2", **_default_kwargs())
        assert explicit == omitted
        assert "network of other agents" in omitted

    def test_solo_base_drops_broadcast_in_grid(self):
        text = render_system_prompt("grid.j2", **_default_kwargs(solo=True))
        assert "broadcast" not in text.lower()


def _agent_prompt_kwargs(**overrides):
    kw = dict(
        history="",
        observation="obs",
        messages="<none>",
        energy=10,
        time=5,
        inventory="<empty>",
        additional_info="",
        actions="{}",
        action_keys="wait",
        memory="",
        use_internal_memory=False,
        use_inventory=False,
        food_mechanism=False,
        max_message_length=200,
    )
    kw.update(overrides)
    return kw


class TestSoloAgentPrompt:
    def test_message_field_present_by_default(self):
        text = AGENT_PROMPT.render(**_agent_prompt_kwargs())
        assert '"message"' in text
        assert "Incoming messages:" in text

    def test_solo_drops_message_field_and_incoming_section(self):
        text = AGENT_PROMPT.render(**_agent_prompt_kwargs(solo=True))
        assert '"message"' not in text
        assert "Incoming messages:" not in text
        assert '"action"' in text
        assert '"params"' in text

    def test_solo_error_msg_drops_message_field(self):
        text = ERROR_MSG.render(
            error="bad",
            action_keys=["wait"],
            use_internal_memory=False,
            solo=True,
        )
        assert '"message"' not in text
        assert '"action"' in text


# ---------------------------------------------------------------------------
# The movement rule names the directions the grid world accepts
# ---------------------------------------------------------------------------


class TestMovementDirections:
    def test_grid_template_names_the_directions_the_env_accepts(self):
        text = render_system_prompt("grid.j2", **_default_kwargs())
        assert "up / down / left / right" in text
        assert "north / south / east / west" not in text
