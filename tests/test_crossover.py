"""Tests for genome crossover and two-parent spawn env wiring."""

from dataclasses import fields
from unittest.mock import MagicMock, patch

import pytest

from terralingua.environment.grid_env import OpenGridWorld
from terralingua.genome.ocean_5 import Genome as Ocean5Genome
from terralingua.genome.sentence_directed import Genome as SentenceDirectedGenome
from terralingua.genome.sentence_mutate import Genome as SMGenome

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_env(tmp_path, **kwargs):
    defaults = dict(
        grid_size=10,
        vision_radius=5,
        init_agent_energy=200,
        lifespan=500,
        food_mechanism=False,
        log_path=tmp_path,
        reproduction_cost=20,
        headless=True,
    )
    defaults.update(kwargs)
    return OpenGridWorld(**defaults)  # type: ignore


# ---------------------------------------------------------------------------
# Ocean5 crossover
# ---------------------------------------------------------------------------

class TestOcean5Crossover:
    def test_child_genes_come_from_parents(self):
        a = Ocean5Genome(honesty=1.0, neuroticism=-1.0, extraversion=0.5,
                         agreeableness=-0.5, conscientiousness=0.8,
                         openness=-0.8, dominance=0.3, fertility=0.9)
        b = Ocean5Genome(honesty=-1.0, neuroticism=1.0, extraversion=-0.5,
                         agreeableness=0.5, conscientiousness=-0.8,
                         openness=0.8, dominance=-0.3, fertility=0.1)
        child = a.crossover(b)
        for f in fields(child):
            val = getattr(child, f.name)
            assert val in (getattr(a, f.name), getattr(b, f.name)), (
                f"{f.name}={val} is not from either parent"
            )


# ---------------------------------------------------------------------------
# SentenceMutate crossover
# ---------------------------------------------------------------------------

class TestSentenceMutateCrossover:
    def test_child_length_and_words_come_from_parents(self):
        # Build genomes without the embedding model by constructing directly
        a = SMGenome(words=("brave", "kind", "honest"))
        b = SMGenome(words=("shy", "calm", "proud", "wise", "fair"))
        child = a.crossover(b)
        assert len(child.words) == max(len(a.words), len(b.words))
        assert set(child.words) <= set(a.words) | set(b.words)


# ---------------------------------------------------------------------------
# SentenceDirected crossover
# ---------------------------------------------------------------------------

class TestSentenceDirectedCrossover:
    def test_llm_crossover_called_and_result_used(self):
        a = SentenceDirectedGenome(sentence="brave and curious explorer")
        b = SentenceDirectedGenome(sentence="calm and methodical thinker")

        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "curious yet methodical adventurer"

        with patch("terralingua.genome.sentence_directed.litellm.completion", return_value=mock_resp) as mock_llm:
            child = a.crossover(b)

        mock_llm.assert_called_once()
        assert child.sentence == "curious yet methodical adventurer"

    def test_falls_back_to_word_crossover_on_empty_sentence(self):
        a = SentenceDirectedGenome(sentence="")
        b = SentenceDirectedGenome(sentence="calm thinker")
        # No LLM call expected — empty sentence triggers immediate fallback
        with patch("terralingua.genome.sentence_directed.litellm.completion") as mock_llm:
            child = a.crossover(b)
        mock_llm.assert_not_called()
        assert isinstance(child, SentenceDirectedGenome)

    def test_retries_on_failure_then_falls_back(self):
        a = SentenceDirectedGenome(sentence="brave explorer")
        b = SentenceDirectedGenome(sentence="calm thinker")

        with patch("terralingua.genome.sentence_directed.litellm.completion", side_effect=Exception("network error")) as mock_llm:
            child = a.crossover(b)

        assert mock_llm.call_count == SentenceDirectedGenome._crossover_retries
        # Fell back to word-level crossover — result is still a valid genome
        assert isinstance(child, SentenceDirectedGenome)
        words_a = set(a.sentence.split())
        words_b = set(b.sentence.split())
        for token in child.sentence.split():
            assert token in words_a | words_b

    def test_retries_on_empty_response_then_succeeds(self):
        a = SentenceDirectedGenome(sentence="brave explorer")
        b = SentenceDirectedGenome(sentence="calm thinker")

        empty_resp = MagicMock()
        empty_resp.choices[0].message.content = "   "
        good_resp = MagicMock()
        good_resp.choices[0].message.content = "brave thinker"

        with patch("terralingua.genome.sentence_directed.litellm.completion",
                   side_effect=[empty_resp, good_resp]) as mock_llm:
            child = a.crossover(b)

        assert mock_llm.call_count == 2
        assert child.sentence == "brave thinker"


# ---------------------------------------------------------------------------
# Env: two-parent spawn action wiring
# ---------------------------------------------------------------------------

class TestTwoParentSpawnEnvWiring:
    def test_partner_param_shown_when_same_genome_type_nearby(self, tmp_path):
        env = make_env(tmp_path, two_parent_spawn=True, vision_radius=5)
        env.add_agent("a0", "Alice", "ocean_5", position=(5, 5))
        env.add_agent("a1", "Bob", "ocean_5", position=(5, 6))
        env.restart_env(agent_poses={"a0": (5, 5), "a1": (5, 6)})

        actions_a0 = env.agent_avail_actions["a0"]
        assert "spawn" in actions_a0
        params = actions_a0["spawn"]["params"]
        assert "partner" in params
        assert "Bob" in params["partner"]["choices"]
        assert "" in params["partner"]["choices"]

    @pytest.mark.parametrize("two_parent_spawn,partner_genome", [
        (True, None),
        (True, "no_traits"),
        (False, "ocean_5"),
    ])
    def test_partner_param_absent(self, tmp_path, two_parent_spawn, partner_genome):
        env = make_env(tmp_path, two_parent_spawn=two_parent_spawn, vision_radius=5)
        env.add_agent("a0", "Alice", "ocean_5", position=(5, 5))
        poses = {"a0": (5, 5)}
        if partner_genome is not None:
            env.add_agent("a1", "Bob", partner_genome, position=(5, 6))
            poses["a1"] = (5, 6)
        env.restart_env(agent_poses=poses)

        actions_a0 = env.agent_avail_actions["a0"]
        assert "spawn" in actions_a0
        assert "partner" not in actions_a0["spawn"]["params"]

    def test_two_parent_spawn_fails_for_different_genome_type_in_step(self, tmp_path):
        env = make_env(tmp_path, two_parent_spawn=True, vision_radius=5,
                       food_mechanism=False)
        env.add_agent("a0", "Alice", "ocean_5", position=(5, 5))
        env.add_agent("a1", "Bob", "no_traits", position=(5, 6))
        env.restart_env(agent_poses={"a0": (5, 5), "a1": (5, 6)})

        # Manually inject the partner param (bypassing available_actions schema check)
        env.agent_avail_actions["a0"]["spawn"]["params"]["partner"] = {
            "description": "test", "choices": ["", "Bob"]
        }
        _, _, _, _, infos = env.step({
            "a0": {"action": "spawn", "message": "", "params": {
                "name": "Child", "partner": "Bob",
            }},
        })
        assert infos["a0"]["spawn"]["status"] == "failed"
        assert "genome type" in infos["a0"]["spawn"]["reason"]
