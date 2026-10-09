"""The system prompt only announces what the run can show: the food legend with food, the traits line with traits."""

import pytest

from terralingua.agents.prompt_templates import render_system_prompt
from terralingua.genome.no_traits import Genome as NoTraitsGenome


def render(template, **flags):
    base = dict(agent_name="Alice", use_internal_memory=False, use_inventory=False, artifact_creation=False,
                finite_energy=True, finite_lifespan=True, external_actions=False, scenario_specific_instructions="",
                food_mechanism=True, energy_death=True, energy_upkeep=1, genome_string="")
    base.update(flags)
    return render_system_prompt(template, **base)


@pytest.mark.parametrize("template", ["graph.j2", "grid.j2"])
def test_the_food_legend_follows_the_food_mechanism(template):
    assert "food value" in render(template, food_mechanism=True)
    text = render(template, food_mechanism=False)
    assert "food value" not in text
    assert "'A(type, movable/fixed): name' = artifact" in text  # the rest of the legend stays


def test_the_traits_line_follows_the_genome_string():
    assert "list of traits" not in render("graph.j2", genome_string="")
    assert "list of traits" not in render("graph.j2", genome_string=NoTraitsGenome().as_string())
    with_traits = render("graph.j2", genome_string="Traits: curious, careful.")
    assert "list of traits" in with_traits and "Traits: curious, careful." in with_traits
    assert "the history of your past observations and selected actions" in render("graph.j2")
