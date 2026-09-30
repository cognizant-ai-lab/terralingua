from pathlib import Path
from types import SimpleNamespace

import pytest

from terralingua.config.compose import compose
from terralingua.environment.graph_builder import build_graph
from terralingua.environment.graph_env import OpenGraphWorld
from terralingua.environment.social_graph_env import OpenSocialGraphWorld
from terralingua.experiment import runner as runner_module
from terralingua.experiment.neuro_san_hocon import load_neuro_san_agent_network


def _write_network(tmp_path: Path) -> Path:
    path = tmp_path / "network.hocon"
    path.write_text(
        """
        {
          metadata: {
            description: "A test Neuro-SAN network."
          }
          tools: [
            {
              name: "coordinator"
              display_as: "Team Coordinator"
              function: {
                description: "Coordinate the local network."
              }
              instructions: "Coordinate carefully."
              tools: ["scout", "archivist", "/date_time"]
            },
            {
              name: "scout"
              instructions: "Observe and report."
              tools: ["archivist"]
            },
            {
              name: "archivist"
              instructions: "Preserve shared memory."
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    return path


def test_load_neuro_san_hocon_resolves_local_and_external_tools(tmp_path):
    path = _write_network(tmp_path)

    network = load_neuro_san_agent_network(path)

    assert network.frontman == "coordinator"
    assert network.description == "A test Neuro-SAN network."
    assert network.agent_names == ("coordinator", "scout", "archivist")
    coordinator = network.agent_by_name("coordinator")
    assert coordinator.name == "coordinator"
    assert coordinator.display_as == "Team Coordinator"
    assert coordinator.description == "Coordinate the local network."
    assert coordinator.instructions == "Coordinate carefully."
    assert coordinator.downstream_agents == ("scout", "archivist")
    assert coordinator.external_tools == ("/date_time",)


def test_load_neuro_san_hocon_expands_commondefs(tmp_path):
    path = tmp_path / "commondefs.hocon"
    path.write_text(
        """
        {
          commondefs: {
            replacement_strings: {
              role: "coordination"
            }
            replacement_values: {
              function_template: {
                description: "Perform {role}."
              }
            }
          }
          tools: [
            {
              name: "coordinator"
              function: function_template
              instructions: "Use {role}."
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    network = load_neuro_san_agent_network(path)

    coordinator = network.agent_by_name("coordinator")
    assert coordinator.description == "Perform coordination."
    assert coordinator.instructions == "Use coordination."


def test_load_neuro_san_hocon_ignores_missing_import_substitutions(tmp_path):
    path = tmp_path / "missing_imports.hocon"
    path.write_text(
        """
        {
          include "registries/aaosa.hocon"
          instructions_prefix: "Always consult your local tools first. "
          tools: [
            {
              name: "director"
              function: ${aaosa_call}{
                description: "Lead back-office service delivery."
              }
              instructions: ${instructions_prefix} "Coordinate." ${aaosa_instructions}
              tools: ["analyst"]
            },
            {
              name: "analyst"
              instructions: "Analyze requests."
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    network = load_neuro_san_agent_network(path)

    director = network.agent_by_name("director")
    assert director.description == "Lead back-office service delivery."
    assert director.instructions == "Always consult your local tools first. Coordinate."
    assert "ConfigSubstitution" not in director.instructions
    assert director.downstream_agents == ("analyst",)


def test_agent_network_graph_uses_hocon_tools_edges(tmp_path):
    path = _write_network(tmp_path)

    graph = build_graph("agent_network", path=str(path))

    assert set(graph.all_nodes()) == {"coordinator", "scout", "archivist"}
    assert graph.has_edge("coordinator", "scout")
    assert graph.has_edge("coordinator", "archivist")
    assert graph.has_edge("scout", "archivist")
    assert not graph.has_edge("archivist", "scout")


@pytest.mark.parametrize(
    "world_type,world_class",
    [("graph", OpenGraphWorld), ("social_graph", OpenSocialGraphWorld)],
)
def test_runner_seeds_hocon_identity_and_personality(
    tmp_path, monkeypatch, world_type, world_class
):
    path = _write_network(tmp_path)
    config = compose(
        overrides={
            "agent_network_hocon_path": str(path),
            "world_type": world_type,
            "exp_name": "network_test",
            "genome": "sentence_directed",
            "scenario_specific_instructions": "none",
            "init_food": 0,
            "food_mechanism": False,
            "reproduction_cost": -1,
            "save_video": False,
        }
    )
    monkeypatch.setattr(runner_module, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(runner_module, "LLMRouter", lambda **kwargs: SimpleNamespace())
    runner = runner_module.SimulationRunner(config)
    try:
        assert type(runner.env) is world_class
        assert set(runner.agents) == {"coordinator", "scout", "archivist"}
        expected = {
            "coordinator": (
                "Team Coordinator",
                "Coordinate the local network.\n\nCoordinate carefully.",
            ),
            "scout": ("scout", "Observe and report."),
            "archivist": ("archivist", "Preserve shared memory."),
        }
        for tag, (name, personality) in expected.items():
            agent = runner.agents[tag]
            assert agent.agent_tag == tag
            assert agent.agent_name == name
            assert runner.env.agent_names[tag] == name
            assert runner.env.agent_pos[tag] == tag
            assert agent.genome.sentence == personality
            assert name in agent.system_prompt
            assert personality in agent.system_prompt
            assert "Local downstream agents/tools" not in agent.system_prompt
            assert "A test Neuro-SAN network." not in agent.system_prompt
    finally:
        for agent in runner.agents.values():
            agent.close()
        runner.env.close()
