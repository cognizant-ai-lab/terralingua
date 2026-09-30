"""Dashboard resource fields remain valid JSON without changing simulation state."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from terralingua.agents.agent_logger import AgentLogger
from terralingua.config.models import ExperimentConfig, GraphConfig
from terralingua.environment.graph_env import OpenGraphWorld
from terralingua.environment.grid_env import OpenGridWorld
from terralingua.environment.social_graph_env import OpenSocialGraphWorld
from terralingua.experiment.runner import SimulationRunner
from terralingua.genome.no_traits import Genome


@pytest.fixture
def make_runner(tmp_path):
    worlds = []

    def create(kind, energy, lifespan):
        options = dict(
            lifespan=lifespan, init_agent_energy=energy, food_mechanism=False,
            init_food=0, food_spawn_rate=0, food_decay_rate=0,
            headless=True, log_path=tmp_path,
        )
        if kind == "grid":
            world = OpenGridWorld(grid_size=3, **options)
            position = (0, 0)
        else:
            cls = OpenSocialGraphWorld if kind == "social" else OpenGraphWorld
            world = cls(graph_cfg=GraphConfig.complete(n_nodes=2), **options)
            position = "0"
        worlds.append(world)
        world.add_agent("a", "Alice", "no_traits", position=position)
        obs, _ = world.restart_env(seed=1, agent_poses={"a": position})
        runner = SimulationRunner.__new__(SimulationRunner)
        runner.env = world
        runner.params = ExperimentConfig()
        runner.obs = obs
        runner.pre_step_obs = obs
        runner.agents = {"a": SimpleNamespace(agent_name="Alice", genome=Genome())}
        runner.dones = {}
        runner.last_actions = {}
        runner.dead_agent_traits = {}
        runner.connection_manager = None
        runner.exp_logdir = tmp_path
        runner._run_id = "offline-dashboard-test"
        return runner

    yield create
    for world in worlds:
        world.close()


@pytest.mark.parametrize("kind,energy,lifespan", [
    ("grid", -1, -1), ("graph", -1, -1), ("social", -1, -1), ("grid", 37, 29),
])
def test_real_world_frames_and_commands_are_strict_json(make_runner, kind, energy, lifespan):
    runner = make_runner(kind, energy, lifespan)
    initial_energy = runner.env.agent_energy["a"]
    initial_time = runner.env.agent_time["a"]
    frame = json.loads(json.dumps(runner._build_frame(), allow_nan=False))
    assert len(frame["grid_state"]["agents"]) == 1
    assert frame["agents"][0]["energy"] == (None if energy == -1 else 37)
    assert frame["agents"][0]["time_left"] == (None if lifespan == -1 else 29)

    config = asyncio.run(runner._dispatch_cmd("get_config", {}))
    assert config["ok"]
    json.dumps(config, allow_nan=False)
    assert config["data"]["population"]["init_agent_energy"] == (
        None if energy == -1 else 37
    )
    assert config["data"]["population"]["agent_lifespan"] == lifespan

    status = asyncio.run(runner._dispatch_cmd("get_status", {}))
    assert status["ok"]
    json.dumps(status, allow_nan=False)
    assert status["data"]["agents"][0]["energy"] == (None if energy == -1 else 37)
    assert runner.env.agent_energy["a"] == runner.obs["a"]["energy"] == initial_energy
    assert runner.env.agent_time["a"] == runner.obs["a"]["time"] == initial_time


def test_runtime_infinite_lifespan_config_is_strict_json(make_runner):
    runner = make_runner("grid", -1, -1)
    # Scenario subclasses can express their own unlimited lifetime directly.
    runner.env.lifespan = float("inf")
    response = asyncio.run(runner._dispatch_cmd("get_config", {}))
    assert response["ok"]
    assert response["data"]["population"]["agent_lifespan"] is None
    json.dumps(response, allow_nan=False)
    assert runner.env.lifespan == float("inf")


def test_agent_history_normalizes_resources_without_rewriting_logs(make_runner, tmp_path):
    runner = make_runner("social", -1, -1)
    directory = tmp_path / "agent_logs"
    directory.mkdir(exist_ok=True)
    path = directory / "a.jsonl"
    records = [
        {"timestamp": 0, "observation": {"energy": float("inf"), "time": float("inf")}},
        {"timestamp": 1, "observation": {"energy": 12, "time": 7}},
        {"timestamp": 2, "observation": {}},
    ]
    logger = AgentLogger(directory, agent_tag="a")
    for record in records:
        logger.log(
            agent_name="Alice", agent_tag="a",
            observation={"observation": {}, **record["observation"]},
            available_actions={}, action={"action": "noop"},
            time=str(record["timestamp"]), internal_memory="", input_prompt="",
        )
    logger.close()
    content = path.read_text()
    stored_observation = json.loads(content.splitlines()[0])["observation"]
    assert stored_observation["energy"] == float("inf")
    assert stored_observation["time"] == float("inf")
    response = asyncio.run(runner._dispatch_cmd("get_agent_log", {"tag": "a"}))
    assert response["ok"]
    entries = json.loads(json.dumps(response, allow_nan=False))["data"]["entries"]
    assert [(row["energy"], row["time_left"]) for row in entries] == [
        (None, None), (12, 7), (None, None),
    ]
    assert path.read_text() == content
