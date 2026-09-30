"""Replay captures actual directed topology at completed runner boundaries."""

import asyncio
from types import SimpleNamespace

import pytest

from terralingua.config.models import ExperimentConfig
from terralingua.experiment.runner import SimulationRunner
from terralingua.experiment.social_graph_history import (
    SocialGraphHistoryRecorder,
    read_social_graph_history,
)
from terralingua.genome.no_traits import Genome as NoTraitsGenome
from tests.test_runner_completion import make_runner
from tests.test_social_graph_env import make_social_env, seed_n_agents


@pytest.fixture
def world(tmp_path):
    env = make_social_env(tmp_path, n_nodes=4)
    seed_n_agents(env, ["Alice", "Bob", "Cara", "Dave"])
    yield env
    env.close()


def recorder(tmp_path, run="run", segment="initial"):
    return SocialGraphHistoryRecorder(
        tmp_path / "social_graph.jsonl", run_id=run, segment_id=segment
    )


def edges(row, kind="follow"):
    return {
        (edge["source"], edge["target"])
        for edge in row["graph"]["edges"] if edge["type"] == kind
    }


def assert_actual_graph(row, env):
    assert row["step"] == env.step_count == row["graph"]["step"]
    assert set(row["graph"]["nodes"]) == set(env.world_graph.all_nodes())
    assert edges(row) == {
        (node, other)
        for node in env.world_graph.all_nodes()
        for other in env.world_graph.neighbors(node)
    }
    assert {a["tag"]: a["node"] for a in row["graph"]["agents"]} == {
        tag: env.agent_pos[tag] for tag in env.agent_registry
    }


def act(env, tag, action_name, **params):
    for agent in env.agent_registry:
        env._get_avail_actions(agent)
    result = env.step({tag: {"action": action_name, "message": "", "params": params}})
    assert result[4][tag][action_name]["status"] == "successful"
    return result


def clear_edges(env):
    for node in env.world_graph.all_nodes():
        for other in list(env.world_graph.neighbors(node)):
            env._drop_outgoing_edge(node, other)


def test_snapshot_keeps_both_directions_and_separate_pending_without_messages(world, tmp_path):
    a, b, c, d = (world.agent_pos[f"a{i}"] for i in range(4))
    world.chat = {0: ["PRIVATE CONTENT"]}
    world.chat_senders = {0: ["a0"]}
    world._agent_directives["a0"] = {"role": "planner", "motivation": "PRIVATE CONTENT"}
    world.agent_time["a0"] = float("inf")
    world._node_layout["already-removed"] = (9, 9)
    # The pending relation is recorded separately from the reverse subscription.
    world._drop_outgoing_edge(a, b)
    world._pending_outgoing = {"a0": {"a1": 0}}
    row = recorder(tmp_path).record(world)
    assert_actual_graph(row, world)
    assert (b, a) in edges(row)
    assert (a, b) not in edges(row)
    assert edges(row, "pending") == {(a, b)}
    assert (b, c) in edges(row) and (c, b) in edges(row)
    assert "already-removed" not in row["graph"]["node_layout"]
    assert row["graph"]["agents"][0]["role"] == "planner"
    assert row["graph"]["agents"][0]["time_left"] is None
    assert row["graph"]["recent_messages"] == []
    assert row["graph"]["artifacts"] == []
    assert "PRIVATE CONTENT" not in (tmp_path / "social_graph.jsonl").read_text()


def test_actions_and_replacements_record_actual_connections(world, tmp_path):
    clear_edges(world)
    writer = recorder(tmp_path)
    writer.record(world)
    a, b, c, d = (world.agent_pos[f"a{i}"] for i in range(4))

    act(world, "a0", "follow", target="Bob", replace="")
    assert edges(writer.record(world)) == {(a, b)}
    act(world, "a1", "request_connection", target="Cara", replace="")
    row = writer.record(world)
    assert edges(row) == {(a, b)}
    assert edges(row, "pending") == {(b, c)}
    act(world, "a2", "accept_connection", target="Bob", replace="")
    row = writer.record(world)
    assert edges(row) == {(a, b), (b, c), (c, b)}
    assert edges(row, "pending") == set()
    act(world, "a0", "follow", target="Cara", replace="Bob")
    assert edges(writer.record(world)) == {(a, c), (b, c), (c, b)}
    act(world, "a2", "disconnect", target="Bob")
    assert edges(writer.record(world)) == {(a, c), (b, c)}
    act(world, "a0", "request_connection", target="Dave", replace="Cara")
    row = writer.record(world)
    assert edges(row) == {(b, c)}
    assert edges(row, "pending") == {(a, d)}
    act(world, "a3", "reject_connection", target="Alice")
    row = writer.record(world)
    assert edges(row, "pending") == set()
    assert_actual_graph(row, world)
    assert len(read_social_graph_history(writer.path)["snapshots"]) == 8


def test_decay_snapshot_is_newer_than_cached_agent_observation(world, tmp_path):
    writer = recorder(tmp_path)
    initial = writer.record(world)
    world.edge_decay_steps = 1
    observations, *_ = world.step({})
    row = writer.record(world)
    assert edges(initial)
    assert observations["a0"]["subscriptions"]
    assert edges(row) == set()
    assert_actual_graph(row, world)


def test_death_and_admin_sever_leave_no_dangling_edges(world, tmp_path):
    writer = recorder(tmp_path)
    first = writer.record(world)
    dead_node = world.agent_pos["a3"]
    world.agent_time["a3"] = 1
    world.step({})
    runner = SimulationRunner.__new__(SimulationRunner)
    runner.env = world
    u, v = next((u, v) for u in world.world_graph.all_nodes()
                for v in world.world_graph.neighbors(u))
    runner.inject_event("sever_edge", u=u, v=v)
    row = writer.record(world)
    assert_actual_graph(row, world)
    assert dead_node not in row["graph"]["nodes"]
    assert not any(dead_node in edge for edge in edges(row))
    assert (u, v) not in edges(row)
    assert (v, u) in edges(row)
    assert len(row["graph"]["agents"]) == len(first["graph"]["agents"]) - 1


def test_resume_replaces_abandoned_branch_using_environment_step(world, tmp_path):
    writer = recorder(tmp_path)
    writer.record(world)
    world.step({})
    checkpoint = world.get_state_ckpt()
    writer.record(world)
    world.step({})
    writer.record(world)
    world.set_state_ckpt(checkpoint)
    source = world.agent_pos["a0"]
    target = next(iter(world.world_graph.neighbors(source)))
    world.world_graph.remove_edge(source, target)
    resumed = recorder(tmp_path, segment="resumed")
    restored = resumed.record(world, segment_start=True, reason="resume")
    world.step({})
    resumed.record(world)
    history = read_social_graph_history(writer.path)
    assert history["run_id"] == "run"
    assert history["segment_id"] == "resumed"
    assert [row["step"] for row in history["snapshots"]] == [0, 1, 2]
    assert [row["segment_id"] for row in history["snapshots"]] == ["initial", "resumed", "resumed"]
    assert history["snapshots"][1] == restored
    assert (source, target) not in edges(history["snapshots"][1])


def test_new_run_in_same_directory_does_not_mix_recordings(world, tmp_path):
    old = recorder(tmp_path)
    old.record(world)
    world.step({})
    old.record(world)
    world.step_count = 0
    new = recorder(tmp_path, run="new-run", segment="new-segment")
    row = new.record(world)
    history = read_social_graph_history(old.path)
    assert history["run_id"] == "new-run"
    assert history["snapshots"] == [row]


def test_unterminated_tail_is_ignored_then_repaired_before_resume(world, tmp_path):
    writer = recorder(tmp_path)
    original = writer.record(world)
    with writer.path.open("ab") as stream:
        stream.write(b'{"valid":true}')
    assert read_social_graph_history(writer.path)["snapshots"] == [original]
    world.step({})
    resumed = recorder(tmp_path, segment="resumed")
    second = resumed.record(world, segment_start=True, reason="resume")
    assert read_social_graph_history(writer.path)["snapshots"] == [original, second]
    assert writer.path.read_bytes().endswith(b"\n")


@pytest.mark.parametrize("bad_line", [b"not json\n", b"{}\n", b"\xff\n"])
def test_complete_corrupt_lines_raise_even_at_end(world, tmp_path, bad_line):
    writer = recorder(tmp_path)
    writer.record(world)
    with writer.path.open("ab") as stream:
        stream.write(bad_line)
    with pytest.raises(ValueError, match="line 2"):
        read_social_graph_history(writer.path)


def test_missing_or_empty_recording_is_empty(tmp_path):
    path = tmp_path / "social_graph.jsonl"
    assert read_social_graph_history(path) == {
        "schema_version": 1, "run_id": None, "segment_id": None, "snapshots": [],
    }
    path.touch()
    assert read_social_graph_history(path)["snapshots"] == []


def test_recording_write_errors_propagate(world, tmp_path):
    writer = recorder(tmp_path)
    writer.path.mkdir()
    with pytest.raises(OSError):
        writer.record(world)


def test_non_social_runner_does_not_require_recorder_attributes():
    runner = SimulationRunner.__new__(SimulationRunner)
    runner.env = SimpleNamespace()
    runner._record_social_graph_snapshot()
    assert not hasattr(runner, "_social_graph_recorder")


def test_runner_records_initial_and_post_respawn_state_headlessly(world, tmp_path):
    runner = make_runner(tmp_path, max_steps=2)
    runner.env = world
    runner._run_id = "integration"
    runner.exp_logdir = tmp_path
    runner.params = ExperimentConfig()
    runner.params.run.max_ts = 42
    runner.params.run.empty_countdown = -1
    runner.params.run.save_video = False
    runner.params.run.live_render = False
    runner.params.env.world_type = "social_graph"
    runner.params.env.min_agents = 4
    runner.params.agent.genome = "no_traits"
    runner.params.agent.agents_name_prefix = "newcomer"
    runner.last_spawn_idx = -1
    runner.dead_agent_traits = {}
    runner.obs, runner.infos = world.restart_env()
    runner.dones = {tag: False for tag in world.agent_registry}
    runner.rewards = {}
    def fake_agent(tag, name, genome=None):
        return SimpleNamespace(
            agent_name=name, genome=genome or NoTraitsGenome(), close=lambda: None,
        )
    runner.agents = {
        tag: fake_agent(tag, world.agent_names[tag]) for tag in world.agent_registry
    }
    runner._make_llm_agent = lambda tag, name, genome: fake_agent(tag, name, genome)
    # The real population handler cleans the death and calls bootstrap_follows
    # after env.step; retaining the loop's default noop action uses no LLM.
    del runner._advance_population
    dead_node = world.agent_pos["a3"]
    world.agent_time["a3"] = 1
    async def collect(ts):
        return {"a0": {"action": "noop", "params": {}, "message": ""}}
    runner._collect_actions = collect
    asyncio.run(runner.run_async())
    rows = read_social_graph_history(tmp_path / "social_graph.jsonl")["snapshots"]
    assert [row["step"] for row in rows] == [0, 1, 2]  # NOT loop labels 40,41
    assert len(rows[0]["graph"]["agents"]) == 4
    after = rows[1]
    assert dead_node not in after["graph"]["nodes"]
    newborn = next(a for a in after["graph"]["agents"] if a["tag"] == "newcomer0")
    assert any(newborn["node"] in edge for edge in edges(after))
    assert len(after["graph"]["agents"]) == 4
    assert_actual_graph(rows[-1], world)
