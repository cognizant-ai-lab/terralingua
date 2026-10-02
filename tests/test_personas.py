"""Personas from a file reach the first beings in creation order."""

import json
import pickle
from types import SimpleNamespace

import pytest

from terralingua.agents.personas import load_personas
from terralingua.experiment.checkpoint import CheckpointManager
from terralingua.experiment.runner import SimulationRunner


def write(tmp_path, data):
    path = tmp_path / "personas.json"
    path.write_text(json.dumps(data))
    return path


def test_entries_expand_by_count_in_file_order(tmp_path):
    path = write(tmp_path, [
        {"persona": "You heal.", "name": "Ada"},
        {"persona": "You farm.", "count": 2, "name": "Bo"},
        "You doubt.",
    ])
    assert load_personas(path) == [
        {"persona": "You heal.", "name": "Ada"},
        {"persona": "You farm."},
        {"persona": "You farm."},
        {"persona": "You doubt."},
    ]


@pytest.mark.parametrize("data", [
    {"persona": "not a list"},
    [{"name": "Ada"}],
    [{"persona": "   "}],
    [{"persona": 42}],
    [{"persona": "x", "name": 7}],
    [{"persona": "x", "count": 0}],
    [{"persona": "x", "count": True}],
    [{"persona": "x", "count": "2"}],
    [{"persona": "x", "name": "Ada"}, {"persona": "y", "name": "Ada"}],
    [42],
])
def test_bad_files_are_refused(tmp_path, data):
    with pytest.raises(ValueError):
        load_personas(write(tmp_path, data))


class StubWorld:
    def __init__(self, answers=None, names=()):
        self.answers = answers or {}
        self.agent_names = {f"old{i}": name for i, name in enumerate(names)}

    def agent_identity(self, tag):
        return dict(self.answers.get(tag, {}))


def make_runner(personas, world):
    runner = SimulationRunner.__new__(SimulationRunner)
    runner.env = world
    runner.personas = personas
    runner.personas_given = 0
    return runner


def test_personas_go_to_new_beings_in_order_until_used_up():
    personas = [{"persona": "You heal.", "name": "Ada"}, {"persona": "You farm."}]
    runner = make_runner(personas, StubWorld())
    assert runner._identity_for("being0") == {"persona": "You heal.", "name": "Ada"}
    assert runner._identity_for("being1") == {"persona": "You farm."}
    assert runner._identity_for("being2") == {}
    assert runner.personas_given == 2


def test_a_scenario_persona_wins_and_keeps_the_file_entry_for_the_next_being():
    personas = [{"persona": "You farm."}]
    world = StubWorld({"being0": {"persona": "You watch the sky.", "name": "Sky"}, "being1": {"name": "Kim"}})
    runner = make_runner(personas, world)
    assert runner._identity_for("being0") == {"persona": "You watch the sky.", "name": "Sky"}
    assert runner._identity_for("being1") == {"name": "Kim", "persona": "You farm."}
    assert runner.personas_given == 1


def test_the_file_name_does_not_replace_a_scenario_name():
    runner = make_runner([{"persona": "You farm.", "name": "Ada"}], StubWorld({"being0": {"name": "Kim"}}))
    assert runner._identity_for("being0") == {"name": "Kim", "persona": "You farm."}


def test_a_name_already_in_use_is_not_applied():
    runner = make_runner([{"persona": "You farm.", "name": "Ada"}], StubWorld(names=["Ada"]))
    assert runner._identity_for("being0") == {"persona": "You farm."}


@pytest.mark.asyncio
async def test_the_checkpoint_keeps_the_cursor(tmp_path):
    manager = CheckpointManager(tmp_path)
    env = SimpleNamespace(get_state_ckpt=lambda: {})
    await manager.save_checkpoint(agents={}, ts=3, env=env, last_spawn_idx=4, env_outs={}, personas_given=2)
    with open(manager.checkpoint_path, "rb") as f:
        assert pickle.load(f)["personas_given"] == 2
