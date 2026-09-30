"""The example scenario on a small grid, driven with scripted actions and no model calls."""

import json

from scenarios.example.storms import (
    STORM_HIT,
    STORM_SHELTERED,
    STORM_WARNING,
    WATCHER_PERSONA,
    Storms,
    StormsOptions,
)
from terralingua.config.compose import compose
from terralingua.environment.grid_env import OpenGridWorld
from terralingua.experiment.scenario_loader import load_scenario

STAY = {"action": "move", "params": {"direction": "stay"}}
FAST = dict(period=2, duration=1, radius=1, damage=3, shelter_cost=5, watchers=1)


def make_env(tmp_path, positions):
    """An 8x8 grid with no food, the mechanic attached, and beings at fixed cells."""
    mechanic = Storms(StormsOptions(**FAST))
    env = OpenGridWorld(
        grid_size=8, vision_radius=2, init_agent_energy=50, lifespan=500, init_food=0,
        food_spawn_rate=0, food_mechanism=False, log_path=tmp_path, drop_food_on_death=False,
        use_inventory=True, reproduction_cost=-1, artifact_creation_cost=0, headless=True,
    )
    env.attach(mechanic)
    identities = {}
    for i, pos in enumerate(positions):
        tag = f"a{i}"
        identities[tag] = env.agent_identity(tag)  # what the runner does before it adds a being
        env.add_agent(tag, f"Name{i}", "text", position=pos)
    env.restart_env(seed=3, agent_poses={f"a{i}": pos for i, pos in enumerate(positions)})
    return env, mechanic, identities


def step(env, **actions):
    acts = {tag: STAY for tag in env.agent_registry}
    acts.update(actions)
    return env.step(acts)[-1]


def events(env, name):
    env.logger.fp.flush()
    lines = [json.loads(line) for line in env.logger.save_path.read_text().splitlines()]
    return [line for line in lines if line["event"] == name]


def test_storm_warns_the_watcher_and_spares_the_sheltered(tmp_path):
    env, mechanic, identities = make_env(tmp_path, [(3, 3), (3, 4)])
    assert identities["a0"] == {"persona": WATCHER_PERSONA}
    assert identities["a1"] == {}
    infos = step(env)  # step 0: nothing yet
    assert "Weather" not in infos["a0"]
    infos = step(env)  # step 1: the storm forms next step, so the watcher is warned
    assert infos["a0"]["Weather"] == STORM_WARNING
    assert "Weather" not in infos["a1"]
    infos = step(env, a1={"action": "build_shelter", "params": {}})  # step 2: the storm forms
    assert infos["a1"]["Action outcome"].startswith("You built shelter_a1")
    assert env.agent_energy["a1"] == 45
    assert env.artifacts["shelter_a1"].art_type == "shelter"
    assert "build_shelter" not in env.agent_avail_actions["a1"]
    assert "build_shelter" in env.agent_avail_actions["a0"]
    assert len(events(env, "STORM")) == 1 and len(events(env, "SHELTER_BUILT")) == 1
    infos = step(env)  # step 3: the storm hits both cells; only the unsheltered being pays
    assert infos["a0"]["Weather"] == STORM_HIT.format(damage=3)
    assert infos["a1"]["Weather"] == STORM_SHELTERED
    assert env.agent_energy["a0"] == 47
    assert env.agent_energy["a1"] == 45
    assert mechanic.state["storm"] is None
    assert mechanic.state["next_storm"] == 3 + FAST["period"]


def test_preset_builds_the_mechanic_and_its_state_survives_a_checkpoint(tmp_path):
    cfg = compose("example")
    (mechanic,) = load_scenario(cfg.run.scenario, cfg.run.scenario_options)
    assert isinstance(mechanic, Storms) and mechanic.options.period == 5
    env, mechanic, _ = make_env(tmp_path, [(3, 3)])
    step(env)
    step(env)
    step(env)  # a storm is active now
    assert env.get_state_ckpt()["mechanics"]["storms"]["storm"] is not None
    restored, restored_mechanic, _ = make_env(tmp_path / "restored", [(3, 3)])
    restored.set_state_ckpt(env.get_state_ckpt())
    assert restored_mechanic.state == mechanic.state
