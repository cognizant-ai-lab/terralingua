"""Core additions for the scenario layer: kill reasons, notes, seeds, events, mechanics."""

import json
import sys
import types
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from terralingua.agents.llm_agent import LLMAgent
from terralingua.config.models import GraphConfig
from terralingua.environment.artifact import (
    ARTIFACT_TYPES,
    Artifact,
    register_artifact_type,
)
from terralingua.environment.graph_env import OpenGraphWorld
from terralingua.environment.grid_env import OpenGridWorld
from terralingua.environment.mechanic import Mechanic
from terralingua.experiment.runner import SimulationRunner
from terralingua.experiment.scenario_loader import load_scenario
from terralingua.genome.no_traits import Genome as NoTraitsGenome

STAY = {"action": "move", "params": {"direction": "stay"}}


def make_env(tmp_path, **kwargs):
    defaults = dict(
        grid_size=10,
        vision_radius=2,
        init_agent_energy=100,
        lifespan=200,
        init_food=20,
        max_food_value=5.0,
        food_decay_rate=0.0,
        food_spawn_rate=0,
        log_path=tmp_path,
        drop_food_on_death=False,
        use_inventory=True,
        use_colors=False,
        reproduction_cost=-1,
        artifact_creation_cost=0,
        headless=True,
    )
    defaults.update(kwargs)
    return OpenGridWorld(**defaults)  # type: ignore


def one_agent_env(tmp_path, mechanic=None):
    env = make_env(tmp_path)
    if mechanic is not None:
        env.attach(mechanic)
    env.add_agent("a0", "Alice", "text", position=(2, 2))
    env.restart_env(agent_poses={"a0": (2, 2)})
    return env


def two_agent_env(tmp_path, mechanic=None):
    env = make_env(tmp_path)
    if mechanic is not None:
        env.attach(mechanic)
    env.add_agent("a0", "Alice", "text", position=(2, 2))
    env.add_agent("a1", "Bob", "text", position=(2, 3))
    env.restart_env(agent_poses={"a0": (2, 2), "a1": (2, 3)})
    return env


# Stage 1: kill reasons, notes, seeds, string events
# ---------------------------------------------------------------------------


def test_queued_kill_logs_its_reason(tmp_path):
    env = one_agent_env(tmp_path)
    env._pending_kills["a0"] = "sickness"
    _, _, done, _, _ = env.step({"a0": STAY})
    assert done["a0"]
    assert env.logger.data["a0"]["death_reason"] == "sickness"
    assert not env._pending_kills


def test_queued_kill_without_reason_keeps_the_default(tmp_path):
    env = one_agent_env(tmp_path)
    env._pending_kills["a0"] = None
    env.step({"a0": STAY})
    assert env.logger.data["a0"]["death_reason"] == "unknown"


def test_note_appends_to_the_next_observation_only(tmp_path):
    env = one_agent_env(tmp_path)
    env.note("a0", "Health", "You feel feverish.")
    env.note("a0", "Health", "Your energy drains faster.")
    *_, infos = env.step({"a0": STAY})
    assert infos["a0"]["Health"] == "You feel feverish.\nYour energy drains faster."
    *_, infos = env.step({"a0": STAY})
    assert "Health" not in infos["a0"]


def test_restart_seed_gives_the_same_world(tmp_path):
    def build():
        env = make_env(tmp_path)
        for i in range(3):
            env.add_agent(f"a{i}", f"Name{i}", "text", position=(1, 1 + i))
        env.restart_env(seed=5)
        return env

    first, second = build(), build()
    assert first.agent_pos == second.agent_pos
    assert first.food == second.food


def test_logger_accepts_string_event_names(tmp_path):
    env = one_agent_env(tmp_path)
    env.logger.log(time=0, event_type="VIRAL_INFECTION", agent_tag="a0", source_tag=None)
    env.logger.fp.flush()
    last = json.loads(env.logger.save_path.read_text().splitlines()[-1])
    assert last["event"] == "VIRAL_INFECTION"
    assert last["agent_tag"] == "a0"


# Stage 2: the Mechanic, its records, and the scenario loader
# ---------------------------------------------------------------------------


class CountingMechanic(Mechanic):
    name = "counter"

    def __init__(self):
        super().__init__()
        self.calls = []

    def on_menu(self, env, tag, menu):
        self.calls.append("menu")
        menu.pop("create_artifact", None)
        menu["wave"] = {"description": "Wave at everyone.", "params": {}}
        return menu

    def on_action(self, env, tag, action, params):
        self.calls.append("action")
        self.state["energy_seen"] = {t: float(e) for t, e in env.agent_energy.items()}
        if action == "move" and params.get("direction") == "north":
            return "You cannot go north."
        if action == "wave":
            self.state["waves"] = self.state.get("waves", 0) + 1
            return "You waved."
        return None

    def on_step(self, env, infos):
        self.calls.append("step")
        for tag in env.agent_registry:
            env.note(tag, "Counter", "step ran")
        self.state["steps"] = self.state.get("steps", 0) + 1
        self.state["transfers"] = list(env.transfers)
        self.state["deaths"] = [f"{d['tag']}:{d['reason']}" for d in env.deaths]


def test_mechanic_filters_the_menu_refuses_actions_and_runs_each_step(tmp_path):
    mechanic = CountingMechanic()
    env = one_agent_env(tmp_path, mechanic)
    assert "create_artifact" not in env.agent_avail_actions["a0"]
    *_, infos = env.step({"a0": {"action": "move", "params": {"direction": "north"}}})
    assert infos["a0"]["Action outcome"] == "You cannot go north."
    assert env.agent_pos["a0"] == (2, 2)
    assert mechanic.state["steps"] == 1
    assert mechanic.calls[-3:] == ["action", "step", "menu"]
    *_, infos = env.step({"a0": {"action": "wave", "params": {}}})
    assert infos["a0"]["Action outcome"] == "You waved."
    assert mechanic.state["waves"] == 1
    env.logger.fp.flush()
    events = [json.loads(line)["event"] for line in env.logger.save_path.read_text().splitlines()]
    assert events.count("MECHANIC_ACTION") == 2


def test_mechanic_state_survives_a_checkpoint_round_trip(tmp_path):
    mechanic = CountingMechanic()
    env = one_agent_env(tmp_path, mechanic)
    env.step({"a0": STAY})
    env.step({"a0": STAY})
    ckpt = env.get_state_ckpt()
    assert ckpt["mechanics"] == {"counter": mechanic.state}
    restored = one_agent_env(tmp_path / "second", CountingMechanic())
    restored.set_state_ckpt(ckpt)
    assert restored.mechanics[0].state == mechanic.state


def test_transfers_and_deaths_are_recorded_for_the_mechanic(tmp_path):
    mechanic = CountingMechanic()
    env = two_agent_env(tmp_path, mechanic)
    *_, infos = env.step(
        {"a0": {"action": "give", "params": {"target": "Bob", "amount": 5}}, "a1": STAY}
    )
    assert mechanic.state["transfers"] == [("a0", "a1", "give")]
    assert mechanic.state["energy_seen"]["a1"] == 105  # a1 acted after a0's gift landed
    assert infos["a0"]["Counter"] == "step ran"  # written in on_step, seen the same step
    env.kill("a1", "sickness")
    env.step({"a0": STAY, "a1": STAY})
    assert mechanic.state["deaths"] == []
    assert env.get_state_ckpt()["pending_deaths"][0]["tag"] == "a1"
    env.step({"a0": STAY})
    assert mechanic.state["deaths"] == ["a1:sickness"]
    assert mechanic.state["transfers"] == []


def test_grid_distance_wraps_and_agents_within_uses_it(tmp_path):
    env = make_env(tmp_path)
    env.add_agent("a0", "Alice", "text", position=(0, 0))
    env.add_agent("a1", "Bob", "text", position=(9, 9))
    env.add_agent("a2", "Cid", "text", position=(5, 5))
    env.restart_env(agent_poses={"a0": (0, 0), "a1": (9, 9), "a2": (5, 5)})
    assert env.distance((0, 0), (9, 9)) == 1
    assert env.distance((0, 0), (5, 5)) == 5
    assert env.agents_within("a0", 1) == ["a1"]


def test_graph_distance_counts_hops(tmp_path):
    env = OpenGraphWorld(
        graph_cfg=GraphConfig(topology="complete", n_nodes=4, hop_radius=1),
        log_path=tmp_path,
        headless=True,
    )
    nodes = env.world_graph.all_nodes()
    env.add_agent("a0", "Alice", "text", position=nodes[0])
    env.add_agent("a1", "Bob", "text", position=nodes[1])
    env.restart_env(agent_poses={"a0": nodes[0], "a1": nodes[1]})
    assert env.distance(nodes[0], nodes[0]) == 0
    assert env.distance(nodes[0], nodes[1]) == 1
    assert env.agents_within("a0", 1) == ["a1"]
    assert env.agents_within("a0", float("inf")) == ["a1"]


def test_artifacts_can_be_placed_in_an_inventory(tmp_path):
    env = one_agent_env(tmp_path)
    env.add_artifact((2, 2), "text", "mask", "a mask", "a0", 10, to_inventory="a0")
    _, name = env.seed_artifact((2, 2), "text", "kit", "a kit", 10, to_inventory="a0")
    assert name == "kit"
    assert env.agent_inventories["a0"] == {"mask", "kit"}
    assert env.artifact_location["mask"] == ("inv", "a0")
    assert not env.pos_artifacts[(2, 2)]


def test_load_scenario_validates_options_and_builds_mechanics(monkeypatch):
    class Options(BaseModel):
        model_config = ConfigDict(extra="forbid")
        rate: float = 0.1

    module = types.ModuleType("fake_tl_scenario")
    module.Options = Options
    module.build = lambda options: [CountingMechanic()] if options.rate > 0 else []
    monkeypatch.setitem(sys.modules, "fake_tl_scenario", module)

    mechanics = load_scenario("fake_tl_scenario", {"rate": "0.5"})
    assert [m.name for m in mechanics] == ["counter"]
    with pytest.raises(ValidationError):
        load_scenario("fake_tl_scenario", {"rate": 0.5, "typo": 1})


# Stage 3: the artifact type registry
# ---------------------------------------------------------------------------


class GadgetArtifact(Artifact):
    description = "A gadget with a charge."

    def __init__(self, *args, charge=1, **kwargs):
        self.charge = int(charge)
        super().__init__(*args, **kwargs)

    @property
    def actions(self):
        return {}

    def interact(self, agent_name, action, params, timestamp):
        return ""

    def passive_effect(self, timestamp, agent_name):
        return f"A gadget with charge {self.charge}."

    def verify_payload(self, payload):
        return True, ""

    def serialize(self):
        data = super().serialize()
        data["charge"] = self.charge
        return data

    @classmethod
    def deserialize(cls, data):
        artifact = super().deserialize(data)
        artifact.charge = int(data["charge"])
        return artifact


@pytest.fixture
def gadget_type():
    register_artifact_type("gadget")(GadgetArtifact)
    yield
    ARTIFACT_TYPES.pop("gadget", None)
    GadgetArtifact.creatable = False


def test_registered_type_seeds_saves_and_restores(tmp_path, gadget_type):
    env = one_agent_env(tmp_path)
    _, name = env.seed_artifact((2, 2), "gadget", "g1", "", 5, charge=3)
    assert isinstance(env.artifacts[name], GadgetArtifact)
    assert env.artifacts[name].charge == 3
    restored = one_agent_env(tmp_path / "second")
    restored.set_state_ckpt(env.get_state_ckpt())
    assert isinstance(restored.artifacts["g1"], GadgetArtifact)
    assert restored.artifacts["g1"].charge == 3


def test_only_creatable_types_are_offered_to_agents(tmp_path, gadget_type):
    env = one_agent_env(tmp_path)
    menu = env.agent_avail_actions["a0"]["create_artifact"]["params"]["type"]["choices"]
    assert menu == ["text"]
    assert "not a valid type" in env.add_artifact((2, 2), "gadget", "g2", "", "a0", 5)
    GadgetArtifact.creatable = True
    menu = env._get_avail_actions("a0")["create_artifact"]["params"]["type"]["choices"]
    assert menu == ["text", "gadget"]
    env.add_artifact((2, 2), "gadget", "g2", "", "a0", 5, charge=7)
    assert env.artifacts["g2"].charge == 7


def test_unknown_artifact_type_in_a_checkpoint_raises(tmp_path):
    env = one_agent_env(tmp_path)
    env.seed_artifact((2, 2), "text", "sign", "hello", 5)
    ckpt = env.get_state_ckpt()
    ckpt["artifacts"]["sign"]["art_type"] = "ghost"
    with pytest.raises(ValueError, match="unknown type"):
        one_agent_env(tmp_path / "second").set_state_ckpt(ckpt)


def test_json_seeds_pass_type_parameters(tmp_path, gadget_type):
    env = one_agent_env(tmp_path)
    seeds = tmp_path / "seeds"
    seeds.mkdir()
    (seeds / "a.json").write_text(json.dumps([
        {"pose": [3, 3], "art_type": "gadget", "name": "g1", "payload": "", "params": {"charge": 4}},
        {"pose": [3, 4], "art_type": "text", "name": "sign", "payload": "hello"},
    ]))
    fake_runner = SimpleNamespace(env=env, params=SimpleNamespace(env=SimpleNamespace(world_type="grid")))
    SimulationRunner._seed_artifacts_from_path(fake_runner, str(seeds))
    assert env.artifacts["g1"].charge == 4
    assert env.artifacts["sign"].payload == "hello"


# Stage 4: roles and affordances on every world
# ---------------------------------------------------------------------------

ROLES_HOCON = """
{
  tools: [
    {
      name: "warden"
      capacity: 1
      instructions: "Watch over the yard."
    }
  ]
}
"""

CELL_AFFORDANCES = {
    "warden": {
        "affordances": [
            {
                "action": "bless",
                "mode": "add",
                "description": "Give 5 energy to a nearby being.",
                "params": {"target": "Name of a nearby being."},
                "effect": {"type": "add_energy", "amount": 5},
            }
        ]
    },
    "2,2": {"affordances": [{"action": "create_artifact", "mode": "remove"}]},
}


def test_grid_world_gets_roles_and_cell_affordances(tmp_path):
    hocon = tmp_path / "roles.hocon"
    hocon.write_text(ROLES_HOCON)
    aff = tmp_path / "aff.json"
    aff.write_text(json.dumps(CELL_AFFORDANCES))
    env = make_env(
        tmp_path, food_mechanism=False, init_food=0,
        roles_hocon_path=str(hocon), affordances_file_path=str(aff),
    )
    env.add_agent("a0", "Alice", "text", position=(2, 2))
    env.add_agent("a1", "Bob", "text", position=(2, 3))
    env.restart_env(seed=1, agent_poses={"a0": (2, 2), "a1": (2, 3)})
    warden = next(tag for tag in env.agent_registry if env.role_of(tag) == "warden")
    other = "a1" if warden == "a0" else "a0"
    assert "bless" in env.agent_avail_actions[warden]
    assert "bless" not in env.agent_avail_actions[other]
    assert "create_artifact" not in env.agent_avail_actions["a0"]
    assert "create_artifact" in env.agent_avail_actions["a1"]
    obs, _ = env._build_obs(warden)
    assert obs["observation_text"].startswith("Your role title: warden")
    before = env.agent_energy[other]
    *_, infos = env.step({
        warden: {"action": "bless", "params": {"target": env.agent_names[other]}},
        other: STAY,
    })
    assert env.agent_energy[other] == before + 5
    assert infos[warden]["bless"]["status"] == "successful"
    assert env.get_state_ckpt()["_role_assignment"] == {warden: "warden"}


@pytest.mark.parametrize("key", ["yard", "12,2", "2,-1", "2"])
def test_grid_rejects_affordance_keys_that_are_not_cells(tmp_path, key):
    aff = tmp_path / "aff.json"
    aff.write_text(json.dumps({key: {"affordances": [{"action": "create_artifact", "mode": "remove"}]}}))
    with pytest.raises(ValueError, match="neither a role name nor a 'row,col' cell"):
        make_env(tmp_path, affordances_file_path=str(aff))


def test_grid_move_agent_uses_four_neighbours_bounds_and_occupancy(tmp_path):
    hocon = tmp_path / "roles.hocon"
    hocon.write_text(ROLES_HOCON)
    aff = tmp_path / "aff.json"
    aff.write_text(json.dumps({"warden": {"affordances": [{
        "action": "escort", "mode": "add", "description": "Move a nearby being.",
        "params": {"target": "Name", "destination": "row,col"}, "effect": {"type": "move_agent"},
    }]}}))
    env = make_env(
        tmp_path, food_mechanism=False, init_food=0,
        roles_hocon_path=str(hocon), affordances_file_path=str(aff),
    )
    for i in range(3):
        env.add_agent(f"a{i}", f"Name{i}", "text", position=(5, i))
    env.restart_env(seed=1, agent_poses={f"a{i}": (5, i) for i in range(3)})
    warden = next(tag for tag in env.agent_registry if env.role_of(tag) == "warden")
    other, third = [tag for tag in ("a0", "a1", "a2") if tag != warden]
    for tag, pos in ((warden, (2, 2)), (other, (2, 3)), (third, (2, 5))):
        env._update_agent_pos(tag, pos)
    stay = {other: STAY, third: STAY}

    def escort(destination):
        act = {"action": "escort", "params": {"target": env.agent_names[other], "destination": destination}}
        *_, infos = env.step({warden: act, **stay})
        return infos[warden]["escort"]

    assert escort("2,4")["status"] == "successful"
    assert env.agent_pos[other] == (2, 4)
    assert "not next to" in escort("3,5")["reason"]  # diagonal is not a move
    assert "occupied" in escort("2,5")["reason"]  # third agent stands there
    assert "not next to" in escort("2,10")["reason"]  # outside the grid
    assert env.agent_pos[other] == (2, 4)


class IdentityMechanic(Mechanic):
    name = "identity"

    def identity(self, env, tag):
        return {"name": f"Human-{tag}", "persona": "You like tea."} if tag == "a0" else None


def test_mechanic_can_name_agents_and_block_eating(tmp_path):
    env = make_env(tmp_path, food_mechanism=True)
    env.attach(IdentityMechanic())
    assert env.agent_identity("a0") == {"name": "Human-a0", "persona": "You like tea."}
    assert env.agent_identity("a1") == {}
    env.add_agent("a0", "Alice", "text", position=(2, 2))
    env.add_agent("a1", "Bob", "text", position=(5, 5))
    env.restart_env(agent_poses={"a0": (2, 2), "a1": (5, 5)})
    env.food = {(2, 2): 7.0, (5, 5): 7.0}
    env.no_appetite.add("a0")
    before = dict(env.agent_energy)
    env.step({"a0": STAY, "a1": STAY})
    assert env.food == {(2, 2): 7.0}  # a0 may not eat; a1 ate
    assert "no_appetite" in env.get_state_ckpt()
    assert env.agent_energy["a0"] == before["a0"] - 1
    assert env.agent_energy["a1"] == before["a1"] - 1 + 7


def test_persona_is_part_of_the_personality_and_survives_a_checkpoint(tmp_path):
    agent = LLMAgent(agent_name="Alice", agent_tag="a0", log_dir=tmp_path, genome=NoTraitsGenome(), persona="You like tea.")
    assert "You like tea." in agent.system_prompt
    restored = LLMAgent(agent_name="Alice", agent_tag="a0", log_dir=tmp_path / "b", genome=NoTraitsGenome())
    restored.set_state_ckpt(agent.get_state_ckpt())
    restored.update_system_prompt()
    assert "You like tea." in restored.system_prompt
