"""Storms: a periodic hazard, a shelter action, and a watcher persona.

The mechanic uses the three hooks and the creation-time lookup:

- on_step: every `period` steps a storm forms over a random being and lasts
  `duration` steps. Beings within `radius` of its center lose `damage` energy
  per step, unless a shelter stands on their cell. Watchers get a warning one
  step before a storm forms.
- on_menu: adds `build_shelter` for a being whose cell has no shelter.
- on_action: carries out `build_shelter`. It costs `shelter_cost` energy and
  seeds a shelter artifact on the being's cell.
- identity: the first `watchers` beings created get the watcher persona.
"""

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from terralingua.environment.mechanic import Mechanic

WATCHER_PERSONA = "You read the sky well. You feel a storm one step before it breaks."
STORM_WARNING = "The sky darkens. A storm will break next step."
STORM_HIT = "A storm rages over your cell. You lose {damage} energy."
STORM_SHELTERED = "A storm rages outside, but the shelter keeps you safe."


class StormsOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    period: int = Field(5, ge=1, description="Steps from the end of one storm to the start of the next.")
    duration: int = Field(2, ge=1, description="Steps a storm lasts.")
    radius: int = Field(2, ge=0, description="Distance from the storm center within which beings take damage.")
    damage: int = Field(3, ge=0, description="Energy a being in the storm loses per step.")
    shelter_cost: int = Field(5, ge=0, description="Energy a being pays to build a shelter.")
    watchers: int = Field(1, ge=0, description="How many of the first beings created get the watcher persona.")


class Storms(Mechanic):
    def __init__(self, options: StormsOptions):
        super().__init__()
        self.options = options
        self.state = {"next_storm": options.period, "storm": None, "watchers": []}

    def identity(self, env, tag: str) -> dict | None:
        if len(self.state["watchers"]) >= self.options.watchers:
            return None
        self.state["watchers"].append(tag)
        return {"persona": WATCHER_PERSONA}

    def on_menu(self, env, tag: str, menu: dict) -> dict:
        if not self.sheltered(env, env.agent_pos[tag]):
            menu["build_shelter"] = {
                "description": (
                    f"Build a shelter on your cell. It costs {self.options.shelter_cost} energy. "
                    "Beings on a sheltered cell take no storm damage."
                ),
                "params": {},
            }
        return menu

    def on_action(self, env, tag: str, action: str, params: dict) -> str | None:
        if action != "build_shelter":
            return None
        pose = env.agent_pos[tag]
        if self.sheltered(env, pose):
            return "There is a shelter on this cell already."
        if env.agent_energy[tag] < self.options.shelter_cost:
            return f"You need {self.options.shelter_cost} energy to build a shelter."
        env.agent_energy[tag] -= self.options.shelter_cost
        _, name = env.seed_artifact(pose, "shelter", f"shelter_{tag}", "", -1, movable=False)
        if env.logger:
            env.logger.log(
                time=env.step_count, event_type="SHELTER_BUILT", agent_tag=tag,
                agent_name=env.agent_names[tag], artifact=name, position=pose,
            )
        return f"You built {name}. Beings on this cell are safe from storms."

    def on_step(self, env, infos: dict) -> None:
        o = self.options
        storm = self.state["storm"]
        if storm is not None:
            center = tuple(storm["center"]) if isinstance(storm["center"], list) else storm["center"]
            for tag in sorted(env.agent_registry):
                if env.distance(env.agent_pos[tag], center) > o.radius:
                    continue
                if self.sheltered(env, env.agent_pos[tag]):
                    env.note(tag, "Weather", STORM_SHELTERED)
                else:
                    env.agent_energy[tag] -= o.damage
                    env.note(tag, "Weather", STORM_HIT.format(damage=o.damage))
            storm["remaining"] -= 1
            if storm["remaining"] <= 0:
                self.state["storm"] = None
                self.state["next_storm"] = env.step_count + o.period
            return
        if env.step_count + 1 == self.state["next_storm"]:
            for tag in self.state["watchers"]:
                if tag in env.agent_registry:
                    env.note(tag, "Weather", STORM_WARNING)
        if env.step_count >= self.state["next_storm"] and env.agent_registry:
            tags = sorted(env.agent_registry)
            center = env.agent_pos[tags[int(self.rng(env).integers(len(tags)))]]
            self.state["storm"] = {
                "center": list(center) if isinstance(center, tuple) else center,
                "remaining": o.duration,
            }
            if env.logger:
                env.logger.log(time=env.step_count, event_type="STORM", position=center, duration=o.duration)

    @staticmethod
    def sheltered(env, pose) -> bool:
        return any(env.artifacts[name].art_type == "shelter" for name in env.pos_artifacts.get(pose, ()))

    @staticmethod
    def rng(env) -> np.random.Generator:
        if env.rng is None:
            env.rng = np.random.default_rng()
        return env.rng
