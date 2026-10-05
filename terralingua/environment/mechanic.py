"""Scenario mechanics: one class with three hooks and a state dict.

A mechanic adds a rule set to a world without editing the engine. The world
calls its hooks at fixed moments. Every hook does nothing by default.

Timing rules:

on_menu:   called whenever the world builds an agent's action menu: at the end
           of each step N for use in step N+1, at restart, and when an agent is
           added. It receives the menu after affordances, removals, and
           exclusions were applied. It may drop, add, or reword entries. The
           social graph world rewords its own `set_role` entry afterwards.
on_action: called right before the world executes one agent's action in step N,
           after the action passed the menu and parameter checks. It sees the
           world with earlier agents' actions of the same step already applied.
           Return None to let the world run the action as usual. Return a string
           to take the action over: the world skips its own handling, shows the
           string to the agent under "Action outcome", and logs a
           MECHANIC_ACTION event. Use it to refuse an action with a reason, or
           to carry out an action the mechanic added to the menu. It cannot
           stop an external tool action (`<server>_<tool>`): the runner sends
           those to the MCP server before the world step.
on_step:   called once in step N, after all actions and artifact updates,
           before energy drain and deaths. Changes made here appear in the
           observations sent at the end of step N.

identity:  called once when the runner creates a new agent, before the world
           adds it. Return {"name": str, "persona": str} or None. The name
           replaces the generated one. The persona is appended to the agent's
           personality in the system prompt. Not called on resume; the agent
           checkpoint keeps both.

Mechanics run in the order they were attached. The menu one mechanic returns is
the input of the next. The first string from on_action wins. The first
identity wins.

What a mechanic may read: env.transfers (this step's successful give, take,
and give_artifact as (giver, receiver, kind)), env.deaths (records of the
previous step's deaths), env.agents_within(tag, r), env.distance(a, b), and the
public agent and artifact tables. env.transfers is rebuilt every step and is
not checkpointed.

What a mechanic may write: env.note(tag, key, text), env.kill(tag, reason),
env.add_artifact(..., to_inventory=tag), env.seed_artifact(...), agent energy,
and env.no_appetite, the set of agents that leave food on their cell untouched.
The world checks that set right after each move, so a tag added in on_step of
step N takes effect from step N+1.

`state` is saved and restored with the world checkpoint under the mechanic's
name. Restore updates the dict in place. Subclasses that define `__init__`
must call `super().__init__()`.
"""


class Mechanic:
    name: str = ""

    def __init__(self) -> None:
        if not self.name:
            self.name = type(self).__name__.lower()
        self.state: dict = {}

    def on_menu(self, env, tag: str, menu: dict) -> dict:
        return menu

    def on_action(self, env, tag: str, action: str, params: dict) -> str | None:
        return None

    def on_step(self, env, infos: dict) -> None:
        return None

    def identity(self, env, tag: str) -> dict | None:
        return None

    def on_identity(self, env, tag: str, identity: dict) -> dict | None:
        """The identity a new agent ends up with: its entry from the personas file, other
        keys such as a role included, or what identity() answered. Return keys that
        complete it, or None to leave it."""
        return None
