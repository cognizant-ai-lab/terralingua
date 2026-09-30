"""roles.py — RolesMixin: persistent roles for every world.

A role is a labeled position that outlives its occupants. Roles come from the
entries of the roles HOCON file (`env.roles_hocon_path`). Each entry may
carry:

- `name`, `instructions`, and `function.description`. Together they form the
  charter every occupant reads.
- `capacity` (default 1): the number of occupant slots.
- `on_vacancy` (default `open`): what a slot does when its occupant dies or
  leaves. `open` reopens the slot for requests. `retire` destroys the slot; at
  zero slots the role leaves the roster. `succeed` passes the slot to the
  successor the occupant named with `name_successor`, or reopens it when no
  valid successor exists. A `retire` role cannot be left by choice.
- `transferable` (default false): occupants may hand the role to a roleless
  agent they can act on with `transfer_role`. The hand-over is immediate and
  does not trigger `on_vacancy`.
- `start_nodes` (graph world only): node ids where the occupants stand after
  the initial draw, filled round-robin.

Permissions come from the entries of `env.affordances_file_path` whose
keys match role names. They follow the occupant: the host world merges them
into the occupant's action menu for as long as the assignment lasts, wherever
the occupant stands.

The mixin owns the role state and the role rules. The host world provides
`agent_registry`, `agent_names`, `name_to_tag`, `rng`, `step_count`, and the
two target hooks `_role_target` and `_role_target_names`, which decide which
agents an occupant may hand a role to or name as successor.
"""

import json
import logging
from typing import Dict, List

import numpy as np

from terralingua.environment.actions import (
    LocationAffordance,
    build_leave_role_action,
    build_name_successor_action,
    build_request_role_action,
    build_transfer_role_action,
)
from terralingua.environment.env_logger import Event
from terralingua.experiment.neuro_san_hocon import load_neuro_san_agent_network

log = logging.getLogger(__name__)

ROLE_VACANCY_POLICIES = ("open", "retire", "succeed")


class RolesMixin:
    """Persistent roles: state, rules, menu entries, observation text, checkpoint."""

    _roles: Dict[str, dict]
    _role_assignment: Dict[str, str]
    _role_successor: Dict[str, str]
    _initial_roles_assigned: bool

    # ---------- setup ----------
    def _init_roles(self, roles_hocon_path, affordances_file_path, network=None) -> None:
        #   _roles: role name -> {charter, capacity, mandate, affordances,
        #           on_vacancy, transferable, start_nodes}
        #   _role_assignment: agent_tag -> role name
        #   _role_successor: agent_tag -> agent_tag named to inherit the role
        self._roles: Dict[str, dict] = {}
        self._role_assignment: Dict[str, str] = {}
        self._role_successor: Dict[str, str] = {}
        self._initial_roles_assigned = False
        self.execution_context: dict = {}
        if roles_hocon_path:
            role_affordances: dict = {}
            if affordances_file_path:
                with open(affordances_file_path) as f:
                    role_affordances = json.load(f)
            if network is None:
                network = load_neuro_san_agent_network(roles_hocon_path)
            for spec in network.agents:
                self._roles[spec.name] = self._role_from_spec(spec, role_affordances)

    @staticmethod
    def _role_from_spec(spec, role_affordances: dict) -> dict:
        charter = "\n\n".join(
            text for text in (spec.description, spec.instructions) if text
        ).strip()
        capacity = int(spec.raw.get("capacity", 1))
        if capacity < 1:
            raise ValueError(
                f"Role '{spec.name}': capacity must be at least 1, got {capacity}."
            )
        on_vacancy = str(spec.raw.get("on_vacancy", "open"))
        if on_vacancy not in ROLE_VACANCY_POLICIES:
            raise ValueError(
                f"Role '{spec.name}': on_vacancy must be one of "
                f"{list(ROLE_VACANCY_POLICIES)}, got '{on_vacancy}'."
            )
        transferable = spec.raw.get("transferable", False)
        if not isinstance(transferable, bool):
            raise ValueError(
                f"Role '{spec.name}': transferable must be true or false, "
                f"got {transferable!r}."
            )
        start_nodes = spec.raw.get("start_nodes", []) or []
        if not isinstance(start_nodes, list) or not all(
            isinstance(n, str) for n in start_nodes
        ):
            raise ValueError(
                f"Role '{spec.name}': start_nodes must be a list of node ids, "
                f"got {start_nodes!r}."
            )
        return {
            "charter": charter,
            "capacity": capacity,
            "mandate": "",
            "affordances": list(
                role_affordances.get(spec.name, {}).get("affordances", [])
            ),
            "on_vacancy": on_vacancy,
            "transferable": transferable,
            "start_nodes": list(start_nodes),
        }

    # ---------- queries ----------
    def _active_roles(self) -> Dict[str, dict]:
        """Roles with at least one slot. A retired role keeps its record only."""
        return {n: r for n, r in self._roles.items() if r["capacity"] > 0}

    def role_of(self, agent: str) -> str | None:
        return self._role_assignment.get(agent)

    def _role_occupants(self, role_name: str) -> List[str]:
        return [
            tag for tag, role in self._role_assignment.items() if role == role_name
        ]

    def _open_roles(self) -> Dict[str, int]:
        """Role name -> free slot count, only roles with at least one free."""
        open_roles: Dict[str, int] = {}
        for name, role in self._roles.items():
            free = role["capacity"] - len(self._role_occupants(name))
            if free > 0:
                open_roles[name] = free
        return open_roles

    def _role_affordances_for(self, agent: str) -> List[LocationAffordance]:
        """The permissions the agent's current role grants or blocks."""
        role_name = self._role_assignment.get(agent)
        if role_name is None:
            return []
        return [
            LocationAffordance.from_dict(aff)
            for aff in self._roles[role_name]["affordances"]
        ]

    # ---------- target hooks (the host world overrides these) ----------
    def _role_target_names(self, agent: str) -> List[str]:
        """Names of the agents the actor may hand a role to or name as successor."""
        return sorted(self.agent_names[t] for t in self._get_nearby_agents(agent))

    def _role_target(
        self, agent: str, params: dict, action: str, infos: dict
    ) -> str | None:
        """Resolve params['target'] to an agent tag the actor may act on.

        Writes a failure under infos[agent][action] and returns None when the
        target is unknown, is the actor, or is out of reach."""
        target_name = params.get("target", "")
        target_tag = self.name_to_tag.get(target_name) if target_name else None
        if target_tag is None or target_tag not in self.agent_registry:
            infos[agent] = {
                action: {
                    "status": "failed",
                    "reason": f"Target '{target_name}' not found.",
                }
            }
            return None
        if target_tag == agent:
            infos[agent] = {
                action: {"status": "failed", "reason": "Cannot target yourself."}
            }
            return None
        if target_tag not in self._get_nearby_agents(agent):
            infos[agent] = {
                action: {
                    "status": "failed",
                    "reason": f"'{target_name}' is not within your reach.",
                }
            }
            return None
        return target_tag

    # ---------- assignment ----------
    def _ensure_initial_roles(self) -> None:
        """Randomly fill role slots from the current population, once.

        Runs at the first reset, when the whole founding population exists.
        Agents beyond the slot count stay roleless; mid-run newborns stay
        roleless until they request a role.
        """
        if not self._roles or self._initial_roles_assigned:
            return
        self._initial_roles_assigned = True
        if self.rng is None:
            self.rng = np.random.default_rng()
        agents = sorted(self.agent_registry)
        slots: List[str] = []
        for name in sorted(self._roles):
            slots.extend([name] * self._roles[name]["capacity"])
        order = [agents[i] for i in self.rng.permutation(len(agents))]
        for tag, role_name in zip(order, slots):
            self._assign_role(tag, role_name)

    def _assign_role(self, agent: str, role_name: str) -> None:
        self._role_assignment[agent] = role_name
        log.info(
            "[Roles] %s now occupies role %s (step %s)",
            self.agent_names.get(agent, agent),
            role_name,
            self.step_count,
        )
        self._log_role_event(Event.ROLE_ASSIGNED, agent, role_name)

    def _log_role_event(self, event: Event, agent: str, role_name: str, **data) -> None:
        logger = getattr(self, "logger", None)
        if logger:
            logger.log(
                time=self.step_count,
                event_type=event,
                agent_tag=agent,
                agent_name=self.agent_names.get(agent, agent),
                role=role_name,
                **data,
            )

    def _release_role(self, agent: str, cause: str) -> str | None:
        """Drop the agent's assignment, then apply the role's vacancy policy.

        `cause` is "death", "left", or "transfer". A transfer never triggers
        the policy: the slot goes straight to the receiver."""
        role_name = self._role_assignment.pop(agent, None)
        if role_name is None:
            return None
        heir = self._role_successor.pop(agent, None)
        log.info(
            "[Roles] %s left role %s (%s, step %s)",
            self.agent_names.get(agent, agent),
            role_name,
            cause,
            self.step_count,
        )
        self._log_role_event(Event.ROLE_RELEASED, agent, role_name, cause=cause)
        if cause == "transfer":
            return role_name
        role = self._roles[role_name]
        if role["on_vacancy"] == "retire":
            role["capacity"] -= 1
            log.info(
                "[Roles] role %s lost a slot, %d left (step %s)",
                role_name,
                role["capacity"],
                self.step_count,
            )
        elif role["on_vacancy"] == "succeed":
            if (
                heir is not None
                and heir in self.agent_registry
                and heir != agent
                and heir not in self._role_assignment
            ):
                self._assign_role(heir, role_name)
        return role_name

    def _forget_successor(self, agent: str) -> None:
        """Drop every successor pointer that names a departed agent."""
        for holder, heir in list(self._role_successor.items()):
            if heir == agent:
                self._role_successor.pop(holder, None)

    # ---------- action handlers ----------
    def _handle_request_role(self, agent: str, params: dict, infos: dict) -> None:
        role_name = params.get("role", "")
        if role_name not in self._active_roles():
            infos[agent] = {
                "request_role": {
                    "status": "failed",
                    "reason": f"Unknown role '{role_name}'. Roles: "
                    f"{sorted(self._active_roles())}.",
                }
            }
            return
        if agent in self._role_assignment:
            infos[agent] = {
                "request_role": {
                    "status": "failed",
                    "reason": f"You already hold the role "
                    f"'{self._role_assignment[agent]}'. Leave it first "
                    "(leave_role).",
                }
            }
            return
        if role_name not in self._open_roles():
            infos[agent] = {
                "request_role": {
                    "status": "failed",
                    "reason": f"Role '{role_name}' has no free slot.",
                }
            }
            return
        self._assign_role(agent, role_name)
        infos[agent] = {
            "request_role": {"status": "successful", "role": role_name}
        }

    def _handle_leave_role(self, agent: str, infos: dict) -> None:
        role_name = self._role_assignment.get(agent)
        if role_name is None:
            infos[agent] = {
                "leave_role": {"status": "failed", "reason": "You hold no role."}
            }
            return
        if self._roles[role_name]["on_vacancy"] == "retire":
            infos[agent] = {
                "leave_role": {
                    "status": "failed",
                    "reason": f"Role '{role_name}' is bound to its occupant and "
                    "cannot be left.",
                }
            }
            return
        self._release_role(agent, cause="left")
        infos[agent] = {"leave_role": {"status": "successful", "role": role_name}}

    def _handle_transfer_role(self, agent: str, params: dict, infos: dict) -> None:
        role_name = self._role_assignment.get(agent)
        if role_name is None:
            infos[agent] = {
                "transfer_role": {"status": "failed", "reason": "You hold no role."}
            }
            return
        if not self._roles[role_name]["transferable"]:
            infos[agent] = {
                "transfer_role": {
                    "status": "failed",
                    "reason": f"Role '{role_name}' cannot be transferred.",
                }
            }
            return
        target_tag = self._role_target(agent, params, "transfer_role", infos)
        if target_tag is None:
            return
        target_name = self.agent_names[target_tag]
        held = self._role_assignment.get(target_tag)
        if held is not None:
            infos[agent] = {
                "transfer_role": {
                    "status": "failed",
                    "reason": f"'{target_name}' already holds the role '{held}'.",
                }
            }
            return
        self._release_role(agent, cause="transfer")
        self._assign_role(target_tag, role_name)
        infos[agent] = {
            "transfer_role": {
                "status": "successful",
                "role": role_name,
                "target": target_name,
            }
        }

    def _handle_name_successor(self, agent: str, params: dict, infos: dict) -> None:
        role_name = self._role_assignment.get(agent)
        if role_name is None:
            infos[agent] = {
                "name_successor": {"status": "failed", "reason": "You hold no role."}
            }
            return
        if self._roles[role_name]["on_vacancy"] != "succeed":
            infos[agent] = {
                "name_successor": {
                    "status": "failed",
                    "reason": f"Role '{role_name}' does not pass to a successor.",
                }
            }
            return
        target_tag = self._role_target(agent, params, "name_successor", infos)
        if target_tag is None:
            return
        self._role_successor[agent] = target_tag
        infos[agent] = {
            "name_successor": {
                "status": "successful",
                "role": role_name,
                "successor": self.agent_names[target_tag],
            }
        }

    def _set_role_mandate(self, agent: str, params: dict, infos: dict) -> None:
        """Roles-mode `set_role`: rewrite a role's standing mandate.

        The target is a role name. Every current and future occupant sees the
        new mandate. The role's charter is fixed."""
        role_name = params.get("target", "")
        if role_name not in self._active_roles():
            infos[agent] = {
                "set_role": {
                    "status": "failed",
                    "reason": f"Unknown role '{role_name}'. Roles: "
                    f"{sorted(self._active_roles())}.",
                }
            }
            return
        mandate = params.get("motivation", "") or params.get("role", "")
        try:
            metadata = self._mandate_metadata(agent, params)
        except ValueError as error:
            infos[agent] = {"set_role": {"status": "failed", "reason": str(error)}}
            return
        self._roles[role_name]["mandate"] = mandate
        self._roles[role_name]["mandate_meta"] = metadata
        log.info(
            "[Roles] %s set the mandate of role %s (step %s)",
            self.agent_names.get(agent, agent),
            role_name,
            self.step_count,
        )
        infos[agent] = {
            "set_role": {
                "status": "successful",
                "target": role_name,
                "mandate_meta": metadata,
                "occupants": [
                    self.agent_names[t] for t in self._role_occupants(role_name)
                ],
            }
        }

    def _mandate_metadata(self, agent: str, params: dict) -> dict:
        """Record the scope and sources of temporary guidance."""
        expires = params.get("expires_at")
        if expires in (None, ""):
            expires = None
        else:
            if isinstance(expires, bool):
                raise ValueError("expires_at must be a nonnegative integer step.")
            try:
                parsed = int(expires)
            except (TypeError, ValueError):
                raise ValueError("expires_at must be a nonnegative integer step.") from None
            if parsed < 0 or (isinstance(expires, float) and expires != parsed):
                raise ValueError("expires_at must be a nonnegative integer step.")
            expires = parsed
        context = params.get("context")
        if context in (None, ""):
            context = dict(getattr(self, "execution_context", {}))
        elif isinstance(context, str):
            try:
                context = json.loads(context)
            except ValueError:
                raise ValueError("context must be a JSON object with scalar values.") from None
        if not isinstance(context, dict) or any(
            not isinstance(key, str) or isinstance(value, (dict, list))
            for key, value in context.items()
        ):
            raise ValueError("context must be a JSON object with scalar values.")
        return {
            "created_at": self.step_count, "created_by": agent,
            "expires_at": expires, "context": dict(context),
        }

    def _mandate_review_reasons(self, metadata: dict | None) -> List[str]:
        """State why recorded guidance needs review. Keep its text and its sources."""
        if not metadata:
            return ["Its context was not recorded."]
        reasons = []
        expires = metadata.get("expires_at")
        if expires is not None and self.step_count >= expires:
            reasons.append(f"It expired at step {expires}.")
        current = getattr(self, "execution_context", {})
        for field, value in metadata.get("context", {}).items():
            if field not in current:
                reasons.append(f"Current context {field!r} is unknown.")
            elif type(current[field]) is not type(value) or current[field] != value:
                reasons.append(f"Context {field!r} changed from {value!r} to {current[field]!r}.")
        return reasons

    def _render_mandate(self, text: str, metadata: dict | None) -> str:
        if not text:
            return ""
        reasons = self._mandate_review_reasons(metadata)
        if reasons:
            return (
                "Mandate REVIEW NEEDED. Historical guidance, not a current requirement. "
                + " ".join(reasons) + f"\nRecorded guidance: {text}\n"
                "Reconsider its premise before following it. Request an updated mandate if needed."
            )
        return f"Current mandate: {text}"

    def refresh_role_context(self, agent: str, obs: dict) -> None:
        """Refresh only the role description. Preserve observations and social messages."""
        previous = self._render_directive(obs)
        if self._roles:
            self._stamp_role_obs(agent, obs)
        elif hasattr(self, "_stamp_legacy_directive_obs"):
            self._stamp_legacy_directive_obs(agent, obs)
        current = self._render_directive(obs)
        if previous == current:
            return
        text = obs.get("observation_text", "")
        if previous and text.startswith(previous):
            text = text[len(previous):].lstrip("\n")
        obs["observation_text"] = "\n\n".join(part for part in (current, text) if part)

    def _on_roles_action(
        self, agent: str, action_name: str, action_params: dict, infos: dict
    ) -> bool:
        """Dispatch the role actions.

        Returns True when the action was consumed."""
        if not self._roles:
            return False
        if action_name == "request_role":
            self._handle_request_role(agent, action_params, infos)
            return True
        if action_name == "leave_role":
            self._handle_leave_role(agent, infos)
            return True
        if action_name == "transfer_role":
            self._handle_transfer_role(agent, action_params, infos)
            return True
        if action_name == "name_successor":
            self._handle_name_successor(agent, action_params, infos)
            return True
        return False

    # ---------- action menu ----------
    def _role_actions(self, agent: str) -> dict:
        """The role actions the agent may take this step."""
        actions: dict = {}
        if not self._roles:
            return actions
        role_name = self._role_assignment.get(agent)
        if role_name is None:
            open_roles = self._open_roles()
            if open_roles:
                actions.update([build_request_role_action(open_roles)])
            return actions
        role = self._roles[role_name]
        if role["on_vacancy"] != "retire":
            actions.update([build_leave_role_action(role_name, role["on_vacancy"])])
        if role["transferable"] or role["on_vacancy"] == "succeed":
            names = self._role_target_names(agent)
            roleless = [
                n for n in names if self._role_assignment.get(self.name_to_tag[n]) is None
            ]
            if role["transferable"] and roleless:
                actions.update([build_transfer_role_action(role_name, roleless)])
            if role["on_vacancy"] == "succeed" and names:
                heir = self._role_successor.get(agent)
                heir_name = self.agent_names.get(heir, "") if heir else ""
                actions.update(
                    [build_name_successor_action(role_name, names, heir_name)]
                )
        return actions

    # ---------- observation ----------
    def _stamp_role_obs(self, agent: str, obs: dict) -> None:
        """Set obs['role'] and obs['motivation'] from the agent's role."""
        if not self._roles:
            return
        role_name = self._role_assignment.get(agent, "")
        obs["role"] = role_name
        parts = []
        if role_name:
            role = self._roles[role_name]
            parts.append(role["charter"])
            if role["mandate"]:
                parts.append(self._render_mandate(role["mandate"], role.get("mandate_meta")))
        obs["motivation"] = "\n\n".join(parts)

    @staticmethod
    def _render_directive(obs: dict) -> str:
        """Render the agent's role title and description, if any."""
        lines = []
        if obs.get("role"):
            lines.append(f"Your role title: {obs['role']}")
        if obs.get("motivation"):
            lines.append(f"Your role description: {obs['motivation']}")
        return "\n".join(lines)

    def _render_roles_overview(self, agent: str) -> str:
        """The role board: every active role, its occupancy, and free slots."""
        lines = ["Organization roles (persistent positions agents occupy):"]
        active = self._active_roles()
        for name in sorted(active):
            role = active[name]
            occupants = sorted(
                self.agent_names[t] for t in self._role_occupants(name)
            )
            held = ", ".join(occupants) if occupants else "vacant"
            free = role["capacity"] - len(occupants)
            slot_note = f", {free} slot(s) open" if free > 0 else ""
            lines.append(
                f"- {name} ({len(occupants)}/{role['capacity']}: {held}{slot_note})"
            )
        heir = self._role_successor.get(agent)
        if heir is not None and heir in self.agent_names:
            lines.append(f"Your named successor: {self.agent_names[heir]}")
        if agent not in self._role_assignment:
            lines.append(
                "You hold no role. You may request an open one with "
                "request_role; a role gives you its charter, mandate, and "
                "permissions."
            )
        return "\n".join(lines)

    # ---------- checkpoint ----------
    def _roles_ckpt(self, ckpt: dict) -> None:
        ckpt["execution_context"] = dict(getattr(self, "execution_context", {}))
        ckpt["_roles"] = {name: dict(role) for name, role in self._roles.items()}
        ckpt["_role_assignment"] = dict(self._role_assignment)
        ckpt["_role_successor"] = dict(self._role_successor)
        ckpt["_initial_roles_assigned"] = self._initial_roles_assigned

    def _restore_roles_ckpt(self, state_ckpt: dict) -> None:
        # `.get` with defaults keeps older checkpoints loadable.
        self.execution_context = dict(state_ckpt.get("execution_context", {}))
        if state_ckpt.get("_roles"):
            self._roles = {
                name: dict(role) for name, role in state_ckpt["_roles"].items()
            }
            for role in self._roles.values():
                role.setdefault("on_vacancy", "open")
                role.setdefault("transferable", False)
        self._role_assignment = dict(state_ckpt.get("_role_assignment", {}))
        self._role_successor = dict(state_ckpt.get("_role_successor", {}))
        self._initial_roles_assigned = state_ckpt.get(
            "_initial_roles_assigned", self._initial_roles_assigned
        )
