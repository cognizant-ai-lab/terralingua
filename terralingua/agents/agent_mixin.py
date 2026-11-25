"""
Shared observation-formatting and prompt-building logic for LLMAgent and RemoteAgent.
"""

import json

import numpy as np
import tiktoken

from terralingua.agents import prompt_templates
from terralingua.agents.agent_logger import AgentLogger
from terralingua.genome.base_genome import Genome


def _name_agents(value, names_by_tag: dict):
    """Return *value* with every ``agent_tag`` field turned into an ``agent`` field
    that holds the being's name. A tag without a name keeps its value."""
    if isinstance(value, dict):
        return {
            ("agent" if key == "agent_tag" else key): (
                names_by_tag.get(item, item)
                if key == "agent_tag" and isinstance(item, str)
                else _name_agents(item, names_by_tag)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_name_agents(item, names_by_tag) for item in value]
    return value


def _visible_response(raw: str, hidden_keys: list, names_by_tag: dict) -> str:
    """Show a JSON external-server response as the agent reads it: without the
    top-level ``hidden_keys``, and with beings named instead of tagged.

    Non-JSON responses pass through unchanged. The runner keeps the full response
    in obs/logs, so this only affects the agent's view.
    """
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
    stripped = isinstance(payload, dict) and any(k in payload for k in hidden_keys)
    if stripped:
        for k in hidden_keys:
            payload.pop(k, None)
    named = _name_agents(payload, names_by_tag)
    if not stripped and named == payload:
        return raw
    return json.dumps(named, indent=2 if stripped else None)


# Receipts are the runner's bookkeeping for external tool calls. A being without
# external actions never reads them; they stay in the logs and the history records.
RECEIPT_KEYS = frozenset({"action_receipts", "execution_request", "execution_receipt"})


def _visible_info(info: dict, external_actions: bool) -> dict:
    return info if external_actions else {k: v for k, v in info.items() if k not in RECEIPT_KEYS}


# Keys stored by the world, shown to the being under a clearer label. The being reads
# the info of step N while choosing the action of step N+1, so an outcome is always
# the outcome of its previous action.
DISPLAY_LABELS = {"Action outcome": "Previous action outcome"}


def _format_info_fields(info: dict, social_graph: bool) -> str:
    """Render fields without changing their stored keys or values.

    The same rendering serves the current state and every history entry."""
    labels = {
        "Artifacts here": "Your public artifact contents",
        "Artifacts in your inventory": "Your private artifact contents",
    }
    lines = []
    for key, value in info.items():
        if key == "external_context" and isinstance(value, dict):
            lines.extend(f"Context from the {server} service:\n{text}" for server, text in value.items())
        elif social_graph and key in labels:
            content = (
                "\n\n".join(value)
                if isinstance(value, list) and all(isinstance(item, str) for item in value)
                else str(value)
            )
            lines.append(f"{labels[key]}:\n{content}")
        else:
            lines.append(f"{DISPLAY_LABELS.get(key, key)}: {value}")
    return "\n".join(lines)


class AgentMixin:
    """Mixin providing observation formatting, prompt building, and memory validation."""

    # Attributes that subclasses must set in their __init__:
    agent_name: str
    agent_tag: str
    model: str | None = None  # Human and remote agents do not use a runner-hosted model.
    logger: AgentLogger
    history: list
    max_history: int
    genome: Genome
    use_inventory: bool
    internal_memory_size: int
    internal_memory_encoder: tiktoken.Encoding
    internal_memory: str
    names_by_tag: dict  # tag -> name of every being; the runner refreshes it each step
    # The last turn, kept until the runner applies it and writes the log line.
    turn_record: dict | None = None

    @property
    def use_internal_memory(self) -> bool:
        """Enable private memory only when its token budget is positive."""
        return self.internal_memory_size > 0

    def record_default_action(
        self, obs: dict, available_actions: dict, info: dict | None, time: int, action: dict,
    ) -> dict:
        """Record a turn the runner decided for this being, because every model
        call failed or no answer came in time. The history and the log file get
        the same line as for a chosen action."""
        self.history.append((
            self._format_observation(obs), action["action"],
            action.get("messages", {}), action.get("params", {}), info,
        ))
        self.history = self.history[-self.max_history:] if self.max_history > 0 else []
        self.turn_record = {
            "available_actions": available_actions,
            "observation": obs,
            "action": action,
            "input_prompt": "",
        }
        return action

    def log_turn(self, step: int) -> None:
        """Write the log line of the last turn, with the step at which the runner
        applied the action. Nothing is written when no turn is waiting."""
        if self.turn_record is None:
            return
        self.logger.log(
            agent_name=self.agent_name,
            agent_tag=self.agent_tag,
            model=self.model,
            internal_memory=self.internal_memory,
            time=str(step),
            **self.turn_record,
        )
        self.turn_record = None

    def _format_sent(self, sent) -> str:
        """The messages a being sent in one turn, one line per receiver. A plain
        string counts as a broadcast."""
        if isinstance(sent, str):
            return sent or "<none>"
        return self._format_messages(sent or {})

    def _format_messages(self, msg_dict: dict) -> str:
        if not msg_dict:
            return "<none>"
        msg_lines = []
        for sender, msg in msg_dict.items():
            msg_str = (
                msg
                if isinstance(msg, str)
                else np.array2string(np.array(msg), precision=2)
            )
            msg_lines.append(f"{sender}: {msg_str}")
        return " \n".join(msg_lines)

    @staticmethod
    def life_spent(obs: dict) -> int | None:
        """The share of its life a being has lived, in whole percent. None when the
        life is unlimited."""
        total, remaining = obs.get("lifespan", np.inf), obs["time"]
        if not np.isfinite(total) or not np.isfinite(remaining) or total <= 0:
            return None
        return int(round(100 * min(max(total - remaining, 0), total) / total))

    def _update_resource_visibility(self, obs: dict) -> None:
        """Keep survival instructions aligned with this agent's actual resources."""
        finite_energy = bool(np.isfinite(obs["energy"]))
        finite_lifespan = bool(np.isfinite(obs["time"]))
        changed = (
            finite_energy != getattr(self, "finite_energy", True)
            or finite_lifespan != getattr(self, "finite_lifespan", True)
        )
        self.finite_energy = finite_energy
        self.finite_lifespan = finite_lifespan
        if changed and hasattr(self, "update_system_prompt"):
            self.update_system_prompt()

    def _format_observation(self, obs: dict) -> dict:
        self._update_resource_visibility(obs)
        sections = []
        if getattr(self, "allow_broadcasts", True):
            sections.append(
                f"Broadcasts:\n{self._format_messages(obs.get('incoming_broadcasts', {}))}"
            )
        sections.append(
            f"Private messages:\n{self._format_messages(obs.get('incoming_dms', {}))}"
        )
        messages = "\n\n".join(sections)
        formatted_obs = {
            "observation": obs["observation_text"],
            "message": messages,
            "inventory": "\n".join(obs["inventory"]) if obs.get("inventory") else "<empty>",
            "energy": obs["energy"],
            "time": obs["time"],
            "life_spent": self.life_spent(obs),
            "step": obs.get("step"),
        }
        ext = obs.get("external_response", [])
        if ext:
            hidden = getattr(self, "hidden_external_keys", None) or []
            ext = [_visible_response(r, hidden, self.names_by_tag) for r in ext]
            formatted_obs["external_response"] = "\n".join(ext)
        return formatted_obs

    def _make_prompt(
        self,
        formatted_obs: dict,
        available_actions: dict,
        internal_memory: str,
        info: dict | None,
    ) -> str:
        solo = getattr(self, "solo", False)
        social_graph = getattr(self, "system_prompt_template", "") == "social_graph.j2"
        external = bool(getattr(self, "external_actions", False))
        inventory_label = "Inventory (private artifacts)" if social_graph else "Inventory"
        history_txt = ""
        if self.history and self.max_history > 0:
            shown = min(len(self.history), self.max_history)
            history_txt = f"=== History (t-{shown} to t-1, oldest first) ===\n"
            for i, (past_obs, past_action, past_msg, past_params, past_info) in enumerate(
                self.history[-self.max_history :], 1
            ):
                history_txt += f"History t-{shown - i + 1}:\n"
                if past_obs.get("step") is not None:
                    history_txt += f"\tStep: {past_obs['step']}\n"
                if np.isfinite(formatted_obs["energy"]) and np.isfinite(past_obs["energy"]):
                    history_txt += f"\tEnergy: {past_obs['energy']}\n"
                if past_obs.get("life_spent") is not None:
                    history_txt += f"\tLife lived: {past_obs['life_spent']}%\n"
                if self.use_inventory:
                    history_txt += f"\t{inventory_label}:\n{past_obs['inventory']}\n"
                history_txt += (
                    ("" if solo else f"\tIncoming msgs: {past_obs['message']}\n")
                    + f"\tObservation:\n{past_obs['observation']}\n"
                )
                # Keep inventory names above; full contents and published context
                # belong in the current decision input, not every history entry.
                if isinstance(past_info, dict):
                    past_info = _visible_info(
                        {
                            k: v for k, v in past_info.items()
                            if k not in {"external_context", "Artifacts in your inventory"}
                        }, external
                    )
                if past_info is not None and len(past_info):
                    displayed_info = (
                        _format_info_fields(past_info, social_graph=social_graph)
                        if isinstance(past_info, dict)
                        else str(past_info)
                    )
                    history_txt += f"\tAdditional info:\n{displayed_info}\n"
                history_txt += (
                    f"\tAction selected/requested: {past_action}\n"
                    f"\tAction parameters: {past_params}\n"
                    + ("" if solo else f"\tMessages sent: {self._format_sent(past_msg)}\n")
                )
                if "external_response" in past_obs:
                    history_txt += f"\tExternal response received:\n{past_obs['external_response']}\n"
                history_txt += "\n"

        additional_info = ""
        if isinstance(info, dict):
            info = _visible_info(info, external)
        if info is not None and len(info):
            info_list = _format_info_fields(info, social_graph=social_graph)
            additional_info = f"\n=== Additional info from the environment ===\n{info_list}\n"

        render_kwargs = dict(
            history=history_txt,
            observation=formatted_obs["observation"],
            messages=formatted_obs["message"],
            finite_energy=bool(np.isfinite(formatted_obs["energy"])),
            finite_lifespan=bool(np.isfinite(formatted_obs["time"])),
            energy=formatted_obs["energy"],
            time=formatted_obs["time"],
            life_spent=formatted_obs.get("life_spent"),
            step=formatted_obs.get("step"),
            inventory=formatted_obs["inventory"],
            additional_info=additional_info.strip(),
            actions=json.dumps(available_actions, indent=4),
            action_keys=", ".join(available_actions.keys()),
            memory=internal_memory,
            use_internal_memory=self.use_internal_memory,
            internal_memory_size=self.internal_memory_size,
            use_inventory=self.use_inventory,
            max_message_length=getattr(self, "max_message_length", 200),
            allow_broadcasts=getattr(self, "allow_broadcasts", True),
            solo=solo,
            social_graph=social_graph,
        )
        if "external_response" in formatted_obs:
            render_kwargs["external_responses"] = formatted_obs["external_response"]
        render_kwargs.setdefault("external_actions", bool(getattr(self, "external_actions", False)))
        return prompt_templates.AGENT_PROMPT.render(**render_kwargs)

    def _build_prompt_sync(
        self,
        obs: dict,
        available_actions: dict,
        info: dict | None,
    ) -> tuple[dict, str]:
        """Format observation and render prompt in one call.

        Designed to be passed to ``asyncio.to_thread`` so the CPU-bound work
        (history serialization, Jinja2 rendering) runs in a thread instead of
        blocking the event loop before the first LLM await.
        """
        formatted_obs = self._format_observation(obs)
        prompt = self._make_prompt(
            formatted_obs=formatted_obs,
            available_actions=available_actions,
            internal_memory=self.internal_memory,
            info=info,
        )
        return formatted_obs, prompt

    def validate_internal_memory(self, internal_memory: str) -> str:
        """Ensure internal memory stays within token limits."""
        if self.internal_memory_size <= 0:
            return ""
        tokens = self.internal_memory_encoder.encode(str(internal_memory))
        if len(tokens) > self.internal_memory_size:
            tokens = tokens[-self.internal_memory_size :]
            internal_memory = self.internal_memory_encoder.decode(tokens)
        return internal_memory
