"""
Shared observation-formatting and prompt-building logic for LLMAgent and RemoteAgent.
"""

import json

import numpy as np
import tiktoken

from terralingua.agents import prompt_templates
from terralingua.genome.base_genome import Genome


def _strip_response_keys(raw: str, hidden_keys: list) -> str:
    """Drop top-level ``hidden_keys`` from a JSON external-server response before
    it is shown to an agent.

    Non-JSON or non-object responses pass through unchanged. The runner keeps the
    full, unstripped response in obs/logs, so this only affects the agent's view.
    """
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
    if not isinstance(payload, dict) or not any(k in payload for k in hidden_keys):
        return raw
    for k in hidden_keys:
        payload.pop(k, None)
    return json.dumps(payload, indent=2)


def _format_info_fields(info: dict, social_graph: bool) -> str:
    """Render fields without changing their stored keys or values."""
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
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


class AgentMixin:
    """Mixin providing observation formatting, prompt building, and memory validation."""

    # Attributes that subclasses must set in their __init__:
    history: list
    max_history: int
    genome: Genome
    use_inventory: bool
    internal_memory_size: int
    internal_memory_encoder: tiktoken.Encoding
    internal_memory: str

    @property
    def use_internal_memory(self) -> bool:
        """Enable private memory only when its token budget is positive."""
        return self.internal_memory_size > 0

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
        broadcasts = self._format_messages(obs["incoming_broadcasts"])
        # Direct messages are a social-env-only channel; when present, show the
        # messages block as two labelled subsections instead of broadcasts alone.
        if "incoming_dms" in obs:
            messages = (
                f"Broadcasts (from agents you follow):\n{broadcasts}\n\n"
                f"Direct messages (private, one-to-one):\n"
                f"{self._format_messages(obs['incoming_dms'])}"
            )
        else:
            messages = broadcasts
        formatted_obs = {
            "observation": obs["observation_text"],
            "message": messages,
            "inventory": "\n".join(obs["inventory"]) if obs.get("inventory") else "<empty>",
            "energy": obs["energy"],
            "time": obs["time"],
        }
        ext = obs.get("external_response", [])
        if ext:
            hidden = getattr(self, "hidden_external_keys", None)
            if hidden:
                ext = [_strip_response_keys(r, hidden) for r in ext]
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
        inventory_label = "Inventory (private artifacts)" if social_graph else "Inventory"
        history_txt = ""
        if self.history and self.max_history > 0:
            shown = min(len(self.history), self.max_history)
            history_txt = f"=== History (t-{shown} to t-1, oldest first) ===\n"
            for i, (past_obs, past_action, past_msg, past_params, past_info) in enumerate(
                self.history[-self.max_history :], 1
            ):
                history_txt += f"History t-{shown - i + 1}:\n"
                if np.isfinite(formatted_obs["energy"]) and np.isfinite(past_obs["energy"]):
                    history_txt += f"\tEnergy: {past_obs['energy']}\n"
                if np.isfinite(formatted_obs["time"]) and np.isfinite(past_obs["time"]):
                    history_txt += f"\tRemaining lifespan (steps): {past_obs['time']}\n"
                if self.use_inventory:
                    history_txt += f"\t{inventory_label}:\n{past_obs['inventory']}\n"
                history_txt += (
                    ("" if solo else f"\tIncoming msgs: {past_obs['message']}\n")
                    + f"\tObservation:\n{past_obs['observation']}\n"
                )
                # Published context appears once, in the current decision input.
                if isinstance(past_info, dict):
                    past_info = {k: v for k, v in past_info.items() if k != "external_context"}
                if past_info is not None and len(past_info):
                    displayed_info = (
                        _format_info_fields(past_info, social_graph=True)
                        if social_graph and isinstance(past_info, dict)
                        else str(past_info)
                    )
                    history_txt += f"\tAdditional info:\n{displayed_info}\n"
                history_txt += (
                    f"\tAction selected/requested: {past_action}\n"
                    f"\tAction parameters: {past_params}\n"
                    + ("" if solo else f"\tOutgoing message proposed: {past_msg or '<none>'}\n")
                )
                if "external_response" in past_obs:
                    history_txt += f"\tExternal response received:\n{past_obs['external_response']}\n"
                history_txt += "\n"

        additional_info = ""
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
            inventory=formatted_obs["inventory"],
            additional_info=additional_info.strip(),
            actions=json.dumps(available_actions, indent=4),
            action_keys=", ".join(available_actions.keys()),
            memory=internal_memory,
            use_internal_memory=self.use_internal_memory,
            internal_memory_size=self.internal_memory_size,
            use_inventory=self.use_inventory,
            max_message_length=getattr(self, "max_message_length", 200),
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
