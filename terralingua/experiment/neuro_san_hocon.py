"""Helpers for importing Neuro-SAN agent-network HOCON files.

The importer intentionally reads the data format only. It does not depend on or
execute the Neuro-SAN runtime; TerraLingua uses the parsed network as an initial
agent roster and graph topology.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NeuroSanAgentSpec:
    """A TerraLingua-friendly view of one Neuro-SAN tool/agent entry."""

    name: str
    instructions: str = ""
    description: str = ""
    downstream_agents: tuple[str, ...] = ()
    external_tools: tuple[str, ...] = ()
    command: str | None = None
    display_as: str | None = None
    llm_config: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NeuroSanNetworkSpec:
    """Parsed Neuro-SAN network with local agent connections resolved."""

    path: Path
    agents: tuple[NeuroSanAgentSpec, ...]
    frontman: str
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    llm_config: dict[str, Any] = field(default_factory=dict)

    @property
    def agent_names(self) -> tuple[str, ...]:
        return tuple(agent.name for agent in self.agents)

    def agent_by_name(self, name: str) -> NeuroSanAgentSpec:
        for agent in self.agents:
            if agent.name == name:
                return agent
        raise KeyError(name)

    def graph_data(self, bidirectional: bool = False) -> dict[str, Any]:
        """Return JSON-like graph data accepted by the file graph loader."""
        nodes = []
        edges = []
        for agent in self.agents:
            nodes.append(
                {
                    "id": agent.name,
                    "label": agent.display_as or agent.name,
                    "description": agent.description,
                    "frontman": agent.name == self.frontman,
                    "neuro_san": {
                        "instructions": agent.instructions,
                        "command": agent.command,
                        "downstream_agents": list(agent.downstream_agents),
                        "external_tools": list(agent.external_tools),
                        "llm_config": agent.llm_config,
                    },
                }
            )
            for target in agent.downstream_agents:
                edges.append(
                    {
                        "from": agent.name,
                        "to": target,
                        "label": target,
                        "source": "neuro_san_tools",
                    }
                )
                if bidirectional and target != agent.name:
                    edges.append(
                        {
                            "from": target,
                            "to": agent.name,
                            "label": agent.name,
                            "source": "neuro_san_tools_reverse",
                        }
                    )
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "source": "neuro_san_hocon",
                "path": str(self.path),
                "description": self.description,
                "frontman": self.frontman,
            },
        }


_UNRESOLVED = object()


def _is_config_values(value: Any) -> bool:
    return value.__class__.__name__ == "ConfigValues" and hasattr(value, "tokens")


def _is_config_substitution(value: Any) -> bool:
    return value.__class__.__name__ == "ConfigSubstitution" and hasattr(
        value, "variable"
    )


def _is_config_quoted_string(value: Any) -> bool:
    return value.__class__.__name__ == "ConfigQuotedString" and hasattr(value, "value")


def _config_child(value: Any, key: str) -> tuple[bool, Any]:
    if isinstance(value, dict) and key in value:
        return True, value[key]
    if hasattr(value, "keys") and key in value.keys():
        try:
            return True, value[key]
        except (KeyError, TypeError):
            pass
    return False, None


def _lookup_config_substitution(root: Any, variable: str, seen: set[str]) -> Any:
    if root is None or variable in seen:
        return _UNRESOLVED

    current = root
    for part in str(variable).split("."):
        part = part.strip().strip('"')
        if not part:
            continue
        found, current = _config_child(current, part)
        if not found:
            return _UNRESOLVED

    return _to_plain(current, root=root, seen={*seen, variable})


def _merge_dict_parts(parts: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for part in parts:
        merged.update(part)
    return merged


def _to_plain_config_values(value: Any, root: Any, seen: set[str]) -> Any:
    parts = []
    for token in value.tokens:
        plain = _to_plain(token, root=root, seen=seen)
        if plain is _UNRESOLVED or plain is None:
            continue
        parts.append(plain)

    if not parts:
        return ""
    if all(isinstance(part, dict) for part in parts):
        return _merge_dict_parts(parts)
    if all(isinstance(part, list) for part in parts):
        merged_list = []
        for part in parts:
            merged_list.extend(part)
        return merged_list
    if len(parts) == 1:
        return parts[0]
    if all(not isinstance(part, (dict, list)) for part in parts):
        return "".join(str(part) for part in parts)

    dict_parts = [part for part in parts if isinstance(part, dict)]
    scalar_parts = [
        str(part) for part in parts if not isinstance(part, (dict, list)) and str(part)
    ]
    if dict_parts and not "".join(scalar_parts).strip():
        return _merge_dict_parts(dict_parts)
    return "".join(str(part) for part in parts if not isinstance(part, (dict, list)))


def _to_plain(value: Any, root: Any = None, seen: set[str] | None = None) -> Any:
    """Convert PyHOCON containers into plain Python dictionaries/lists."""
    if seen is None:
        seen = set()
    if _is_config_substitution(value):
        return _lookup_config_substitution(root, str(value.variable), seen)
    if _is_config_values(value):
        return _to_plain_config_values(value, root, seen)
    if _is_config_quoted_string(value):
        return str(value.value)
    if hasattr(value, "items") and hasattr(value, "keys") and not isinstance(
        value, (str, bytes)
    ):
        effective_root = value if root is None else root
        result = {}
        for key, child in value.items():
            plain = _to_plain(child, root=effective_root, seen=seen)
            if plain is not _UNRESOLVED:
                result[str(key).strip('"')] = plain
        return result
    if isinstance(value, list):
        result = []
        for child in value:
            plain = _to_plain(child, root=root, seen=seen)
            if plain is not _UNRESOLVED:
                result.append(plain)
        return result
    return value


@contextlib.contextmanager
def _suppress_pyhocon_warnings():
    logger = logging.getLogger("pyhocon.config_parser")
    disabled = logger.disabled
    logger.disabled = True
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The 'warn' method is deprecated.*",
                category=DeprecationWarning,
            )
            yield
    finally:
        logger.disabled = disabled


def _parse_hocon_file(config_factory: Any, hocon_path: Path) -> Any:
    """Parse HOCON, tolerating optional Neuro-SAN include fragments.

    Neuro-SAN registries often include shared snippets such as aaosa.hocon. For
    TerraLingua initialization, missing shared snippets should not block loading
    the local agent graph; unresolved substitutions are resolved locally when
    possible and otherwise dropped by ``_to_plain``.
    """
    try:
        with contextlib.redirect_stderr(io.StringIO()), _suppress_pyhocon_warnings():
            return config_factory.parse_file(str(hocon_path))
    except Exception as resolved_exc:
        try:
            with (
                contextlib.redirect_stderr(io.StringIO()),
                _suppress_pyhocon_warnings(),
            ):
                return config_factory.parse_file(str(hocon_path), resolve=False)
        except Exception:
            raise resolved_exc


def _stringify_tool_ref(tool_ref: Any) -> str:
    if isinstance(tool_ref, str):
        return tool_ref
    if isinstance(tool_ref, dict):
        for key in ("name", "tool", "agent", "server", "url"):
            value = tool_ref.get(key)
            if isinstance(value, str):
                return value
    return json.dumps(tool_ref, sort_keys=True, default=str)


def _function_description(agent_data: dict[str, Any]) -> str:
    function = agent_data.get("function", {})
    if isinstance(function, dict):
        description = function.get("description", "")
        if description is not None:
            return str(description).strip()
    return ""


def _replace_strings_once(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        result = value
        for key, replacement in replacements.items():
            result = result.replace("{" + key + "}", str(replacement))
        return result
    if isinstance(value, dict):
        return {k: _replace_strings_once(v, replacements) for k, v in value.items()}
    if isinstance(value, list):
        return [_replace_strings_once(v, replacements) for v in value]
    return value


def _replace_strings(value: Any, replacements: dict[str, str]) -> Any:
    """Apply Neuro-SAN commondefs string replacements to fixed point."""
    current = value
    for _ in range(20):
        updated = _replace_strings_once(current, replacements)
        if updated == current:
            return updated
        current = updated
    return current


def _replace_values_once(value: Any, replacements: dict[str, Any]) -> Any:
    if isinstance(value, str) and value in replacements:
        return copy.deepcopy(replacements[value])
    if isinstance(value, dict):
        return {k: _replace_values_once(v, replacements) for k, v in value.items()}
    if isinstance(value, list):
        return [_replace_values_once(v, replacements) for v in value]
    return value


def _expand_commondefs(parsed: dict[str, Any]) -> dict[str, Any]:
    commondefs = parsed.get("commondefs", {})
    if not isinstance(commondefs, dict):
        return parsed

    replacement_strings = commondefs.get("replacement_strings", {})
    if not isinstance(replacement_strings, dict):
        replacement_strings = {}
    replacement_strings = {
        str(k): str(v) for k, v in replacement_strings.items() if v is not None
    }

    replacement_values = commondefs.get("replacement_values", {})
    if not isinstance(replacement_values, dict):
        replacement_values = {}
    replacement_values = {
        str(k): _replace_strings(v, replacement_strings)
        for k, v in replacement_values.items()
    }

    expanded = _replace_values_once(parsed, replacement_values)
    return _replace_strings(expanded, replacement_strings)


def load_neuro_san_agent_network(path: str | Path) -> NeuroSanNetworkSpec:
    """Load a Neuro-SAN agent-network HOCON file.

    Only local agent-to-agent references in each agent's ``tools`` list become
    graph edges. External Neuro-SAN agents, URLs, MCP dictionaries, and other
    non-local tool references are preserved as metadata for prompts, but are not
    executable TerraLingua graph nodes.
    """
    try:
        from pyhocon import ConfigFactory
    except ImportError as exc:
        raise ImportError(
            "Reading Neuro-SAN HOCON files requires pyhocon. "
            "Install project requirements before using --agent_network_hocon_path."
        ) from exc

    hocon_path = Path(path).expanduser()
    if not hocon_path.exists():
        raise FileNotFoundError(f"Neuro-SAN HOCON file not found: {hocon_path}")

    parsed = _expand_commondefs(_to_plain(_parse_hocon_file(ConfigFactory, hocon_path)))
    tools = parsed.get("tools")
    if not isinstance(tools, list) or not tools:
        raise ValueError(
            f"{hocon_path} must define a non-empty top-level 'tools' list."
        )

    raw_agents = []
    names = []
    for idx, entry in enumerate(tools):
        if not isinstance(entry, dict):
            raise ValueError(
                f"{hocon_path}: tools[{idx}] must be an object, "
                f"got {type(entry).__name__}."
            )
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{hocon_path}: tools[{idx}] is missing a string name.")
        name = name.strip()
        if name in names:
            raise ValueError(f"{hocon_path}: duplicate agent/tool name '{name}'.")
        names.append(name)
        raw_agents.append((name, entry))

    local_names = set(names)
    agent_specs = []
    for name, entry in raw_agents:
        local_tools = []
        external_tools = []
        for tool_ref in entry.get("tools", []) or []:
            tool_name = _stringify_tool_ref(tool_ref).strip()
            if not tool_name:
                continue
            if tool_name in local_names:
                local_tools.append(tool_name)
            else:
                external_tools.append(tool_name)

        agent_specs.append(
            NeuroSanAgentSpec(
                name=name,
                instructions=str(entry.get("instructions", "") or "").strip(),
                description=_function_description(entry),
                downstream_agents=tuple(dict.fromkeys(local_tools)),
                external_tools=tuple(dict.fromkeys(external_tools)),
                command=(
                    str(entry["command"]).strip()
                    if entry.get("command") is not None
                    else None
                ),
                display_as=(
                    str(entry["display_as"]).strip()
                    if entry.get("display_as") is not None
                    else None
                ),
                llm_config=entry.get("llm_config", {})
                if isinstance(entry.get("llm_config", {}), dict)
                else {},
                raw=entry,
            )
        )

    metadata = parsed.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    description = metadata.get("description", "")
    if description is None:
        description = ""

    llm_config = parsed.get("llm_config", {})
    if not isinstance(llm_config, dict):
        llm_config = {}

    return NeuroSanNetworkSpec(
        path=hocon_path,
        agents=tuple(agent_specs),
        frontman=agent_specs[0].name,
        description=str(description).strip(),
        metadata=metadata,
        llm_config=llm_config,
    )
