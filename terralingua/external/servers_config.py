import json
import re
from dataclasses import dataclass, field, replace
from typing import List
from urllib.parse import urlparse

from terralingua.external.external_state import state_handler_kinds

# Valid aggregation modes for an external server.
VALID_MODES = ("independent", "unanimous")

# Server names become the prefix of flattened `<server>_<tool>` actions, so they
# must be safe identifiers: lowercase, alphanumeric, starting with a letter.
_NAME_RE = re.compile(r"^[a-z][a-z0-9]*$")


@dataclass
class ServerSpec:
    """One external MCP server declared in a scenario's servers.json."""

    name: str
    url: str
    # Port the runner binds locally to receive this server's pushes (callbacks).
    # The server POSTs to http://runner_host:callback_port/.
    callback_port: int
    cost: int = 0
    mode: str = "independent"
    # Shared-state handler kind registered by the scenario; empty => none.
    state: str = ""
    # For mode "unanimous": the request-arg field whose value groups requests
    # into one election each (e.g. a world slot index). Empty => one global election.
    vote_key: str = ""
    # Top-level keys in this server's responses to strip from what agents see in
    # their prompts, while the runner keeps the full response for logging/replay.
    # E.g. a server may send a "difficulty" field that would bias agents but
    # must survive in external_messages.jsonl for trajectory metadata.
    hidden_keys: List[str] = field(default_factory=list)
    # When True, the runner sends ONE MCP call per timestep to this server's
    # `batch_round` tool, carrying every agent's request as an entry
    # ({agent_tag, tool, args}). The single call marks the end of the round, so
    # the server knows all actions have arrived (e.g. to advance a stepped
    # game exactly once). The server replies with per-agent results keyed by
    # agent_tag, which the runner routes back to each caller.
    batch: bool = False

    def __post_init__(self):
        if not _NAME_RE.match(self.name):
            raise ValueError(
                f"Server name {self.name!r} is invalid: names must be lowercase "
                "alphanumeric and start with a letter (used as an action prefix)."
            )
        if not self.url:
            raise ValueError(f"Server {self.name!r} is missing a 'url'.")
        if not (1 <= self.callback_port <= 65535):
            raise ValueError(
                f"Server {self.name!r} has invalid callback_port {self.callback_port!r}; "
                "must be between 1 and 65535."
            )
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"Server {self.name!r} has invalid mode {self.mode!r}; "
                f"must be one of {VALID_MODES}."
            )
        if self.cost < 0:
            raise ValueError(
                f"Server {self.name!r} has invalid cost {self.cost!r}; must be >= 0."
            )
        if self.state and self.state not in state_handler_kinds():
            raise ValueError(
                f"Server {self.name!r} has invalid state {self.state!r}; "
                f"must be one of {state_handler_kinds()}."
            )
        if not isinstance(self.hidden_keys, list) or not all(
            isinstance(k, str) for k in self.hidden_keys
        ):
            raise ValueError(
                f"Server {self.name!r} has invalid hidden_keys {self.hidden_keys!r}; "
                "must be a list of strings."
            )
        if not isinstance(self.batch, bool):
            raise ValueError(
                f"Server {self.name!r} has invalid batch {self.batch!r}; "
                "must be a boolean."
            )


def load_servers_config(path: str) -> List[ServerSpec]:
    """Load and validate a servers.json file into a list of ServerSpec.

    The file is a JSON array of objects, each with required keys ``name``,
    ``url``, and ``callback_port`` (the port the runner binds to receive this
    server's pushes) and optional ``cost`` (default 0), ``mode`` (default
    ``"independent"``), ``state`` (default ``""`` => no shared-state handler),
    ``vote_key`` (default ``""``; for ``"unanimous"`` mode, the request-arg
    field whose value groups requests into one election each),
    ``hidden_keys`` (default ``[]``; top-level response keys to hide from agent
    prompts while still logging the full response), and ``batch`` (default
    ``False``; send all agents' requests as one ``batch_round`` call per
    timestep instead of one call per agent).
    Raises ValueError on malformed input or duplicate names/callback_ports.
    """
    with open(path) as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError(
            f"servers config {path!r} must be a JSON array of server objects."
        )

    specs: List[ServerSpec] = []
    seen: set[str] = set()
    seen_ports: set[int] = set()
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(
                f"servers config entry {i} in {path!r} must be an object, "
                f"got {type(entry).__name__}."
            )
        if "name" not in entry or "url" not in entry or "callback_port" not in entry:
            raise ValueError(
                f"servers config entry {i} in {path!r} must include 'name', 'url', "
                "and 'callback_port'."
            )
        spec = ServerSpec(
            name=entry["name"],
            url=entry["url"],
            callback_port=entry["callback_port"],
            cost=entry.get("cost", 0),
            mode=entry.get("mode", "independent"),
            state=entry.get("state", ""),
            vote_key=entry.get("vote_key", ""),
            hidden_keys=entry.get("hidden_keys", []),
            batch=entry.get("batch", False),
        )
        if spec.name in seen:
            raise ValueError(
                f"Duplicate server name {spec.name!r} in {path!r}; names must be unique."
            )
        if spec.callback_port in seen_ports:
            raise ValueError(
                f"Duplicate callback_port {spec.callback_port!r} in {path!r}; "
                "each server needs its own callback_port."
            )
        seen.add(spec.name)
        seen_ports.add(spec.callback_port)
        specs.append(spec)

    if not specs:
        raise ValueError(f"servers config {path!r} declares no servers.")

    return specs


def apply_port_overrides(
    specs: List[ServerSpec],
    url_port: int | None = None,
    callback_port: int | None = None,
) -> List[ServerSpec]:
    """Supersede the sole server's URL port and/or callback_port.

    Lets parallel runs each use their own ports without editing servers.json (see
    ``run.external_server_port`` / ``run.external_server_callback_port``). Only the
    port component of the URL is replaced; host and path are preserved. Requires
    exactly one declared server, since the override is unambiguous only then.
    Returns ``specs`` unchanged when neither override is set.
    """
    if url_port is None and callback_port is None:
        return specs
    if len(specs) != 1:
        raise ValueError(
            "external_server_port / external_server_callback_port overrides require "
            f"exactly one declared server, found {len(specs)}."
        )
    spec = specs[0]
    url = spec.url
    if url_port is not None:
        parsed = urlparse(url)
        netloc = f"{parsed.hostname or 'localhost'}:{url_port}"
        url = parsed._replace(netloc=netloc).geturl()
    # replace() re-runs ServerSpec.__post_init__, so an out-of-range override port
    # is validated the same as one read from the file.
    return [
        replace(
            spec,
            url=url,
            callback_port=callback_port if callback_port is not None else spec.callback_port,
        )
    ]
