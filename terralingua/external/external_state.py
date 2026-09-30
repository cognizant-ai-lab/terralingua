"""External-server shared-state handlers.

A state handler maintains cross-agent state derived from one external MCP
server's responses and surfaces it back into every agent's per-step ``info``
dict — so any agent can pick up where another left off. The runner builds one
handler per server that opts in via the ``state`` field of its servers.json
entry, then drives it each timestep through ``observe`` (fold in new responses),
``inject`` (write the summary into agent infos), and ``note_step`` (post-step
bookkeeping). Concrete handlers register under a ``kind`` string via
:func:`register_state_handler` when their module is imported. A scenario
module imports its handlers, and the runner imports the scenario module before
it reads servers.json, so the ``state`` values validate.
"""


class ExternalStateHandler:
    """Base class for per-server shared-state handlers.

    Subclasses set ``kind`` (the servers.json selector) and ``info_key`` (the
    agent-info key they populate), and override :meth:`observe`,
    :meth:`summary`, and optionally :meth:`note_step`.
    """

    kind: str = ""
    info_key: str = ""

    def observe(self, messages: list[str]) -> None:
        """Fold a batch of incoming external responses into handler state."""

    def summary(self) -> str:
        """Return the state summary to inject (empty string to inject nothing)."""
        return ""

    def inject(self, infos: dict) -> None:
        """Write :meth:`summary` under :attr:`info_key` into every agent's info."""
        if not self.info_key:
            return
        summary = self.summary()
        if not summary:
            return
        for agent_tag in infos:
            infos[agent_tag][self.info_key] = summary

    def note_step(self, active_topics: set) -> None:
        """Post-step bookkeeping given the topics acted on this step (no-op by default)."""

    def capture_call_context(self, tool: str, params: dict) -> dict:
        """Snapshot available state before dispatch, without taking another action.

        The runner can add ``concurrent_calls`` when several requests share this
        state. Adapters must not attribute another request's effects to this one.
        """
        return {}

    def execution_context(self) -> dict:
        """Return known environment identities for temporary guidance lifetimes."""
        return {}

    def assess_effect(self, tool: str, params: dict, context: dict, responses: list[str]) -> dict:
        """Report observed effect separately from delivery/acceptance.

        No generic inference from a successful response to an intended effect
        exists. Domain adapters may return pending, confirmed, not_observed or
        failed with explicit evidence; incomplete observations stay unknown.
        """
        return {"effect_status": "unknown", "effect_evidence": {}}


# ── handler registry ──────────────────────────────────────────────────────────

_HANDLER_REGISTRY: dict[str, type[ExternalStateHandler]] = {}


def register_state_handler(
    cls: type[ExternalStateHandler],
) -> type[ExternalStateHandler]:
    """Class decorator registering *cls* under its ``kind``."""
    if not cls.kind:
        raise ValueError(f"{cls.__name__} must set a non-empty 'kind' to register.")
    _HANDLER_REGISTRY[cls.kind] = cls
    return cls


def state_handler_kinds() -> tuple[str, ...]:
    """Return the registered handler kinds (valid servers.json ``state`` values)."""
    return tuple(_HANDLER_REGISTRY)


def make_state_handler(kind: str) -> ExternalStateHandler:
    """Instantiate the handler registered under *kind*."""
    try:
        cls = _HANDLER_REGISTRY[kind]
    except KeyError:
        raise ValueError(
            f"Unknown external state handler {kind!r}; "
            f"known kinds: {sorted(_HANDLER_REGISTRY)}"
        )
    return cls()

