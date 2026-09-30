"""Standard MCP server metadata reaches agents through TL's generic client.

A live FastMCP server runs in a thread. It sends instructions at connection
and publishes one context resource per agent. TL subscribes to each agent's
resource and re-reads it only after the server sends
notifications/resources/updated.
"""

import asyncio
import socket
import threading
import time
from types import SimpleNamespace
from urllib.parse import quote, unquote

import pytest
from mcp import types
from mcp.server.fastmcp import FastMCP
from pydantic import AnyUrl

from terralingua.agents.prompt_templates import render_system_prompt
from terralingua.experiment.runner import SimulationRunner, render_server_instructions
from terralingua.external.request_manager import ExternalRequestManager


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve(service: FastMCP) -> str:
    threading.Thread(target=lambda: service.run(transport="streamable-http"), daemon=True).start()
    deadline = time.time() + 10
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", service.settings.port)) == 0:
                return f"http://127.0.0.1:{service.settings.port}/mcp"
        time.sleep(0.1)
    raise TimeoutError("test server did not start")


def _notes_server(subscribable: bool = True):
    """A server that keeps one note per agent and announces changed notes."""
    service = FastMCP("notes", instructions="Write short notes.", host="127.0.0.1", port=_free_port())
    low = service._mcp_server
    state = SimpleNamespace(notes={}, reads=[], subscribers={})

    @service.resource("notes://{agent_tag}")
    def note(agent_tag: str) -> str:
        tag = unquote(agent_tag)
        state.reads.append(tag)
        return state.notes.get(tag, "")

    @service.tool()
    async def write_note(text: str, agent_tag: str = "") -> str:
        """Replace the caller's note."""
        state.notes[agent_tag] = text
        request = low.request_context
        uri = f"notes://{quote(agent_tag, safe='')}"
        for session in state.subscribers.get(uri, set()):
            await session.send_notification(
                types.ServerNotification(types.ResourceUpdatedNotification(
                    params=types.ResourceUpdatedNotificationParams(uri=AnyUrl(uri)))),
                related_request_id=request.request_id if session is request.session else None,
            )
        return "saved"

    if subscribable:
        @low.subscribe_resource()
        async def subscribe(uri: AnyUrl) -> None:
            state.subscribers.setdefault(str(uri), set()).add(low.request_context.session)

        @low.unsubscribe_resource()
        async def unsubscribe(uri: AnyUrl) -> None:
            state.subscribers.get(str(uri), set()).discard(low.request_context.session)

    return service, state


@pytest.fixture
def notes():
    service, state = _notes_server()
    manager = ExternalRequestManager(server_url=_serve(service), listen_port=_free_port(), request_timeout_s=15)
    state.notes.update({"alice": "Alice's note", "big bob": "Bob's note"})
    yield SimpleNamespace(manager=manager, state=state)
    manager.close()


def test_instructions_reach_the_system_prompt(notes):
    assert notes.manager.instructions == "Write short notes."
    text = render_server_instructions({"notes": notes.manager.instructions, "silent": ""})
    assert text == (
        "=== Instructions from the notes service ===\n"
        "Its actions are the ones whose names start with notes_.\n"
        "Write short notes."
    )
    prompt = render_system_prompt(
        "social_graph.j2", agent_name="Alice", max_message_length=100,
        scenario_specific_instructions="", genome_string="", server_instructions=text,
    )
    assert text in prompt


def test_context_is_read_at_join_and_after_update_notices_only(notes):
    manager, state = notes.manager, notes.state
    assert manager.context_templates == ("notes://{agent_tag}",)
    assert manager.agent_contexts(["alice", "big bob"]) == {"alice": "Alice's note", "big bob": "Bob's note"}
    assert sorted(state.reads) == ["alice", "big bob"]
    manager.agent_contexts(["alice", "big bob"])
    assert len(state.reads) == 2  # no reads without a notice

    manager.send_outgoing_messages({"big bob": {"text": "Bob's new note"}}, tool_name="write_note")
    assert manager.agent_contexts(["alice", "big bob"])["big bob"] == "Bob's new note"
    assert state.reads[2:] == ["big bob"]


def test_departed_agents_are_unsubscribed_and_empty_text_is_omitted_by_the_runner(notes):
    manager, state = notes.manager, notes.state
    manager.agent_contexts(["alice", "big bob"])
    manager.agent_contexts(["alice"])
    assert not state.subscribers.get("notes://big%20bob")
    manager.send_outgoing_messages({"big bob": {"text": "unseen"}}, tool_name="write_note")
    assert "big bob" not in state.reads[2:]

    runner = SimulationRunner.__new__(SimulationRunner)
    runner.external_managers = {"notes": manager}
    runner.agents = {"alice": object(), "carol": object()}
    runner.obs = {"alice": {}, "carol": {}}
    runner.infos = {"alice": {}, "carol": {"external_context": {"notes": "stale"}}}
    asyncio.run(runner._refresh_external_context())
    assert runner.infos == {"alice": {"external_context": {"notes": "Alice's note"}}, "carol": {}}


def test_server_without_subscriptions_is_read_once():
    service, state = _notes_server(subscribable=False)
    manager = ExternalRequestManager(server_url=_serve(service), listen_port=_free_port(), request_timeout_s=15)
    try:
        state.notes["alice"] = "first"
        assert manager.agent_contexts(["alice"]) == {"alice": "first"}
        manager.send_outgoing_messages({"alice": {"text": "second"}}, tool_name="write_note")
        assert manager.agent_contexts(["alice"]) == {"alice": "first"}
        assert state.reads == ["alice"]
    finally:
        manager.close()


def test_server_without_context_templates_publishes_nothing():
    service = FastMCP("plain", host="127.0.0.1", port=_free_port())

    @service.tool()
    def echo(text: str) -> str:
        return text

    manager = ExternalRequestManager(server_url=_serve(service), listen_port=_free_port(), request_timeout_s=15)
    try:
        assert manager.instructions == ""
        assert manager.context_templates == ()
        assert manager.agent_contexts(["alice"]) == {}
    finally:
        manager.close()
