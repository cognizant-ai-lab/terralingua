"""Integration test for the generalized MCP bridge (M3).

Spins up a tiny three-tool FastMCP server in a background thread, connects an
ExternalRequestManager to it, and verifies tool discovery, calling a tool
*by name* with agent_tag auto-injected, and that close() waits for queued
deliveries.
"""

import asyncio
import socket
import threading
import time

import pytest
from mcp.server.fastmcp import FastMCP

from terralingua.external.request_manager import ExternalRequestManager


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_for_port(port: int, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise TimeoutError(f"server on port {port} did not come up")


@pytest.fixture(scope="module")
def mcp_server():
    port = _free_port()
    mcp = FastMCP("Test Bridge Server", host="127.0.0.1", port=port)

    def greet(name: str, agent_tag: str | None = None) -> str:
        return f"world:{name}:{agent_tag}"

    def echo_tool(text: str, agent_tag: str | None = None) -> str:
        return f"echo:{text}:{agent_tag}"

    async def slow_echo(text: str, agent_tag: str | None = None) -> str:
        await asyncio.sleep(0.3)
        return f"slow:{text}:{agent_tag}"

    mcp.tool()(greet)
    mcp.tool()(echo_tool)
    mcp.tool()(slow_echo)

    thread = threading.Thread(
        target=lambda: mcp.run(transport="streamable-http"), daemon=True
    )
    thread.start()
    _wait_for_port(port)
    yield f"http://127.0.0.1:{port}/mcp"


@pytest.fixture
def manager(mcp_server):
    mgr = ExternalRequestManager(
        server_url=mcp_server, listen_port=_free_port(), request_timeout_s=15
    )
    yield mgr
    mgr.close()


def test_discovers_all_tools(manager):
    names = {t.name for t in manager.list_tools()}
    assert {"greet", "echo_tool"} <= names


def test_default_tool_call(manager):
    # No tool_name -> the manager's primary (first discovered) tool; structured
    # args passed natively, agent_tag auto-injected from the schema.
    statuses = manager.send_outgoing_messages({"a0": {"name": "hello"}})
    assert statuses["a0"][0]["status"] == "sent and received response"
    msgs = manager.get_incoming_messages("a0")
    assert msgs == ["world:hello:a0"]


def test_named_tool_call(manager):
    manager.send_outgoing_messages({"a1": {"text": "ping"}}, tool_name="echo_tool")
    msgs = manager.get_incoming_messages("a1")
    assert msgs == ["echo:ping:a1"]


def test_close_waits_for_queued_deliveries_and_returns_quickly(mcp_server):
    mgr = ExternalRequestManager(
        server_url=mcp_server, listen_port=_free_port(), request_timeout_s=15, sync_send=False
    )
    statuses = mgr.send_outgoing_messages({"a2": {"text": "late"}}, tool_name="slow_echo")
    assert statuses["a2"][0]["status"] == "queued for delivery"
    started = time.time()
    mgr.close()
    assert time.time() - started < 5
    assert mgr.get_incoming_messages("a2") == ["slow:late:a2"]
