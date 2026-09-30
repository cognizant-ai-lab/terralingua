import argparse
import asyncio
import json
import logging
import re
import sys
import threading
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pprint import pprint
from typing import Any, Deque, Dict, List, Literal, Optional, Tuple
from urllib.parse import quote

log = logging.getLogger(__name__)

from flask import Flask, jsonify
from flask import request as flask_request
from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client
from pydantic import AnyUrl, BaseModel, ValidationError

# ---------------------------
# Envelope (JSON on the wire)
# ---------------------------

MessageType = Literal["agent_request", "agent_response", "event", "ack", "error"]

# Seconds to wait for the initial MCP session handshake. If the server doesn't
# complete it in time, construction fails fast with a clear error instead of
# hanging forever on tool discovery.
CONNECT_TIMEOUT_S = 10.0

# A resource template with exactly this variable publishes one context text per
# agent. TL subscribes to each agent's URI and re-reads it when the server sends
# notifications/resources/updated.
AGENT_CONTEXT_VARIABLE = "agent_tag"

# Tool a batch-mode server (servers.json `batch: true`) must expose. It receives
# every agent's request for the timestep in one call, so the call itself marks
# the end of the round. It is never offered to agents as an action.
BATCH_TOOL_NAME = "batch_round"


class MessageSource(BaseModel):
    agent_tag: Optional[str] = None


class MessageTarget(BaseModel):
    agent_tag: Optional[str] = None  # None => broadcast


class MessageEnvelope(BaseModel):
    """
    Wire format for incoming messages.

    Notes:
      - payload is plain text
      - servers POST these to the listener's root path (http://runner_host:callback_port/)
    """

    v: int = 1
    type: MessageType
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    source: MessageSource
    target: Optional[MessageTarget] = None
    payload: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeliveryStatus:
    message_id: str
    status: str
    timestamp: str = field(
        default_factory=lambda: (
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )
    )


# ---------------------------
# MCP client / bridge
# ---------------------------


class ExternalRequestManager:
    """
    MCP protocol bridge using native mcp.ClientSession.

    Agent-facing API:
      - send_outgoing_messages({agent_tag: "plain text"}) ->
          {agent_tag: [{"message_id": <uuid>, "status": <status_line>}]}
        Sends messages via MCP tools and returns delivery status.

      - get_incoming_messages(agent_tag) -> List[str]
      - get_broadcast_messages() -> List[str]

    Uses native MCP ClientSession with streamable_http transport.
    """

    # Standard MCP server metadata, read at connection time.
    instructions: str = ""
    context_templates: Tuple[str, ...] = ()

    def __init__(
        self,
        server_url: str,
        listen_port: int,
        max_workers: int = 10,
        verbose: bool = False,
        request_timeout_s: float | None = None,
        tool_name: str | None = None,
        sync_send: bool = True,
    ):
        self.server_url = server_url.rstrip("/")
        self.listen_port = listen_port
        self.request_timeout_s = request_timeout_s
        self.tool_name = tool_name
        self.verbose = verbose
        self.sync_send = sync_send

        self._lock = threading.Lock()
        self._agent_queues: Dict[str, Deque[dict]] = defaultdict(deque)
        self._broadcast_queue: Deque[str] = deque()

        # Per-agent context resources; touched only on the event loop thread.
        self._agent_uris: Dict[str, List[str]] = {}
        self._contexts: Dict[str, str] = {}
        self._context_reads: Dict[str, asyncio.Future] = {}
        # Fire-and-forget deliveries still running; touched only on the event loop thread.
        self._deliveries: set[asyncio.Future] = set()

        # Tools discovered on the server (name -> tool object). `tool_info` is the
        # default call target; other tools are reachable by name via
        # send_outgoing_messages(tool_name=...).
        self.tool_info = None
        self.tools: Dict[str, Any] = {}

        # Used to deliver multiple agent requests concurrently
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        # Event loop for async operations
        self._event_loop = None
        self._loop_thread = None
        self._session = None
        self._session_ready = threading.Event()

        # Start async event loop in background thread
        log.debug("[ExternalRequestManager] Starting event loop thread...")
        self._start_event_loop()

        log.debug(
            "[ExternalRequestManager] Starting listener on port %d...", self.listen_port
        )
        self._start_listener()

        # Discover the tools the server exposes (the runner routes each call to a
        # specific tool by name).
        self.tool_ok = False
        if not self.discover_tools():
            raise RuntimeError(
                f"[ExternalRequestManager] No tools available on MCP server at "
                f"{self.server_url}"
            )
        self.tool_ok = True
        log.info(
            "[ExternalRequestManager] Ready: tools %s on MCP server at %s",
            list(self.tools),
            self.server_url,
        )

    def _start_event_loop(self):
        """Start event loop in a background thread."""

        def run_loop():
            # Create a brand new event loop for this thread
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)

            # Initialize MCP session
            self._event_loop.run_until_complete(self._init_session())

            # Keep the loop running forever to handle future async tasks
            self._event_loop.run_forever()

        # Start the loop in a background thread
        self._loop_thread = threading.Thread(target=run_loop, daemon=True)
        self._loop_thread.start()

        # Fail fast if the server never completes the session handshake, rather than
        # proceeding with a half-open session (which leaves the loop wedged inside
        # _init_session, so tool discovery would block forever).
        if not self._session_ready.wait(timeout=CONNECT_TIMEOUT_S):
            raise RuntimeError(
                f"[ExternalRequestManager] Timed out after {CONNECT_TIMEOUT_S:g}s "
                f"establishing an MCP session with {self.server_url}. Is the server "
                f"running and reachable at that URL?"
            )

    async def _init_session(self):
        """Initialize MCP session."""
        try:
            # Both context managers stay referenced for the loop's lifetime; dropping
            # them would let garbage collection finalize the transports mid-run.
            self._client_context = streamable_http_client(self.server_url)
            read_stream, write_stream, _ = await self._client_context.__aenter__()

            # Create session
            self._session_context = ClientSession(
                read_stream, write_stream, message_handler=self._on_server_message
            )
            self._session = await self._session_context.__aenter__()

            # Initialize connection
            init = await self._session.initialize()
            self.instructions = init.instructions or ""
            if init.capabilities.resources is not None:
                try:
                    templates = (await self._session.list_resource_templates()).resourceTemplates
                except Exception as e:
                    log.warning("[ExternalRequestManager] Listing resource templates failed: %s", e)
                    templates = []
                self.context_templates = tuple(
                    t.uriTemplate for t in templates
                    if re.findall(r"{([^}]*)}", t.uriTemplate) == [AGENT_CONTEXT_VARIABLE]
                )

            log.debug(
                "[ExternalRequestManager] Connected to MCP server at %s",
                self.server_url,
            )
            self._session_ready.set()

        except Exception as e:
            raise RuntimeError(
                f"[ExternalRequestManager] Failed to initialize MCP session: {e}"
            )

    def _run_in_event_loop(self, coro):
        """Run a coroutine in the background event loop and wait for result."""
        if not self._event_loop:
            raise RuntimeError("Event loop not started")

        future = asyncio.run_coroutine_threadsafe(coro, self._event_loop)
        return future.result(timeout=self.request_timeout_s)

    # ---------------------------
    # Agent-facing API
    # ---------------------------

    def send_outgoing_messages(
        self, requests: Dict[str, dict], tool_name: str | None = None, *, before_send=None
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Call an MCP tool for each agent with that agent's structured arguments.

        When sync_send=True (default): delivers one call at a time by running
        _deliver_message_async in the event loop and blocking until each response
        is received before moving to the next. Prevents flooding the MCP server.

        When sync_send=False: fire-and-forget — schedules each delivery in the
        event loop without waiting and returns immediately with "queued" statuses.

        Args:
            requests: {agent_tag: {param: value, ...}} — the tool's native args;
                `agent_tag` is auto-injected when the tool declares it.
            tool_name: which tool to call (defaults to this manager's primary tool).

        Returns:
            {agent_tag: [{"message_id": <uuid>, "status": <status>}]}
        """
        message_status_queue: Dict[str, List[dict]] = defaultdict(list)

        for agent_tag, args in requests.items():
            msg_id = str(uuid.uuid4())
            if before_send is not None:
                before_send([{"agent": agent_tag, "tool": tool_name or self.tool_name,
                              "params": args}], msg_id)
            if self.sync_send:
                # Block until the MCP tool responds
                try:
                    _, status_line = self._run_in_event_loop(
                        self._deliver_message_async(agent_tag, args, msg_id, tool_name)
                    )
                except Exception as e:
                    status_line = f"failed with error {e}"
            else:
                self._schedule_delivery(
                    self._deliver_message_async(agent_tag, args, msg_id, tool_name)
                )
                status_line = "queued for delivery"
            message_status_queue[agent_tag].append(
                DeliveryStatus(message_id=msg_id, status=status_line).__dict__
            )

        return message_status_queue

    def _schedule_delivery(self, coro) -> None:
        """Start *coro* on the event loop and track it until it finishes."""
        if not self._event_loop:
            raise RuntimeError("Event loop not started")

        def start():
            task = asyncio.ensure_future(coro)
            self._deliveries.add(task)
            task.add_done_callback(self._deliveries.discard)

        self._event_loop.call_soon_threadsafe(start)

    def send_outgoing_batch(
        self, requests: Dict[str, dict], tools: Dict[str, str], *, before_send=None
    ) -> Dict[str, List[Dict[str, str]]]:
        """Send every agent's request for this timestep as ONE ``batch_round`` call.

        The single call is the round boundary: the server knows no more actions
        are coming and can settle the round (e.g. advance a stepped game once).
        Blocks until the server replies, regardless of ``sync_send``.

        Args:
            requests: {agent_tag: args} — each agent's native tool arguments.
            tools: {agent_tag: tool_name} — which tool each agent invoked.

        The call carries ``{"entries": [{"agent_tag", "tool", "args"}, ...]}``.
        The server must reply with JSON ``{"results": {agent_tag: result, ...}}``;
        each result is queued for its agent (re-serialized to a JSON string when
        it is not already one). An agent missing from the reply receives nothing.

        Returns:
            {agent_tag: [{"message_id": <uuid>, "status": <status>}]}
        """
        entries = [
            {"agent_tag": tag, "tool": tools.get(tag, ""), "args": args}
            for tag, args in requests.items()
        ]
        msg_id = str(uuid.uuid4())
        if before_send is not None:
            before_send([{"agent": e["agent_tag"], "tool": e["tool"], "params": e["args"]}
                         for e in entries], msg_id)
        try:
            per_agent = self._run_in_event_loop(
                self._deliver_batch_async(entries, message_id=msg_id)
            )
            statuses = {
                tag: "sent and received response"
                if tag in per_agent
                else "sent (no response for this agent)"
                for tag in requests
            }
        except Exception as e:
            statuses = {tag: f"failed with error {e}" for tag in requests}

        return {
            tag: [DeliveryStatus(message_id=msg_id, status=status).__dict__]
            for tag, status in statuses.items()
        }

    async def _deliver_batch_async(
        self, entries: List[dict], message_id: str = ""
    ) -> Dict[str, Any]:
        """Call ``batch_round`` once, split the reply, and queue per-agent results.

        Returns the per-agent results that were queued, keyed by agent_tag.
        Raises on transport errors or a reply that is not the documented shape,
        so the caller can mark every agent's delivery as failed.
        """
        if not self._session:
            raise RuntimeError("MCP session not initialized")
        if BATCH_TOOL_NAME not in self.tools:
            raise RuntimeError(
                f"server exposes no '{BATCH_TOOL_NAME}' tool; required for batch mode"
            )

        result = await self._session.call_tool(
            BATCH_TOOL_NAME, arguments={"entries": entries}
        )
        response_text = self._result_text(result)
        if getattr(result, "isError", False):
            for entry in entries:
                self._push_agent(
                    entry["agent_tag"], response_text, message_id=message_id,
                    tool_error=True,
                )
            raise ValueError(f"MCP tool returned an error: {response_text}")

        reply = json.loads(response_text)
        per_agent = reply.get("results") if isinstance(reply, dict) else None
        if not isinstance(per_agent, dict):
            raise ValueError(
                f"'{BATCH_TOOL_NAME}' reply must contain a 'results' object keyed "
                f"by agent_tag, got: {type(per_agent).__name__}"
            )
        requested_tags = {entry["agent_tag"] for entry in entries}
        for tag, agent_result in per_agent.items():
            text = (
                agent_result
                if isinstance(agent_result, str)
                else json.dumps(agent_result)
            )
            self._push_agent(
                tag, text.strip(),
                message_id=message_id if tag in requested_tags else "",
            )
        return per_agent

    def agent_contexts(self, agent_tags: List[str]) -> Dict[str, str]:
        """Return the context text this server publishes for each agent.

        New agents are subscribed and read once; departed agents are
        unsubscribed. Reads started by update notifications finish first.
        """
        if not self.context_templates:
            return {}
        return self._run_in_event_loop(self._agent_contexts(list(agent_tags)))

    async def _agent_contexts(self, agent_tags: List[str]) -> Dict[str, str]:
        for tag in set(self._agent_uris) - set(agent_tags):
            for uri in self._agent_uris.pop(tag):
                self._contexts.pop(uri, None)
                self._context_reads.pop(uri, None)
                try:
                    await self._session.unsubscribe_resource(AnyUrl(uri))
                except Exception as e:
                    log.debug("[ExternalRequestManager] Unsubscribe %s failed: %s", uri, e)
        for tag in agent_tags:
            if tag in self._agent_uris:
                continue
            uris = [t.replace("{%s}" % AGENT_CONTEXT_VARIABLE, quote(tag, safe=""))
                    for t in self.context_templates]
            self._agent_uris[tag] = uris
            for uri in uris:
                # Subscribe before the first read, so no change is missed.
                # FastMCP servers do not advertise subscribe support, so try it.
                try:
                    await self._session.subscribe_resource(AnyUrl(uri))
                except Exception as e:
                    log.warning("[ExternalRequestManager] %s is read once: subscribe failed: %s", uri, e)
                self._schedule_context_read(uri)
        pending = [read for read in self._context_reads.values() if not read.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        return {
            tag: "\n\n".join(self._contexts[uri] for uri in self._agent_uris[tag] if self._contexts.get(uri))
            for tag in agent_tags
        }

    async def _on_server_message(self, message) -> None:
        if (isinstance(message, types.ServerNotification)
                and isinstance(message.root, types.ResourceUpdatedNotification)):
            self._schedule_context_read(str(message.root.params.uri))

    def _schedule_context_read(self, uri: str) -> None:
        """Read *uri* after any earlier read of it, so the newest text is kept."""
        if not any(uri in uris for uris in self._agent_uris.values()):
            return
        previous = self._context_reads.get(uri)

        async def read():
            if previous is not None:
                await asyncio.gather(previous, return_exceptions=True)
            try:
                result = await self._session.read_resource(AnyUrl(uri))
                text = "\n".join(c.text for c in result.contents if isinstance(c, types.TextResourceContents))
            except Exception as e:
                log.warning("[ExternalRequestManager] Reading %s failed: %s", uri, e)
                text = ""
            if any(uri in uris for uris in self._agent_uris.values()):
                self._contexts[uri] = text

        self._context_reads[uri] = asyncio.ensure_future(read())

    def get_incoming_messages(self, agent_tag: str) -> List[str]:
        """Pop and return all queued plain-text messages for this agent."""
        return [rec["text"] for rec in self.get_incoming_records(agent_tag) if rec["text"]]

    def get_incoming_records(self, agent_tag: str) -> List[dict]:
        """Pop replies with their request IDs and transport error flags.

        An empty request ID means an unsolicited or uncorrelated observation.
        Empty replies still carry completion metadata. They provide no outcome evidence.
        This and get_incoming_messages drain the same queue; call only one.
        """
        with self._lock:
            q = self._agent_queues.get(agent_tag)
            if not q:
                return []
            msgs = list(q)
            q.clear()
            return [dict(msg) for msg in msgs]

    def get_broadcast_messages(self) -> List[str]:
        """Pop and return all queued broadcast plain-text messages."""
        with self._lock:
            if not self._broadcast_queue:
                return []
            msgs = list(self._broadcast_queue)
            self._broadcast_queue.clear()
            return msgs

    # ---------------------------
    # MCP delivery
    # ---------------------------

    async def _deliver_message_async(
        self,
        agent_tag: str,
        args: dict,
        msg_id: str,
        tool_name: str | None = None,
    ) -> Tuple[bool, str]:
        """
        Call an MCP tool natively with the agent's structured ``args``.
        ``tool_name`` selects which tool to call (defaults to this manager's
        primary tool); ``agent_tag`` is injected when the tool declares it.
        """
        if not self._session:
            self._push_agent(agent_tag, "MCP session not initialized",
                             message_id=msg_id, delivery_failed=True)
            return False, "MCP session not initialized"

        target_tool = tool_name or (self.tool_info.name if self.tool_info else None)
        if not target_tool:
            self._push_agent(agent_tag, "no target tool on MCP server",
                             message_id=msg_id, delivery_failed=True)
            return False, "no target tool on MCP server"
        arguments = dict(args)
        # Inject the caller identity only if the tool's schema declares it.
        tool = self.tools.get(target_tool)
        schema = getattr(tool, "inputSchema", None) or {}
        if "agent_tag" in schema.get("properties", {}):
            arguments["agent_tag"] = agent_tag
        try:
            result = await self._session.call_tool(
                target_tool,
                arguments=arguments,
            )

            response_text = self._result_text(result)
            tool_error = bool(getattr(result, "isError", False))

            # Keep the originating request ID even if another call finished first.
            self._push_agent(
                agent_tag, response_text, message_id=msg_id, tool_error=tool_error,
                structured_content=getattr(result, "structuredContent", None),
            )
            if tool_error:
                return False, "sent and received tool error"
            if response_text:
                return True, "sent and received response"

            return True, "sent (no response content)"

        except Exception as e:
            self._push_agent(agent_tag, str(e), message_id=msg_id, delivery_failed=True)
            return False, f"failed with error {str(e)}"

    @staticmethod
    def _result_text(result: Any) -> str:
        """Keep tool text, or structured content when text is absent."""
        text = "".join(
            item.text if hasattr(item, "text") else str(item)
            for item in (getattr(result, "content", None) or [])
        ).strip()
        structured = getattr(result, "structuredContent", None)
        if not text and structured is not None:
            return json.dumps(structured)
        return text

    # ---------------------------
    # Incoming listener (for external responses)
    # ---------------------------

    def _start_listener(self) -> None:
        app = Flask(__name__)

        @app.get("/health")
        def health():
            return jsonify({"status": "ok"}), 200

        @app.post("/")
        def receive():
            data = flask_request.get_json(silent=True)
            if not data:
                return (
                    jsonify(
                        {
                            "status": "error",
                            "code": "BAD_JSON",
                            "message": "Expected JSON envelope body",
                        }
                    ),
                    400,
                )

            try:
                env = MessageEnvelope.model_validate(data)
            except ValidationError as ve:
                return (
                    jsonify(
                        {
                            "status": "error",
                            "code": "VALIDATION_ERROR",
                            "message": str(ve),
                        }
                    ),
                    400,
                )

            self._queue_callback(env)

            return jsonify({"status": "ok", "received_id": env.id}), 200

        def run():
            app.run(
                host="0.0.0.0",
                port=self.listen_port,
                debug=False,
                use_reloader=False,
            )

        t = threading.Thread(target=run, daemon=True)
        t.start()
        log.debug(
            "[ExternalRequestManager] Listener started on port %d", self.listen_port
        )

    # ---------------------------
    # Queue helpers
    # ---------------------------

    def _queue_callback(self, envelope: MessageEnvelope) -> None:
        """Keep callback identity and error flags even when its payload is empty."""
        target = envelope.target.agent_tag if envelope.target else None
        payload = (envelope.payload or "").strip()
        if target:
            request_id = envelope.meta.get("request_id", "")
            self._push_agent(
                target, payload,
                message_id=request_id if isinstance(request_id, str) else "",
                tool_error=envelope.type == "error",
                response_complete=envelope.type in ("agent_response", "error"),
            )
        elif payload:
            self._push_broadcast(payload)

    def set_response_observer(self, observer) -> None:
        """Observe replies before queueing, without consuming them.

        The observer is host instrumentation, never an agent callback. A
        recording failure must not turn an executed tool call into a transport
        failure or cause the world action to be sent again.
        """
        self._response_observer = observer

    def _push_agent(
        self, agent_tag: str, text: str, *, message_id: str = "",
        tool_error: bool = False, delivery_failed: bool = False,
        structured_content: Any = None,
        response_complete: bool = True,
    ) -> None:
        record = {
            "text": text, "message_id": message_id, "tool_error": tool_error,
            "delivery_failed": delivery_failed,
        }
        if structured_content is not None:
            record["structured_content"] = structured_content
        if not response_complete:
            # ACKs and events can precede the actual correlated result. Their
            # payload (including an empty one) does not establish completion.
            record["response_complete"] = False
        observer = getattr(self, "_response_observer", None)
        if observer is not None:
            try:
                observer(agent_tag, record)
            except Exception:
                log.exception("Could not record completed external response for %s", agent_tag)
        with self._lock:
            self._agent_queues[agent_tag].append(record)

    def _push_broadcast(self, text: str) -> None:
        with self._lock:
            self._broadcast_queue.append(text)

    # ---------------------------
    # Utility methods
    # ---------------------------

    def discover_tools(self) -> bool:
        """Cache the tools the server exposes (name → tool). Returns False if the
        server exposes none. ``tool_info`` is the primary tool (``tool_name`` if
        present, else the first discovered) used as the default call target."""

        async def _list():
            if not self._session:
                return []
            tools_result = await self._session.list_tools()
            return tools_result.tools

        tools = self._run_in_event_loop(_list())
        if not tools:
            log.debug("[ExternalRequestManager] No tools available on MCP server")
            return False
        self.tools = {t.name: t for t in tools}
        primary = self.tools.get(self.tool_name) if self.tool_name else None
        self.tool_info = primary or next(iter(self.tools.values()))
        return True

    def list_tools(self) -> List[Any]:
        """Return the tool objects discovered on this server (name, description,
        inputSchema), as cached at connection time."""
        return list(self.tools.values())

    # ---------------------------
    # Cleanup
    # ---------------------------

    async def _cleanup_session(self):
        """Wait for in-flight deliveries and context reads, then cancel the transport tasks."""
        in_flight = [
            t for t in (*self._deliveries, *self._context_reads.values()) if not t.done()
        ]
        if in_flight:
            await asyncio.gather(*in_flight, return_exceptions=True)

        # The MCP client and session run receive loops that only end when the
        # connection drops, so they are cancelled rather than awaited.
        current = asyncio.current_task()
        transport = [t for t in asyncio.all_tasks() if t is not current and not t.done()]
        for task in transport:
            task.cancel()
        if transport:
            await asyncio.wait(transport, timeout=5)

    def close(self) -> None:
        """Close all resources.

        Waits for deliveries and context reads that are still running, cancels
        the MCP transport tasks, then stops the background event loop and joins
        its thread.
        """
        self._executor.shutdown(wait=False)

        if self._event_loop and self._event_loop.is_running():
            future = asyncio.run_coroutine_threadsafe(
                self._cleanup_session(), self._event_loop
            )
            try:
                future.result(timeout=10)
            except Exception as e:
                log.warning(
                    "[ExternalRequestManager] cleanup did not finish during close: %r", e
                )

            self._event_loop.call_soon_threadsafe(self._event_loop.stop)

        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=5)

        log.debug("[ExternalRequestManager] Closed")


if __name__ == "__main__":
    MCP_URL = "http://localhost:8080/mcp"
    LISTENER_PORT = 9000

    # Create manager
    print("Creating manager...")
    manager = ExternalRequestManager(
        server_url=MCP_URL,
        listen_port=LISTENER_PORT,
        verbose=True,
    )

    outgoing_messages = {
        "agent_1": "what is the amount in the bank account?",
        "agent_2": "What is the richest company?",
    }

    print("Sending messages...")
    status = manager.send_outgoing_messages(outgoing_messages)

    print("\nDelivery status:")
    pprint(status)

    time.sleep(2)
    not_received = []
    for agent_tag in outgoing_messages.keys():
        messages = manager.get_incoming_messages(agent_tag)
        if messages:
            print(f"\n{agent_tag} received:")
            for msg in messages:
                print(f"  {msg}")
        else:
            print(f"\n{agent_tag}: No response received yet")
            not_received.append(agent_tag)

    if not_received:
        print("\nWaiting for delayed responses...")
        time.sleep(5)
        for agent_tag in not_received:
            messages = manager.get_incoming_messages(agent_tag)
            if messages:
                print(f"\n{agent_tag} received delayed response:")
                for msg in messages:
                    print(f"  {msg}")
            else:
                print(f"\n{agent_tag}: Still no response received")
    else:
        print("\nAll responses received immediately.")
