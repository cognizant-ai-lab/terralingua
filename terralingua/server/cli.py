"""
TerraLingua API server (web UI process).

Connects to Redis to receive simulation frames from the runner process
and to route dashboard commands back to the runner.

Start alongside the runner process, or manually:
  # terminal 1
  terralingua --remote_api_enabled <experiment args>
  # terminal 2
  terralingua-dashboard [--host HOST] [--port PORT] [--workers N]

Environment variables:
  REDIS_URL              Redis connection URL (default: redis://localhost:6379)
  API_HOST               Bind host (default: 0.0.0.0)
  API_PORT               Bind port (default: 8765)
  API_WORKERS            Number of uvicorn worker processes (default: 1).
                         Set >1 to spread server-side LLM fanout across cores.
                         Cross-worker coordination uses Redis (ogw:bg_owner:*,
                         CHANNEL_AGENT_EVENT, CHANNEL_PUSH).
  FORWARDED_ALLOW_IPS    Comma-separated IPs/CIDRs trusted to set X-Forwarded-*
                         headers. Default 127.0.0.1. No-op when uvicorn is
                         directly internet-facing; set to the proxy's IP/CIDR
                         when fronted by a load balancer (k8s ingress, ALB, …).
  SSL_CERTFILE           TLS cert path (enables HTTPS when paired with key).
  SSL_KEYFILE            TLS key path.
  COOKIE_SECURE          Whether auth cookies carry the Secure flag (default:
                         true). Set to "false" for local HTTP development, or
                         pass --dev on the command line.
"""

import argparse
import logging
import os
import secrets
import sys

from dotenv import find_dotenv, load_dotenv

# Load .env and apply early CLI flags BEFORE importing anything from
# terralingua.server, because terralingua.server.user_manager builds the auth backend
# (reading COOKIE_SECURE) at module import time.
load_dotenv(find_dotenv(usecwd=True), override=True)
if "--dev" in sys.argv:
    os.environ["COOKIE_SECURE"] = "false"

import uvicorn  # noqa: E402

from terralingua.server.api import create_app  # noqa: E402
from terralingua.server.dashboard_manager import DashboardManager  # noqa: E402
from terralingua.utils.logging_setup import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


class _ScanFilter(logging.Filter):
    _SCAN_PREFIXES = (
        "/private", "/server.", "/my.key", "/key.pem", "/ssl/", "/id_rsa",
        "/id_dsa", "/.ssh/", "/php", "/PHP", "/wp-", "/admin", "/.env",
        "/.git", "/config.", "/.well-known/",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(p in msg for p in self._SCAN_PREFIXES)


logging.getLogger("uvicorn.access").addFilter(_ScanFilter())


def create_app_from_env():
    """Factory used by uvicorn workers (--workers > 1 path).

    uvicorn cannot fork a pre-instantiated app instance across workers, so
    multi-worker mode imports this factory by string and calls it inside each
    worker process after fork.
    """
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8765"))
    ssl_enabled = bool(os.environ.get("SSL_CERTFILE") and os.environ.get("SSL_KEYFILE"))
    testing = os.environ.get("API_TESTING", "").lower() == "true"
    setup_logging(verbose=1)
    return create_app(
        dashboard_manager=DashboardManager(),
        host=host,
        port=port,
        ssl=ssl_enabled,
        testing=testing,
    )


def main():
    parser = argparse.ArgumentParser(description="TerraLingua dashboard and remote-agent API server")
    parser.add_argument("--host", default=os.environ.get("API_HOST", "0.0.0.0"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("API_PORT", "8765"))
    )
    parser.add_argument("--ssl-certfile", default=os.environ.get("SSL_CERTFILE"))
    parser.add_argument("--ssl-keyfile", default=os.environ.get("SSL_KEYFILE"))
    parser.add_argument(
        "--testing",
        action="store_true",
        help="Gate the entire server behind DASHBOARD_PASSWORD (no user accounts needed).",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help=(
            "Local development mode: disables the Secure flag on auth cookies so "
            "they work over http://localhost. Equivalent to COOKIE_SECURE=false."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("API_WORKERS", "1")),
        help=(
            "Number of uvicorn worker processes. >1 enables multi-process mode "
            "(SO_REUSEPORT). Cross-worker state is coordinated via Redis."
        ),
    )
    args = parser.parse_args()

    ssl_enabled = bool(args.ssl_certfile and args.ssl_keyfile)
    scheme = "https" if ssl_enabled else "http"
    logger.info("[API] Dashboard: %s://localhost:%s/dashboard", scheme, args.port)

    common_kwargs = dict(
        host=args.host,
        port=args.port,
        log_level="info",
        proxy_headers=True,
        forwarded_allow_ips=os.environ.get("FORWARDED_ALLOW_IPS", "127.0.0.1"),
        # Don't advertise the underlying ASGI server in responses.
        server_header=False,
        # Per-message WebSocket cap. Uvicorn's default is 16 MiB; 5 MiB still
        # comfortably covers every legitimate dashboard payload (longest is on
        # the order of tens of KB) while shrinking the per-message blast
        # radius for a misbehaving / hostile client.
        ws_max_size=5 * 1024 * 1024,
    )
    if ssl_enabled:
        common_kwargs["ssl_certfile"] = args.ssl_certfile
        common_kwargs["ssl_keyfile"] = args.ssl_keyfile

    if args.workers > 1:
        # Workers can't share a pre-built app instance; pass a factory by import
        # string so each fork builds its own. Surface CLI args via env so the
        # factory sees the right config.
        os.environ["API_HOST"] = args.host
        os.environ["API_PORT"] = str(args.port)
        if args.ssl_certfile:
            os.environ["SSL_CERTFILE"] = args.ssl_certfile
        if args.ssl_keyfile:
            os.environ["SSL_KEYFILE"] = args.ssl_keyfile
        if args.testing:
            os.environ["API_TESTING"] = "true"
            # Share the gate-cookie nonce across workers so cookies validate uniformly.
            os.environ["OGW_GATE_NONCE"] = secrets.token_hex(32)
        logger.info("[API] Starting %d uvicorn workers", args.workers)
        uvicorn.run(
            "terralingua.server.cli:create_app_from_env",
            factory=True,
            workers=args.workers,
            **common_kwargs,
        )
    else:
        # Single-worker path preserves the in-process app instance for
        # dev/test flows that rely on lifespan side-effects in the parent.
        app = create_app(
            dashboard_manager=DashboardManager(),
            host=args.host,
            port=args.port,
            ssl=ssl_enabled,
            testing=args.testing,
        )
        setup_logging(verbose=1)
        uvicorn.run(app, **common_kwargs)


if __name__ == "__main__":
    main()
