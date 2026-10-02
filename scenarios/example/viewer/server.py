"""A small viewer for runs of the example scenario.

It follows the convention the launcher uses to start a scenario's viewer:

    python -m scenarios.example.viewer --logs <folder> --port <n>

It lists the runs under the logs folder. For one run it shows the main
settings, the latest frame and the last lines of the world log.
"""

import argparse
import html
import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

WORLD_LOGS = ("open_gridworld.log", "open_social_graph_world.log")
TAIL_LINES = 60
STYLE = (
    "body{font-family:sans-serif;margin:2rem;max-width:60rem}"
    "pre{background:#f4f4f4;padding:1rem;overflow:auto}img{max-width:100%}"
    "th{text-align:left;padding-right:1rem}"
)


def runs(logs: Path) -> list[Path]:
    """The run folders, newest first. A run folder holds a params.json."""
    if not logs.is_dir():
        return []
    found = [path for path in logs.iterdir() if (path / "params.json").is_file()]
    return sorted(found, key=lambda path: path.stat().st_mtime, reverse=True)


def latest_frame(run: Path) -> Path | None:
    frames = sorted((run / "frames").glob("*.png")) if (run / "frames").is_dir() else []
    return frames[-1] if frames else None


def world_log_tail(run: Path) -> str:
    for name in WORLD_LOGS:
        path = run / name
        if path.is_file():
            return "\n".join(path.read_text(errors="replace").splitlines()[-TAIL_LINES:])
    return "No world log yet."


def page(title: str, body: str) -> HTMLResponse:
    return HTMLResponse(
        f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
        f"<style>{STYLE}</style></head><body>{body}</body></html>"
    )


def create_app(logs: Path) -> FastAPI:
    app = FastAPI(title="Example scenario viewer")

    def run_folder(name: str) -> Path:
        run = logs / name
        if "/" in name or not (run / "params.json").is_file():
            raise HTTPException(404, f"no run named {name}")
        return run

    @app.get("/")
    def index():
        items = "".join(
            f"<li><a href='/runs/{html.escape(run.name)}'>{html.escape(run.name)}</a></li>"
            for run in runs(logs)
        )
        return page("Runs", f"<h1>Runs under {html.escape(str(logs))}</h1><ul>{items or '<li>No runs yet.</li>'}</ul>")

    @app.get("/runs/{name}")
    def show(name: str):
        run = run_folder(name)
        params = json.loads((run / "params.json").read_text())
        env = params.get("env", {})
        facts = {
            "World": env.get("world_type"),
            "Grid size": env.get("grid_size"),
            "Initial beings": env.get("init_agents"),
            "Steps": params.get("run", {}).get("max_ts"),
        }
        rows = "".join(
            f"<tr><th>{html.escape(key)}</th><td>{html.escape(str(value))}</td></tr>"
            for key, value in facts.items() if value is not None
        )
        frame = (
            f"<img src='/runs/{html.escape(name)}/frame' alt='latest frame'>"
            if latest_frame(run) else "<p>No frame yet.</p>"
        )
        return page(name, (
            f"<p><a href='/'>All runs</a></p><h1>{html.escape(name)}</h1><table>{rows}</table>{frame}"
            f"<h2>World log, last lines</h2><pre>{html.escape(world_log_tail(run))}</pre>"
        ))

    @app.get("/runs/{name}/frame")
    def frame(name: str):
        path = latest_frame(run_folder(name))
        if path is None:
            raise HTTPException(404, "no frame yet")
        return FileResponse(path)

    return app


def main():
    parser = argparse.ArgumentParser(description="Viewer for runs of the example scenario.")
    parser.add_argument("--logs", type=Path, default=Path("logs"), help="Folder that holds the runs")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    print(f"Example viewer: http://{args.host}:{args.port} (runs under {args.logs})")
    uvicorn.run(create_app(args.logs.resolve()), host=args.host, port=args.port, log_level="warning")
