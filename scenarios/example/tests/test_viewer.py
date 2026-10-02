import json

from fastapi.testclient import TestClient

from scenarios.example.viewer.server import create_app


def make_run(logs, name):
    run = logs / name
    (run / "frames").mkdir(parents=True)
    (run / "params.json").write_text(json.dumps({"env": {"world_type": "grid", "grid_size": 15, "init_agents": 4}, "run": {"max_ts": 30}}))
    (run / "open_gridworld.log").write_text("step 1\nstep 2\n")
    (run / "frames" / "0001.png").write_bytes(b"\x89PNG not a real image")
    return run


def test_the_viewer_lists_and_shows_runs(tmp_path):
    logs = tmp_path / "logs"
    make_run(logs, "trial")
    (logs / "not_a_run").mkdir()
    client = TestClient(create_app(logs))
    index = client.get("/").text
    assert "trial" in index and "not_a_run" not in index
    shown = client.get("/runs/trial").text
    assert "Grid size" in shown and "15" in shown and "step 2" in shown and "/runs/trial/frame" in shown
    assert client.get("/runs/trial/frame").status_code == 200
    assert client.get("/runs/missing").status_code == 404


def test_an_empty_logs_folder_is_fine(tmp_path):
    client = TestClient(create_app(tmp_path / "nowhere"))
    assert "No runs yet" in client.get("/").text
