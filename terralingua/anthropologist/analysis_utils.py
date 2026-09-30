import json
import os
from copy import deepcopy
from pathlib import Path
from typing import List

import numpy as np


def received_messages(observation: dict) -> dict:
    """Messages an agent received in one step, by sender: broadcasts and direct messages."""
    messages = dict(observation.get("incoming_broadcasts", {}))
    for sender, text in observation.get("incoming_dms", {}).items():
        messages[sender] = f"{messages[sender]}\n{text}" if sender in messages else text
    return messages


def load_agent_log(filepath: Path | str, reduce: bool) -> dict:
    """Load agent log

    Args:
        filepath (Path | str): _description_
        reduce (bool): if True, reduce the log size by removing some fields like available_actions and reasoning.

    Returns:
        dict: Loaded agent log data {ts: log_entry}
    """
    with open(filepath, "r") as f:
        agent_data = {}
        for line in f:
            if line.strip():  # skip empty lines
                obj = json.loads(line)

                ts = (
                    int(obj["timestamp"])
                    if obj["timestamp"].isdigit()
                    else obj["timestamp"]
                )
                if reduce:
                    if obj["action"]["action"] == "move":
                        del obj["action"]["params"]

                    obj.pop("available_actions", None)
                    obj.pop("input_prompt", None)

                    if "reasoning" in obj["action"]:
                        del obj["action"]["reasoning"]

                    action_dict = deepcopy(obj["action"])
                    obj["action"] = action_dict["action"]
                    obj["action_params"] = action_dict.get("params", {})
                    obj["sent_message"] = action_dict.get("message", "")

                    obs_dict = deepcopy(obj["observation"])
                    obj["observation"] = obs_dict.get("observation", {})
                    obj["received_messages"] = received_messages(obs_dict)
                    obj["energy"] = obs_dict["energy"]
                    obj["time"] = obs_dict["time"]
                    if "inventory" in obs_dict:
                        obj["inventory"] = obs_dict["inventory"]

                agent_data[ts] = obj
    return agent_data


def load_worldlog(
    filepath: Path | str,
    agent_name: str | None = None,
    agent_tag: str | None = None,
    event: List[str] | None = None,
):
    with open(filepath, "r") as f:
        data = []
        for line in f:
            if agent_tag is None and agent_name is None and event is None:
                obj = json.loads(line)
                data.append(obj)
            else:

                def check_presence(line):
                    if agent_name is not None and agent_name in str(line):
                        return True
                    if agent_tag is not None and agent_tag in str(line):
                        return True
                    if event is not None and any(e in str(line) for e in event):
                        return True
                    return False

                if check_presence(line=line):
                    obj = json.loads(line)
                    data.append(obj)
        return data


def get_last_ts(worldlog_path: Path | str) -> int:
    """Return the step at which the run ended (END_RUN or last AGENT_DIED).

    For post-hoc analysis where you want the final step of a completed run.
    For tracking the live simulation cursor, use get_current_step instead.
    """
    with open(worldlog_path, "r") as f:
        last_logged_ts = 0
        for line in reversed(list(f)):
            try:
                entry = json.loads(line.strip())
            except json.JSONDecodeError:
                print(repr(line))
                raise
            if last_logged_ts == 0:
                last_logged_ts = entry["timestamp"]
            if entry.get("event") == "END_RUN":
                return entry["timestamp"]
            if entry.get("event") == "AGENT_DIED":
                return entry["timestamp"]
        return last_logged_ts


def get_current_step(worldlog_path: Path | str) -> int:
    """Return the timestamp of the most recent entry in the world log.

    Use this in the live server to track simulation progress — unlike
    get_last_ts it is not confused by AGENT_DIED events that have newer
    entries logged after them.

    World log lines can exceed 20 KB. We read only the last 64 KB
    (enough for at least 2 complete lines), falling back to a full
    read if no parseable timestamp is found in the tail.
    """
    worldlog_path = Path(worldlog_path)
    with open(worldlog_path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        if size == 0:
            return 0
        chunk = min(size, 65536)
        f.seek(size - chunk)
        tail = f.read().decode("utf-8", errors="replace")

    for line in reversed(tail.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return int(json.loads(line)["timestamp"])
        except Exception:
            continue

    # Tail didn't yield a parseable line (e.g. log entry > 64 KB).
    # Fall back to reading the whole file.
    with open(worldlog_path, "r") as f:
        for line in reversed(list(f)):
            line = line.strip()
            if not line:
                continue
            try:
                return int(json.loads(line)["timestamp"])
            except Exception:
                continue
    return 0


def agents_time(worldlog_path: Path | str) -> np.ndarray:
    agent_count = np.zeros(get_last_ts(worldlog_path) + 1)
    worldlog = load_worldlog(filepath=worldlog_path, agent_name=None, agent_tag=None)
    for ts in worldlog:
        if ts["event"] == "AGENT_ADDED":
            agent_count[ts["timestamp"]] += 1
        elif ts["event"] == "AGENT_DIED":
            agent_count[ts["timestamp"]] -= 1
    return np.cumsum(agent_count)


def get_exp_folders(
    log_path: Path | str, exp_name: str, only_completed: bool = True
) -> List[str]:
    log_path = Path(log_path)
    assert log_path.exists(), f"Log path {log_path} does not exist."
    exp_folders = []
    for fold in os.listdir(log_path):
        if exp_name in fold and (log_path / fold).is_dir():
            if not only_completed:
                exp_folders.append(fold)
            elif (log_path / fold / "video.mp4").exists():
                exp_folders.append(fold)
    return sorted(exp_folders)
