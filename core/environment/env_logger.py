import json
from enum import Enum, auto
from pathlib import Path

import numpy as np


class Event(Enum):
    AGENT_DIED = auto()
    AGENT_ADDED = auto()
    AGENT_REPRODUCED = auto()
    RESET_AGENT = auto()
    ENV_RESET = auto()
    TAKE_ENERGY = auto()
    GIFT_ENERGY = auto()
    ARTIFACT_ADDED = auto()
    ARTIFACT_REMOVED = auto()
    ARTIFACT_INTERACTION = auto()
    ARTIFACT_PASSIVE_INTERACTION = auto()
    ARTIFACT_PICKUP = auto()
    ARTIFACT_DROP = auto()
    GIVE_ARTIFACT = auto()
    SET_COLOR = auto()
    SET_STATE_CKPT = auto()
    END_RUN = auto()


class JSONLogger:
    def __init__(self, filepath: Path | str):
        self.save_path = Path(filepath)
        if not self.save_path.parent.exists():
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = open(filepath, "a", buffering=1)  # line-buffered
        self.data = {}

    def log(self, time: int, event_type: Event, **data):
        """
        Write one log entry as a JSON object with:
          - timestamp: wall-clock time
          - event: a short event name
          - any other keyword data fields
        """
        entry = {"timestamp": time, "event": event_type, **data}
        safe_entry = self._sanitize(entry)
        try:
            self.fp.write(json.dumps(safe_entry) + "\n")
        except:
            print(f"Failed logging: {safe_entry}")

        if event_type == Event.AGENT_ADDED:
            # If an agent is added, we can also log the current state of the environment
            self.data[data["agent_tag"]] = {
                "spawn_position": data.get("position", None),
                "agent_type": data["agent_type"],
                "spawn_time": time,
                "agent_name": data["agent_name"],
            }
        if event_type == Event.AGENT_DIED:
            agent_tag = data["agent_tag"]
            if agent_tag in self.data:
                self.data[agent_tag]["death_time"] = time
                self.data[agent_tag]["death_position"] = data["position"]
                self.data[agent_tag]["age"] = time - self.data[agent_tag]["spawn_time"]
                self.data[agent_tag]["energy"] = data["energy"]
                self.data[agent_tag]["death_reason"] = data["reason"]
                self.data[agent_tag]["agent_name"] = data["agent_name"]
            else:
                raise ValueError(f"Dead agent {agent_tag} was never logged in")

        if event_type == Event.AGENT_REPRODUCED:
            agent_tag = data["agent_tag"]
            if agent_tag in self.data:
                reprod = {"time": time}
                if data["successful"]:
                    reprod["child_name"] = data["child_name"]
                    reprod["child_tag"] = data["child_tag"]
                else:
                    reprod["fail_reason"] = data["fail_reason"]
                self.data[agent_tag].setdefault("reproduction", []).append(reprod)
            else:
                raise ValueError(f"Reproducing agent {agent_tag} was never logged in")

        if event_type in [Event.TAKE_ENERGY, Event.GIFT_ENERGY]:
            agent_tag = data["agent_tag"]
            if agent_tag in self.data:
                event = {
                    "time": time,
                    "target_tag": data["target_tag"],
                    "target_name": data["target_name"],
                    "amount": data["amount"],
                }
                if event_type == Event.TAKE_ENERGY:
                    self.data[agent_tag].setdefault("take", []).append(event)
                else:
                    self.data[agent_tag].setdefault("gift", []).append(event)
            else:
                raise ValueError(f"Agent {agent_tag} was never logged in")

    def close(self):
        self.fp.close()
        with open(self.save_path.parent / "agent_events.json", "w") as f:
            json.dump(self._sanitize(self.data), f, indent=4)

    def _sanitize(self, obj):
        """
        Recursively convert NumPy types to native Python ones:
          - np.generic  -> .item()
          - np.ndarray  -> list()
          - tuple/list  -> same type with sanitized elements
          - dict        -> sanitized keys/values
        """
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Event):
            return obj.name
        elif isinstance(obj, dict):
            return {self._sanitize(k): self._sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            cls = list if isinstance(obj, list) else tuple
            return cls(self._sanitize(v) for v in obj)
        else:
            return obj
