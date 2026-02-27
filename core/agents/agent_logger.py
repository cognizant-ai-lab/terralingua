import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from core.utils import ROOT


class AgentLogger:
    def __init__(self, log_dir: Path | str | None, agent_tag: str = "agent"):
        if log_dir is None:
            self.log_dir = (
                ROOT / "logs" / datetime.now().strftime("%Y%m%d_%H%M") / "agent_logs"
            )
        else:
            self.log_dir = Path(log_dir)

        os.makedirs(self.log_dir, exist_ok=True)
        self.filename = self.log_dir / f"{agent_tag}.jsonl"
        self.data_dict: Dict[str, Dict[str, Any]] = {}

    def save_genome(self, agent_tag: str, genome: dict):
        genome_filename = self.log_dir / f"{agent_tag}_genome.json"
        with open(genome_filename, "w") as f:
            json.dump(genome, f, indent=4)

    def log(
        self,
        agent_name: str,
        agent_tag: str,
        observation: dict,
        available_actions: dict,
        action: dict,
        time: str,
        internal_memory: str,
        input_prompt: str,
    ):
        """
        Record the agent’s observation and action, both in a live file and in-memory dictionary.
        """
        observation["observation"] = {
            str(k): v for k, v in observation["observation"].items()
        }

        record = {
            "timestamp": time,
            "agent": agent_name,
            "agent_tag": agent_tag,
            "action": action,
            "observation": observation,
            "internal_memory": internal_memory,
            "available_actions": available_actions,
            "input_prompt": input_prompt,
        }

        # Save line to file
        with open(self.filename, "a") as f:
            f.write(json.dumps(record) + "\n")

        # Save to internal dict
        self.data_dict[time] = {
            "agent": agent_name,
            "agent_tag": agent_tag,
            "action": action,
            "internal_memory": internal_memory,
            "available_actions": available_actions,
            "observation": observation,
            "input_prompt": input_prompt,
        }

    def close(self):
        return
