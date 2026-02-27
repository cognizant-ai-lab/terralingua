import json
import pickle
from pathlib import Path
from typing import Dict

from core.agents.human_agent import HumanAgent
from core.agents.llm_agent import LLMAgent
from core.environment.env import OpenGridWorld
from core.experiment.config import ExperimentConfig, build_config


class CheckpointManager:
    def __init__(self, exp_logdir: Path):
        self.exp_logdir = exp_logdir
        self.checkpoint_path = exp_logdir / "checkpoint_latest.pkl"

    def save_checkpoint(
        self,
        agents: Dict[str, LLMAgent | HumanAgent],
        ts: int,
        env: OpenGridWorld,
        last_spawn_idx: int,
        env_outs: dict,
    ):
        ckpt_data = {}
        ckpt_data["ts"] = ts
        ckpt_data["env_outs"] = env_outs
        ckpt_data["env"] = env.get_state_ckpt()
        ckpt_data["agents"] = {}
        for agent_tag, agent in agents.items():
            ckpt_data["agents"][agent_tag] = agent.get_state_ckpt()
        ckpt_data["last_spawn_idx"] = last_spawn_idx

        with open(self.checkpoint_path, "wb") as f:
            pickle.dump(ckpt_data, f)
        print(f"💾 Saved checkpoint at timestep {ts} to {self.checkpoint_path}")

    def load_checkpoint(self) -> dict:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"No checkpoint file found at {self.checkpoint_path}"
            )

        with open(self.checkpoint_path, "rb") as f:
            ckpt_data = pickle.load(f)
        print(f"💾 Loaded checkpoint from {self.checkpoint_path}")
        return ckpt_data

    def update_parameters(self) -> ExperimentConfig:
        with open(self.exp_logdir / "params.json", "r") as f:
            saved_params = json.load(f)
        flat_params = {}
        for p in saved_params.values():
            flat_params.update(p)

        return build_config(flat_params)
