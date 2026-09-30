import asyncio
import json
import pickle
from pathlib import Path
from typing import Dict

from terralingua.agents.human_agent import HumanAgent
from terralingua.agents.llm_agent import LLMAgent
from terralingua.agents.remote_agent import RemoteAgent
from terralingua.config.models import ExperimentConfig
from terralingua.environment.graph_env import OpenGraphWorld
from terralingua.environment.grid_env import OpenGridWorld


class CheckpointManager:
    def __init__(self, exp_logdir: Path):
        self.exp_logdir = exp_logdir
        self.checkpoint_path = exp_logdir / "checkpoint_latest.pkl"
        self.previous_checkpoint_path = exp_logdir / "checkpoint_previous.pkl"

    async def save_checkpoint(
        self,
        agents: Dict[str, LLMAgent | HumanAgent | RemoteAgent],
        ts: int,
        env: OpenGridWorld | OpenGraphWorld,
        last_spawn_idx: int,
        env_outs: dict,
        run_id: str = "",
    ):
        ckpt_data = {}
        ckpt_data["ts"] = ts
        ckpt_data["run_id"] = run_id
        ckpt_data["env_outs"] = env_outs
        ckpt_data["env"] = env.get_state_ckpt()
        ckpt_data["agents"] = {}
        for agent_tag, agent in agents.items():
            ckpt_data["agents"][agent_tag] = agent.get_state_ckpt()
        ckpt_data["last_spawn_idx"] = last_spawn_idx

        # Serialize synchronously (no await = no context switches, consistent snapshot),
        # then write the bytes to disk in a thread so the event loop stays responsive.
        raw = pickle.dumps(ckpt_data)

        def _rotate_and_write(path: Path, previous_path: Path, data: bytes):
            if path.exists():
                path.rename(previous_path)
            path.write_bytes(data)

        await asyncio.get_event_loop().run_in_executor(
            None, _rotate_and_write, self.checkpoint_path, self.previous_checkpoint_path, raw
        )
        print(f"💾 Saved checkpoint at timestep {ts} to {self.checkpoint_path}")

    def load_checkpoint(self) -> dict:
        for path in (self.checkpoint_path, self.previous_checkpoint_path):
            if not path.exists():
                continue
            try:
                with open(path, "rb") as f:
                    ckpt_data = pickle.load(f)
                print(f"💾 Loaded checkpoint from {path}")
                return ckpt_data
            except Exception as e:
                print(f"⚠️  Checkpoint at {path} is corrupted ({e}), trying previous...")
        raise FileNotFoundError(
            f"No valid checkpoint found in {self.exp_logdir}"
        )

    def update_parameters(self) -> ExperimentConfig:
        with open(self.exp_logdir / "params.json", "r") as f:
            saved_params = json.load(f)
        return ExperimentConfig.model_validate(saved_params)
