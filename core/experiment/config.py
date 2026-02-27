from argparse import Namespace
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from typing import List, Tuple

from core.agents.prompt_templates import AVAILABLE_EX_MOTIVATIONS, OBS_STYLE
from core.environment.env import AVAILABLE_DEAD_AGENT_FOOD
from core.genome import AVAILABLE_GENOMES
from core.utils import ROOT


@dataclass
class AgentConfig:
    agents_name_prefix: str = field(
        default="being",
        metadata={"help": "Prefix for agent names (e.g. being0, being1)"},
    )
    exogenous_motivation: str = field(
        default="base",
        metadata={
            "help": "Type of exogenous motivations",
            "choices": AVAILABLE_EX_MOTIVATIONS,
        },
    )
    genome: str = field(
        default="ocean_5",
        metadata={
            "help": "Agent genome type",
            "choices": AVAILABLE_GENOMES,
        },
    )
    internal_memory_size: int = field(
        default=150,
        metadata={"help": "Size in tokens of internal memory"},
    )
    max_history: int = field(
        default=1,
        metadata={"help": "Max number of interactions stored per agent"},
    )
    model: str = field(
        default="claude-sonnet-4-6",
        metadata={"help": "Model name used for agent decision making"},
    )
    obs_style: str = field(
        default="list",
        metadata={"help": "Observation style", "choices": list(OBS_STYLE.keys())},
    )
    use_colors: bool = field(
        default=False,
        metadata={"help": "Allow agents to choose their own color"},
    )
    use_internal_memory: bool = field(
        default=True,
        metadata={"help": "Use agent-internal memory"},
    )
    use_inventory: bool = field(
        default=True,
        metadata={"help": "Enable inventory system"},
    )

    def __post_init__(self):
        avail_obs_styles = list(OBS_STYLE.keys())
        if self.obs_style not in avail_obs_styles:
            raise ValueError(
                f"Obs_style is {self.obs_style} - Available: {avail_obs_styles}"
            )

        assert self.genome in AVAILABLE_GENOMES, (
            f"Genome must be one of {AVAILABLE_GENOMES}, got {self.genome}"
        )

        assert self.exogenous_motivation in AVAILABLE_EX_MOTIVATIONS, (
            f"Exogenous motivation must be one of {AVAILABLE_EX_MOTIVATIONS}, got {self.exogenous_motivation}"
        )


@dataclass
class EnvConfig:
    agent_lifespan: int = field(default=100, metadata={"help": "Max lifespan"})
    artifact_creation: bool = field(
        default=True, metadata={"help": "Enable artifact creation"}
    )
    artifact_creation_cost: int = field(
        default=0, metadata={"help": "Cost to create artifact"}
    )
    dead_agent_food: str = field(
        default="single",
        metadata={
            "help": "Food from dead agents",
            "choices": AVAILABLE_DEAD_AGENT_FOOD,
        },
    )
    food_decay_rate: float = field(default=0.05, metadata={"help": "Food decay rate"})
    food_mechanism: bool = field(
        default=True, metadata={"help": "Enable energy mechanic"}
    )
    food_spawn_rate: int = field(default=1, metadata={"help": "Food spawn per step"})
    food_zones: int | List[Tuple[int, int]] | None = field(
        default=None,
        metadata={
            "help": "Food zones. Accepts integer OR list of 'x,y' pairs",
            "autocoerce": "food_zones",
        },
    )
    grid_size: int = field(default=50, metadata={"help": "Grid dimension"})
    inert_artifacts: bool = field(
        default=False, metadata={"help": "Artifacts cannot be interacted with"}
    )
    init_agents: int = field(default=20, metadata={"help": "Initial agent count"})
    init_human_agents: int = field(
        default=0, metadata={"help": "Initial human agent count"}
    )
    init_agent_energy: int = field(
        default=50, metadata={"help": "Initial energy per agent"}
    )
    init_food: int = field(default=100, metadata={"help": "Initial food count"})
    min_agents: int = field(default=0, metadata={"help": "Minimum agent population"})
    reproduction_allowed: bool = field(
        default=True, metadata={"help": "Enable reproduction"}
    )
    reproduction_cost: int = field(
        default=50, metadata={"help": "Energy cost to reproduce"}
    )
    static_food: bool = field(
        default=False, metadata={"help": "Food always spawns in same positions"}
    )
    vision_radius: int = field(default=6, metadata={"help": "Vision radius"})

    def __post_init__(self):
        assert self.dead_agent_food in AVAILABLE_DEAD_AGENT_FOOD, (
            f"Dead agent food must be one of {AVAILABLE_DEAD_AGENT_FOOD}, got {self.dead_agent_food}"
        )

        assert self.min_agents <= self.init_agents, (
            "min_agents cannot be greater than init_agents"
        )


@dataclass
class RunConfig:
    ckpt_interval: int = field(default=100, metadata={"help": "Checkpoint interval"})
    empty_countdown: int = field(default=20, metadata={"excluded": True})
    exp_description: str = field(
        default="", metadata={"help": "Experiment description"}
    )
    exp_name: str | None = field(
        default='TEST', metadata={"help": "Experiment name", "arg_type": str}
    )
    live_render: bool = field(
        default=False, metadata={"help": "Render simulation live"}
    )
    max_parallel_workers: int = field(
        default=8, metadata={"help": "Max worker threads"}
    )
    max_ts: int = field(default=3000, metadata={"help": "Max simulation timesteps"})
    ports: tuple = field(
        default=(9000, 9001, 9002, 9003, 9010, 9011, 9012),
        metadata={
            "help": "Ports hosting LLM models",
            "arg_type": int,
            "nargs": "+",
        },
    )
    save_root: str | None = field(
        default=None, metadata={"help": "Output directory root", "arg_type": str}
    )
    save_video: bool = field(default=True, metadata={"help": "Save video"})
    video_fps: int = field(default=10, metadata={"help": "FPS for video output"})

    def __post_init__(self):
        if self.exp_name is None:
            self.exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.save_root is None:
            self.save_root = str(ROOT)

        self.ports = tuple(self.ports)


@dataclass
class ExperimentConfig:
    agent: AgentConfig
    env: EnvConfig
    run: RunConfig

    def to_json(self):
        return {
            "agent": asdict(self.agent),
            "env": asdict(self.env),
            "run": asdict(self.run),
        }


def build_config(args: dict | Namespace) -> ExperimentConfig:
    agent = AgentConfig()
    env = EnvConfig()
    run = RunConfig()

    if isinstance(args, Namespace):
        args = vars(args)

    for k, v in args.items():
        if v is None or k == "resume":
            continue
        if hasattr(agent, k):
            agent = replace(agent, **{k: v})
        elif hasattr(env, k):
            env = replace(env, **{k: v})
        elif hasattr(run, k):
            run = replace(run, **{k: v})

    return ExperimentConfig(agent, env, run)
