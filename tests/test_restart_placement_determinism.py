"""Agent placement at restart depends only on the seed, not on Python's hash-randomised set order."""

import json
import os
import subprocess
import sys

SCRIPT = r"""
import json, sys
from terralingua.config.models import GraphConfig
from terralingua.environment.graph_env import OpenGraphWorld
env = OpenGraphWorld(graph_cfg=GraphConfig.complete(n_nodes=40), log_path=sys.argv[1], headless=True,
                     food_mechanism=False, init_agent_energy=50, lifespan=100, reproduction_cost=-1)
for i in range(8):
    env.add_agent(f"a{i}", f"Being{i}", "no_traits")
env.restart_env(seed=3)
print(json.dumps(env.agent_pos, sort_keys=True))
"""


def placements(tmp_path, hash_seed):
    env = dict(os.environ, PYTHONHASHSEED=str(hash_seed))
    out = subprocess.run([sys.executable, "-c", SCRIPT, str(tmp_path / f"h{hash_seed}")],
                         env=env, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_restart_places_the_same_agents_on_the_same_nodes_under_any_hash_seed(tmp_path):
    first = placements(tmp_path, 0)
    assert len(first) == 8
    for hash_seed in (1, 2, 3):
        assert placements(tmp_path, hash_seed) == first
