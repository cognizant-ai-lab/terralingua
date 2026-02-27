#!/bin/bash

python main.py \
    \
    `# Experiment` \
    --exp_name              "abundant_resources" \
    --exp_description       "Ablation extending the temporal context of the agents and providing them with more resources." \
    --max_ts                3000 \
    \
    `# Agent LLM` \
    --model                 "DeepSeek-R1-32" \
    \
    `# Agents` \
    --exogenous_motivation  "base" \
    --genome                "ocean_5" \
    --max_history           20 \
    \
    `# Environment` \
    --grid_size             50 \
    --init_agents           20 \
    --init_agent_energy     50 \
    --agent_lifespan        100 \
    --food_zones            "null" \
    --vision_radius         6 \
    --dead_agent_food       "single" \
    --artifact_creation_cost 0 \
    --reproduction_cost     50 \
    \
    `# Output` \
    --save_video \
    --video_fps             10 \
    # --save_root           "/path/to/output"   # defaults to project root
