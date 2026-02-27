#!/bin/bash

python main.py \
    \
    `# Experiment` \
    --exp_name              "artifact_cost" \
    --exp_description       "Ablation introducing a cost for creating artifacts." \
    --max_ts                3000 \
    \
    `# Agent LLM` \
    --model                 "DeepSeek-R1-32" \
    \
    `# Agents` \
    --exogenous_motivation  "base" \
    --genome                "ocean_5" \
    --max_history           1 \
    \
    `# Environment` \
    --grid_size             50 \
    --init_agents           20 \
    --init_agent_energy     50 \
    --food_zones            1 \
    --agent_lifespan        100 \
    --vision_radius         6 \
    --dead_agent_food       "single" \
    --artifact_creation_cost 10 \
    --reproduction_cost     50 \
    \
    `# Output` \
    --save_video \
    --video_fps             10 \
    # --save_root           "/path/to/output"   # defaults to project root
