#!/bin/bash

python main.py \
    \
    `# Experiment` \
    --exp_name              "experiment_name" \
    --exp_description       "Experiment description" \
    --max_ts                300 \
    \
    `# Agent LLM` \
    --model                 "claude-haiku-4-5" \
    \
    `# Agents` \
    --agents_name_prefix    "being" \    # name prefix for agents, e.g. being_0, being_1, etc.
    --exogenous_motivation  "base" \     # motivation mechanism for agents, e.g. "base", "creative", "survival", "none".
    --genome                "ocean_5" \  # genome configuration for agents, e.g. "ocean_5", "no_traits", "single_word".
    --max_history           1 \          # number of past timesteps to include in agent observations
    --internal_memory_size  150 \        # size of the internal memory for agents
    --use_internal_memory \              # flag to enable internal memory for agents
    --use_inventory \                    # flag to enable inventory for agents
    --no-use_colors \                    # flag to disable color usage for agents (agents can set a color for themselves that other agents can see)
    \
    `# Environment` \
    --grid_size             50 \         # size of the grid environment (grid_size x grid_size)
    --init_agents           10 \         # initial number of agents in the environment
    --init_human_agents     0 \          # initial number of human agents in the environment
    --min_agents            0 \          # minimum number of agents in the environment
    --init_agent_energy     50 \         # initial energy for each agent
    --init_food             100 \        # initial amount of food in the environment
    --food_zones            1 \          # number of food zones in the environment (areas where food can spawn more frequently)
    --food_mechanism \                   # flag to enable the food mechanism
    --agent_lifespan        100 \        # lifespan of agents in the environment
    --vision_radius         6 \          # vision radius of agents
    --dead_agent_food       "single" \   # food type from dead agents ("single": dead agent leaves all its energy as food in its cell, "none": dead agents do not leave food, "area": a 3x3 area around the dead agent position is filled with food)
    --artifact_creation \                # flag to enable artifact creation
    --artifact_creation_cost 0 \         # cost of creating artifacts
    --no-inert_artifacts \               # flag to disable inert artifacts. To have inert artifacts use the `inert_artifacts` flag instead of this one.
    --reproduction_allowed \             # flag to allow agents to reproduce. To disable reproduction, use the `no-reproduction_allowed` flag instead.
    --reproduction_cost     50 \         # energy cost for agents to reproduce
