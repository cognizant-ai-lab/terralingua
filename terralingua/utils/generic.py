import os
import subprocess
from pathlib import Path


def get_render_state(env) -> dict:
    """Extract all data needed to render the grid from an OpenGridWorld instance."""
    r = env.vision_radius
    max_food = float(env._max_food_value) if env._max_food_value else 1.0

    vision_cells = set()
    for tag in env.agent_registry:
        cx, cy = env.agent_pos[tag]
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                vision_cells.add(env.wrap_xy(cx + dx, cy + dy))

    food = []
    for (x, y), val in env.food.items():
        if 0 <= x < env.grid_size and 0 <= y < env.grid_size:
            ratio = max(0.0, min(1.0, float(val) / max_food))
            food.append({"x": int(x), "y": int(y), "ratio": float(ratio), "value": int(round(ratio * max_food))})

    artifacts = []
    for (x, y), arts in env.pos_artifacts.items():
        if 0 <= x < env.grid_size and 0 <= y < env.grid_size and arts:
            artifacts.append({"x": int(x), "y": int(y)})

    agents = []
    for tag in env.agent_registry:
        ax, ay = env.agent_pos[tag]
        agents.append({"x": int(ax), "y": int(ay), "tag": tag, "name": env.agent_names.get(tag, tag)})

    step = int(env.step_count)
    recent_messages = []
    for s in range(step - 1, max(-1, step - 11), -1):
        msgs = env.chat.get(s)
        if msgs:
            senders = env.chat_senders.get(s, [])
            step_recipients = env.chat_recipients.get(s, {})
            recipients_per_msg = [
                sorted(step_recipients.get(senders[i], set())) if i < len(senders) else []
                for i in range(len(msgs))
            ]
            recent_messages.append({
                "step": s,
                "messages": list(msgs),
                "recipients": recipients_per_msg,
            })

    return {
        "grid_size": int(env.grid_size),
        "vision_radius": int(r),
        "step": step,
        "vision_cells": [[int(x), int(y)] for x, y in vision_cells],
        "food": food,
        "artifacts": artifacts,
        "agents": agents,
        "recent_messages": recent_messages,
    }

def get_graph_render_state(env) -> dict:
    """Extract all data needed to render the graph from an OpenGraphWorld instance."""
    hop_radius = env.hop_radius
    max_food = float(env._max_food_value) if env._max_food_value else 1.0
    step = int(env.step_count)

    hop_nodes: set = set()
    for tag in env.agent_registry:
        neighborhood = env.world_graph.nodes_within_hops(env.agent_pos[tag], hop_radius)
        hop_nodes.update(neighborhood.keys())

    food = []
    for node_id, val in env.food.items():
        ratio = max(0.0, min(1.0, float(val) / max_food))
        food.append({"node": node_id, "ratio": float(ratio), "value": int(round(ratio * max_food))})

    artifacts = []
    for node_id, arts in env.pos_artifacts.items():
        if arts:
            artifacts.append({"node": node_id})

    agents = []
    for tag in env.agent_registry:
        agents.append({
            "node": env.agent_pos[tag],
            "tag": tag,
            "name": env.agent_names.get(tag, tag),
        })

    recent_messages = []
    for s in range(step - 1, max(-1, step - 11), -1):
        msgs = env.chat.get(s)
        if msgs:
            senders = env.chat_senders.get(s, [])
            step_recipients = env.chat_recipients.get(s, {})
            recipients_per_msg = [
                sorted(step_recipients.get(senders[i], set())) if i < len(senders) else []
                for i in range(len(msgs))
            ]
            recent_messages.append({
                "step": s,
                "messages": list(msgs),
                "recipients": recipients_per_msg,
            })

    all_nodes = env.world_graph.all_nodes()
    seen_pairs: set = set()
    edges = []
    for u in all_nodes:
        for v in env.world_graph.neighbors(u):
            pair = (min(u, v), max(u, v))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            edges.append({
                "source": u,
                "target": v,
                "type": "mutual" if env.world_graph.has_edge(v, u) else "follow",
            })

    # Pending connection requests (social-graph world only): directional
    # sender -> target, drawn as dashed edges. A pending request never coexists
    # with a follow edge in the same direction, so these don't overlap solids.
    pending_outgoing = getattr(env, "_pending_outgoing", None)
    agent_pos = getattr(env, "agent_pos", {})
    if pending_outgoing:
        for sender_tag, targets in pending_outgoing.items():
            source = agent_pos.get(sender_tag)
            if source is None:
                continue
            for target_tag in targets:
                target = agent_pos.get(target_tag)
                if target is not None:
                    edges.append({"source": source, "target": target, "type": "pending"})

    node_layout = {
        node_id: list(xy)
        for node_id, xy in (env._node_layout or {}).items()
    }

    return {
        "env_type": "graph",
        "topology": env.graph_cfg.topology,
        "hop_radius": int(hop_radius),
        "step": step,
        "nodes": all_nodes,
        "edges": edges,
        "node_layout": node_layout,
        "hop_nodes": list(hop_nodes),
        "food": food,
        "artifacts": artifacts,
        "agents": agents,
        "recent_messages": recent_messages,
    }


# The working directory at start. Analysis output is written under it.
ROOT = Path.cwd().resolve()

# Where simulation logs, checkpoints, and artifacts live. Defaults to `logs/`
# under the working directory; set TL_LOGS_DIR to relocate (useful when running
# under Docker with a bind-mounted volume).
LOGS_DIR = Path(os.environ.get("TL_LOGS_DIR") or (ROOT / "logs")).resolve().absolute()


def create_video(
    input_pattern="%05d.png", output_file="video.mp4", fps=10, crf=18, preset="medium"
):
    scale_filter = "scale=500:500:flags=neighbor"
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file without asking
        "-framerate",
        str(fps),
        "-i",
        input_pattern,
        "-c:v",
        "libx264",
        "-preset",
        preset,  # options: ultrafast, fast, medium, slow, slower
        "-crf",
        str(crf),  # lower is better quality (18-28 range typical)
        "-pix_fmt",
        "yuv420p",  # ensures compatibility
        output_file,
        "-vf",
        scale_filter,
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
