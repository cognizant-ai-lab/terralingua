// Modal/dialog utilities used across the dashboard UI.

import { escHtml } from './state.js';

// `message` and the labels are plain text. They are set with textContent, so
// agent and artifact names can never be parsed as HTML.
export function _confirm(message, okLabel = "OK", cancelLabel = "Cancel") {
  return new Promise((resolve) => {
    const overlay = document.createElement("div");
    overlay.style.cssText = "position:fixed;inset:0;background:rgba(0,0,0,0.7);display:flex;align-items:center;justify-content:center;z-index:9999;";
    const box = document.createElement("div");
    box.style.cssText = "background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:1.5rem;max-width:380px;width:90%;";
    const text = document.createElement("p");
    text.style.cssText = "font-size:0.9rem;color:var(--text);margin-bottom:1.25rem;line-height:1.5;";
    text.textContent = message;
    const buttons = document.createElement("div");
    buttons.style.cssText = "display:flex;gap:0.75rem;justify-content:flex-end;";
    const close = (val) => { overlay.remove(); resolve(val); };
    if (cancelLabel) {
      const cancelBtn = document.createElement("button");
      cancelBtn.style.cssText = "padding:0.5rem 1rem;border-radius:6px;border:1px solid var(--border);background:transparent;color:var(--muted);cursor:pointer;font-size:0.85rem;";
      cancelBtn.textContent = cancelLabel;
      cancelBtn.onclick = () => close(false);
      buttons.appendChild(cancelBtn);
    }
    const okBtn = document.createElement("button");
    okBtn.style.cssText = "padding:0.5rem 1rem;border-radius:6px;border:none;background:var(--accent);color:#fff;cursor:pointer;font-size:0.85rem;font-weight:500;";
    okBtn.textContent = okLabel;
    okBtn.onclick = () => close(true);
    buttons.appendChild(okBtn);
    box.appendChild(text);
    box.appendChild(buttons);
    overlay.appendChild(box);
    document.body.appendChild(overlay);
    overlay.addEventListener("click", e => { if (e.target === overlay) close(false); });
  });
}


export function _showConfigModal(cfg) {
  const overlay = document.createElement("div");
  overlay.className = "cfg-overlay";

  const SECTION_LABELS = {
    world: "World",
    population: "Population",
    food: "Food & Energy",
    artifacts: "Artifacts",
    agents: "Agents",
  };
  const KEY_LABELS = {
    // world
    world_type: "World type", topology: "Topology",
    agent_network_hocon_path: "Agent network HOCON",
    grid_size: "Current grid size", vision_radius: "Agent vision radius",
    hop_radius: "Hop radius", nodes: "Nodes",
    // population
    min_agents: "Minimum agents", agent_lifespan: "Agent lifespan",
    init_agent_energy: "Starting energy",
    reproduction: "Reproduction enabled", reproduction_cost: "Reproduction cost",
    // food
    enabled: "Enabled", spawn_rate: "Food spawned per step", decay_rate: "Decay rate",
    // artifacts
    creation: "Creation enabled", creation_cost: "Creation cost", inert: "Inert (no interaction)",
    // agents
    max_history: "History depth", internal_memory: "Internal memory",
  };

  function fmtVal(v) {
    if (v === true)  return `<span class="cfg-val cfg-val--true">yes</span>`;
    if (v === false) return `<span class="cfg-val cfg-val--false">no</span>`;
    return `<span class="cfg-val">${escHtml(String(v))}</span>`;
  }

  let sectionsHtml = "";
  for (const [sec, vals] of Object.entries(cfg)) {
    if (typeof vals !== "object" || vals === null) continue;
    const rows = Object.entries(vals)
      .filter(([, v]) => v !== null && v !== undefined)
      .map(([k, v]) => `<div class="cfg-row"><span class="cfg-key">${escHtml(KEY_LABELS[k] || k)}</span>${fmtVal(v)}</div>`)
      .join("");
    if (!rows) continue;
    sectionsHtml += `<div><div class="cfg-section-title">${escHtml(SECTION_LABELS[sec] || sec)}</div>${rows}</div>`;
  }

  overlay.innerHTML = `
    <div class="cfg-box">
      <div class="cfg-header">
        <span>⚙ Simulation Config</span>
        <button class="cfg-close" id="_cfgClose">✕</button>
      </div>
      <div class="cfg-body">${sectionsHtml}</div>
    </div>`;
  document.body.appendChild(overlay);
  overlay.querySelector("#_cfgClose").onclick = () => overlay.remove();
  overlay.addEventListener("click", e => { if (e.target === overlay) overlay.remove(); });
}
