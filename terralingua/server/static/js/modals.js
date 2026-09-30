// Modal/dialog utilities used across the dashboard UI.

export function _confirm(message, okLabel = "OK", cancelLabel = "Cancel") {
  return new Promise((resolve) => {
    const overlay = document.createElement("div");
    overlay.style.cssText = "position:fixed;inset:0;background:rgba(0,0,0,0.7);display:flex;align-items:center;justify-content:center;z-index:9999;";
    const box = document.createElement("div");
    box.style.cssText = "background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:1.5rem;max-width:380px;width:90%;";
    box.innerHTML = `
      <p style="font-size:0.9rem;color:var(--text);margin-bottom:1.25rem;line-height:1.5;">${message}</p>
      <div style="display:flex;gap:0.75rem;justify-content:flex-end;">
        ${cancelLabel ? `<button id="_confirmCancelBtn" style="padding:0.5rem 1rem;border-radius:6px;border:1px solid var(--border);background:transparent;color:var(--muted);cursor:pointer;font-size:0.85rem;">${cancelLabel}</button>` : ""}
        <button id="_confirmOkBtn" style="padding:0.5rem 1rem;border-radius:6px;border:none;background:var(--accent);color:#fff;cursor:pointer;font-size:0.85rem;font-weight:500;">${okLabel}</button>
      </div>`;
    overlay.appendChild(box);
    document.body.appendChild(overlay);
    const close = (val) => { overlay.remove(); resolve(val); };
    box.querySelector("#_confirmOkBtn").onclick = () => close(true);
    const cancelBtn = box.querySelector("#_confirmCancelBtn");
    if (cancelBtn) cancelBtn.onclick = () => close(false);
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
    return `<span class="cfg-val">${v}</span>`;
  }

  let sectionsHtml = "";
  for (const [sec, vals] of Object.entries(cfg)) {
    if (typeof vals !== "object" || vals === null) continue;
    const rows = Object.entries(vals)
      .filter(([, v]) => v !== null && v !== undefined)
      .map(([k, v]) => `<div class="cfg-row"><span class="cfg-key">${KEY_LABELS[k] || k}</span>${fmtVal(v)}</div>`)
      .join("");
    if (!rows) continue;
    sectionsHtml += `<div><div class="cfg-section-title">${SECTION_LABELS[sec] || sec}</div>${rows}</div>`;
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
