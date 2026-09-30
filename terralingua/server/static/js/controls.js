import { state, familyTags, escHtml } from './state.js';
import { agentStore } from './agents.js';
import { drawGrid, getGraphNodePositions } from './grid.js';
import { renderRoster, selectAgent, deselectAgent } from './roster.js';
import { _computeFamilyTags, renderHistoryList, renderFamilyTree, renderAnalysis, updateFollowDetail } from './follow.js';
import { myArtifacts, _saveMyArtifacts,
         renderArtifacts, openArtDetail, closeArtDetail } from './artifacts.js';
import { _confirm, _showConfigModal } from './modals.js';
import { snapshotElement } from './screenshot.js';
import { openArtDrawerEdit, stopPicking } from './drawers.js';
import { wsRequest } from './ws.js';

// ── Roster ────────────────────────────────────────────────────────────────────
document.getElementById("killAllBtn").addEventListener("click", async () => {
  const mine = [...agentStore.keys()].map(t => [t, agentStore.get(t)]).filter(([,d]) => !d.isHuman);
  if (!mine.length) return;
  const n = mine.length;
  if (!await _confirm(`Kill all ${n} of your agent${n===1?"":"s"}? They will die on the next simulation step.`, "Kill All", "Cancel")) return;
  await Promise.allSettled(mine.map(([tag, data]) => wsRequest("kill_agent", {tag, token: data.token})));
  deselectAgent();
});

document.getElementById("traitsBtn").addEventListener("click", () => {
  state.traitsExpanded = !state.traitsExpanded;
  document.getElementById("traitsBtn").classList.toggle("on", state.traitsExpanded);
  if(state.lastFrame) updateFollowDetail(state.lastFrame.agents||[]);
});

document.getElementById("familyBtn").addEventListener("click", () => {
  state.familyHighlight = !state.familyHighlight;
  document.getElementById("familyBtn").classList.toggle("on", state.familyHighlight);
  if (state.familyHighlight) {
    state.analysisMode = false;
    document.getElementById("analysisBtn").classList.remove("on");
  }
  familyTags.clear();
  if (state.familyHighlight && state.selectedTag && state.lastFrame) {
    for (const t of _computeFamilyTags(state.selectedTag, state.lastFrame.agents)) familyTags.add(t);
    renderFamilyTree();
  } else {
    renderHistoryList();
  }
  if (state.lastFrame || state.graphReplay) drawGrid(state.lastFrame?.grid_state);
});

document.getElementById("analysisBtn").addEventListener("click", () => {
  state.analysisMode = !state.analysisMode;
  document.getElementById("analysisBtn").classList.toggle("on", state.analysisMode);
  if (state.analysisMode) {
    state.familyHighlight = false;
    document.getElementById("familyBtn").classList.remove("on");
    familyTags.clear();
    renderAnalysis();
    if (state.lastFrame || state.graphReplay) drawGrid(state.lastFrame?.grid_state);
  } else {
    renderHistoryList();
  }
});

// ── Artifacts ────────────────────────────────────────────────────────────────
document.getElementById("showExpiredBtn").addEventListener("click", () => {
  state._showExpired = !state._showExpired;
  document.getElementById("showExpiredBtn").classList.toggle("on", state._showExpired);
  if (state.lastFrame) renderArtifacts(state.lastFrame.artifacts||[], state.lastFrame.expired_artifacts||[]);
});

document.getElementById("showDeadBtn").addEventListener("click", () => {
  state._showDead = !state._showDead;
  document.getElementById("showDeadBtn").classList.toggle("on", state._showDead);
  if (state.lastFrame) renderRoster(state.lastFrame.agents, selectAgent, state.lastFrame.dead_agents||[]);
});

document.getElementById("artDetailCloseBtn").addEventListener("click",closeArtDetail);
document.getElementById("artDetailBody").addEventListener("click", e => {
  const btn = e.target.closest(".art-access-name[data-tag]");
  if(btn) selectAgent(btn.dataset.tag);
});
document.getElementById("artDetailEditBtn").addEventListener("click",()=>{ if(state.selectedArtifact) openArtDrawerEdit(state.selectedArtifact); });
document.getElementById("artDetailDeleteBtn").addEventListener("click",async()=>{
  const art=state.selectedArtifact; if(!art) return;
  if(!await _confirm(`Delete artifact "${art.name}"? It will disappear at the next timestep.`, "Delete", "Cancel")) return;
  try{
    await wsRequest("destroy_artifact",{name:art.name});
    myArtifacts.delete(art.name); _saveMyArtifacts();
    closeArtDetail();
  }catch(err){ alert(err.message); }
});
document.getElementById("followCloseBtn").addEventListener("click",deselectAgent);

document.getElementById("snapshotMsgBtn").addEventListener("click", function() {
  snapshotElement(document.getElementById("msgSection"), this);
});

// ── Theme ─────────────────────────────────────────────────────────────────────
(function(){
  const btn       = document.getElementById("themeBtn");
  const stateEl   = document.getElementById("themeBtnState");
  const saved     = localStorage.getItem("ogw-theme");
  if(saved === "light") {
    document.body.classList.add("light");
    btn.classList.add("on");
    stateEl.textContent = "Light";
  }
  btn.addEventListener("click", function(){
    const isLight = document.body.classList.toggle("light");
    stateEl.textContent = isLight ? "Light" : "Dark";
    btn.classList.toggle("on", isLight);
    localStorage.setItem("ogw-theme", isLight ? "light" : "dark");
    if(state.lastFrame || state.graphReplay) drawGrid(state.lastFrame?.grid_state);
  });
})();


// ── Cell tooltip ──────────────────────────────────────────────────────────────
function showCellTooltip(clientX, clientY, html){
  const tip = document.getElementById("cellTooltip");
  tip.innerHTML = html;
  tip.style.left = "-9999px";
  tip.style.top  = "-9999px";
  tip.style.display = "block";
  const {width, height} = tip.getBoundingClientRect();
  const x = clientX + 14 + width > window.innerWidth  - 8 ? clientX - width  - 8 : clientX + 14;
  const y = clientY + 14 + height > window.innerHeight - 8 ? clientY - height - 8 : clientY + 14;
  tip.style.left = Math.max(8, x) + "px";
  tip.style.top  = Math.max(8, y) + "px";
}
function hideCellTooltip(){
  document.getElementById("cellTooltip").style.display = "none";
}
document.addEventListener("click", e => {
  if(!e.target.closest("#grid-canvas") && !e.target.closest("#cellTooltip")) hideCellTooltip();
});
document.getElementById("cellTooltip").addEventListener("click", e => {
  const btn = e.target.closest("[data-tip-action]");
  if(!btn) return;
  hideCellTooltip();
  if(btn.dataset.tipAction === "agent"){
    selectAgent(btn.dataset.tipTag);
  } else if(btn.dataset.tipAction === "artifact"){
    const art = (state.lastFrame?.artifacts||[]).find(a => a.name === btn.dataset.tipName);
    if(art){ openArtDetail(art); deselectAgent(); }
  }
});

// ── Canvas click-to-select ────────────────────────────────────────────────────
(()=>{
  const canvas = document.getElementById("grid-canvas");
  function canvasPoint(e){
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width, scaleY = canvas.height / rect.height;
    const cx = (e.clientX - rect.left) * scaleX;
    const cy = (e.clientY - rect.top) * scaleY;
    return {
      x: (cx - state.mapPanX) / state.mapZoom,
      y: (cy - state.mapPanY) / state.mapZoom,
    };
  }
  function canvasToGrid(e){
    const gs = state.lastFrame.grid_state.grid_size;
    const cW = canvas.width / gs, cH = canvas.height / gs;
    const pt = canvasPoint(e);
    const wrap = (n, m) => ((n % m) + m) % m;
    return {
      row: wrap(Math.floor(pt.y / cH), gs),
      col: wrap(Math.floor(pt.x / cW), gs),
    };
  }
  function graphHitAt(e){
    const graphData = state.lastFrame.grid_state;
    const pt = canvasPoint(e);
    const positions = getGraphNodePositions(graphData, canvas.width, canvas.height);
    const radius = Math.max(18, Math.min(canvas.width, canvas.height) * 0.045);
    let bestNode = null, bestDist = Infinity;
    for (const [node, pos] of positions.entries()) {
      const dist = Math.hypot(pt.x - pos.x, pt.y - pos.y);
      if (dist < bestDist) { bestDist = dist; bestNode = node; }
    }
    if (!bestNode || bestDist > radius) return null;
    const hitAgents = graphData.agents.filter(ag => ag.node === bestNode);
    const hitFood = (graphData.food || []).find(f => f.node === bestNode);
    const hitArts = (state.lastFrame.artifacts || [])
      .filter(a => a.pose === bestNode || a.node === bestNode);
    return { node: bestNode, hitAgents, hitFood, hitArts };
  }
  function handleGraphClick(e){
    const hit = graphHitAt(e);
    if(!hit){
      hideCellTooltip();
      deselectAgent();
      return;
    }
    const { hitAgents, hitFood, hitArts, node } = hit;
    const selectable = hitAgents.length + hitArts.length;
    if(!selectable && !hitFood){ hideCellTooltip(); deselectAgent(); return; }
    if(selectable === 1 && !hitFood){
      if(hitAgents.length){
        const ag = hitAgents[0];
        if(ag.tag === state.selectedTag){ deselectAgent(); hideCellTooltip(); }
        else {
          const agFull = state.lastFrame.agents.find(a => a.tag === ag.tag);
          showCellTooltip(e.clientX, e.clientY,
            `<div class="tip-type" style="color:var(--accent)">Agent &nbsp; `+
            `time ${agFull?.time_left??"-"} energy ${agFull?.energy??"-"}</div>`+
            `<div class="tip-name">${escHtml(ag.name||ag.tag)}</div>`+
            `<div class="tip-type">Node ${escHtml(node)}</div>`
          );
          selectAgent(ag.tag);
        }
      } else {
        showCellTooltip(e.clientX, e.clientY,
          `<div class="tip-type" style="color:#fb923c">Artifact</div>`+
          `<div class="tip-name">${escHtml(hitArts[0].name)}</div>`+
          `<div class="tip-type">Node ${escHtml(node)}</div>`
        );
        openArtDetail(hitArts[0]); deselectAgent();
      }
      return;
    }
    const parts = [
      ...hitAgents.map(ag => {
        const agFull = state.lastFrame.agents.find(a => a.tag === ag.tag);
        return `<div class="tip-type" style="color:var(--accent)">Agent &nbsp; `+
          `time ${agFull?.time_left??"-"} energy ${agFull?.energy??"-"}</div>`+
          `<button class="tip-link" data-tip-action="agent" `+
          `data-tip-tag="${escHtml(ag.tag)}">${escHtml(ag.name||ag.tag)}</button>`;
      }),
      ...hitArts.map(art =>
        `<div class="tip-type" style="color:#fb923c">Artifact</div>`+
        `<button class="tip-link" data-tip-action="artifact" `+
        `data-tip-name="${escHtml(art.name)}">${escHtml(art.name)}</button>`),
      ...(hitFood ? [
        `<div class="tip-type" style="color:var(--green)">Food</div>`+
        `<div class="tip-name">${hitFood.value??hitFood.ratio.toFixed(2)}</div>`] : []),
    ];
    showCellTooltip(e.clientX, e.clientY, parts.join('<hr class="tip-sep">'));
  }
  canvas.addEventListener("click", (e) => {
    if(!state.lastFrame) return;
    if(state.panHasMoved){ state.panHasMoved=false; return; }
    if(state.lastFrame.grid_state.env_type === "graph"){
      handleGraphClick(e);
      return;
    }
    const {row, col} = canvasToGrid(e);
    if(state.pickingFor){
      if(state.pickingFor==="artifact"){ document.getElementById("aRow").value=row; document.getElementById("aCol").value=col; }
      else if(state.pickingFor==="agent"){ document.getElementById("dRow").value=row; document.getElementById("dCol").value=col; }
      stopPicking();
      hideCellTooltip();
      return;
    }
    const hitAgents = state.lastFrame.grid_state.agents.filter(ag => ag.x===row && ag.y===col);
    const hitFood   = (state.lastFrame.grid_state.food||[]).find(f => f.x===row && f.y===col);
    const hitArts   = (state.lastFrame.artifacts||[]).filter(a => a.pose && a.pose[0]===row && a.pose[1]===col);
    const selectable = hitAgents.length + hitArts.length;
    if(!selectable && !hitFood){ hideCellTooltip(); deselectAgent(); return; }
    if(selectable === 1 && !hitFood){
      if(hitAgents.length){
        const ag = hitAgents[0];
        if(ag.tag === state.selectedTag){ deselectAgent(); hideCellTooltip(); }
        else {
          const agFull = state.lastFrame.agents.find(a => a.tag === ag.tag);
          showCellTooltip(e.clientX, e.clientY,
            `<div class="tip-type" style="color:var(--accent)">Agent &nbsp; ⏱ ${agFull?.time_left??"-"} ⚡ ${agFull?.energy??"-"}</div>`+
            `<div class="tip-name">${escHtml(ag.name||ag.tag)}</div>`
          );
          selectAgent(ag.tag);
        }
      } else {
        showCellTooltip(e.clientX, e.clientY,
          `<div class="tip-type" style="color:#fb923c">Artifact</div>`+
          `<div class="tip-name">${escHtml(hitArts[0].name)}</div>`
        );
        openArtDetail(hitArts[0]); deselectAgent();
      }
      return;
    }
    if(selectable === 0){
      showCellTooltip(e.clientX, e.clientY,
        `<div class="tip-type" style="color:var(--green)">Food</div>`+
        `<div class="tip-name">${hitFood.value??hitFood.ratio.toFixed(2)}</div>`);
      deselectAgent(); return;
    }
    // Multiple selectable entities — interactive tooltip
    const parts = [
      ...hitAgents.map(ag => {
        const agFull = state.lastFrame.agents.find(a => a.tag === ag.tag);
        return `<div class="tip-type" style="color:var(--accent)">Agent &nbsp; ⏱ ${agFull?.time_left??"-"} ⚡ ${agFull?.energy??"-"}</div>`+
          `<button class="tip-link" data-tip-action="agent" data-tip-tag="${escHtml(ag.tag)}">${escHtml(ag.name||ag.tag)}</button>`;
      }),
      ...hitArts.map(art =>
        `<div class="tip-type" style="color:#fb923c">Artifact</div>`+
        `<button class="tip-link" data-tip-action="artifact" data-tip-name="${escHtml(art.name)}">${escHtml(art.name)}</button>`),
      ...(hitFood ? [
        `<div class="tip-type" style="color:var(--green)">Food</div>`+
        `<div class="tip-name">${hitFood.value??hitFood.ratio.toFixed(2)}</div>`] : []),
    ];
    showCellTooltip(e.clientX, e.clientY, parts.join('<hr class="tip-sep">'));
  });
  canvas.addEventListener("mousemove", (e) => {
    if (state.graphReplay) { canvas.style.cursor = "pointer"; return; }
    if(!state.lastFrame) return;
    if(state.lastFrame.grid_state.env_type === "graph"){
      state.hoverCell = null;
      const hit = graphHitAt(e);
      canvas.style.cursor = hit && (hit.hitAgents.length || hit.hitArts.length)
        ? "pointer"
        : "default";
      return;
    }
    const {row, col} = canvasToGrid(e);
    if(state.pickingFor){
      canvas.style.cursor = "crosshair";
      state.hoverCell = {row, col};
      drawGrid(state.lastFrame.grid_state);
      return;
    }
    state.hoverCell = null;
    const hit = state.lastFrame.grid_state.agents.find(ag => ag.x===row && ag.y===col);
    canvas.style.cursor = hit ? "pointer" : "default";
  });
  canvas.addEventListener("mouseleave", () => {
    if(state.pickingFor){ state.hoverCell=null; if(state.lastFrame || state.graphReplay) drawGrid(state.lastFrame?.grid_state); }
  });
})();

// ── Toggles ───────────────────────────────────────────────────────────────────
document.getElementById("namesBtn").addEventListener("click",function(){
  state.showNames=!state.showNames;
  document.getElementById("namesBtnState").textContent = state.showNames ? "ON" : "OFF";
  this.classList.toggle("on", state.showNames);
  if(state.lastFrame || state.graphReplay) drawGrid(state.lastFrame?.grid_state);
});
document.getElementById("snapshotBtn").addEventListener("click", function() {
  snapshotElement(document.getElementById("main"), this);
});
document.getElementById("configBtn").addEventListener("click", async function() {
  try {
    const data = await wsRequest("get_config");
    _showConfigModal(data);
  } catch(e) {
    await _confirm(`Could not load config: ${e.message}`, "OK", "");
  }
});
(function(){
  const allBtn  = document.getElementById("filterAllBtn");
  const mineBtn = document.getElementById("filterMineBtn");
  function setFilter(mine){
    state.filterMine = mine;
    allBtn.classList.toggle("on", !mine);
    mineBtn.classList.toggle("on", mine);
    if (state.lastFrame) renderRoster(state.lastFrame.agents, selectAgent, state.lastFrame.dead_agents||[]);
  }
  allBtn .addEventListener("click", () => setFilter(false));
  mineBtn.addEventListener("click", () => setFilter(true));
})();

// ── Arrow-key navigation of the selected agent / artifact ──────────────────
// Up/Down moves the selection to the previous/next visible row in whichever
// list the user last clicked (state.lastListNav). Respects current filters
// (All/Mine, Dead, Expired, search). No-op when no row is selected in the
// target list or when the user is typing in an input field.
document.addEventListener("keydown", (e) => {
  if (e.key !== "ArrowUp" && e.key !== "ArrowDown") return;
  const t = document.activeElement?.tagName;
  if (t === "INPUT" || t === "TEXTAREA" || t === "SELECT") return;

  // Pick the target list. Honor the last-clicked context if that list still
  // has a selection; otherwise fall back to whichever single list is selected.
  let mode = state.lastListNav;
  const haveAgent    = !!state.selectedTag;
  const haveArtifact = !!state.selectedArtifact;
  if (mode === "agents"    && !haveAgent)    mode = null;
  if (mode === "artifacts" && !haveArtifact) mode = null;
  if (!mode) {
    if      (haveAgent    && !haveArtifact) mode = "agents";
    else if (haveArtifact && !haveAgent)    mode = "artifacts";
    else return;
  }

  if (mode === "agents") {
    const list = document.getElementById("rosterList");
    const rows = Array.from(list.querySelectorAll(".agent-row"))
      .filter(r => r.style.display !== "none" && r.dataset.tag);
    const i = rows.findIndex(r => r.dataset.tag === state.selectedTag);
    if (i < 0) return;
    const next = e.key === "ArrowDown" ? rows[i + 1] : rows[i - 1];
    if (!next) return;
    e.preventDefault();
    selectAgent(next.dataset.tag);
    next.scrollIntoView({ block: "nearest" });
  } else if (mode === "artifacts") {
    const list = document.getElementById("artifactList");
    const rows = Array.from(list.querySelectorAll(".art-row"))
      .filter(r => r.style.display !== "none" && r._art);
    const i = rows.findIndex(r => r.dataset.name === state.selectedArtifact.name);
    if (i < 0) return;
    const next = e.key === "ArrowDown" ? rows[i + 1] : rows[i - 1];
    if (!next || !next._art) return;
    e.preventDefault();
    openArtDetail(next._art);
    next.scrollIntoView({ block: "nearest" });
  }
});

// ── Canvas: initial connecting state ─────────────────────────────────────────
(()=>{
  const canvas = document.getElementById("grid-canvas");
  const ctx = canvas.getContext("2d");
  const dark = !document.body.classList.contains("light");
  ctx.fillStyle = dark ? "#0d1117" : "#f0f2f5";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = dark ? "#3a3f5a" : "#b0b4c8";
  ctx.font = '500 14px "Inter", system-ui, sans-serif';
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("Connecting…", canvas.width / 2, canvas.height / 2);
})();

// ── Canvas auto-resize ────────────────────────────────────────────────────────
(()=>{
  const canvas  = document.getElementById("grid-canvas");
  const wrap    = document.getElementById("gridWrap");
  const padding = 8; // matches 0.5rem padding on gridWrap
  const legend = document.getElementById("gridLegend");
  const banner = document.getElementById("pickBanner");
  const observer = new ResizeObserver(()=>{
    const extraHeight = legend.offsetHeight + banner.offsetHeight + 14;
    const size = Math.max(100, Math.min(wrap.clientWidth - padding*2, wrap.clientHeight - padding*2 - extraHeight));
    if(canvas.width===size && canvas.height===size) return;
    canvas.width=size; canvas.height=size;
    if(state.lastFrame || state.graphReplay) drawGrid(state.lastFrame?.grid_state);
  });
  observer.observe(wrap);
  observer.observe(legend);
  observer.observe(banner);
})();

// ── Canvas zoom & pan ─────────────────────────────────────────────────────────
(()=>{
  const canvas = document.getElementById("grid-canvas");
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width, scaleY = canvas.height / rect.height;
    const cx = (e.clientX - rect.left) * scaleX;
    const cy = (e.clientY - rect.top) * scaleY;
    const factor = e.deltaY < 0 ? 1.2 : 1/1.2;
    const newZoom = Math.max(0.5, Math.min(10, state.mapZoom * factor));
    state.mapPanX = cx - (cx - state.mapPanX) * newZoom / state.mapZoom;
    state.mapPanY = cy - (cy - state.mapPanY) * newZoom / state.mapZoom;
    state.mapZoom = newZoom;
    if(state.lastFrame || state.graphReplay) drawGrid(state.lastFrame?.grid_state);
  }, { passive: false });

  let isPanning = false, panStartX = 0, panStartY = 0, panStartMapX = 0, panStartMapY = 0;
  canvas.addEventListener("mousedown", (e) => {
    if(e.button !== 0 || state.pickingFor) return;
    isPanning = true; state.panHasMoved = false;
    panStartX = e.clientX; panStartY = e.clientY;
    panStartMapX = state.mapPanX; panStartMapY = state.mapPanY;
    canvas.style.cursor = "grab";
  });
  window.addEventListener("mousemove", (e) => {
    if(!isPanning) return;
    const dx = e.clientX - panStartX, dy = e.clientY - panStartY;
    if(!state.panHasMoved && Math.abs(dx) < 4 && Math.abs(dy) < 4) return;
    state.panHasMoved = true;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width, scaleY = canvas.height / rect.height;
    state.mapPanX = panStartMapX + dx * scaleX;
    state.mapPanY = panStartMapY + dy * scaleY;
    canvas.style.cursor = "grabbing";
    if(state.lastFrame || state.graphReplay) drawGrid(state.lastFrame?.grid_state);
  });
  window.addEventListener("mouseup", (e) => {
    if(e.button !== 0 || !isPanning) return;
    isPanning = false;
    if(!state.pickingFor) canvas.style.cursor = "default";
  });
  canvas.addEventListener("dblclick", () => {
    state.mapZoom=1.0; state.mapPanX=0; state.mapPanY=0;
    if(state.lastFrame || state.graphReplay) drawGrid(state.lastFrame?.grid_state);
  });

  function zoomAround(factor){
    const cx = canvas.width / 2, cy = canvas.height / 2;
    const newZoom = Math.max(0.5, Math.min(10, state.mapZoom * factor));
    state.mapPanX = cx - (cx - state.mapPanX) * newZoom / state.mapZoom;
    state.mapPanY = cy - (cy - state.mapPanY) * newZoom / state.mapZoom;
    state.mapZoom = newZoom;
    if(state.lastFrame || state.graphReplay) drawGrid(state.lastFrame?.grid_state);
  }
  document.getElementById("zoomInBtn").addEventListener("click",  () => zoomAround(1.3));
  document.getElementById("zoomOutBtn").addEventListener("click",  () => zoomAround(1/1.3));
  document.getElementById("zoomResetBtn").addEventListener("click", () => {
    state.mapZoom=1.0; state.mapPanX=0; state.mapPanY=0;
    if(state.lastFrame || state.graphReplay) drawGrid(state.lastFrame?.grid_state);
  });
})();

// ── Row resize handles ────────────────────────────────────────────────────────
document.querySelectorAll('.row-resize').forEach(handle => {
  handle.addEventListener('mousedown', e => {
    e.preventDefault();
    const topSection = handle.previousElementSibling;
    const startY = e.clientY;
    const startH = topSection.getBoundingClientRect().height;
    handle.classList.add('dragging');
    const onMove = e => {
      const colH = handle.parentElement.getBoundingClientRect().height;
      let bottomSection = handle.nextElementSibling;
      while (bottomSection && getComputedStyle(bottomSection).display === 'none')
        bottomSection = bottomSection.nextElementSibling;
      const minBottom = bottomSection
        ? (bottomSection.querySelector('.sec-header')?.offsetHeight ?? 0)
        : 0;
      const minTop = topSection.querySelector('.sec-header')?.offsetHeight ?? 0;
      const newH = Math.min(colH - 4 - minBottom, Math.max(minTop, startH + (e.clientY - startY)));
      topSection.style.flex = `0 0 ${newH}px`;
    };
    const onUp = () => {
      handle.classList.remove('dragging');
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
    };
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  });
});

// ── Column resize handles ──────────────────────────────────────────────────────
document.querySelectorAll('.col-resize').forEach(handle => {
  handle.addEventListener('mousedown', e => {
    e.preventDefault();
    let leftCol  = handle.previousElementSibling;
    let rightCol = handle.nextElementSibling;
    while (leftCol  && getComputedStyle(leftCol).display  === 'none') leftCol  = leftCol.previousElementSibling;
    while (rightCol && getComputedStyle(rightCol).display === 'none') rightCol = rightCol.nextElementSibling;
    if (!leftCol || !rightCol) return;
    const startX   = e.clientX;
    const startL   = leftCol.getBoundingClientRect().width;
    const startR   = rightCol.getBoundingClientRect().width;
    handle.classList.add('dragging');
    const onMove = e => {
      const dx = e.clientX - startX;
      const newL = Math.max(140, startL + dx);
      const newR = Math.max(140, startR - dx);
      leftCol.style.flex  = `0 0 ${newL}px`;
      rightCol.style.flex = `0 0 ${newR}px`;
    };
    const onUp = () => {
      handle.classList.remove('dragging');
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
    };
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  });
});
