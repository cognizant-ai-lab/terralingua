import { state, escHtml, rosterRowCache, deadAgentRowCache, familyTags, agentHistory } from './state.js';
import { agentStore } from './agents.js';
import { drawGrid } from './grid.js';
import { renderMessages } from './messages.js';
import { renderHistoryList, updateFollowDetail, _computeFamilyTags, renderFamilyTree, renderAnalysis } from './follow.js';
import { renderHumanControlPanel } from './human.js';
import { refreshAgentHistory } from './ws.js';

let _rosterSearch = '';
document.getElementById('rosterSearch').addEventListener('input', e => {
  _rosterSearch = e.target.value;
  if (state.lastFrame) renderRoster(state.lastFrame.agents, selectAgent, state.lastFrame.dead_agents || []);
});

export function dirArrow(d){ return {up:"↑",down:"↓",left:"←",right:"→",stay:"·"}[d]||d; }

export function renderRoster(agents, onSelect, deadAgents=[]) {
  const list = document.getElementById("rosterList");
  const killableCount = [...agentStore.values()].filter(d => !d.isHuman).length;
  document.getElementById("killAllBtn").style.display = killableCount > 0 ? "" : "none";
  const all = state.filterMine ? agents.filter(a=>agentStore.has(a.tag)) : agents;
  const visible = [...all.filter(a=>agentStore.has(a.tag)), ...all.filter(a=>!agentStore.has(a.tag))];
  const q = _rosterSearch.toLowerCase().trim();
  const filteredVisible = q ? visible.filter(a =>
    a.name.toLowerCase().includes(q) || (a.type||'').toLowerCase().includes(q)
  ) : visible;
  const filteredDead = q ? deadAgents.filter(a => a.name.toLowerCase().includes(q)) : deadAgents;
  if (!filteredVisible.length && !filteredDead.length) {
    list.innerHTML=`<div class="empty-note">${q ? `No agents match "${escHtml(q)}".` : state.filterMine?"No agents connected from this tab.":"No agents in simulation."}</div>`;
    rosterRowCache.clear(); deadAgentRowCache.clear(); return;
  }
  // Clear any static HTML placeholder on the first real render
  if(rosterRowCache.size===0 && deadAgentRowCache.size===0) list.innerHTML="";
  // ── Alive agents ──
  const newTagSet = new Set(filteredVisible.map(a=>a.tag));
  for(const [tag, {row}] of rosterRowCache)
    if(!newTagSet.has(tag)){ list.removeChild(row); rosterRowCache.delete(tag); }
  for(const ag of filteredVisible){
    const isMine = agentStore.has(ag.tag);
    const act = ag.last_action
      ? `${ag.last_action}${ag.last_params?.direction?" "+dirArrow(ag.last_params.direction):""}` : "-";
    if(rosterRowCache.has(ag.tag)){
      const {row, dotEl, actionEl, statsEl} = rosterRowCache.get(ag.tag);
      row.className = "agent-row"+(ag.tag===state.selectedTag?" selected":"");
      dotEl.className = "agent-dot "+(isMine&&ag.connected?"connected":"disconnected");
      actionEl.textContent = act;
      statsEl.textContent = `⏱ ${ag.time_left??"-"} ⚡ ${ag.energy??"-"}`;
    } else {
      const row = document.createElement("div");
      row.className = "agent-row"+(ag.tag===state.selectedTag?" selected":"");
      row.dataset.tag = ag.tag;
      const dotEl = document.createElement("div");
      dotEl.className = "agent-dot "+(isMine&&ag.connected?"connected":"disconnected");
      const nameEl = document.createElement("div"); nameEl.className="agent-name"; nameEl.textContent=ag.name;
      const typeEl = document.createElement("div"); typeEl.className="agent-type"; typeEl.textContent=ag.type.replace("Agent","");
      const actionEl = document.createElement("div"); actionEl.className="agent-action"; actionEl.textContent=act;
      const statsEl = document.createElement("div"); statsEl.className="agent-stats"; statsEl.textContent=`⏱${ag.time_left??"-"} ⚡${ag.energy??"-"}`;
      row.append(dotEl, nameEl, typeEl, actionEl, statsEl);
      row.addEventListener("click", ()=>{ state.lastListNav = "agents"; onSelect(ag.tag); });
      rosterRowCache.set(ag.tag, {row, dotEl, actionEl, statsEl});
    }
  }
  for(const ag of filteredVisible) list.appendChild(rosterRowCache.get(ag.tag).row);
  // ── Dead agents ──
  const deadTagSet = new Set(filteredDead.map(a=>a.tag));
  for(const [tag, {row}] of deadAgentRowCache)
    if(!deadTagSet.has(tag)){ if(row.parentNode===list) list.removeChild(row); deadAgentRowCache.delete(tag); }
  for(const ag of filteredDead){
    if(!deadAgentRowCache.has(ag.tag)){
      const row = document.createElement("div");
      row.className = "agent-row agent-row--dead"+(ag.tag===state.selectedTag?" selected":"");
      row.dataset.tag = ag.tag;
      const dotEl = document.createElement("div"); dotEl.className = "agent-dot";
      const nameEl = document.createElement("div"); nameEl.className = "agent-name"; nameEl.textContent = ag.name;
      row.append(dotEl, nameEl);
      row.addEventListener("click", ()=>{ state.lastListNav = "agents"; onSelect(ag.tag); });
      deadAgentRowCache.set(ag.tag, {row});
    } else {
      const {row} = deadAgentRowCache.get(ag.tag);
      row.className = "agent-row agent-row--dead"+(ag.tag===state.selectedTag?" selected":"");
    }
    const {row} = deadAgentRowCache.get(ag.tag);
    row.style.display = state._showDead ? "" : "none";
    if(state._showDead) list.appendChild(row);
  }
}

export async function selectAgent(tag){
  if(state.selectedTag===tag){ deselectAgent(); return; }
  const keepFamily = state.familyHighlight;
  const keepAnalysis = state.analysisMode;
  state.selectedTag=tag;
  state.traitsExpanded=false;
  state.historyLastStep=-1;
  state.historyFetchSeq++;
  agentHistory.delete(tag);
  document.getElementById("followCloseBtn").style.display = "";
  if(keepFamily && !keepAnalysis){
    familyTags.clear();
    if(state.lastFrame)
      for(const t of _computeFamilyTags(tag, state.lastFrame.agents)) familyTags.add(t);
    renderFamilyTree();
  } else if(keepAnalysis){
    familyTags.clear();
    document.getElementById("familyBtn").classList.remove("on");
    state.familyHighlight=false;
    renderAnalysis();
  } else {
    familyTags.clear();
    document.getElementById("familyBtn").classList.remove("on");
    renderHistoryList(); // clear stale DOM immediately before async fetch
  }
  renderHumanControlPanel();
  if(state.lastFrame){
    renderRoster(state.lastFrame.agents, selectAgent, state.lastFrame.dead_agents||[]);
    renderMessages();
    updateFollowDetail(state.lastFrame.agents);
    requestAnimationFrame(() => { if(state.lastFrame) drawGrid(state.lastFrame.grid_state); });
  }
  await refreshAgentHistory(tag);
}

export function deselectAgent(){
  state.selectedTag=null;
  state.familyHighlight=false; familyTags.clear();
  state.analysisMode=false;
  state.traitsExpanded=false;
  document.getElementById("familyBtn").classList.remove("on");
  document.getElementById("analysisBtn").classList.remove("on");
  document.getElementById("traitsBtn").classList.remove("on");
  document.getElementById("analysisBtn").style.display="none";
  state.historyLastStep=-1;
  document.getElementById("followCloseBtn").style.display = "none";
  document.getElementById("followSnapshotBtn").style.display = "none";
  document.getElementById("followTitle").textContent="Following — none";
  document.getElementById("followEmpty").style.display="";
  document.getElementById("followDetail").style.display="none";
  renderHumanControlPanel();
  if(state.lastFrame){
    drawGrid(state.lastFrame.grid_state);
    renderRoster(state.lastFrame.agents, selectAgent, state.lastFrame.dead_agents||[]);
    renderMessages();
  }
}
