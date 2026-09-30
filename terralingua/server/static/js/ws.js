import { state, agentHistory,
         allMessages, seenAgentNames, humanPendingPrompts, humanWaiting,
         WS_BASE, DASHBOARD_TOKEN } from './state.js';
import { getAnonId } from './crypto.js';
import { drawGrid } from './grid.js';
import { accumulateMessages, renderMessages } from './messages.js';
import { renderRoster, selectAgent } from './roster.js';
import { renderHistoryList, updateFollowDetail, renderFamilyTree, renderAnalysis } from './follow.js';
import { myArtifacts, myArtifactVersions, _saveMyArtifacts, _loadMyArtifacts,
         renderArtifacts, openArtDetail, refreshSelectedArtifact, setArtifactAnalysisData } from './artifacts.js';
import { agentStore } from './agents.js';
import { renderHumanControlPanel } from './human.js';
import { updateAgentList, onFieldNote, onFieldNotesSnapshot, onPostmortem, onPostmortemUnavailable, onObituariesIndex, onLiveAnnotations, clearLiveAnnotationHistory } from './anthropologist.js';
import { reconnectUserAgents } from './auth.js';
import { addStepError, flushStepErrors, showAnthroErrorToast } from './notifications.js';

// ── Dashboard WebSocket RPC layer ─────────────────────────────────────────────
let _dashWS = null;
const _dashPending = new Map();
let _heartbeatInterval = null;
const HEARTBEAT_INTERVAL_MS = 5000;

// Labels shown in the buffered error notice when an anthropologist LLM call
// fails at runtime (rate limit, invalid key, …). Mirrors the operation
// strings emitted by the anthropologist server.
const _ANTHRO_OP_LABELS = {
  postmortem: "Anthropologist (postmortem)",
  artifact_phylogeny: "Anthropologist (phylogeny)",
};

function _genReqId() {
  return ([1e7]+-1e3+-4e3+-8e3+-1e11).replace(/[018]/g, c =>
    (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16)
  );
}

export function wsRequest(cmd, params = {}) {
  return new Promise((resolve, reject) => {
    if (!_dashWS || _dashWS.readyState !== WebSocket.OPEN) {
      reject(new Error("Dashboard WebSocket not connected"));
      return;
    }
    const req = _genReqId();
    _dashPending.set(req, { resolve, reject });
    _dashWS.send(JSON.stringify({ req, cmd, ...params }));
    setTimeout(() => {
      if (_dashPending.has(req)) {
        _dashPending.delete(req);
        reject(new Error(`Request '${cmd}' timed out`));
      }
    }, 30000);
  });
}

// ── Sim watchdog ──────────────────────────────────────────────────────────────
function _resetSimWatchdog() {
  const now = Date.now();
  if(state._lastFrameTime !== null){
    const iv = now - state._lastFrameTime;
    state._frameInterval = state._frameInterval ? state._frameInterval * 0.7 + iv * 0.3 : iv;
  }
  state._lastFrameTime = now;
  clearTimeout(state._simOfflineTimer);
  const timeout = Math.max(10_000, (state._frameInterval || 10_000) * 3);
  state._simOfflineTimer = setTimeout(() => {
    state._simStale = true;
    document.getElementById("sseDot").className = "error";
    document.getElementById("sseStatus").textContent = "Simulation offline";
  }, timeout);
}

function _clearSimWatchdog() {
  clearTimeout(state._simOfflineTimer);
  state._simStale = state._lastFrameTime = state._frameInterval = null;
}

// ── Full message history ──────────────────────────────────────────────────────
async function fetchAllMessages(since=0){
  try{ return (await wsRequest("get_messages",{since})).messages||[]; }
  catch(_){ return []; }
}

// ── Agent log (server-backed) ─────────────────────────────────────────────────
async function fetchAgentLog(tag, since=0){
  try{ return (await wsRequest("get_agent_log",{tag,since})).entries||[]; }
  catch(_){ return []; }
}

export async function refreshAgentHistory(tag){
  if(!tag) return;
  const mySeq=state.historyFetchSeq;
  const since=state.historyLastStep>=0?state.historyLastStep+1:0;
  const entries=await fetchAgentLog(tag,since);
  if(state.selectedTag!==tag) return;
  if(mySeq!==state.historyFetchSeq) return;
  const newEntries=entries.filter(e=>e.step>state.historyLastStep);
  if(!newEntries.length) return;
  const hist=agentHistory.get(tag)||[];
  const combined=[...hist,...newEntries];
  agentHistory.set(tag,combined);
  state.historyLastStep=combined[combined.length-1].step;
  if(!state.familyHighlight && !state.analysisMode) renderHistoryList();
}

// ── WebSocket connection ───────────────────────────────────────────────────────
function drawConnecting() {
  if (state.graphReplay) { drawGrid(); return; }
  const canvas = document.getElementById("grid-canvas");
  const ctx = canvas.getContext("2d");
  const dark = !document.body.classList.contains("light");
  ctx.fillStyle = dark ? "#10131f" : "rgb(245,245,245)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = dark ? "rgba(255,255,255,0.18)" : "rgba(0,0,0,0.25)";
  ctx.font = '500 14px "Inter",sans-serif';
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("Connecting…", canvas.width / 2, canvas.height / 2);
}

function safeCall(label, fn) {
  try {
    return fn();
  } catch (e) {
    console.error(`[dashboard] ${label} failed`, e);
    return undefined;
  }
}

function onFrame(f) {
  if(state._simStale){
    state._simStale = false;
    document.getElementById("sseDot").className = "live";
    document.getElementById("sseStatus").textContent = "";
  }
  _resetSimWatchdog();
  // Prune localStorage entries for agents/artifacts that no longer exist
  const liveTags = new Set(f.agents.map(a=>a.tag));
  for (const t of [...agentStore.keys()])
    if (!liveTags.has(t)) agentStore.close(t);
  let artOwnershipChanged = false;
  for(const art of (f.artifacts||[])){
    if(myArtifacts.has(art.name) && art.version > (myArtifactVersions.get(art.name)??0)){
      myArtifacts.delete(art.name); myArtifactVersions.delete(art.name); artOwnershipChanged=true;
    }
  }
  const liveArtNames = new Set((f.artifacts||[]).map(a=>a.name));
  for(const n of [...myArtifacts]) if(!liveArtNames.has(n)){ myArtifacts.delete(n); myArtifactVersions.delete(n); artOwnershipChanged=true; }
  if(artOwnershipChanged) _saveMyArtifacts();
  state.tagToAgent = new Map(f.agents.map(a => [a.tag, a]));
  for(const a of f.agents) seenAgentNames.set(a.tag, a.name);
  state.currentStep = f.step;
  document.getElementById("stepVal").textContent       = f.step.toLocaleString();
  document.getElementById("agentCountVal").textContent = f.agents.length;
  if (f.total_agents_ever != null) document.getElementById("agentTotalVal").textContent = f.total_agents_ever;
  document.title = `TerraLingua (${f.agents.length} agent${f.agents.length===1?"":"s"})`;
  document.getElementById("artCountVal").textContent = (f.artifacts||[]).length;
  if (f.total_artifacts_ever != null) document.getElementById("artTotalVal").textContent = f.total_artifacts_ever;
  const clientEl = document.getElementById("clientCountVal");
  if (clientEl && f.connected_clients != null) clientEl.textContent = f.connected_clients;
  accumulateMessages(f.grid_state.recent_messages||[]);
  safeCall("renderArtifacts", () => renderArtifacts(f.artifacts || [], f.expired_artifacts || []));
  safeCall("drawGrid", () => drawGrid(f.grid_state));
  safeCall("renderRoster", () => renderRoster(f.agents, selectAgent, f.dead_agents||[]));
  safeCall("renderMessages", () => renderMessages());
  safeCall("updateFollowDetail", () => updateFollowDetail(f.agents));
  safeCall("renderHumanControlPanel", () => renderHumanControlPanel());
  safeCall("updateAgentList", () => updateAgentList([
    ...f.agents.map(a => ({ tag: a.tag, name: a.name, alive: true })),
    ...(f.dead_agents || []).map(a => ({ tag: a.tag, name: a.name, alive: false })),
  ]));
  if(state.selectedTag) safeCall("refreshAgentHistory", () => refreshAgentHistory(state.selectedTag));
  if(state.selectedTag && state.familyHighlight) safeCall("renderFamilyTree", () => renderFamilyTree());
  if(state.selectedArtifact){
    const updated = (f.artifacts||[]).find(a=>a.name===state.selectedArtifact.name);
    if(updated) safeCall("refreshSelectedArtifact", () => refreshSelectedArtifact(updated));
  }
}

export function connectDashboardWS() {
  const dot    = document.getElementById("sseDot");
  const status = document.getElementById("sseStatus");
  dot.className = "connecting";
  status.textContent = "Connecting…";
  drawConnecting();

  const ws = new WebSocket(`${WS_BASE}/ws/dashboard?token=${DASHBOARD_TOKEN}`);
  _dashWS = ws;

  ws.onopen = () => {
    dot.className = "live";
    status.textContent = "";
    _clearSimWatchdog();
    reconnectUserAgents();
    _loadMyArtifacts();
    // Anonymous users: keep the server-side bg loops alive while this tab is
    // open. Logged-in users persist regardless, so don't bother heartbeating.
    if (!state.currentUser) {
      const anonId = getAnonId();
      const send = () => {
        if (_dashWS && _dashWS.readyState === WebSocket.OPEN) {
          _dashWS.send(JSON.stringify({ cmd: "heartbeat", anon_id: anonId }));
        }
      };
      send();
      clearInterval(_heartbeatInterval);
      _heartbeatInterval = setInterval(send, HEARTBEAT_INTERVAL_MS);
    }
  };

  ws.onmessage = async (e) => {
    try {
      const msg = JSON.parse(e.data);

      // Response to one of our requests
      if (msg.res !== undefined) {
        const pend = _dashPending.get(msg.res);
        if (pend) {
          _dashPending.delete(msg.res);
          if (msg.ok) pend.resolve(msg.data);
          else pend.reject(new Error(msg.error || "Server error"));
        }
        return;
      }

      // Server push: initial config
      if (msg.type === "config") {
        const allowed = msg.allow_human_agents !== false;
        document.getElementById("dTypeToggle").style.display = allowed ? "" : "none";
        // Load full message history on (re)connect
        const groups = await fetchAllMessages(0);
        accumulateMessages(groups);
        renderMessages();
        return;
      }

      // Server push: child agent spawned. The server already spawned the
      // child's bg loop via _watch_child_agents (logged-in case only); we just
      // mirror the parent's tracking so the dashboard treats the child as ours.
      if (msg.type === "child_agent") {
        if (agentStore.has(msg.parent_tag)) {
          const pd = agentStore.get(msg.parent_tag) || {};
          agentStore.trackAgent(msg.agent_tag, msg.token, { model: pd.model || '' });
        }
        return;
      }

      // Server push: runner lifecycle
      if (msg.type === "runner_status") {
        if (msg.status === "running") {
          dot.className = "live";
          status.textContent = "";
        } else if (msg.status === "stopped") {
          dot.className = "error";
          status.textContent = "Simulation finished";
        }
        return;
      }

      // Server push: anthropologist events
      if (msg.type === "field_notes_snapshot") { onFieldNotesSnapshot(msg); return; }
      if (msg.type === "field_note") { onFieldNote(msg); return; }
      if (msg.type === "agent_postmortem") {
        onPostmortem(msg);
        if (state.analysisMode && state.selectedTag === msg.agent_tag) renderAnalysis();
        return;
      }
      if (msg.type === "agent_postmortem_unavailable") { onPostmortemUnavailable(msg); return; }
      if (msg.type === "obituaries_index") { onObituariesIndex(msg); return; }
      if (msg.type === "live_annotations") {
        onLiveAnnotations(msg);
        if (state.analysisMode && state.selectedTag && msg.annotations[state.selectedTag]) renderAnalysis();
        return;
      }
      if (msg.type === "agent_api_error") {
        const name = state.tagToAgent?.get(msg.agent_tag)?.name || msg.agent_tag;
        addStepError(msg.error_type, msg.error_message, name);
        return;
      }
      // Anthropologist notifications are targeted: the publisher includes
      // target_user_id and/or target_session_token. Drop the message unless
      // it concerns this dashboard.
      if (msg.type === "anthro_error" || msg.type === "anthro_api_error") {
        const myUserId = state.currentUser ? String(state.currentUser.id) : null;
        const userMatch = msg.target_user_id != null
          && myUserId != null
          && String(msg.target_user_id) === myUserId;
        const sessionMatch = msg.target_session_token != null
          && DASHBOARD_TOKEN
          && msg.target_session_token === DASHBOARD_TOKEN;
        if (!userMatch && !sessionMatch) return;
        if (msg.type === "anthro_error") {
          showAnthroErrorToast(msg.operation, msg.reason);
        } else {
          // Runtime LLM failure: feed into the same buffered error notice as
          // agent_api_error so identical error_type strings coalesce.
          const label = _ANTHRO_OP_LABELS[msg.operation] || msg.operation || "anthropologist";
          addStepError(msg.error_type, msg.error_message, label);
        }
        return;
      }

      if (msg.type === "artifact_classifications") {
        setArtifactAnalysisData(msg.categories, msg.phylogeny, msg.done || [], msg.pending || []); return;
      }

      // Server push: simulation frame
      if (msg.type === "frame") {
        // New run detected: clear all state that the frame won't overwrite
        if (state.lastFrame && msg.run_id && msg.run_id !== state.lastFrame.run_id) {
          allMessages.length = 0;
          state.lastMsgStep = -1;
          state.selectedTag = null;
          state.selectedArtifact = null;
          agentStore.clear();
          humanPendingPrompts.clear();
          humanWaiting.clear();
          state.historyFetchSeq++;
          agentHistory.clear();
          seenAgentNames.clear();
          clearLiveAnnotationHistory();
          document.getElementById("followTitle").textContent = "Following — none";
          document.getElementById("followEmpty").style.display = "";
          document.getElementById("followDetail").style.display = "none";
        }
        state.lastFrame = msg;
        document.dispatchEvent(new CustomEvent("dashboard-frame", { detail: msg }));
        onFrame(state.lastFrame);
        flushStepErrors();
      }
    } catch(_) {}
  };

  ws.onerror = () => {
    dot.className = "error";
    status.textContent = "Reconnecting…";
  };

  ws.onclose = (evt) => {
    _dashWS = null;
    _clearSimWatchdog();
    if (_heartbeatInterval) { clearInterval(_heartbeatInterval); _heartbeatInterval = null; }
    for (const [, {reject}] of _dashPending)
      reject(new Error("WebSocket closed"));
    _dashPending.clear();

    if (evt.code === 4001) {
      // Token rejected — need a fresh page load to get a new token
      dot.className = "error";
      status.textContent = "Session expired, reloading…";
      setTimeout(() => window.location.reload(), 2000);
    } else {
      dot.className = "error";
      status.textContent = "Reconnecting…";
      setTimeout(connectDashboardWS, 3000);
    }
  };
}
