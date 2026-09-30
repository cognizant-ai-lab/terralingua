// ── Constants ─────────────────────────────────────────────────────────────────
export const SERVER          = window.location.origin;
export const WS_BASE         = SERVER.replace(/^http/, "ws");
export const DASHBOARD_TOKEN = document.querySelector('meta[name="dashboard-token"]')?.content || "";

// ── Scalar state (read/write via state.xxx) ───────────────────────────────────
export const state = {
  lastFrame:                 null,
  graphReplay:               null,
  selectedTag:               null,
  familyHighlight:           false,
  selectedArtifact:          null,
  showNames:                 false,
  filterMine:                false,
  editingTag:                null,
  historyLastStep:           -1,
  historyFetchSeq:           0,
  lastMsgStep:               -1,
  currentStep:               0,
  genomeInfo:                null,
  pickingFor:                null,
  hoverCell:                 null,
  mapZoom:                   1.0,
  mapPanX:                   0,
  mapPanY:                   0,
  panHasMoved:               false,
  tagToAgent:                new Map(),
  actionsInfo:               [],
  _showExpired:              false,
  _showDead:                 false,
  analysisMode:              false,
  traitsExpanded:            false,
  _simOfflineTimer:          null,
  _simStale:                 false,
  _lastFrameTime:            null,
  _frameInterval:            null,
  _humanAgentType:           "remote_llm",
  currentUser:               null,  // null = anonymous; object = logged-in user
  // Which list arrow-key navigation should target. Updated on click of an
  // agent or artifact row; persists until the user clicks the other list.
  lastListNav:               null,  // "agents" | "artifacts" | null
};

// ── Collections (mutated in-place, never reassigned) ──────────────────────────
export const familyTags          = new Set();
export const agentHistory        = new Map(); // tag → [{step,...}]
export const allMessages         = [];        // [{step, messages:[...]}]
export const rosterRowCache      = new Map(); // tag → {row, dotEl, actionEl, statsEl}
export const deadAgentRowCache   = new Map(); // tag → {row}
export const artifactRowCache    = new Map(); // name → row element
export const expiredArtRowCache  = new Map(); // name → row element
export const seenAgentNames      = new Map(); // tag → name (survives death, cleared on new run)
export const humanPendingPrompts = new Map(); // tag → {payload, ws}
export const humanWaiting        = new Map(); // tag → {payload, ws}

// ── Utilities ─────────────────────────────────────────────────────────────────
export function escHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

export function agentName(tag) {
  return state.tagToAgent.get(tag)?.name || seenAgentNames.get(tag) || tag;
}
