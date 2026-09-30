import { escHtml } from './state.js';

const _STYLE = `
#ogw-error-notice{position:fixed;bottom:1rem;right:1rem;z-index:9000;background:rgba(239,68,68,0.10);border:1px solid rgba(248,113,113,0.30);color:#f87171;border-radius:8px;padding:0.6rem 0.9rem;font-size:0.8rem;line-height:1.7;max-width:360px;display:none;}
#ogw-error-notice .ogw-err-title{font-weight:600;margin-bottom:0.15rem;}
#ogw-anthro-toasts{position:fixed;top:4.5rem;right:1rem;z-index:9100;display:flex;flex-direction:column;gap:0.5rem;max-width:420px;}
#ogw-anthro-toasts .ogw-anthro-toast{background:#2c0e0e;border:1px solid rgba(248,113,113,0.65);color:#fecaca;border-radius:10px;padding:0.85rem 1.05rem;font-size:0.9rem;line-height:1.5;font-weight:500;cursor:pointer;transition:opacity 0.4s ease;box-shadow:0 6px 18px rgba(0,0,0,0.45);}
#ogw-anthro-toasts .ogw-anthro-toast.fading{opacity:0;}
#ogw-anthro-toasts .ogw-anthro-toast::before{content:"⚠ ";font-weight:700;margin-right:0.15rem;}
`;

let _el = null;
let _toastContainer = null;
const _buffer = new Map(); // errorType -> {label, agents: Set}

function _init() {
  if (_el) return;
  const style = document.createElement("style");
  style.textContent = _STYLE;
  document.head.appendChild(style);
  _el = document.createElement("div");
  _el.id = "ogw-error-notice";
  document.body.appendChild(_el);
  _toastContainer = document.createElement("div");
  _toastContainer.id = "ogw-anthro-toasts";
  document.body.appendChild(_toastContainer);
}

// rawMessage from the first occurrence of each type is kept for display.
export function addStepError(errorType, rawMessage, agentName) {
  _init();
  if (!_buffer.has(errorType)) _buffer.set(errorType, { rawMessage, agents: new Set() });
  _buffer.get(errorType).agents.add(agentName);
}

export function flushStepErrors() {
  _init();
  if (_buffer.size === 0) { _el.style.display = "none"; return; }
  const lines = [..._buffer.values()]
    .map(({ rawMessage, agents }) => `<div>• ${escHtml(rawMessage)} — ${[...agents].map(escHtml).join(", ")}</div>`)
    .join("");
  _el.innerHTML = `<div class="ogw-err-title">⚠ Agent errors</div>${lines}`;
  _el.style.display = "";
  _buffer.clear();
}

const _ANTHRO_OPERATION_MESSAGES = {
  missing_api_key: {
    postmortem: "Cannot run postmortem — set your API key in your dashboard.",
    artifact_phylogeny: "Cannot compute phylogeny — set your API key in your dashboard.",
  },
};

function _anthroErrorText(operation, reason) {
  const byReason = _ANTHRO_OPERATION_MESSAGES[reason];
  if (byReason && byReason[operation]) return byReason[operation];
  return `Anthropologist task failed (${reason || "unknown reason"}).`;
}

// Track at most one live toast per (operation, reason) key so a burst of
// rejections (e.g. user clicks several artifacts before setting the key)
// doesn't stack identical messages — same key just refreshes the fade timer.
const _liveToasts = new Map(); // key -> {node, fadeTimer, removeTimer}

// Public: surface an anthropologist precondition-fail (e.g. missing key) as a
// human-readable text toast for the requester. The dispatcher in ws.js gates
// this on target_user_id / target_session_token so the toast is private.
export function showAnthroErrorToast(operation, reason) {
  _init();
  const key = `${operation}|${reason}`;
  const text = _anthroErrorText(operation, reason);

  // If an identical toast is already on-screen, just reset its timers so it
  // stays visible — don't add a duplicate.
  const existing = _liveToasts.get(key);
  if (existing && existing.node.isConnected) {
    clearTimeout(existing.fadeTimer);
    clearTimeout(existing.removeTimer);
    existing.node.classList.remove("fading");
    existing.fadeTimer = setTimeout(() => { existing.node.classList.add("fading"); }, 8000);
    existing.removeTimer = setTimeout(() => {
      existing.node.remove();
      _liveToasts.delete(key);
    }, 8500);
    return;
  }

  const node = document.createElement("div");
  node.className = "ogw-anthro-toast";
  node.textContent = text;
  node.addEventListener("click", () => {
    node.remove();
    _liveToasts.delete(key);
  });
  _toastContainer.appendChild(node);
  const entry = {
    node,
    fadeTimer: setTimeout(() => { node.classList.add("fading"); }, 8000),
    removeTimer: setTimeout(() => {
      node.remove();
      _liveToasts.delete(key);
    }, 8500),
  };
  _liveToasts.set(key, entry);
}

