import { humanPendingPrompts, humanWaiting, WS_BASE } from './state.js';
import { renderHumanControlPanel } from './human.js';

/**
 * AgentStore: tracks agents the current browser owns or is responsible for.
 *
 * Two flavors:
 *   - LLM agents (logged-in OR anonymous): tracked via `trackAgent(...)`. The
 *     server runs a `server_llm_loop` for them, so the browser does NOT open a
 *     /ws/{tag} WebSocket — entries are just metadata used by the dashboard UI
 *     (kill button, "your agents" badge).
 *   - Human-controlled agents: opened via `openHumanAgent(...)` which DOES hold
 *     a /ws/{tag} WebSocket — observations come over it and the human's chosen
 *     action is sent back over it.
 */
class AgentStore {
  constructor() { this._m = new Map(); }

  has(tag)   { return this._m.has(tag); }
  get(tag)   { return this._m.get(tag); }
  ws(tag)    { return this._m.get(tag)?.ws ?? null; }
  keys()     { return this._m.keys(); }
  values()   { return this._m.values(); }

  update(tag, fields) {
    const entry = this._m.get(tag);
    if (entry) Object.assign(entry, fields);
  }

  // Track an LLM agent in the local store. No WS — the server drives it.
  trackAgent(tag, token, opts = {}) {
    this._m.set(tag, { token, ...opts, isHuman: false });
  }

  openHumanAgent(tag, token, opts = {}) {
    const { _reconnectDelay = 1000 } = opts;
    const ws = new WebSocket(`${WS_BASE}/ws/${tag}?token=${token}`);
    this._m.set(tag, { ws, token, isHuman: true });

    ws.onmessage = (evt) => {
      try {
        const payload = JSON.parse(evt.data); if (payload.step === undefined) return;
        humanPendingPrompts.set(tag, { payload, ws });
        renderHumanControlPanel();
      } catch (_) {}
    };

    ws.onclose = (evt) => {
      humanPendingPrompts.delete(tag); humanWaiting.delete(tag);
      renderHumanControlPanel();
      if (!this._m.has(tag)) return;
      if (evt.code === 4001) { this._m.delete(tag); return; }
      const e = this._m.get(tag); if (e) e.ws = null;
      const nextDelay = Math.min(_reconnectDelay * 2, 16000);
      setTimeout(() => {
        if (this._m.has(tag) && !this._m.get(tag)?.ws)
          this.openHumanAgent(tag, token, { ...opts, _reconnectDelay: nextDelay });
      }, _reconnectDelay);
    };

    ws.onerror = () => {
      humanPendingPrompts.delete(tag); humanWaiting.delete(tag);
      renderHumanControlPanel();
      const e = this._m.get(tag); if (e) e.ws = null;
    };
    return ws;
  }

  // Delete entry BEFORE closing WS so onclose doesn't schedule a reconnect.
  close(tag) {
    const entry = this._m.get(tag);
    this._m.delete(tag);
    if (entry?.ws) try { entry.ws.close(); } catch (_) {}
  }

  clear() {
    const entries = [...this._m.values()];
    this._m.clear();
    for (const { ws } of entries) try { ws?.close(); } catch (_) {}
  }
}

export const agentStore = new AgentStore();
