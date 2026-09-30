/**
 * anthropologist.js — Anthropologist mode toggle + field notes rendering.
 *
 * Two entry points for field notes:
 *   1. Full anthropologist mode (both trays) — toggled by #anthropoBtn
 *   2. Inline mode (message panel only) — toggled by the #msgModeSeg tabs
 *
 * Obituaries (agent postmortems) are rendered in #anthropoDetailBody when a
 * dead agent is selected in #anthropoAgentList. Live agents show their
 * live annotation commentary there instead.
 */

// ── State ────────────────────────────────────────────────────────────────────

import { addSnapshotBtn, snapshotElement } from './screenshot.js';

let _anthropoMode = false;          // full tray mode active
let _inlineMode   = false;          // standalone field notes in message panel

// Saved column flex values for each mode so layouts are fully independent.
// Each entry: { midOrLeft, grid, rightOrNotes } — null means "use default flex:1".
let _normalLayout = null;
let _anthroLayout = null;

function _saveLayout(midOrLeft, grid, rightOrNotes) {
  return {
    midOrLeft: midOrLeft.style.flex,
    grid:      grid.style.flex,
    rightOrNotes: rightOrNotes.style.flex,
  };
}

function _restoreLayout(layout, midOrLeft, grid, rightOrNotes) {
  if (!layout) return;
  midOrLeft.style.flex    = layout.midOrLeft;
  grid.style.flex         = layout.grid;
  rightOrNotes.style.flex = layout.rightOrNotes;
}
const _fieldNotes = [];             // accumulated events, newest first
const _sevFilter  = new Set([1, 2]); // severity levels currently shown (default: med+high)
let _fnSearch     = '';             // field notes text search term
let _agentSearch  = '';             // obituaries agent name search term
const _obituaries      = {};         // agent_tag → postmortem payload (loaded content)
const _obituariesIndex = new Set();  // agent_tags with obituaries on disk (may not be loaded yet)
const _unavailable     = new Set();  // agent_tags with no log file — postmortem impossible
let _allAgents    = [];             // [{tag, name, alive}] — alive + dead
let _selectedTag  = null;           // selected agent in anthropo left tray
let _unreadNotes  = 0;
let _unreadObitCount = 0;
const _requested       = new Set();  // tags for which we already sent a fetch/generation request
let _wsRequest         = null;       // injected from dashboard.js
const _liveAnnotations = {};         // tag → {text, step}[] (history, oldest first)
const _LS_LIVE_KEY = 'ogw:live_annotation_history';

function _saveLiveHistory() {
  try { localStorage.setItem(_LS_LIVE_KEY, JSON.stringify(_liveAnnotations)); } catch(_) {}
}
function _loadLiveHistory() {
  try {
    const raw = localStorage.getItem(_LS_LIVE_KEY);
    if (raw) Object.assign(_liveAnnotations, JSON.parse(raw));
  } catch(_) {}
}

export function clearLiveAnnotationHistory() {
  for (const k of Object.keys(_liveAnnotations)) delete _liveAnnotations[k];
  try { localStorage.removeItem(_LS_LIVE_KEY); } catch(_) {}
}

// ── Kind metadata ─────────────────────────────────────────────────────────────

const _KIND_META = {
  // ── Graph / social ──
  community_split:            { color: '#fb923c', icon: '⬡' },
  community_merge:            { color: '#6e8efb', icon: '⬡' },
  community_contraction:      { color: '#fb923c', icon: '◎' },
  coalition_shift:            { color: '#a78bfa', icon: '⇄' },
  bridge_agent_lost:          { color: '#f87171', icon: '◈' },
  polarization_spike:         { color: '#f87171', icon: '⚡' },
  polarization_creep:         { color: '#fb923c', icon: '⟶' },
  reciprocity_collapse:       { color: '#fb923c', icon: '↺' },
  interaction_rate_surge:     { color: '#fbbf24', icon: '≋' },
  interaction_rate_collapse:  { color: '#fbbf24', icon: '≋' },
  interaction_rate_erosion:   { color: '#fbbf24', icon: '↘' },
  interaction_rate_growth:    { color: '#34d399', icon: '↗' },
  isolation_wave:             { color: '#f87171', icon: '◌' },
  // ── Agent ──
  agent_isolated:             { color: '#5a607a', icon: '○' },
  agent_emerged_hub:          { color: '#34d399', icon: '●' },
  behavioral_drift:           { color: '#a78bfa', icon: '↝' },
  // ── Artifact ──
  new_institutional_artifact: { color: '#a78bfa', icon: '📜' },
  artifact_category_changed:  { color: '#a78bfa', icon: '◈' },
  artifact_burst:             { color: '#2dd4bf', icon: '📦' },
  artifact_rise:              { color: '#2dd4bf', icon: '↑' },
  artifact_abandoned:         { color: '#5a607a', icon: '⊘' },
  artifact_death:             { color: '#5a607a', icon: '✗' },
  artifact_destroyed:         { color: '#f87171', icon: '✗' },
  contested_artifact:         { color: '#f87171', icon: '⚔' },
  artifact_diffusion:         { color: '#34d399', icon: '⟳' },
};

function _kindLabel(e) {
  if (e.kind === 'llm_single_emergence') {
    const kws = (e.data?.keywords || []).filter(k => k !== 'none');
    if (kws.length) return kws.join(' · ').replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    return 'Single Emergence';
  }
  return e.kind.replace(/^llm_/, '').replace(/_/g, ' ').replace(/^./, c => c.toUpperCase());
}

function _kindMeta(kind) {
  if (_KIND_META[kind]) return _KIND_META[kind];
  if (kind === 'llm_single_emergence')  return { color: '#f59e0b', icon: '◉' };
  if (kind && kind.startsWith('llm_'))  return { color: '#c084fc', icon: '✦' };
  return { color: '#5a607a', icon: '•' };
}

// ── Public API ────────────────────────────────────────────────────────────────

export function initAnthropologist(wsRequest) {
  _wsRequest = wsRequest;
  _loadLiveHistory();
  document.getElementById('anthropoBtn').addEventListener('click', _toggleAnthroMode);
  document.getElementById('msgModeMessagesBtn').addEventListener('click', () => _setInlineMode(false));
  document.getElementById('msgModeNotesBtn')   .addEventListener('click', () => _setInlineMode(true));
  document.querySelectorAll('.fn-fb-btn').forEach(btn => {
    btn.addEventListener('click', _onFilterClick);
  });
  document.getElementById('fnSearchInputTray').addEventListener('input', e => _onFnSearchInput(e.target.value));
  document.getElementById('fnSearchInputInline').addEventListener('input', e => _onFnSearchInput(e.target.value));
  document.getElementById('anthropoAgentSearch').addEventListener('input', e => {
    _agentSearch = e.target.value;
    if (_anthropoMode) _renderAgentList();
  });
  document.getElementById('anthropoDetailSnapshotBtn').addEventListener('click', function () {
    snapshotElement(document.getElementById('anthropoDetailSection'), this);
  });
  document.getElementById('fieldNotesTraySnapshotBtn').addEventListener('click', function () {
    snapshotElement(document.getElementById('fieldNotesTraySection'), this);
  });
}

function _onFnSearchInput(val) {
  _fnSearch = val;
  const tray   = document.getElementById('fnSearchInputTray');
  const inline = document.getElementById('fnSearchInputInline');
  if (tray   && tray.value   !== val) tray.value   = val;
  if (inline && inline.value !== val) inline.value = val;
  _renderFieldNotes();
}

function _onFilterClick(e) {
  const sev = parseInt(e.currentTarget.dataset.sev);
  if (_sevFilter.has(sev)) {
    _sevFilter.delete(sev);
  } else {
    _sevFilter.add(sev);
  }
  _syncFilterBtns();
  _renderFieldNotes();
}

function _syncFilterBtns() {
  document.querySelectorAll('.fn-fb-btn').forEach(btn => {
    btn.classList.toggle('fn-fb-on', _sevFilter.has(parseInt(btn.dataset.sev)));
  });
}

/** Called from ws.js on every frame with all known agents (alive + dead via seenAgentNames). */
export function updateAgentList(agents) {
  // agents already contains alive + previously-dead agents from seenAgentNames.
  // Also merge in dead agents from loaded obituaries and from the index (known
  // to exist on disk but not yet fetched this session).
  const knownTags = new Set(agents.map(a => a.tag));
  const fromObits = Object.keys(_obituaries)
    .filter(tag => !knownTags.has(tag))
    .map(tag => ({ tag, name: _obituaries[tag].annotation?.name || tag, alive: false }));
  const fromIndex = [..._obituariesIndex]
    .filter(tag => !knownTags.has(tag) && !_obituaries[tag])
    .map(tag => ({ tag, name: tag, alive: false }));
  _allAgents = [...agents, ...fromObits, ...fromIndex];
  if (_anthropoMode) _renderAgentList();
}

/** Called from ws.js on connect with the full field notes history. */
export function onFieldNotesSnapshot(msg) {
  // msg.events is newest-first; replace the local array entirely.
  _fieldNotes.length = 0;
  msg.events.forEach(e => _fieldNotes.push(e));
  if (_anthropoMode || _inlineMode) _renderFieldNotes();
}

/** Called from ws.js when a field_note message arrives. */
export function onFieldNote(msg) {
  msg.events.forEach(e => _fieldNotes.unshift(e));
  if (_anthropoMode || _inlineMode) {
    _renderFieldNotes();
  }
  if (!_anthropoMode && !_inlineMode) {
    _unreadNotes += msg.events.length;
    _updateNotesBadge();
  }
}

/** Called from ws.js when an agent_postmortem message arrives. */
export function onPostmortem(msg) {
  const tag = msg.agent_tag;
  _obituaries[tag] = msg;
  _obituariesIndex.add(tag);
  // Mark agent as dead in our list if not already
  const existing = _allAgents.find(a => a.tag === tag);
  if (existing) {
    existing.alive = false;
  } else {
    _allAgents.push({ tag, name: msg.annotation?.name || tag, alive: false });
  }
  if (_anthropoMode) {
    _renderAgentList();
    if (_selectedTag === tag) _renderDetail(tag);
    _unreadObitCount++;
    _updateMainBadge();
  } else {
    _unreadObitCount++;
    _updateMainBadge();
  }
}

/** Called from ws.js when live_annotations message arrives. */
export function onLiveAnnotations(msg) {
  for (const [tag, entry] of Object.entries(msg.annotations)) {
    const text = typeof entry === 'string' ? entry : entry.text;
    const step = typeof entry === 'string' ? -1  : entry.step;
    if (!_liveAnnotations[tag]) _liveAnnotations[tag] = [];
    const hist = _liveAnnotations[tag];
    if (!hist.length || hist[hist.length - 1].text !== text) hist.push({ text, step });
  }
  _saveLiveHistory();
  // Refresh detail panel if a live agent is currently selected
  if (_anthropoMode && _selectedTag) {
    const agent = _allAgents.find(a => a.tag === _selectedTag);
    if (agent && agent.alive) _renderDetail(_selectedTag);
  }
}

/** Called from ws.js on obituaries_index (WS connect). Sets badge status without loading content. */
export function onObituariesIndex(msg) {
  for (const tag of msg.tags) {
    _obituariesIndex.add(tag);
    _requested.delete(tag);  // un-stick any pending request so clicking re-fetches
    const existing = _allAgents.find(a => a.tag === tag);
    if (!existing) _allAgents.push({ tag, name: tag, alive: false });
    else existing.alive = false;
  }
  if (_anthropoMode) {
    _renderAgentList();
    if (_selectedTag && _obituariesIndex.has(_selectedTag)) _renderDetail(_selectedTag);
  }
}

/** Called from ws.js when an agent_postmortem_unavailable message arrives. */
export function onPostmortemUnavailable(msg) {
  const tag = msg.agent_tag;
  _unavailable.add(tag);
  _requested.delete(tag);
  if (_anthropoMode) {
    _renderAgentList();
    if (_selectedTag === tag) _renderDetail(tag);
  }
}

// ── Toggle logic ──────────────────────────────────────────────────────────────

function _toggleAnthroMode() {
  _anthropoMode = !_anthropoMode;
  const btn = document.getElementById('anthropoBtn');
  const leftTray  = document.getElementById('anthropoLeftTray');
  const rightTray = document.getElementById('anthropoRightTray');
  const midCol    = document.getElementById('midCol');
  const rightCol  = document.getElementById('rightCol');
  const msgModeSeg = document.getElementById('msgModeSeg');

  const gridCol = document.getElementById('gridCol');

  if (_anthropoMode) {
    // Exit inline mode if it was active
    if (_inlineMode) _setInlineMode(false);

    // Save normal layout, restore anthropologist layout
    _normalLayout = _saveLayout(midCol, gridCol, rightCol);
    _restoreLayout(_anthroLayout, leftTray, gridCol, rightTray);

    btn.classList.add('on');
    document.body.classList.add('anthropo-active');
    leftTray.classList.remove('hidden');
    rightTray.classList.remove('hidden');
    midCol.classList.add('anthro-hidden');
    rightCol.classList.add('anthro-hidden');
    msgModeSeg.style.display = 'none';
    _unreadNotes = 0;
    _unreadObitCount = 0;
    _updateMainBadge();
    _renderAgentList();
    _renderFieldNotes();
  } else {
    // Save anthropologist layout, restore normal layout
    _anthroLayout = _saveLayout(leftTray, gridCol, rightTray);
    _restoreLayout(_normalLayout, midCol, gridCol, rightCol);

    btn.classList.remove('on');
    document.body.classList.remove('anthropo-active');
    leftTray.classList.add('hidden');
    rightTray.classList.add('hidden');
    midCol.classList.remove('anthro-hidden');
    rightCol.classList.remove('anthro-hidden');
    msgModeSeg.style.display = '';
  }
}

function _setInlineMode(on) {
  _inlineMode = on;
  const msgsBtn   = document.getElementById('msgModeMessagesBtn');
  const notesBtn  = document.getElementById('msgModeNotesBtn');
  const msgList   = document.getElementById('msgList');
  const notesList = document.getElementById('fieldNotesListInline');
  const filterBar = document.getElementById('fnFilterInline');
  const searchBar = document.getElementById('fnSearchInline');
  const snapshotBtn  = document.getElementById('snapshotMsgBtn');

  msgsBtn .classList.toggle('on', !_inlineMode);
  notesBtn.classList.toggle('on',  _inlineMode);

  if (_inlineMode) {
    msgList.style.display = 'none';
    notesList.classList.remove('hidden');
    if (filterBar) filterBar.classList.remove('hidden');
    if (searchBar) searchBar.classList.remove('hidden');
    if (snapshotBtn)  snapshotBtn.title = 'Share field notes';
    _unreadNotes = 0;
    _updateNotesBadge();
    _renderFieldNotes();
  } else {
    msgList.style.display = '';
    notesList.classList.add('hidden');
    if (filterBar) filterBar.classList.add('hidden');
    if (searchBar) searchBar.classList.add('hidden');
    if (snapshotBtn)  snapshotBtn.title = 'Share messages';
  }
}

// ── Rendering ─────────────────────────────────────────────────────────────────

const _SEV_LABEL = ['low', 'medium', 'high'];

const _KEY_LABEL = {
  agent_tag:   'Agent Name',
  agent_tags:  'Agent Name(s)',
  tag_changes: 'Behavioral Changes',
};

function _fmtValue(v) {
  if (Array.isArray(v)) {
    if (v.length === 0) return '<em>[]</em>';
    if (typeof v[0] === 'object' && v[0] !== null) {
      // Array of objects (e.g. tag_changes: [{step, tags}, ...])
      // Show one row per actual change: full tag union, grey=unchanged, green=added, red=removed.
      const rows = v.map((item, idx) => {
        if (!Array.isArray(item.tags)) return null;
        if (idx === 0) return null; // baseline — shown via first change
        const prev = new Set(v[idx - 1].tags || []);
        const curr = new Set(item.tags);
        const added   = item.tags.filter(t => !prev.has(t));
        const removed = [...prev].filter(t => !curr.has(t));
        if (added.length === 0 && removed.length === 0) return null;
        const pills = [
          ...item.tags.filter(t => prev.has(t)).map(t => `<span class="fn-tag-pill fn-tag-dim">${_esc(t)}</span>`),
          ...added.map(t => `<span class="fn-tag-pill fn-tag-added">${_esc(t)}</span>`),
          ...removed.map(t => `<span class="fn-tag-pill fn-tag-removed">${_esc(t)}</span>`),
        ].join('');
        return `<div class="fn-tc-row fn-tc-changed">${pills}</div>`;
      }).filter(Boolean);
      // Fallback: if no changes were detected, show the final state
      if (!rows.length) {
        const last = v[v.length - 1];
        if (Array.isArray(last?.tags)) {
          const pills = last.tags.map(t => `<span class="fn-tag-pill fn-tag-dim">${_esc(t)}</span>`).join('');
          rows.push(`<div class="fn-tc-row">${pills}</div>`);
        }
      }
      return `<div class="fn-tag-changes">${rows.join('')}</div>`;
    }
    // Array of strings → tag pills
    return v.map(t => `<span class="fn-tag-pill">${_esc(String(t))}</span>`).join(' ');
  }
  if (typeof v === 'object' && v !== null) {
    return _esc(JSON.stringify(v));
  }
  if (typeof v === 'number' && !Number.isInteger(v)) {
    return _esc(parseFloat(v.toFixed(2)).toString());
  }
  return _esc(String(v));
}

function _fmtAgentTag(tag) {
  const agent = _allAgents.find(a => a.tag === tag);
  const name = agent ? agent.name : tag;
  return `<span title="${_esc(tag)}">${_esc(name)}</span>`;
}

function _fmtNarrative(text) {
  const lines = text.split('\n');
  let heading = '';
  let body = text;
  if (lines[0].startsWith('# ')) {
    heading = lines[0].slice(2).trim();
    body = lines.slice(1).join('\n').replace(/^\n+/, '');
  }
  const h = heading ? `<div class="fn-narrative-heading">${_esc(heading)}</div>` : '';
  const b = body.trim() ? `<div class="fn-narrative-body">${_esc(body)}</div>` : '';
  return h + b;
}

function _syntheticNarrative(e) {
  if (e.kind !== 'coalition_shift') return '';
  const d = e.data || {};
  const agent = _allAgents.find(a => a.tag === d.agent_tag);
  const name = agent ? agent.name : (d.agent_tag || 'Unknown agent');
  const pct = Math.round((d.overlap ?? 0) * 100);
  const oldC = d.old_community != null && d.old_community !== -1 ? `community ${d.old_community}` : 'their previous group';
  const newC = d.new_community != null && d.new_community !== -1 ? `community ${d.new_community}` : 'a new group';
  let sentence;
  if (pct === 0) {
    sentence = `${name} completely left ${oldC} and joined ${newC} — no members in common.`;
  } else if (pct < 50) {
    sentence = `${name} shifted from ${oldC} to ${newC}, with only ${pct}% of members in common.`;
  } else {
    sentence = `${name} moved toward ${newC} while retaining ${pct}% overlap with ${oldC}.`;
  }
  return `<div class="fn-narrative"><div class="fn-narrative-body">${_esc(sentence)}</div></div>`;
}

function _renderFieldNotes() {
  const container = _anthropoMode
    ? document.getElementById('fieldNotesList')
    : document.getElementById('fieldNotesListInline');
  if (!container) return;

  if (!_fieldNotes.length) {
    container.innerHTML = '<div class="empty-note">No field notes yet.</div>';
    return;
  }

  const fnQ = _fnSearch.toLowerCase().trim();
  const visible = _fieldNotes.filter(e => {
    if (!_sevFilter.has(e.severity)) return false;
    if (!fnQ) return true;
    return (
      e.kind.toLowerCase().includes(fnQ) ||
      (e.narrative && e.narrative.toLowerCase().includes(fnQ)) ||
      (e.data && JSON.stringify(e.data).toLowerCase().includes(fnQ))
    );
  });
  if (!visible.length) {
    let msg;
    if (_sevFilter.size === 0)  msg = 'No severities selected — pick low / med / high above to show notes.';
    else if (fnQ)               msg = `No notes match "${_esc(fnQ)}".`;
    else                        msg = 'No notes match the current filter.';
    container.innerHTML = `<div class="empty-note">${msg}</div>`;
    return;
  }

  container.innerHTML = visible.map(e => {
    const meta = _kindMeta(e.kind);
    const sevLabel = _SEV_LABEL[e.severity] ?? 'low';
    const narrative = e.narrative ? _fmtNarrative(e.narrative) : _syntheticNarrative(e);
    const dataStr = Object.entries(e.data || {})
      .filter(([k]) => !(k === 'description' && narrative))
      .map(([k, v]) => {
        const isComplex = Array.isArray(v) && v.length > 0
          && (typeof v[0] === 'object' || typeof v[0] === 'string');
        const cls = isComplex ? 'fn-datum fn-datum-block' : 'fn-datum';
        let displayVal;
        if (k === 'agent_tag') {
          displayVal = _fmtAgentTag(v);
        } else if (k === 'agent_tags' && Array.isArray(v)) {
          displayVal = v.map(t => `<span title="${_esc(t)}">${_esc((_allAgents.find(a => a.tag === t) || {}).name || t)}</span>`).join(', ');
        } else {
          displayVal = _fmtValue(v);
        }
        const label = _KEY_LABEL[k] ?? k;
        return `<span class="${cls}"><span class="fn-key">${_esc(label)}</span> ${displayVal}</span>`;
      })
      .join('');
    return `
      <div class="fn-card" data-sev="${e.severity}">
        <div class="fn-header">
          <span class="fn-step">step ${e.step}</span>
          <span class="fn-kind-pill" style="color:${meta.color};border-color:${meta.color};background:${meta.color}1a">
            ${meta.icon} ${_esc(_kindLabel(e))}
          </span>
          <span class="fn-sev fn-sev-${sevLabel}">${sevLabel}</span>
        </div>
        ${dataStr ? `<div class="fn-data">${dataStr}</div>` : ''}
        ${narrative ? `<div class="fn-narrative">${narrative}</div>` : ''}
      </div>`;
  }).join('');
  container.querySelectorAll('.fn-card').forEach(card => addSnapshotBtn(card));
}

function _renderAgentList() {
  const el = document.getElementById('anthropoAgentList');
  if (!el) return;

  if (!_allAgents.length) {
    el.innerHTML = '<div class="empty-note">No agents yet.</div>';
    return;
  }

  // Alive agents first, then dead
  const sorted = [..._allAgents].sort((a, b) => {
    if (a.alive !== b.alive) return a.alive ? -1 : 1;
    return a.tag.localeCompare(b.tag);
  });

  const q = _agentSearch.toLowerCase().trim();
  const filtered = q
    ? sorted.filter(a => (a.name || '').toLowerCase().includes(q) || a.tag.toLowerCase().includes(q))
    : sorted;

  if (!filtered.length) {
    el.innerHTML = `<div class="empty-note">No agents match "${_esc(q)}".</div>`;
    return;
  }

  el.innerHTML = filtered.map(a => {
    const active  = _selectedTag === a.tag ? ' anthro-agent-active' : '';
    const deadCls = a.alive ? '' : ' anthro-agent-dead';
    const dot = a.alive
      ? '<span class="anthro-alive-dot"></span>'
      : '<span class="anthro-dead-dot"></span>';

    // Status badge
    let obitStatus = '';
    if (a.alive) {
      obitStatus = '<span class="anthro-obit-badge obit-live" title="Live analysis in progress">live analysis…</span>';
    } else if (_obituaries[a.tag] || _obituariesIndex.has(a.tag)) {
      obitStatus = '<span class="anthro-obit-badge obit-done" title="Obituary available">done</span>';
    } else if (_unavailable.has(a.tag)) {
      obitStatus = '<span class="anthro-obit-badge obit-unavailable" title="No agent log — obituary unavailable">unavailable</span>';
    } else if (_requested.has(a.tag)) {
      obitStatus = '<span class="anthro-obit-badge obit-writing" title="Working on obituary…">analysing…</span>';
    } else {
      obitStatus = '<span class="anthro-obit-badge obit-none" title="No obituary yet">none</span>';
    }

    return `<div class="anthro-agent-row${active}${deadCls}" data-tag="${_esc(a.tag)}">
      ${dot} <span class="anthro-agent-name">${_esc(a.name)}</span>${obitStatus}
    </div>`;
  }).join('');

  el.querySelectorAll('.anthro-agent-row').forEach(row => {
    row.addEventListener('click', () => {
      _selectedTag = row.dataset.tag;
      _renderDetail(_selectedTag);
      _renderAgentList();
    });
  });
}

function _renderDetail(tag) {
  const titleEl = document.getElementById('anthropoDetailTitle');
  const bodyEl  = document.getElementById('anthropoDetailBody');
  if (!titleEl || !bodyEl) return;

  const agent = _allAgents.find(a => a.tag === tag);
  if (!agent) return;

  titleEl.textContent = agent.alive ? `Live — ${agent.name}` : `Obituary — ${agent.name}`;

  const pm = _obituaries[tag];
  if (!agent.alive && pm) {
    bodyEl.innerHTML = _renderObituaryHtml(pm);
    addSnapshotBtn(bodyEl);
  } else if (!agent.alive && _unavailable.has(tag)) {
    bodyEl.innerHTML = '<div class="empty-note" style="color:var(--error,#f87171)">No agent log recorded — obituary unavailable.</div>';
  } else if (!agent.alive) {
    if (!_requested.has(tag) && _wsRequest) {
      _requested.add(tag);
      const cmd = _obituariesIndex.has(tag) ? 'fetch_obituary' : 'request_postmortem';
      const params = cmd === 'request_postmortem'
        ? { agent_tag: tag, agent_name: agent.name }
        : { agent_tag: tag };
      _wsRequest(cmd, params).catch(e => {
        console.error(`[anthro] ${cmd} failed:`, e);
        _requested.delete(tag);
        _renderAgentList();
        if (_selectedTag === tag) {
          const msg = cmd === 'fetch_obituary'
            ? 'Could not load obituary file.'
            : 'Could not queue obituary — is the anthropologist server running?';
          bodyEl.innerHTML = `<div class="empty-note" style="color:var(--error,#f87171)">${msg}</div>`;
        }
      });
    }
    const msg = _requested.has(tag)
      ? 'Obituary requested — working on it…'
      : 'No obituary yet.';
    bodyEl.innerHTML = `<div class="empty-note">${msg}</div>`;
  } else {
    const hist = _liveAnnotations[tag] || [];
    if (!hist.length) {
      bodyEl.innerHTML = '<div class="empty-note">Agent is alive — annotation pending.</div>';
    } else {
      const reversed = [...hist].reverse();
      const entries = reversed.map(({ text, step }, i) => {
        const stepLabel = step >= 0 ? `Step ${step}` : 'Earlier';
        return i === 0
          ? `<div class="live-annotation-latest">
               <div class="obit-section-title">Live Annotation</div>
               <div class="obit-narrative">${_esc(text)}</div>
             </div>`
          : `<div class="live-annotation-entry">
               <div class="live-annotation-divider"><span>${_esc(stepLabel)}</span></div>
               <div class="live-annotation-past-text">${_esc(text)}</div>
             </div>`;
      }).join('');
      bodyEl.innerHTML = entries;
      addSnapshotBtn(bodyEl);
    }
  }
}

/** Format a tag name: underscores → spaces, title case. */
function _fmtTag(s) {
  return s.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

/** Render **bold** and *italic* markdown in escaped text. */
function _md(str) {
  return _esc(str)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g,     '<em>$1</em>');
}

/**
 * Full markdown renderer for the anthropologist block.
 * Handles headings (#/##/###), ordered/unordered lists, and paragraphs.
 */
function _mdFull(str) {
  const lines = str.split('\n');
  let html = '';
  let listType = null;   // 'ol' | 'ul' | null

  const flushList = () => {
    if (listType) { html += `</${listType}>`; listType = null; }
  };

  for (const raw of lines) {
    const line = raw.trim();
    const h1 = line.match(/^#\s+(.+)/);
    const h2 = line.match(/^##\s+(.+)/);
    const h3 = line.match(/^###\s+(.+)/);
    const ol = line.match(/^\d+\.\s+(.+)/);
    const ul = line.match(/^[-*]\s+(.+)/);

    if (h3) {
      flushList();
      html += `<div class="obit-md-h3">${_md(h3[1])}</div>`;
    } else if (h2) {
      flushList();
      html += `<div class="obit-md-h2">${_md(h2[1])}</div>`;
    } else if (h1) {
      flushList();
      html += `<div class="obit-md-h1">${_md(h1[1])}</div>`;
    } else if (ol) {
      if (listType !== 'ol') { flushList(); html += '<ol class="obit-md-list">'; listType = 'ol'; }
      html += `<li>${_md(ol[1])}</li>`;
    } else if (ul) {
      if (listType !== 'ul') { flushList(); html += '<ul class="obit-md-list">'; listType = 'ul'; }
      html += `<li>${_md(ul[1])}</li>`;
    } else if (line === '') {
      // don't flush list on blank lines — LLMs often emit blank lines between items
    } else {
      flushList();
      html += `<p class="obit-anthro-text">${_md(line)}</p>`;
    }
  }

  flushList();
  return html;
}

function _renderObituaryHtml(pm) {
  const ann = pm.annotation || {};

  // ── Header ──────────────────────────────────────────────────────────────────
  const parts = [];
  if (pm.birth_step != null)  parts.push(`Born at step <strong>${pm.birth_step}</strong>`);
  if (pm.steps_lived != null) parts.push(`Died after <strong>${pm.steps_lived}</strong> steps`);
  if (pm.energy_at_death != null) parts.push(`Energy at death: <strong>${pm.energy_at_death - 1}</strong>`);
  const statChips = parts.map(p => `<span class="obit-stat">${p}</span>`).join('');

  // ── Summary ──────────────────────────────────────────────────────────────────
  const summary = ann.comment
    ? `<div class="obit-narrative">${_md(ann.comment)}</div>` : '';

  // ── Behaviors ────────────────────────────────────────────────────────────────
  const behaviorItems = (ann.behaviors || []).map(b =>
    `<div class="obit-behavior">
       <div class="obit-behavior-header">
         <span class="obit-tag">${_fmtTag(_esc(b.behavior))}</span>
       </div>
       ${b.description ? `<div class="obit-desc">${_esc(b.description)}</div>` : ''}
     </div>`
  ).join('');

  // ── Emergence ────────────────────────────────────────────────────────────────
  const hasEmergence = ann.emergence?.keywords?.length && ann.emergence.keywords[0] !== 'none';
  const emergenceBlock = hasEmergence
    ? `<div class="obit-section-title">Emergence</div>
       <div class="obit-emergence">
         ${ann.emergence.keywords.map(k => `<span class="obit-tag obit-emerge-tag">${_fmtTag(_esc(k))}</span>`).join('')}
         ${ann.emergence.comment ? `<div class="obit-emerge-comment">${_esc(ann.emergence.comment)}</div>` : ''}
       </div>` : '';

  // ── Anthropologist note ──────────────────────────────────────────────────────
  const anthropologistBlock = ann.anthropologist
    ? `<div class="obit-section-title">Anthropologist</div>
       <div class="obit-anthro-block">
         ${_mdFull(ann.anthropologist)}
       </div>` : '';

  return `
    <div class="obit-card">
      <div class="obit-header-meta">${statChips}</div>
      ${summary ? `<div class="obit-section-title">Summary</div>${summary}` : ''}
      ${behaviorItems ? `<div class="obit-section-title">Behaviors</div>${behaviorItems}` : ''}
      ${emergenceBlock}
      ${anthropologistBlock}
    </div>`;
}

/** Render analysis content (obituary or live annotation) into a DOM element. */
export function renderAnalysisInto(el, tag, isAlive, name) {
  const pm = _obituaries[tag];
  if (!isAlive && pm) {
    el.innerHTML = _renderObituaryHtml(pm);
  } else if (!isAlive) {
    if (!_requested.has(tag) && _wsRequest) {
      _requested.add(tag);
      _wsRequest('request_postmortem', { agent_tag: tag, agent_name: name })
        .then(r  => console.log('[anthro] postmortem queued:', r))
        .catch(e => {
          // Server rejected (e.g. missing API key). Drop the optimistic
          // in-flight marker so a re-click after the user fixes the issue
          // can re-fire the request, and refresh the embedded view.
          console.error('[anthro] postmortem request failed:', e);
          _requested.delete(tag);
          el.innerHTML = `<div class="empty-note">No obituary yet.</div>`;
        });
    }
    const msg = _requested.has(tag) ? 'Obituary requested — working on it…' : 'No obituary yet.';
    el.innerHTML = `<div class="empty-note">${msg}</div>`;
  } else {
    const hist = _liveAnnotations[tag] || [];
    if (!hist.length) {
      el.innerHTML = '<div class="empty-note">Agent is alive — annotation pending.</div>';
    } else {
      const reversed = [...hist].reverse();
      el.innerHTML = reversed.map(({ text, step }, i) => {
        const stepLabel = step >= 0 ? `Step ${step}` : 'Earlier';
        return i === 0
          ? `<div class="live-annotation-latest">
               <div class="obit-section-title">Live Annotation</div>
               <div class="obit-narrative">${_esc(text)}</div>
             </div>`
          : `<div class="live-annotation-entry">
               <div class="live-annotation-divider"><span>${_esc(stepLabel)}</span></div>
               <div class="live-annotation-past-text">${_esc(text)}</div>
             </div>`;
      }).join('');
    }
  }
}

// ── Badge helpers ─────────────────────────────────────────────────────────────

function _updateMainBadge() {
  const el = document.getElementById('anthropoBadge');
  if (!el) return;
  if (_unreadObitCount > 0) {
    el.textContent = _unreadObitCount;
    el.classList.remove('hidden');
  } else {
    el.classList.add('hidden');
  }
}

function _updateNotesBadge() {
  const el = document.getElementById('fieldNotesBadge');
  if (!el) return;
  if (_unreadNotes > 0) {
    el.textContent = _unreadNotes;
    el.classList.remove('hidden');
  } else {
    el.classList.add('hidden');
  }
}

// ── Utility ───────────────────────────────────────────────────────────────────

function _esc(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
