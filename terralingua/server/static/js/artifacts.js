import { state, escHtml, agentName, seenAgentNames, artifactRowCache, expiredArtRowCache, SERVER } from './state.js';
import { snapshotElement } from './screenshot.js';
import { drawGrid } from './grid.js';

// ── Artifact ownership ────────────────────────────────────────────────────────
export const myArtifacts        = new Set();    // names of artifacts created by this user
export const myArtifactVersions = new Map();    // name → version we expect after last user edit
const _MY_ARTIFACTS_KEY = `ogw-artifacts-${SERVER}`;

// Superusers can edit/delete any artifact as if they owned it.
export function ownsArtifact(name) {
  return state.currentUser?.is_superuser === true || myArtifacts.has(name);
}

export function _saveMyArtifacts(){
  const obj={};
  for(const n of myArtifacts) obj[n]=myArtifactVersions.get(n)??0;
  localStorage.setItem(_MY_ARTIFACTS_KEY,JSON.stringify(obj));
}
export function _loadMyArtifacts(){
  try{
    const raw=localStorage.getItem(_MY_ARTIFACTS_KEY); if(!raw) return;
    const data=JSON.parse(raw);
    if(Array.isArray(data)){ for(const n of data){ myArtifacts.add(n); myArtifactVersions.set(n,0); } }
    else{ for(const [n,v] of Object.entries(data)){ myArtifacts.add(n); myArtifactVersions.set(n,v); } }
  }catch(_){}
}

// ── wsRequest late-binding ────────────────────────────────────────────────────
let _wsRequest = null;
export function initArtifacts(wsReq){
  _wsRequest = wsReq;
  document.getElementById('artSearch').addEventListener('input', e => {
    _artSearch = e.target.value;
    if (state.lastFrame) renderArtifacts(state.lastFrame.artifacts || [], state.lastFrame.expired_artifacts || []);
  });
}

let _artSearch = '';

// ── Anthropologist data (injected via setArtifactAnalysisData) ────────────────
let _artCategories = {};   // artifact_name → category string ("1"–"4" or "-1")
let _artPhylogeny  = {};   // artifact_name → {ancestor_name: score}
const _artPhyloDone = new Set(); // artifact names whose phylogeny has been computed (even if no ancestors)
let _phyloMode     = false; // whether the phylogeny panel is currently shown
const _phyloRequested = new Set(); // artifact names for which we've sent a request
let _phyloPanCtl   = null; // AbortController for pan/zoom listeners; aborted on each re-render
const _phyloView   = new Map(); // artifact_name → {scale, tx, ty} — preserves pan/zoom across re-renders

export function setArtifactAnalysisData(categories, phylogeny, done = [], pending = []) {
  _artCategories = categories || {};
  _artPhylogeny  = phylogeny  || {};
  // _artPhyloDone is fully authoritative from the server — replace, don't accumulate.
  // Accumulating leaks stale entries that lie about state and make the panel show
  // "No ancestor artifacts found." while the LLM is still computing.
  _artPhyloDone.clear();
  for (const name of done) _artPhyloDone.add(name);
  // Server's pending list is also authoritative; merge it with any optimistic
  // local adds from clicks that haven't reached the server yet (otherwise the
  // tight publish-on-click race would briefly drop "Computing…").
  for (const name of pending) _phyloRequested.add(name);
  for (const name of done) _phyloRequested.delete(name);
  // Refresh detail panel if one is open
  if (state.selectedArtifact) _renderArtDetail(state.selectedArtifact);
}

const _CAT_LABELS = {
  '1':  { label: 'basic',         color: '#6b7280' },
  '2':  { label: 'procedural',    color: '#3b82f6' },
  '3':  { label: 'institutional', color: '#f59e0b' },
  '4':  { label: 'governance',    color: '#a78bfa' },
  '-1': { label: 'unclassified',  color: '#6b7280' },
};

// ── Rendering ─────────────────────────────────────────────────────────────────
function _makeArtRow(art, expired){
  const isSelected = state.selectedArtifact&&state.selectedArtifact.name===art.name;
  const carrierName = art.carrier ? (state.tagToAgent.get(art.carrier)?.name || art.carrier) : null;
  const pos = carrierName ? "" : (art.pose?`(${art.pose[0]},${art.pose[1]})`:"");
  const carrierHtml = carrierName ? `<span class="art-carrier">🎒 ${escHtml(carrierName)}</span>` : "";
  const row = document.createElement("div");
  row.className = "art-row"+(isSelected?" selected":"")+(expired?" art-row--expired":"");
  row.dataset.name = art.name;
  row.innerHTML = `<span class="art-icon${!expired&&ownsArtifact(art.name)?" art-icon--mine":""}">◈</span>
    <span class="art-name">${escHtml(art.name)}</span>
    ${carrierHtml}<span class="art-pos">${pos}</span>`;
  row._art = art;
  row.addEventListener("click", ()=>{
    state.lastListNav = "artifacts";
    const sel = state.selectedArtifact&&state.selectedArtifact.name===row._art.name;
    if (sel) { closeArtDetail(); return; }
    const keepPhylo = _phyloMode;
    openArtDetail(row._art);
    if (keepPhylo) {
      _phyloMode = true;
      document.getElementById("artPhyloBtn").classList.add("on");
      _renderArtDetail(state.selectedArtifact);
    }
  });
  return row;
}

export function renderArtifacts(artifacts, expiredArtifacts=[]){
  const total = state._showExpired ? artifacts.length + expiredArtifacts.length : artifacts.length;
  document.getElementById("artCountBadge").textContent = total || "";
  const list=document.getElementById("artifactList");
  const q = _artSearch.toLowerCase().trim();
  if (q) {
    artifacts = artifacts.filter(a =>
      a.name.toLowerCase().includes(q) ||
      (a.carrier ? (state.tagToAgent.get(a.carrier)?.name || a.carrier).toLowerCase().includes(q) : false)
    );
    expiredArtifacts = expiredArtifacts.filter(a =>
      a.name.toLowerCase().includes(q)
    );
  }

  const hasAny = artifacts.length || expiredArtifacts.length;
  if(!hasAny){
    list.innerHTML=`<div class="empty-note">${q ? `No artifacts match "${q}".` : 'No artifacts yet.'}</div>`;
    artifactRowCache.clear(); expiredArtRowCache.clear(); return;
  }
  if(artifactRowCache.size===0 && expiredArtRowCache.size===0) list.innerHTML="";

  // ── Active artifacts ──
  const newNameSet = new Set(artifacts.map(a=>a.name));
  for(const [name, row] of artifactRowCache)
    if(!newNameSet.has(name)){ if(row.parentNode===list) list.removeChild(row); artifactRowCache.delete(name); }
  for(const art of artifacts){
    const isSelected = state.selectedArtifact&&state.selectedArtifact.name===art.name;
    const carrierName = art.carrier ? (state.tagToAgent.get(art.carrier)?.name || art.carrier) : null;
    const pos = carrierName ? "" : (art.pose?`(${art.pose[0]},${art.pose[1]})`:"");
    if(artifactRowCache.has(art.name)){
      const row = artifactRowCache.get(art.name);
      row.className = "art-row"+(isSelected?" selected":"");
      const iconEl=row.querySelector(".art-icon");
      if(iconEl) iconEl.className="art-icon"+(ownsArtifact(art.name)?" art-icon--mine":"");
      const posEl2 = row.querySelector(".art-pos");
      if(carrierName){
        posEl2.textContent = "";
        let carrierBadge = row.querySelector(".art-carrier");
        if(!carrierBadge){ carrierBadge = document.createElement("span"); carrierBadge.className="art-carrier"; row.insertBefore(carrierBadge, posEl2); }
        carrierBadge.textContent = `🎒 ${carrierName}`;
      } else {
        posEl2.textContent = pos;
        const carrierBadge = row.querySelector(".art-carrier");
        if(carrierBadge) carrierBadge.remove();
      }
      row._art = art;
      if(isSelected){
        state.selectedArtifact = art;
        const posEl = document.getElementById("artDetailPos");
        if(posEl) posEl.textContent = art.pose ? `(${art.pose[0]}, ${art.pose[1]})` : "—";
      }
    } else {
      artifactRowCache.set(art.name, _makeArtRow(art, false));
    }
  }
  for(const art of artifacts) list.appendChild(artifactRowCache.get(art.name));

  // ── Expired artifacts ──
  const expiredNameSet = new Set(expiredArtifacts.map(a=>a.name));
  for(const [name, row] of expiredArtRowCache)
    if(!expiredNameSet.has(name)){ if(row.parentNode===list) list.removeChild(row); expiredArtRowCache.delete(name); }
  for(const art of expiredArtifacts){
    const isSelected = state.selectedArtifact&&state.selectedArtifact.name===art.name;
    if(expiredArtRowCache.has(art.name)){
      const row = expiredArtRowCache.get(art.name);
      row.className = "art-row art-row--expired"+(isSelected?" selected":"");
      row._art = art;
    } else {
      expiredArtRowCache.set(art.name, _makeArtRow(art, true));
    }
    const row = expiredArtRowCache.get(art.name);
    row.style.display = state._showExpired ? "" : "none";
    if(state._showExpired) list.appendChild(row);
  }
}

async function _resolveArtTags(art){
  const tags = new Set();
  if(art.creator_tag) tags.add(art.creator_tag);
  if(art.carrier) tags.add(art.carrier);
  for(const v of art.past_versions||[]) if(v.modified_by) tags.add(v.modified_by);
  for(const tag of Object.keys(art.users_tag||{})) tags.add(tag);
  const unknown = [...tags].filter(t => !seenAgentNames.has(t) && !state.tagToAgent.has(t));
  if(unknown.length){
    try{
      const result = await _wsRequest("get_agent_names", {tags: unknown});
      for(const [t, name] of Object.entries(result)) seenAgentNames.set(t, name);
    }catch(_){}
  }
}

function _renderArtDetail(art){
  if(state.selectedArtifact?.name !== art.name) return; // stale render, skip
  const body=document.getElementById("artDetailBody");
  // Reset layout overrides from phylo mode; _renderPhylogenyPanel re-applies them if needed
  body.style.display = "";
  body.style.flexDirection = "";
  body.style.overflowY = "";

  if (_phyloMode) { _renderPhylogenyPanel(art); return; }

  const pos=art.pose?`(${art.pose[0]}, ${art.pose[1]})`:"—";
  const creator=escHtml(agentName(art.creator_tag)||"—");
  const carrierName = art.carrier ? (agentName(art.carrier) || art.carrier) : null;
  const kvs=[
    ["Creator",creator],["Created",art.creation_time??"-"],
    ["Version",art.version??"-"],["Lifespan",art.remaining_time??art.lifespan??"-"],
  ];
  if(carrierName) kvs.push(["Carrier", `🎒 ${escHtml(carrierName)}`]);

  // Category badge from anthropologist classifications
  const catRaw = _artCategories[art.name];
  const catMeta = catRaw ? (_CAT_LABELS[String(catRaw)] || _CAT_LABELS['-1']) : null;
  const catHtml = catMeta
    ? `<div class="art-detail-kv"><div class="art-detail-kv-label">Category</div>`
      + `<div class="art-detail-kv-value"><span class="art-cat-badge" style="background:${catMeta.color}18">`
      + `<span class="art-cat-dot" style="background:${catMeta.color}"></span>${escHtml(catMeta.label)}</span></div></div>`
    : '';

  const metaHtml=`<div class="art-detail-kv"><div class="art-detail-kv-label">Position</div><div class="art-detail-kv-value" id="artDetailPos">${pos}</div></div>`
    +kvs.map(([l,v])=>`<div class="art-detail-kv"><div class="art-detail-kv-label">${l}</div><div class="art-detail-kv-value">${v}</div></div>`).join("")
    +catHtml;
  const payloadText=typeof art.payload==="object"?JSON.stringify(art.payload,null,2):String(art.payload??"(empty)");
  const hist=art.past_versions;
  const histHtml=hist&&hist.length
    ?`<div class="art-detail-block"><div class="art-detail-label">Version history</div>`
      +[...hist].reverse().map((v,ri,arr)=>{
        const i=arr.length-1-ri;
        const payload=typeof v.payload==="object"?JSON.stringify(v.payload,null,2):String(v.payload??"");
        const modifier=v.modified_by?`<span class="art-version-modifier">Modified by: ${escHtml(agentName(v.modified_by)||v.modified_by)}</span>`:"";
        return `<div class="art-version-tile"><div class="art-version-hdr"><span class="art-version-tag">v${i}</span>${modifier}</div><div class="art-detail-text">${escHtml(payload)}</div></div>`;
      }).join("")+`</div>`
    :"";
  const userEntries=Object.entries(art.users_tag||{})
    .map(([tag,times])=>[tag,times.length])
    .sort((a,b)=>b[1]-a[1]);
  const accessHtml=userEntries.length
    ?`<div class="art-detail-block"><div class="art-detail-label">Accessed by</div>`
      +userEntries.map(([tag,count])=>
        `<div class="art-access-row"><button class="art-access-name" data-tag="${escHtml(tag)}">${escHtml(agentName(tag)||tag)}</button><span class="art-access-count">${count}×</span></div>`
      ).join("")+`</div>`
    :"";
  body.innerHTML=`<div class="art-detail-meta">${metaHtml}</div>
    <div class="art-detail-block"><div class="art-detail-label">Content</div><div class="art-detail-text">${escHtml(payloadText)}</div></div>
    ${histHtml}${accessHtml}`;
}

function _renderPhylogenyPanel(art) {
  const body = document.getElementById("artDetailBody");
  // Tear down any previous render's window listeners up-front so early-return
  // paths (Computing…, no ancestors) don't leak the mousemove/mouseup hooks.
  if (_phyloPanCtl) { _phyloPanCtl.abort(); _phyloPanCtl = null; }
  const ancestors = _artPhylogeny[art.name];
  const hasAncestors = ancestors && Object.keys(ancestors).length > 0;

  const hasDescendants = Object.values(_artPhylogeny).some(a => art.name in a);

  // Trigger a request when we have neither ancestors nor a done-marker for this
  // artifact, and haven't already asked.
  if (!hasAncestors && !_artPhyloDone.has(art.name) && !_phyloRequested.has(art.name)) {
    _phyloRequested.add(art.name);
    if (_wsRequest) {
      _wsRequest("request_artifact_phylogeny", { artifact_name: art.name })
        .catch(e => {
          // Server rejected (e.g. missing API key) — undo the optimistic
          // in-flight marker so a user-initiated re-click can fire a new
          // request after they fix the issue. Do NOT re-render here: that
          // would immediately satisfy the request-gate and re-fire the
          // request in a tight loop. The red toast (published by the
          // server on this same rejection) tells the user what to do.
          console.error("[artifacts] phylo request failed:", e);
          _phyloRequested.delete(art.name);
        });
    }
  }

  // "Still computing" = we've requested AND the server hasn't reported it done.
  // Strong signal — use it to gate the "Computing…" message; relying on
  // !_artPhyloDone alone misfires when the server has stale done entries.
  const stillComputing = _phyloRequested.has(art.name) && !_artPhyloDone.has(art.name);

  if (!hasAncestors && !hasDescendants) {
    body.innerHTML = stillComputing
      ? '<div class="empty-note">Computing phylogeny… this may take a moment.</div>'
      : '<div class="empty-note">No ancestor artifacts found.</div>';
    return;
  }

  body.innerHTML = '';
  // Switch body to a column flex container so the zoom viewport can claim the
  // remaining vertical space. Reset by _renderArtDetail on non-phylo renders.
  body.style.display = "flex";
  body.style.flexDirection = "column";
  body.style.overflowY = "hidden";
  // Show computing notice above the graph when ancestry is still pending
  if (!hasAncestors && stillComputing) {
    const notice = document.createElement('div');
    notice.className = 'empty-note';
    notice.style.marginBottom = '0.5rem';
    notice.textContent = 'Computing ancestry… this may take a moment.';
    body.appendChild(notice);
  }

  const allArts = [...(state.lastFrame?.artifacts || []), ...(state.lastFrame?.expired_artifacts || [])];
  const artByName = new Map(allArts.map(a => [a.name, a]));

  // BFS backward: collect ancestor nodes
  const ancestorNodes = new Set();
  { const q = [art.name];
    while (q.length) {
      const n = q.shift();
      for (const a of Object.keys(_artPhylogeny[n] || {})) {
        if (!ancestorNodes.has(a)) { ancestorNodes.add(a); q.push(a); }
      }
    }
  }

  // Build reverse map: ancestor → direct descendants (from all known phylogenies)
  const directDescOf = new Map();
  for (const [desc, ancsObj] of Object.entries(_artPhylogeny)) {
    for (const anc of Object.keys(ancsObj)) {
      if (!directDescOf.has(anc)) directDescOf.set(anc, []);
      directDescOf.get(anc).push(desc);
    }
  }

  // BFS forward from art.name: collect descendant nodes
  const descendantNodes = new Set();
  { const q = [art.name]; const vis = new Set([art.name]);
    while (q.length) {
      const n = q.shift();
      for (const d of (directDescOf.get(n) || [])) {
        if (!vis.has(d)) { vis.add(d); descendantNodes.add(d); q.push(d); }
      }
    }
  }

  // All nodes in the subgraph
  const allNodes = new Set([...ancestorNodes, art.name, ...descendantNodes]);

  // Collect directed edges (ancestor → descendant) within the subgraph
  const edges = [];
  const edgeSet = new Set();
  for (const [desc, ancsObj] of Object.entries(_artPhylogeny)) {
    if (!allNodes.has(desc)) continue;
    for (const anc of Object.keys(ancsObj)) {
      if (!allNodes.has(anc)) continue;
      const key = `${anc}→${desc}`;
      if (!edgeSet.has(key)) { edgeSet.add(key); edges.push([anc, desc]); }
    }
  }

  // Build adjacency maps
  const childrenOf = new Map();
  const parentsOf = new Map();
  for (const n of allNodes) { childrenOf.set(n, []); parentsOf.set(n, []); }
  for (const [from, to] of edges) {
    childrenOf.get(from).push(to);
    parentsOf.get(to).push(from);
  }

  // Assign ranks via longest-path (Kahn's BFS)
  const rank = new Map();
  const inDeg = new Map();
  for (const n of allNodes) inDeg.set(n, 0);
  for (const [from, to] of edges) inDeg.set(to, inDeg.get(to) + 1);
  const topoQ = [...allNodes].filter(n => inDeg.get(n) === 0);
  for (const n of topoQ) rank.set(n, 0);
  while (topoQ.length) {
    const n = topoQ.shift();
    const r = rank.get(n);
    for (const c of childrenOf.get(n)) {
      const nr = r + 1;
      if (!rank.has(c) || rank.get(c) < nr) rank.set(c, nr);
      inDeg.set(c, inDeg.get(c) - 1);
      if (inDeg.get(c) === 0) topoQ.push(c);
    }
  }

  // Group nodes by rank
  const byRank = new Map();
  for (const [n, r] of rank) {
    if (!byRank.has(r)) byRank.set(r, []);
    byRank.get(r).push(n);
  }
  const maxRank = Math.max(...rank.values(), 0);

  // Barycenter heuristic: reduce crossings by ordering nodes within each rank
  // by the average position of their parents (top-down), then children (bottom-up).
  const xOrder = new Map(); // name → index within its rank
  for (const [, nodes] of byRank) nodes.forEach((n, i) => xOrder.set(n, i));
  for (let pass = 0; pass < 3; pass++) {
    const rankList = [...byRank.entries()].sort(([a], [b]) => a - b);
    if (pass % 2 === 1) rankList.reverse();
    for (const [r, nodes] of rankList) {
      nodes.sort((a, b) => {
        const refs = pass % 2 === 0 ? parentsOf : childrenOf;
        const avgA = refs.get(a).length ? refs.get(a).reduce((s, p) => s + (xOrder.get(p) ?? 0), 0) / refs.get(a).length : xOrder.get(a);
        const avgB = refs.get(b).length ? refs.get(b).reduce((s, p) => s + (xOrder.get(p) ?? 0), 0) / refs.get(b).length : xOrder.get(b);
        return avgA - avgB;
      });
      nodes.forEach((n, i) => xOrder.set(n, i));
    }
  }

  // Layout constants
  const NODE_W = 132, NODE_H = 34, H_GAP = 20, V_GAP = 58, PAD = 16;

  const maxPerRank = Math.max(...[...byRank.values()].map(v => v.length), 1);
  const totalW = maxPerRank * NODE_W + (maxPerRank - 1) * H_GAP;
  const svgW = totalW + 2 * PAD;
  const svgH = (maxRank + 1) * NODE_H + maxRank * V_GAP + 2 * PAD;

  // Compute node centre positions
  const pos = new Map();
  for (const [r, nodes] of byRank) {
    const rowW = nodes.length * NODE_W + (nodes.length - 1) * H_GAP;
    const startX = PAD + (totalW - rowW) / 2;
    for (let i = 0; i < nodes.length; i++) {
      const cx = startX + i * (NODE_W + H_GAP) + NODE_W / 2;
      const cy = PAD + r * (NODE_H + V_GAP) + NODE_H / 2;
      pos.set(nodes[i], { cx, cy });
    }
  }

  // Render SVG
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("width", svgW);
  svg.setAttribute("height", svgH);
  svg.style.display = "block";
  svg.style.overflow = "visible";

  // SVG defs — inject once into document so IDs are stable across re-renders
  if (!document.getElementById("phylo-arrow")) {
    const globalDefs = document.createElementNS(NS, "defs");

    // Arrow marker — accent-coloured, clean triangle
    const marker = document.createElementNS(NS, "marker");
    marker.setAttribute("id", "phylo-arrow");
    marker.setAttribute("markerWidth", "8");
    marker.setAttribute("markerHeight", "8");
    marker.setAttribute("refX", "7");
    marker.setAttribute("refY", "4");
    marker.setAttribute("orient", "auto");
    const arrowPath = document.createElementNS(NS, "path");
    arrowPath.setAttribute("d", "M 1 1 L 7 4 L 1 7 Z");
    arrowPath.setAttribute("fill", "var(--accent)");
    arrowPath.setAttribute("opacity", "0.7");
    marker.appendChild(arrowPath);
    globalDefs.appendChild(marker);

    // Horizontal gradient for the active/self node (accent → accent2)
    const grad = document.createElementNS(NS, "linearGradient");
    grad.setAttribute("id", "phylo-self-grad");
    grad.setAttribute("x1", "0%"); grad.setAttribute("y1", "0%");
    grad.setAttribute("x2", "100%"); grad.setAttribute("y2", "0%");
    const stop1 = document.createElementNS(NS, "stop");
    stop1.setAttribute("offset", "0%"); stop1.setAttribute("stop-color", "var(--accent)");
    const stop2 = document.createElementNS(NS, "stop");
    stop2.setAttribute("offset", "100%"); stop2.setAttribute("stop-color", "var(--accent2)");
    grad.appendChild(stop1); grad.appendChild(stop2);
    globalDefs.appendChild(grad);

    const hiddenSvg = document.createElementNS(NS, "svg");
    hiddenSvg.style.cssText = "position:absolute;width:0;height:0;overflow:hidden";
    hiddenSvg.appendChild(globalDefs);
    document.body.appendChild(hiddenSvg);
  }

  // Edges (drawn before nodes so they appear behind)
  for (const [from, to] of edges) {
    const fp = pos.get(from), tp = pos.get(to);
    if (!fp || !tp) continue;
    const x1 = fp.cx, y1 = fp.cy + NODE_H / 2;
    const x2 = tp.cx, y2 = tp.cy - NODE_H / 2;
    const mid = (y1 + y2) / 2;
    const path = document.createElementNS(NS, "path");
    path.setAttribute("d", `M ${x1} ${y1} C ${x1} ${mid} ${x2} ${mid} ${x2} ${y2}`);
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", "var(--accent)");
    path.setAttribute("stroke-opacity", "0.35");
    path.setAttribute("stroke-width", "1.5");
    path.setAttribute("marker-end", "url(#phylo-arrow)");
    svg.appendChild(path);
  }

  // Nodes
  for (const [name, p] of pos) {
    const isSelf = name === art.name;
    const artObj = artByName.get(name);
    const isExpired = !artObj || !!artObj.deletion_time;
    const g = document.createElementNS(NS, "g");
    if (artObj) g.style.cursor = "pointer";

    // Full name tooltip
    const titleEl = document.createElementNS(NS, "title");
    titleEl.textContent = name;
    g.appendChild(titleEl);

    const nx = p.cx - NODE_W / 2, ny = p.cy - NODE_H / 2;
    const rect = document.createElementNS(NS, "rect");
    rect.setAttribute("x", nx); rect.setAttribute("y", ny);
    rect.setAttribute("width", NODE_W); rect.setAttribute("height", NODE_H);
    rect.setAttribute("rx", "8");
    if (isSelf) {
      rect.setAttribute("fill", "url(#phylo-self-grad)");
      rect.setAttribute("stroke", "var(--accent)");
      rect.setAttribute("stroke-opacity", "0.8");
      rect.setAttribute("stroke-width", "1.5");
    } else if (isExpired) {
      rect.setAttribute("fill", "var(--bg)");
      rect.setAttribute("stroke", "var(--border)");
      rect.setAttribute("stroke-width", "1");
    } else {
      rect.setAttribute("fill", "var(--surface)");
      rect.setAttribute("stroke", "var(--accent)");
      rect.setAttribute("stroke-opacity", "0.3");
      rect.setAttribute("stroke-width", "1.5");
    }

    // Hover highlight for clickable non-self nodes
    if (artObj && !isSelf) {
      g.addEventListener("mouseenter", () => {
        rect.setAttribute("stroke", "var(--accent)");
        rect.setAttribute("stroke-opacity", "0.65");
        rect.setAttribute("stroke-width", "1.5");
      });
      g.addEventListener("mouseleave", () => {
        if (isExpired) {
          rect.setAttribute("stroke", "var(--border)");
          rect.setAttribute("stroke-opacity", "1");
          rect.setAttribute("stroke-width", "1");
        } else {
          rect.setAttribute("stroke", "var(--accent)");
          rect.setAttribute("stroke-opacity", "0.3");
        }
      });
    }

    const text = document.createElementNS(NS, "text");
    text.setAttribute("x", p.cx); text.setAttribute("y", p.cy);
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("dominant-baseline", "middle");
    text.setAttribute("font-size", "11");
    text.setAttribute("font-family", '"JetBrains Mono", "Fira Code", monospace');
    text.setAttribute("fill", isSelf ? "#ffffff" : isExpired ? "var(--muted)" : "var(--text)");
    const label = name.length > 16 ? name.slice(0, 14) + "…" : name;
    text.textContent = label;
    text.style.pointerEvents = "none";

    g.appendChild(rect);
    g.appendChild(text);
    if (artObj) g.addEventListener("click", () => {
      openArtDetail(artObj);
      _phyloMode = true;
      document.getElementById("artPhyloBtn").classList.add("on");
      _renderArtDetail(state.selectedArtifact);
    });
    svg.appendChild(g);
  }

  const wrapper = document.createElement("div");
  wrapper.style.position = "relative";
  wrapper.style.overflow = "hidden";
  wrapper.style.height = "100%";
  wrapper.style.minHeight = "300px";
  wrapper.style.flex = "1";
  wrapper.style.cursor = "grab";
  wrapper.style.userSelect = "none";

  // SVG is transformed via CSS; wrap in a positioning div anchored at top-left
  // so scale/translate are predictable regardless of wrapper size.
  const stage = document.createElement("div");
  stage.style.position = "absolute";
  stage.style.top = "0";
  stage.style.left = "0";
  stage.style.transformOrigin = "0 0";
  stage.style.willChange = "transform";
  stage.appendChild(svg);
  wrapper.appendChild(stage);

  // Floating zoom controls — overlay top-right
  const controls = document.createElement("div");
  controls.style.cssText = "position:absolute;top:0.5rem;right:0.5rem;display:flex;gap:0.25rem;z-index:2;";
  const mkBtn = (label, title) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "toggle-btn";
    b.textContent = label;
    b.title = title;
    b.style.padding = "0.15rem 0.45rem";
    b.style.fontSize = "0.85rem";
    b.style.lineHeight = "1";
    return b;
  };
  const zoomInBtn = mkBtn("+", "Zoom in");
  const zoomOutBtn = mkBtn("−", "Zoom out");
  const resetBtn = mkBtn("⤢", "Fit to view");
  controls.appendChild(zoomOutBtn);
  controls.appendChild(zoomInBtn);
  controls.appendChild(resetBtn);
  wrapper.appendChild(controls);

  // Pan/zoom state — scale and translation applied to `stage`.
  const MIN_SCALE = 0.1, MAX_SCALE = 4;
  const saved = _phyloView.get(art.name);
  let scale = saved?.scale ?? 1, tx = saved?.tx ?? 0, ty = saved?.ty ?? 0;
  const applyTransform = () => {
    stage.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`;
    _phyloView.set(art.name, { scale, tx, ty });
  };
  const clampScale = s => Math.min(MAX_SCALE, Math.max(MIN_SCALE, s));

  // Fit-to-view: centre the graph and scale to fit wrapper bounds (with margin).
  const fitToView = () => {
    const wr = wrapper.getBoundingClientRect();
    if (wr.width <= 0 || wr.height <= 0) { scale = 1; tx = 0; ty = 0; applyTransform(); return; }
    const sx = (wr.width  - 24) / svgW;
    const sy = (wr.height - 24) / svgH;
    scale = clampScale(Math.min(sx, sy, 1));
    tx = (wr.width  - svgW * scale) / 2;
    ty = (wr.height - svgH * scale) / 2;
    applyTransform();
  };

  // Abort previous render's window listeners — this panel re-renders on every frame
  // update while in phylo mode, so without cleanup we'd leak listeners.
  if (_phyloPanCtl) _phyloPanCtl.abort();
  _phyloPanCtl = new AbortController();
  const sig = _phyloPanCtl.signal;

  // Wheel: zoom toward cursor — keep the world-point under the cursor fixed.
  wrapper.addEventListener("wheel", (e) => {
    e.preventDefault();
    const wr = wrapper.getBoundingClientRect();
    const mx = e.clientX - wr.left, my = e.clientY - wr.top;
    const factor = Math.exp(-e.deltaY * 0.0015);
    const newScale = clampScale(scale * factor);
    const k = newScale / scale;
    tx = mx - k * (mx - tx);
    ty = my - k * (my - ty);
    scale = newScale;
    applyTransform();
  }, { passive: false, signal: sig });

  // Drag to pan
  let dragging = false, dragStartX = 0, dragStartY = 0, dragOrigTx = 0, dragOrigTy = 0;
  let dragMoved = false;
  wrapper.addEventListener("mousedown", (e) => {
    if (e.target.closest("button")) return; // let control buttons handle their own clicks
    dragging = true; dragMoved = false;
    dragStartX = e.clientX; dragStartY = e.clientY;
    dragOrigTx = tx; dragOrigTy = ty;
    wrapper.style.cursor = "grabbing";
    e.preventDefault();
  }, { signal: sig });
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    tx = dragOrigTx + (e.clientX - dragStartX);
    ty = dragOrigTy + (e.clientY - dragStartY);
    if (Math.abs(tx - dragOrigTx) > 3 || Math.abs(ty - dragOrigTy) > 3) dragMoved = true;
    applyTransform();
  }, { signal: sig });
  window.addEventListener("mouseup", () => {
    if (!dragging) return;
    dragging = false;
    wrapper.style.cursor = "grab";
  }, { signal: sig });

  // Suppress node-click after a real drag so clicking-then-dragging doesn't open detail
  wrapper.addEventListener("click", (e) => {
    if (dragMoved) { e.stopPropagation(); dragMoved = false; }
  }, { capture: true, signal: sig });

  // Button-driven zoom — anchor at viewport centre
  const zoomBy = (factor) => {
    const wr = wrapper.getBoundingClientRect();
    const mx = wr.width / 2, my = wr.height / 2;
    const newScale = clampScale(scale * factor);
    const k = newScale / scale;
    tx = mx - k * (mx - tx);
    ty = my - k * (my - ty);
    scale = newScale;
    applyTransform();
  };
  zoomInBtn.addEventListener("click", () => zoomBy(1.25), { signal: sig });
  zoomOutBtn.addEventListener("click", () => zoomBy(0.8), { signal: sig });
  resetBtn.addEventListener("click", () => fitToView(), { signal: sig });

  body.appendChild(wrapper);
  // On first open for this artifact, fit to view; on subsequent re-renders preserve
  // the user's pan/zoom. wrapper has no bounds until inserted, so defer one frame.
  if (saved) requestAnimationFrame(applyTransform);
  else requestAnimationFrame(fitToView);
}

// Called on each frame update — preserves the current view mode (_phyloMode) instead of resetting it.
export function refreshSelectedArtifact(art) {
  state.selectedArtifact = art;
  const posEl = document.getElementById("artDetailPos");
  if (posEl) posEl.textContent = art.pose ? `(${art.pose[0]}, ${art.pose[1]})` : "—";
  _renderArtDetail(art); // respects _phyloMode internally
}

export function openArtDetail(art){
  state.selectedArtifact=art;
  _phyloMode = false;
  document.getElementById("artDetailTitle").textContent=art.name;
  _renderArtDetail(art);
  _resolveArtTags(art).then(()=>_renderArtDetail(art));
  const isMine = ownsArtifact(art.name) && !art.deletion_time;
  document.getElementById("artDetailEditBtn").style.display = isMine ? "" : "none";
  document.getElementById("artDetailDeleteBtn").style.display = isMine ? "" : "none";

  const phyloBtn = document.getElementById("artPhyloBtn");
  phyloBtn.style.display = "";
  phyloBtn.classList.remove("on");
  phyloBtn.onclick = () => {
    _phyloMode = !_phyloMode;
    phyloBtn.classList.toggle("on", _phyloMode);
    _renderArtDetail(state.selectedArtifact);
  };

  const snapshotBtn = document.getElementById("artSnapshotBtn");
  snapshotBtn.style.display = "";
  snapshotBtn.onclick = () => snapshotElement(document.getElementById("artDetailSection"), snapshotBtn);

  document.getElementById("msgSection").style.display="none";
  document.getElementById("artDetailSection").style.display="flex";
  if(state.lastFrame) drawGrid(state.lastFrame.grid_state);
  if(state.lastFrame) renderArtifacts(state.lastFrame.artifacts||[], state.lastFrame.expired_artifacts||[]);
}

export function closeArtDetail(){
  state.selectedArtifact=null;
  _phyloMode = false;
  if (_phyloPanCtl) { _phyloPanCtl.abort(); _phyloPanCtl = null; }
  _phyloView.clear();
  document.getElementById("artPhyloBtn").style.display="none";
  document.getElementById("artSnapshotBtn").style.display="none";
  document.getElementById("artDetailSection").style.display="none";
  document.getElementById("msgSection").style.display="";
  if(state.lastFrame){ drawGrid(state.lastFrame.grid_state); renderArtifacts(state.lastFrame.artifacts||[], state.lastFrame.expired_artifacts||[]); }
}
