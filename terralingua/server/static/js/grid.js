import { state, familyTags } from './state.js';

export function getGraphNodePositions(graphData, width, height) {
  const nodes = graphData.nodes || [];
  const layout = graphData.node_layout || {};
  const positions = new Map();
  if (!nodes.length) return positions;

  const layoutNodes = nodes.filter(n => Array.isArray(layout[n]) && layout[n].length >= 2);
  if (layoutNodes.length === nodes.length) {
    const xs = layoutNodes.map(n => Number(layout[n][0]) || 0);
    const ys = layoutNodes.map(n => Number(layout[n][1]) || 0);
    const bounds = graphData.layout_bounds;
    const minX = bounds ? bounds[0] : Math.min(...xs), maxX = bounds ? bounds[2] : Math.max(...xs);
    const minY = bounds ? bounds[1] : Math.min(...ys), maxY = bounds ? bounds[3] : Math.max(...ys);
    const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
    const pad = Math.min(Math.min(width, height) * 0.3, Math.max(graphData.layout_bounds ? 60 : 36, Math.min(width, height) * 0.12));
    const spanX = Math.max(maxX - minX, 0.001);
    const spanY = Math.max(maxY - minY, 0.001);
    const scale = Math.min((width - pad * 2) / spanX, (height - pad * 2) / spanY);
    for (const n of nodes) {
      positions.set(n, {
        x: width / 2 + ((Number(layout[n][0]) || 0) - cx) * scale,
        y: height / 2 + ((Number(layout[n][1]) || 0) - cy) * scale,
      });
    }
    return positions;
  }

  const radius = Math.max(40, Math.min(width, height) * 0.34);
  const centerX = width / 2, centerY = height / 2;
  nodes.forEach((n, i) => {
    const angle = -Math.PI / 2 + (2 * Math.PI * i) / nodes.length;
    positions.set(n, {
      x: centerX + Math.cos(angle) * radius,
      y: centerY + Math.sin(angle) * radius,
    });
  });
  return positions;
}

// Small filled circle just outside the `to` node, marking the followee end of
// an edge. One dot = one-way follow; a dot at each end = mutual connection.
function drawEdgeDot(ctx, from, to, radius, color) {
  const dx = to.x - from.x, dy = to.y - from.y;
  const len = Math.hypot(dx, dy);
  if (!len) return;
  const ux = dx / len, uy = dy / len;
  const r = Math.max(2.5, radius * 0.2);
  const cx = to.x - ux * (radius + r + 1);
  const cy = to.y - uy * (radius + r + 1);
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.fill();
}

function drawGraph(graphData, targetCanvas = null) {
  const canvas = targetCanvas || document.getElementById("grid-canvas");
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  const dark = !document.body.classList.contains("light");
  // Match the body's radial-gradient dot pattern (dashboard.css) so the canvas
  // blends seamlessly with the rest of the page.
  const BG        = dark ? "#0a0a0a"                : "#f0f2f5";
  const DOT       = dark ? "#1e2235"                : "#d8dce6";
  const EDGE      = dark ? "rgba(185,195,220,0.22)" : "rgba(35,45,70,0.22)";
  const EDGE_HI   = dark ? "rgba(110,142,251,0.65)" : "rgba(40,90,140,0.55)";
  // Edge-end dots are near-opaque so the faint line doesn't show through them.
  const EDGE_DOT    = dark ? "rgba(205,213,235,0.95)" : "rgba(30,40,65,0.95)";
  const EDGE_DOT_HI = dark ? "rgba(140,168,255,0.98)" : "rgba(30,75,125,0.97)";
  const NODE      = dark ? "#20283a"                : "#ffffff";
  const NODE_LINE = dark ? "rgba(255,255,255,0.18)" : "rgba(0,0,0,0.16)";
  const HOP       = dark ? "rgba(110,142,251,0.14)" : "rgba(74,110,200,0.12)";
  const AGENT_COL = dark ? "#6e8efb"                : "rgb(40,90,140)";
  const FOOD      = dark ? "#50d184"                : "#41a85f";
  const ARTIFACT  = dark ? "#fb923c"                : "rgb(180,60,70)";
  const TEXT      = dark ? "rgba(255,255,255,0.9)"  : "rgba(15,23,42,0.9)";
  const MUTED     = dark ? "rgba(255,255,255,0.58)" : "rgba(15,23,42,0.58)";

  ctx.fillStyle = BG;
  ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = DOT;
  const step = 24;
  for (let y = step / 2; y < H; y += step) {
    for (let x = step / 2; x < W; x += step) {
      ctx.fillRect(x, y, 1, 1);
    }
  }
  ctx.save();
  ctx.translate(state.mapPanX, state.mapPanY);
  ctx.scale(state.mapZoom, state.mapZoom);

  const positions = getGraphNodePositions(graphData, W, H);
  const nodes = graphData.nodes || [];
  const hopNodes = new Set(graphData.hop_nodes || []);
  const agentsByNode = new Map();
  for (const ag of graphData.agents || []) {
    if (!agentsByNode.has(ag.node)) agentsByNode.set(ag.node, []);
    agentsByNode.get(ag.node).push(ag);
  }
  const foodByNode = new Map((graphData.food || []).map(f => [f.node, f]));
  const artifactNodes = new Set((graphData.artifacts || []).map(a => a.node));
  const selectedNode = state.graphReplay ? state.graphReplay.selectedNode : (graphData.agents || []).find(a => a.tag === state.selectedTag)?.node;
  const nodeRadius = Math.max(15, Math.min(28, Math.min(W, H) / Math.max(22, nodes.length * 1.5)));

  ctx.lineCap = "round";
  for (const e of graphData.edges || []) {
    const u = e.source, v = e.target;
    const a = positions.get(u), b = positions.get(v);
    if (!a || !b) continue;
    const highlighted = u === selectedNode || v === selectedNode;
    const color = highlighted ? EDGE_HI : EDGE;
    const dotColor = highlighted ? EDGE_DOT_HI : EDGE_DOT;
    ctx.strokeStyle = color;
    ctx.lineWidth = (highlighted ? 2.4 : 1.4) / state.mapZoom;
    ctx.setLineDash(e.type === "pending" ? [6 / state.mapZoom, 5 / state.mapZoom] : []);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    ctx.setLineDash([]);
    // Dot marks the followee end. mutual → dot at both ends.
    drawEdgeDot(ctx, a, b, nodeRadius, dotColor);
    if (e.type === "mutual") drawEdgeDot(ctx, b, a, nodeRadius, dotColor);
  }

  for (const n of nodes) {
    const p = positions.get(n);
    if (!p) continue;
    if (hopNodes.has(n)) {
      ctx.fillStyle = HOP;
      ctx.beginPath();
      ctx.arc(p.x, p.y, nodeRadius * 1.85, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  for (const n of nodes) {
    const p = positions.get(n);
    if (!p) continue;
    const isSelected = n === selectedNode;
    ctx.fillStyle = NODE;
    ctx.strokeStyle = isSelected ? AGENT_COL : NODE_LINE;
    ctx.lineWidth = (isSelected ? 3 : 1.4) / state.mapZoom;
    ctx.beginPath();
    ctx.arc(p.x, p.y, nodeRadius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    if (foodByNode.has(n)) {
      ctx.fillStyle = FOOD;
      ctx.beginPath();
      ctx.arc(p.x + nodeRadius * 0.58, p.y - nodeRadius * 0.58, nodeRadius * 0.24, 0, Math.PI * 2);
      ctx.fill();
    }
    if (artifactNodes.has(n)) {
      ctx.fillStyle = ARTIFACT;
      const s = nodeRadius * 0.35;
      ctx.beginPath();
      ctx.moveTo(p.x - nodeRadius * 0.58, p.y - nodeRadius * 0.58 - s);
      ctx.lineTo(p.x - nodeRadius * 0.58 + s, p.y - nodeRadius * 0.58);
      ctx.lineTo(p.x - nodeRadius * 0.58, p.y - nodeRadius * 0.58 + s);
      ctx.lineTo(p.x - nodeRadius * 0.58 - s, p.y - nodeRadius * 0.58);
      ctx.closePath();
      ctx.fill();
    }

    const agents = agentsByNode.get(n) || [];
    agents.forEach((ag, i) => {
      const offset = (i - (agents.length - 1) / 2) * nodeRadius * 0.48;
      const ax = p.x + offset, ay = p.y + nodeRadius * 0.02;
      const selected = state.graphReplay ? n === selectedNode : ag.tag === state.selectedTag;
      ctx.fillStyle = AGENT_COL;
      ctx.strokeStyle = selected ? (dark ? "#ffffff" : "#e53935") : BG;
      ctx.lineWidth = (selected ? 3 : 1.8) / state.mapZoom;
      ctx.beginPath();
      ctx.arc(ax, ay, nodeRadius * 0.43, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    });

    if (state.showNames || nodes.length <= 24) {
      ctx.font = `600 ${Math.max(11, Math.min(15, nodeRadius * 0.68))}px "Inter",sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillStyle = TEXT;
      ctx.fillText(n, p.x, p.y + nodeRadius + 6);
      const agentNames = agents.map(a => a.name || a.tag).filter(Boolean);
      if (agentNames.length && agentNames.join(", ") !== n) {
        ctx.font = `500 ${Math.max(10, Math.min(13, nodeRadius * 0.56))}px "Inter",sans-serif`;
        ctx.fillStyle = MUTED;
        ctx.fillText(agentNames.join(", "), p.x, p.y + nodeRadius + 22);
      }
    }
  }

  if (!nodes.length) {
    ctx.fillStyle = MUTED;
    ctx.font = '500 14px "Inter",sans-serif';
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("No graph nodes", W / 2, H / 2);
  }
  ctx.restore();
}

// Parse "rgb(r,g,b)" / "rgba(r,g,b,a)" / "#rrggbb" / "#rgb" → [r,g,b].
function parseColor(str, fallback) {
  if (!str) return fallback;
  const s = str.trim();
  if (s.startsWith('#')) {
    const h = s.slice(1);
    if (h.length === 3) return [parseInt(h[0]+h[0],16), parseInt(h[1]+h[1],16), parseInt(h[2]+h[2],16)];
    if (h.length === 6) return [parseInt(h.slice(0,2),16), parseInt(h.slice(2,4),16), parseInt(h.slice(4,6),16)];
    return fallback;
  }
  const m = s.match(/rgba?\(([^)]+)\)/);
  if (!m) return fallback;
  const parts = m[1].split(/[ ,/]+/).slice(0, 3).map(n => parseFloat(n));
  if (parts.length === 3 && parts.every(n => !isNaN(n))) return parts.map(n => Math.round(n));
  return fallback;
}

// Mix two [r,g,b] colors; `fraction` is the weight of `a`.
function mixRgb(a, b, fraction) {
  return [0,1,2].map(i => Math.round(a[i] * fraction + b[i] * (1 - fraction)));
}

export function drawGrid(gridData, targetCanvas = null) {
  if (state.graphReplay) gridData = state.graphReplay.graph;
  if (!gridData) {
    const canvas = targetCanvas || document.getElementById("grid-canvas");
    const ctx = canvas.getContext("2d");
    const light = document.body.classList.contains("light");
    ctx.fillStyle = light ? "#f0f2f5" : "#0a0a0a";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = light ? "#667085" : "#9ca3af";
    ctx.font = '500 14px "Inter",sans-serif';
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText("No live frame available", canvas.width / 2, canvas.height / 2);
    return;
  }
  if (gridData?.env_type === "graph") {
    drawGraph(gridData, targetCanvas);
    if (state.graphReplay) {
      const canvas = targetCanvas || document.getElementById("grid-canvas");
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = document.body.classList.contains("light") ? "#f0f2f5" : "#0a0a0a";
      ctx.fillRect(0, 0, canvas.width, 34);
      ctx.fillStyle = document.body.classList.contains("light") ? "#172033" : "#e5e9f3";
      ctx.font = '600 14px "Inter",sans-serif';
      ctx.textAlign = "left"; ctx.textBaseline = "middle";
      ctx.fillText(`Recorded graph · step ${state.graphReplay.step}`, 12, 17);
    }
    return;
  }
  const canvas = targetCanvas || document.getElementById("grid-canvas");
  const ctx = canvas.getContext("2d");
  const gs = gridData.grid_size, W = canvas.width, H = canvas.height;
  const cW = W/gs, cH = H/gs;

  // Entity colors come from the single source of truth in dashboard.css
  // (--agent-color, --food-color, --artifact-color, --bg). Theme switches the
  // CSS vars; this code is theme-agnostic. To change colors, edit dashboard.css.
  const styles     = getComputedStyle(document.body);
  const BG         = styles.getPropertyValue('--bg').trim()             || '#0a0a0a';
  const AGENT_COL  = styles.getPropertyValue('--agent-color').trim()    || '#6e8efb';
  const ARTIFACT   = styles.getPropertyValue('--artifact-color').trim() || '#fb923c';
  const FOOD_COLOR = styles.getPropertyValue('--food-color').trim()     || '#10b981';

  // Food gradient: high-density food = --food-color; low-density = mixed with bg.
  const FD     = parseColor(FOOD_COLOR, [16,185,129]);
  const BG_RGB = parseColor(BG,         [10,10,10]);
  const FL     = mixRgb(FD, BG_RGB, 0.35);

  // Rgba helpers derived from entity colors so tints follow theme.
  const AG_RGB = parseColor(AGENT_COL, [110,142,251]);
  const AR_RGB = parseColor(ARTIFACT,  [251,146,60]);
  const agRgba  = (a) => `rgba(${AG_RGB[0]},${AG_RGB[1]},${AG_RGB[2]},${a})`;
  const artRgba = (a) => `rgba(${AR_RGB[0]},${AR_RGB[1]},${AR_RGB[2]},${a})`;
  const VISION  = agRgba(0.10);

  // Theme flag — used only for the few highlight colors that aren't tied to
  // an entity (selected/family agent borders go white in dark mode, red in light).
  const dark = !document.body.classList.contains("light");

  ctx.fillStyle=BG; ctx.fillRect(0,0,W,H);
  ctx.save();
  ctx.translate(state.mapPanX, state.mapPanY);
  ctx.scale(state.mapZoom, state.mapZoom);

  // Compute which grid tiles are visible so we can draw the toroidal wrap
  const visX0 = -state.mapPanX / state.mapZoom, visX1 = visX0 + W / state.mapZoom;
  const visY0 = -state.mapPanY / state.mapZoom, visY1 = visY0 + H / state.mapZoom;
  const tiX0 = Math.floor(visX0 / W), tiX1 = Math.ceil(visX1 / W);
  const tiY0 = Math.floor(visY0 / H), tiY1 = Math.ceil(visY1 / H);

  for(let ti = tiX0; ti < tiX1; ti++){
    for(let tj = tiY0; tj < tiY1; tj++){
      ctx.save();
      ctx.translate(ti * W, tj * H);

      // ── Shape helpers — hybrid style:
      //   food/artifacts → rounded squares filling cells (continuous patches)
      //   agents         → circles        (alive things moving through the world)
      const cornerR   = cW * 0.20;
      const agentR    = Math.min(cW, cH) * 0.40;
      const cellCx    = (gy) => gy * cW + cW / 2;
      const cellCy    = (gx) => gx * cH + cH / 2;
      const fillRound = (gy, gx) => {
        ctx.beginPath();
        ctx.roundRect(gy * cW, gx * cH, cW, cH, cornerR);
        ctx.fill();
      };
      const strokeRound = (gy, gx, sw) => {
        ctx.beginPath();
        ctx.roundRect(gy * cW + sw/2, gx * cH + sw/2,
                      cW - sw, cH - sw, Math.max(0, cornerR - sw/2));
        ctx.stroke();
      };
      const fillAgent = (gy, gx) => {
        ctx.beginPath();
        ctx.arc(cellCx(gy), cellCy(gx), agentR, 0, Math.PI * 2);
        ctx.fill();
      };
      const strokeAgent = (gy, gx, sw) => {
        ctx.beginPath();
        ctx.arc(cellCx(gy), cellCy(gx), Math.max(1, agentR - sw/2), 0, Math.PI * 2);
        ctx.stroke();
      };

      // Vision cells — translucent territory highlight, stays as full-cell
      // fillRect (it's a property of cells, not an object).
      ctx.fillStyle=VISION;
      for(const [x,y] of gridData.vision_cells) ctx.fillRect(y*cW,x*cH,cW,cH);

      // Food — green rounded squares, full-size (ratio drives color, not size,
      // since the dashboard's data already encodes intensity as color).
      for(const f of gridData.food){
        const r=Math.round(FL[0]+(FD[0]-FL[0])*f.ratio),g=Math.round(FL[1]+(FD[1]-FL[1])*f.ratio),b=Math.round(FL[2]+(FD[2]-FL[2])*f.ratio);
        ctx.fillStyle=`rgb(${r},${g},${b})`;
        fillRound(f.y, f.x);
      }

      // Artifacts — orange rounded squares.
      ctx.fillStyle=ARTIFACT;
      for(const a of gridData.artifacts) fillRound(a.y, a.x);

      if(state.selectedArtifact?.pose){
        const [ax,ay]=state.selectedArtifact.pose;
        ctx.shadowColor=ARTIFACT; ctx.shadowBlur=Math.max(8,cW*0.8);
        ctx.fillStyle=ARTIFACT; fillRound(ay, ax);
        ctx.shadowBlur=0;
        const sw=Math.max(2,cW*0.14);
        ctx.strokeStyle=dark?"#ffffff":"#ffb347"; ctx.lineWidth=sw;
        strokeRound(ay, ax, sw);
      }
      if(state.pickingFor && state.hoverCell){
        const {row:hr,col:hc}=state.hoverCell;
        const pickColor = state.pickingFor==="agent" ? (AGENT_COL) : ARTIFACT;
        const pickFill  = state.pickingFor==="agent" ? agRgba(0.35) : artRgba(0.35);
        const pickLine  = state.pickingFor==="agent" ? agRgba(0.5)  : artRgba(0.5);
        ctx.shadowColor=pickColor; ctx.shadowBlur=Math.max(8,cW*0.8);
        ctx.fillStyle=pickFill;
        if(state.pickingFor==="agent") fillAgent(hc, hr); else fillRound(hc, hr);
        ctx.shadowBlur=0;
        const sw=Math.max(2,cW*0.14);
        ctx.strokeStyle=pickColor; ctx.lineWidth=sw;
        if(state.pickingFor==="agent") strokeAgent(hc, hr, sw); else strokeRound(hc, hr, sw);
        // Crosshair guides — keep so the user can read the picked coordinate.
        ctx.strokeStyle=pickLine; ctx.lineWidth=1/state.mapZoom;
        ctx.setLineDash([3/state.mapZoom,3/state.mapZoom]);
        ctx.beginPath(); ctx.moveTo(hc*cW+cW/2,0); ctx.lineTo(hc*cW+cW/2,H); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(0,hr*cH+cH/2); ctx.lineTo(W,hr*cH+cH/2); ctx.stroke();
        ctx.setLineDash([]);
      }
      ctx.shadowBlur=0;

      // Agents — blue circles. Soft glow on selected/family-highlighted only.
      for(const ag of gridData.agents){
        if(ag.tag===state.selectedTag) continue;
        ctx.fillStyle=AGENT_COL;
        fillAgent(ag.y, ag.x);
      }
      if(state.familyHighlight && familyTags.size){
        const familyCol=dark?"#ef4444":"#f97316";
        for(const ag of gridData.agents){
          if(!familyTags.has(ag.tag)) continue;
          ctx.shadowColor=familyCol; ctx.shadowBlur=Math.max(6,cW*0.7);
          ctx.fillStyle=AGENT_COL; fillAgent(ag.y, ag.x);
          ctx.shadowBlur=0;
          const sw=Math.max(1.5,cW*0.12);
          ctx.strokeStyle=familyCol; ctx.lineWidth=sw;
          strokeAgent(ag.y, ag.x, sw);
        }
      }
      for(const ag of gridData.agents){
        if(ag.tag!==state.selectedTag) continue;
        ctx.shadowColor=AGENT_COL; ctx.shadowBlur=Math.max(8,cW*0.8);
        ctx.fillStyle=AGENT_COL; fillAgent(ag.y, ag.x);
        ctx.shadowBlur=0;
        const sw=Math.max(2,cW*0.14);
        ctx.strokeStyle=dark?"#ffffff":"#e53935"; ctx.lineWidth=sw;
        strokeAgent(ag.y, ag.x, sw);
      }
      if(state.selectedTag && gridData.vision_radius != null){
        const selAg = gridData.agents.find(ag => ag.tag===state.selectedTag);
        if(selAg){
          const r = gridData.vision_radius;
          ctx.fillStyle = agRgba(dark ? 0.18 : 0.14);
          for(let dx=-r; dx<=r; dx++)
            for(let dy=-r; dy<=r; dy++){
              const vx=((selAg.x+dx)%gs+gs)%gs, vy=((selAg.y+dy)%gs+gs)%gs;
              ctx.fillRect(vy*cW, vx*cH, cW, cH);
            }
          if(selAg.x-r>=0 && selAg.y-r>=0 && selAg.x+r<gs && selAg.y+r<gs){
            const sw=Math.max(1.5, cW*0.08);
            ctx.strokeStyle=agRgba(dark ? 0.75 : 0.65);
            ctx.lineWidth=sw;
            ctx.strokeRect((selAg.y-r)*cW+sw/2, (selAg.x-r)*cH+sw/2, (2*r+1)*cW-sw, (2*r+1)*cH-sw);
          }
        }
      }
      if(state.showNames){
        ctx.fillStyle=dark?"rgba(255,255,255,0.9)":"#000"; ctx.textAlign="center"; ctx.textBaseline="bottom";
        ctx.font=`600 ${Math.max(11,Math.min(18,cW*0.65))}px "Inter",sans-serif`;
        for(const ag of gridData.agents)
          ctx.fillText(ag.name||ag.tag, (ag.y+0.5)*cW, ag.x*cH-1);
      }

      ctx.restore();
    }
  }
  ctx.restore();
}
