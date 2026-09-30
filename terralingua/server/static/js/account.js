const SERVER = window.location.origin;

// ── Background simulation ─────────────────────────────────────────────────────
// Grid-based world (positions snap to cells) but rendered with rounded shapes:
// circles for agents/food/artifacts, no grid lines, faint comm lines between
// neighboring agents. Uses the dashboard's color palette.
(function initBgSim() {
  const canvas = document.getElementById('bg-canvas');
  const ctx = canvas.getContext('2d');
  const CELL = 20;              // CSS px per grid cell; HiDPI scales the buffer
  const TICK_MS = 650;
  const AGENT_COUNT = 30;
  const ARTIFACT_PRODUCE_PROB = 0.012;  // per agent per tick
  const ARTIFACT_LIFE_DECAY = 0.005;    // ~200 ticks ≈ 130s lifespan
  const COMM_RADIUS = 2.4;              // cells; pairs closer than this get a line
  const AGENT_RADIUS = 0.40;            // agent circle radius as fraction of CELL
  let GX, GY, food, agents, artifacts;
  let lastTick = 0;
  let cssW = 0, cssH = 0;

  // Palette — read from CSS vars on body. The 3 entity colors plus --bg are
  // the source of truth (see :root in account.css). The low-density food
  // gradient endpoint is derived: dim variant = mix(food, bg, 0.40).
  let BG_COLOR = '#10131f';
  let AGENT_COLOR = '#6e8efb';
  let AGENT_RGB = [110, 142, 251];
  let ARTIFACT_COLOR = '#fb923c';
  let FL = [20, 80, 50];
  let FD = [50, 155, 95];
  let VIG_RGB = '16,19,31';

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

  function mixRgb(a, b, fraction) {
    return [0,1,2].map(i => Math.round(a[i] * fraction + b[i] * (1 - fraction)));
  }

  function applyPalette() {
    const s = getComputedStyle(document.body);
    BG_COLOR       = (s.getPropertyValue('--bg').trim()             || '#10131f');
    AGENT_COLOR    = (s.getPropertyValue('--agent-color').trim()    || '#6e8efb');
    ARTIFACT_COLOR = (s.getPropertyValue('--artifact-color').trim() || '#fb923c');
    const foodStr  = (s.getPropertyValue('--food-color').trim()     || '#10b981');
    const bgRgb    = parseColor(BG_COLOR, [16,19,31]);
    FD = parseColor(foodStr, [50,155,95]);
    FL = mixRgb(FD, bgRgb, 0.40);
    VIG_RGB = bgRgb.join(',');
    AGENT_RGB = parseColor(AGENT_COLOR, [110,142,251]);
  }
  applyPalette();

  function init() {
    // HiDPI: render at devicePixelRatio so cells/lines are crisp on retina.
    // Drawing API stays in CSS pixels via setTransform(dpr, 0, 0, dpr, ...).
    const dpr = window.devicePixelRatio || 1;
    cssW = window.innerWidth;
    cssH = window.innerHeight;
    canvas.width  = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    canvas.style.width  = cssW + 'px';
    canvas.style.height = cssH + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    GX = Math.ceil(cssW / CELL) + 1;
    GY = Math.ceil(cssH / CELL) + 1;

    food = Float32Array.from({ length: GX * GY }, () =>
      Math.random() < 0.08 ? 0.4 + Math.random() * 0.6 : 0
    );
    agents = Array.from({ length: AGENT_COUNT }, () => {
      const r = Math.floor(Math.random() * GY);
      const c = Math.floor(Math.random() * GX);
      return { row: r, col: c, pr: r, pc: c };
    });
    artifacts = [];
  }
  init();
  window.addEventListener('resize', init);

  const DIRS = [[0,1],[0,-1],[1,0],[-1,0]];

  function tick() {
    // 1. Move agents.
    for (const ag of agents) {
      ag.pr = ag.row;
      ag.pc = ag.col;
      if (Math.random() < 0.55) {
        const [dr, dc] = DIRS[Math.floor(Math.random() * 4)];
        ag.row = (ag.row + dr + GY) % GY;
        ag.col = (ag.col + dc + GX) % GX;
      }
    }
    // 2. Agents eat food on the cell they ended up in.
    for (const ag of agents) {
      const idx = ag.row * GX + ag.col;
      if (food[idx] > 0) food[idx] = 0;
    }
    // 3. Agents occasionally drop an artifact at the cell they JUST LEFT.
    //    Placing it at the previous cell (ag.pr/pc) makes it look like the
    //    agent created the artifact and then walked away — during the next
    //    render frames the lerp animation visibly moves the agent off the
    //    artifact, selling the "I made this" illusion. Skipped for agents
    //    that didn't move this tick (otherwise it just spawns under them).
    for (const ag of agents) {
      const moved = (ag.pr !== ag.row || ag.pc !== ag.col);
      if (moved && Math.random() < ARTIFACT_PRODUCE_PROB) {
        artifacts.push({ row: ag.pr, col: ag.pc, life: 1 });
      }
    }
    // 4. Food decay — cells lose a tiny amount each tick so unvisited patches
    //    don't accumulate forever. Dead cells stay at 0.
    for (let i = 0; i < food.length; i++) {
      if (food[i] > 0) {
        food[i] -= 0.008 + Math.random() * 0.004;
        if (food[i] < 0.05) food[i] = 0;
      }
    }
    // 5. Food regrowth on empty cells.
    for (let i = 0; i < food.length; i++) {
      if (food[i] === 0 && Math.random() < 0.0012) food[i] = 0.5 + Math.random() * 0.4;
    }
    // 6. Artifact decay — fade out, then remove.
    for (let i = artifacts.length - 1; i >= 0; i--) {
      artifacts[i].life -= ARTIFACT_LIFE_DECAY;
      if (artifacts[i].life <= 0) artifacts.splice(i, 1);
    }
  }

  function easeOutCubic(t) { return 1 - Math.pow(1 - t, 3); }

  function draw(now) {
    const W = cssW, H = cssH;

    const tRaw = Math.min(1, (now - lastTick) / TICK_MS);
    const t = easeOutCubic(tRaw);

    ctx.fillStyle = BG_COLOR;
    ctx.fillRect(0, 0, W, H);

    // Food — rounded squares filling cells (continuous patches when clustered).
    // Size still shrinks with `f` so decaying cells visibly taper before vanishing.
    const cornerR = CELL * 0.20;
    ctx.shadowBlur = CELL * 0.35;
    for (let r = 0; r < GY; r++) {
      for (let c = 0; c < GX; c++) {
        const f = food[r * GX + c];
        if (f <= 0) continue;
        const rv = Math.round(FL[0] + (FD[0] - FL[0]) * f);
        const gv = Math.round(FL[1] + (FD[1] - FL[1]) * f);
        const bv = Math.round(FL[2] + (FD[2] - FL[2]) * f);
        const col = `rgb(${rv},${gv},${bv})`;
        ctx.shadowColor = col;
        ctx.fillStyle   = col;
        const size = CELL * (0.55 + 0.45 * f);
        const off  = (CELL - size) / 2;
        ctx.beginPath();
        ctx.roundRect(c * CELL + off, r * CELL + off, size, size, cornerR * (0.6 + 0.4 * f));
        ctx.fill();
      }
    }
    ctx.shadowBlur = 0;

    // Artifacts — orange rounded squares filling cells, alpha-fade with life.
    ctx.shadowColor = ARTIFACT_COLOR;
    ctx.shadowBlur = CELL * 0.5;
    ctx.fillStyle = ARTIFACT_COLOR;
    for (const a of artifacts) {
      ctx.globalAlpha = Math.min(1, a.life * 1.4);
      ctx.beginPath();
      ctx.roundRect(a.col * CELL, a.row * CELL, CELL, CELL, cornerR);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
    ctx.shadowBlur = 0;

    // Compute interpolated agent positions once for both line and dot passes.
    const px = new Float32Array(agents.length);
    const py = new Float32Array(agents.length);
    for (let i = 0; i < agents.length; i++) {
      const ag = agents[i];
      // Snap (skip interpolation) when an agent wraps around a torus edge.
      let dr = ag.row - ag.pr;
      let dc = ag.col - ag.pc;
      if (Math.abs(dr) > 1) dr = 0;
      if (Math.abs(dc) > 1) dc = 0;
      const cellR = ag.pr + dr * t;
      const cellC = ag.pc + dc * t;
      px[i] = cellC * CELL + CELL / 2;
      py[i] = cellR * CELL + CELL / 2;
    }

    // Communication lines — faint accent strokes between near-neighbor agents.
    // Hints visually at "language emerging between minds." O(n²), n ≈ 30.
    const commRadiusPx = COMM_RADIUS * CELL;
    const commR2 = commRadiusPx * commRadiusPx;
    ctx.lineWidth = 0.75;
    for (let i = 0; i < agents.length; i++) {
      for (let j = i + 1; j < agents.length; j++) {
        const dx = px[i] - px[j];
        const dy = py[i] - py[j];
        const d2 = dx * dx + dy * dy;
        if (d2 > commR2) continue;
        const alpha = 0.32 * (1 - Math.sqrt(d2) / commRadiusPx);
        ctx.strokeStyle = `rgba(${AGENT_RGB[0]},${AGENT_RGB[1]},${AGENT_RGB[2]},${alpha})`;
        ctx.beginPath();
        ctx.moveTo(px[i], py[i]);
        ctx.lineTo(px[j], py[j]);
        ctx.stroke();
      }
    }

    // Agents — blue dots with soft glow, drawn on top so they stay visible.
    const agentBaseR = CELL * AGENT_RADIUS;
    ctx.shadowColor = AGENT_COLOR;
    ctx.shadowBlur = CELL * 0.6;
    ctx.fillStyle = AGENT_COLOR;
    for (let i = 0; i < agents.length; i++) {
      ctx.beginPath();
      ctx.arc(px[i], py[i], agentBaseR, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.shadowBlur = 0;

    // Radial vignette — dark edges, atmospheric depth (matches canvas bg)
    const vig = ctx.createRadialGradient(W/2, H/2, 0, W/2, H/2, Math.max(W, H) * 0.72);
    vig.addColorStop(0,    `rgba(${VIG_RGB},0.10)`);
    vig.addColorStop(0.55, `rgba(${VIG_RGB},0.30)`);
    vig.addColorStop(1,    `rgba(${VIG_RGB},0.60)`);
    ctx.fillStyle = vig;
    ctx.fillRect(0, 0, W, H);
  }

  function loop(ts) {
    if (ts - lastTick >= TICK_MS) { tick(); lastTick = ts; }
    draw(ts);
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
}());

// ── Live stats ────────────────────────────────────────────────────────────────
// Animate a stat from its current value to `target` over ~700ms using
// rAF. Easing: easeOutCubic — quick start, gentle settle. Skips work if
// already at target.
function animateStat(el, target) {
  const start = Number(el.dataset.target || 0);
  if (start === target) return;
  el.dataset.target = String(target);
  const t0 = performance.now();
  const dur = 700;
  function step(now) {
    const t = Math.min(1, (now - t0) / dur);
    const eased = 1 - Math.pow(1 - t, 3);
    const cur = Math.round(start + (target - start) * eased);
    el.textContent = cur.toLocaleString();
    if (t < 1) requestAnimationFrame(step);
    else el.textContent = target.toLocaleString();
  }
  requestAnimationFrame(step);
}

async function pollStats() {
  try {
    const r = await fetch('/api/stats');
    if (!r.ok) return;
    const d = await r.json();
    if (d.agents    != null) animateStat(document.getElementById('statAgents'),    d.agents);
    if (d.artifacts != null) animateStat(document.getElementById('statArtifacts'), d.artifacts);
    if (d.observers != null) animateStat(document.getElementById('statUsers'),     d.observers);
  } catch (_) {}
}
pollStats();
setInterval(pollStats, 5000);


const params = new URLSearchParams(window.location.search);
const verifyToken = params.get('verify_token');
const resetToken  = params.get('reset_token');

// Already logged in? (skip during token flows — they override the view)
if (!verifyToken && !resetToken) {
  (async () => {
    try {
      const r = await fetch(`${SERVER}/users/me`, { credentials: 'include' });
      if (!r.ok) return;
      const next = params.get('next');
      if (next && next.startsWith('/')) { window.location.replace(next); return; }
      const user = await r.json();
      if (user?.has_api_key) { window.location.replace('/dashboard'); return; }
      showSettings(user);
    } catch (_) {}
  })();
}

// ── Panel switching ────────────────────────────────────────────────────────────
const PANELS = { login: 'loginForm', register: 'registerForm', forgot: 'forgotPanel', reset: 'resetPanel', verify: 'verifyPanel' };

function showPanel(name) {
  const showTabs = name === 'login' || name === 'register';
  document.querySelector('.tabs').style.visibility = showTabs ? '' : 'hidden';
  document.querySelectorAll('.tab').forEach(b =>
    b.classList.toggle('active', b.dataset.tab === name)
  );
  Object.entries(PANELS).forEach(([k, id]) =>
    document.getElementById(id).classList.toggle('hidden', k !== name)
  );
}

document.querySelectorAll('.tab').forEach(btn =>
  btn.addEventListener('click', () => showPanel(btn.dataset.tab))
);

// ── Password show/hide toggles ────────────────────────────────────────────────
document.querySelectorAll('.pwd-toggle').forEach(btn => {
  btn.addEventListener('click', () => {
    const input = document.getElementById(btn.dataset.target);
    const showing = input.type === 'text';
    input.type = showing ? 'password' : 'text';
    btn.querySelector('.eye-show').classList.toggle('hidden', showing);
    btn.querySelector('.eye-hide').classList.toggle('hidden', !showing);
    btn.setAttribute('aria-label', showing ? 'Show password' : 'Hide password');
  });
});

// ── Helpers ───────────────────────────────────────────────────────────────────
function showError(id, msg) {
  const el = document.getElementById(id);
  el.textContent = msg;
  el.classList.remove('hidden');
}

function clearError(id) {
  const el = document.getElementById(id);
  el.textContent = '';
  el.classList.add('hidden');
}

async function doLogin(email, password) {
  const r = await fetch(`${SERVER}/auth/jwt/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({ username: email, password }),
    credentials: 'include',
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    const d = err.detail || '';
    throw new Error(d === 'LOGIN_BAD_CREDENTIALS' ? 'Invalid email or password.' : (d || 'Login failed.'));
  }
}

// ── Login form ────────────────────────────────────────────────────────────────
document.getElementById('loginForm').addEventListener('submit', async e => {
  e.preventDefault();
  clearError('loginError');
  const btn = document.getElementById('loginBtn');
  btn.disabled = true;
  try {
    await doLogin(
      document.getElementById('loginEmail').value.trim(),
      document.getElementById('loginPassword').value,
    );
    const next = params.get('next');
    if (next && next.startsWith('/')) { window.location.replace(next); return; }
    const meRes = await fetch(`${SERVER}/users/me`, { credentials: 'include' });
    const user = meRes.ok ? await meRes.json() : null;
    window.location.replace(user?.has_api_key ? '/dashboard' : '/login');
  } catch (err) {
    showError('loginError', err.message);
    btn.disabled = false;
  }
});

// ── Register form ─────────────────────────────────────────────────────────────
document.getElementById('registerForm').addEventListener('submit', async e => {
  e.preventDefault();
  clearError('registerError');
  const btn = document.getElementById('registerBtn');
  const username = document.getElementById('regUsername').value.trim();
  const email    = document.getElementById('regEmail').value.trim();
  const password = document.getElementById('regPassword').value;
  const confirm  = document.getElementById('regPasswordConfirm').value;

  if (password !== confirm) {
    showError('registerError', 'Passwords do not match.');
    return;
  }

  btn.disabled = true;
  try {
    const r = await fetch(`${SERVER}/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, email, password }),
      credentials: 'include',
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      const d = err.detail || '';
      throw new Error(
        d === 'REGISTER_USER_ALREADY_EXISTS' ? 'Username or email already in use.' :
        (d || 'Registration failed.')
      );
    }
    document.getElementById('registerForm').innerHTML =
      '<p style="text-align:center">Account created! Check your inbox to verify your email address, then <a href="/login">log in</a>.</p>';
  } catch (err) {
    showError('registerError', err.message);
    btn.disabled = false;
  }
});

// ── Email verification ────────────────────────────────────────────────────────

if (verifyToken) {
  showPanel('verify');
  (async () => {
    try {
      const r = await fetch(`${SERVER}/auth/verify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token: verifyToken }),
      });
      if (r.ok) {
        document.getElementById('verifyMessage').innerHTML =
          'Email verified! Your account is active. <a href="/login">Sign in →</a>';
      } else {
        const e = await r.json().catch(() => ({}));
        const msg = e.detail === 'VERIFY_USER_ALREADY_VERIFIED'
          ? 'This email is already verified.'
          : (e.detail || 'Invalid or expired link.');
        document.getElementById('verifyMessage').classList.add('hidden');
        showError('verifyError', msg + ' <a href="/login">Back to sign in</a>');
      }
    } catch (_) {
      document.getElementById('verifyMessage').classList.add('hidden');
      showError('verifyError', 'Could not reach the server.');
    }
  })();
}

// ── Forgot / reset password ───────────────────────────────────────────────────

if (resetToken) showPanel('reset');

document.getElementById('forgotLink').addEventListener('click', e => {
  e.preventDefault();
  showPanel('forgot');
});

document.getElementById('backToLogin').addEventListener('click', e => {
  e.preventDefault();
  showPanel('login');
});

document.getElementById('forgotBtn').addEventListener('click', async () => {
  clearError('forgotError');
  document.getElementById('forgotSuccess').classList.add('hidden');
  const email = document.getElementById('forgotEmail').value.trim();
  if (!email) { showError('forgotError', 'Please enter your email address.'); return; }
  const btn = document.getElementById('forgotBtn');
  btn.disabled = true;
  try {
    await fetch(`${SERVER}/auth/forgot-password`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email }),
    });
    const el = document.getElementById('forgotSuccess');
    el.textContent = 'If that email is registered, a reset link is on its way.';
    el.classList.remove('hidden');
  } catch (_) {
    showError('forgotError', 'Something went wrong. Please try again.');
  } finally {
    btn.disabled = false;
  }
});

document.getElementById('resetBtn').addEventListener('click', async () => {
  clearError('resetError');
  const password = document.getElementById('resetPassword').value;
  const confirm  = document.getElementById('resetPasswordConfirm').value;
  if (password !== confirm) { showError('resetError', 'Passwords do not match.'); return; }
  const btn = document.getElementById('resetBtn');
  btn.disabled = true;
  try {
    const r = await fetch(`${SERVER}/auth/reset-password`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token: resetToken, password }),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      const d = err.detail || '';
      throw new Error(d === 'RESET_PASSWORD_BAD_TOKEN'
        ? 'This link has expired or is invalid. Please request a new one.'
        : (d || 'Reset failed.'));
    }
    document.getElementById('resetPanel').innerHTML =
      '<p style="text-align:center;padding:1rem 0">Password updated! <a href="/login">Sign in</a></p>';
  } catch (err) {
    showError('resetError', err.message);
    btn.disabled = false;
  }
});

// ── Settings panel (logged-in view) ──────────────────────────────────────────

function showSettings(user) {
  document.getElementById('authCard').classList.add('hidden');
  document.getElementById('settingsCard').classList.remove('hidden');
  document.getElementById('settingsUsername').textContent = user.username;
  document.getElementById('settingsEmail').textContent = user.email;
}

document.getElementById('logoutBtn').addEventListener('click', async () => {
  await fetch(`${SERVER}/auth/jwt/logout`, { method: 'POST', credentials: 'include' });
  window.location.reload();
});

document.getElementById('guestLink').addEventListener('click', e => {
  e.preventDefault();
  document.cookie = 'guest_ok=1; path=/; SameSite=Lax';
  window.location.href = '/dashboard';
});
