import { SERVER, state, escHtml } from './state.js';
import { agentStore } from './agents.js';
import { getAnonId } from './crypto.js';
import { wsRequest } from './ws.js';

// ── Session check ─────────────────────────────────────────────────────────────

export async function checkSession() {
  try {
    const resp = await fetch(`${SERVER}/users/me`, { credentials: 'include' });
    if (resp.ok) {
      state.currentUser = await resp.json();
    } else {
      state.currentUser = null;
    }
  } catch (_) {
    state.currentUser = null;
  }
  renderUserBadge();
}

// ── Login / register / logout ─────────────────────────────────────────────────

export async function login(email, password) {
  const body = new URLSearchParams({ username: email, password });
  const resp = await fetch(`${SERVER}/auth/jwt/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body,
    credentials: 'include',
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    const detail = err.detail || '';
    throw new Error(detail === 'LOGIN_BAD_CREDENTIALS' ? 'Invalid email or password.' : (detail || 'Login failed.'));
  }
  location.reload();
}

export async function register(username, email, password) {
  const resp = await fetch(`${SERVER}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, email, password }),
    credentials: 'include',
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    const detail = err.detail || '';
    throw new Error(detail === 'REGISTER_USER_ALREADY_EXISTS' ? 'Username or email already in use.' : (detail || 'Registration failed.'));
  }
  await login(email, password);
}

export async function logout() {
  await fetch(`${SERVER}/auth/jwt/logout`, { method: 'POST', credentials: 'include' });
  state.currentUser = null;
  location.reload();
}

// ── Agent reconnection (logged-in users only) ─────────────────────────────────

export async function reconnectUserAgents() {
  try {
    const headers = {};
    if (!state.currentUser) headers['X-Anon-Id'] = getAnonId();
    const resp = await fetch(`${SERVER}/auth/my-agents`, { credentials: 'include', headers });
    if (!resp.ok) return;
    const { agents } = await resp.json();
    for (const a of agents) {
      if (!agentStore.has(a.agent_tag))
        agentStore.trackAgent(a.agent_tag, a.token, { model: a.model_info || '' });
    }
  } catch (_) {}
}

// ── User badge (topbar) ───────────────────────────────────────────────────────

export function renderUserBadge() {
  const el = document.getElementById('userBadge');
  if (!el) return;
  const drawer = document.getElementById('settingsDrawer');

  if (state.currentUser) {
    const initial = escHtml(state.currentUser.username.charAt(0).toUpperCase());
    el.innerHTML = `
      <button class="sd-topbar-trigger" id="sdTopbarTrigger">
        <span class="sd-topbar-avatar">${initial}</span>
        <span class="sd-topbar-username">${escHtml(state.currentUser.username)}</span>
      </button>`;
  } else {
    const guestIcon = `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>`;
    el.innerHTML = `
      <button class="sd-topbar-trigger" id="sdTopbarTrigger">
        <span class="sd-topbar-avatar sd-topbar-avatar--guest">${guestIcon}</span>
        <span class="sd-topbar-username" style="color:var(--muted)">Guest</span>
      </button>
      <button class="user-badge-btn--signin" id="userSignInBtn">Sign in</button>`;
    document.getElementById('userSignInBtn').addEventListener('click', showAuthModal);
  }

  document.getElementById('sdTopbarTrigger').addEventListener('mousedown', e => {
    e.stopPropagation();
    drawer.classList.toggle('open');
  });

  // API key row in the spawn drawer is only needed for guests; logged-in users
  // have their key stored server-side. This runs after checkSession() resolves,
  // fixing the race where initDrawers() always saw currentUser=null.
  document.getElementById('dApiKeyRow')?.classList.toggle('hidden', !!state.currentUser);
}

// ── Auth redirect ─────────────────────────────────────────────────────────────

export function showAuthModal() {
  window.location.href = `/login?next=${encodeURIComponent(window.location.pathname)}`;
}

// ── Settings drawer ───────────────────────────────────────────────────────────

export function initSettingsDrawer() {
  const user = state.currentUser;

  if (user) {
    // Populate and show signed-in section
    const initial = user.username.charAt(0).toUpperCase();
    document.getElementById('sdAvatar').textContent = initial;
    document.getElementById('sdUsername').textContent = user.username;
    document.getElementById('sdEmail').textContent = user.email;
    document.getElementById('sdApiKeyHint').textContent = user.has_api_key
      ? 'A key is saved — enter a new one to replace it.'
      : 'No key saved. Agents need a key to run.';
    document.getElementById('sdSignedIn').classList.remove('hidden');
    document.getElementById('sdSignedInFooter').classList.remove('hidden');
    document.getElementById('sdApiKeySignedIn').classList.remove('hidden');
  } else {
    document.getElementById('sdSignedOut').classList.remove('hidden');
    document.getElementById('sdApiKeySignedOut').classList.remove('hidden');
  }

  // Eye toggle
  document.getElementById('sdApiKeyToggle')?.addEventListener('click', () => {
    const input = document.getElementById('sdApiKey');
    const showing = input.type === 'password';
    input.type = showing ? 'text' : 'password';
    document.querySelector('.sd-eye-show').classList.toggle('hidden', showing);
    document.querySelector('.sd-eye-hide').classList.toggle('hidden', !showing);
  });

  // Save settings
  document.getElementById('sdSaveBtn')?.addEventListener('click', async () => {
    const btn    = document.getElementById('sdSaveBtn');
    const apiKey = document.getElementById('sdApiKey').value.trim();
    const status = document.getElementById('sdStatus');
    status.className = 'sd-status';
    status.textContent = '';
    btn.disabled = true;
    try {
      if (!apiKey) throw new Error('Enter an API key.');
      // Save via the dashboard WebSocket so the server can mirror the key
      // to the AgentRecords and session entry atomically with the DB write.
      // wsRequest rejects on server-side errors.
      await wsRequest('set_api_key', { api_key: apiKey });
      document.getElementById('sdApiKey').value = '';
      document.getElementById('sdApiKeyHint').textContent = 'A key is saved — enter a new one to replace it.';
      if (state.currentUser) state.currentUser.has_api_key = true;
      status.className = 'sd-status success';
      status.textContent = 'API key saved.';
      setTimeout(() => { status.className = 'sd-status'; status.textContent = ''; }, 3000);
    } catch (err) {
      status.className = 'sd-status error';
      status.textContent = err.message;
    } finally {
      btn.disabled = false;
    }
  });

  // Anonymous session API key save (session-scoped, no DB)
  document.getElementById('sdAnonApiKeyToggle')?.addEventListener('click', () => {
    const input = document.getElementById('sdAnonApiKey');
    const showing = input.type === 'password';
    input.type = showing ? 'text' : 'password';
    document.querySelector('.sd-anon-eye-show').classList.toggle('hidden', showing);
    document.querySelector('.sd-anon-eye-hide').classList.toggle('hidden', !showing);
  });

  document.getElementById('sdAnonSaveBtn')?.addEventListener('click', async () => {
    const btn    = document.getElementById('sdAnonSaveBtn');
    const apiKey = document.getElementById('sdAnonApiKey').value.trim();
    const status = document.getElementById('sdAnonStatus');
    status.className = 'sd-status';
    status.textContent = '';
    btn.disabled = true;
    try {
      if (!apiKey) throw new Error('Enter an API key.');
      await wsRequest('set_api_key', { api_key: apiKey });
      document.getElementById('sdAnonApiKey').value = '';
      status.className = 'sd-status success';
      status.textContent = 'Key saved for this session.';
      setTimeout(() => { status.className = 'sd-status'; status.textContent = ''; }, 3000);
    } catch (err) {
      status.className = 'sd-status error';
      status.textContent = err.message;
    } finally {
      btn.disabled = false;
    }
  });

  // Send password reset email
  document.getElementById('sdChangePwdBtn')?.addEventListener('click', async () => {
    const btn = document.getElementById('sdChangePwdBtn');
    const msg = document.getElementById('sdPwdMsg');
    btn.disabled = true;
    msg.textContent = 'Sending…';
    try {
      await fetch(`${SERVER}/auth/forgot-password`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: state.currentUser.email }),
      });
      msg.textContent = 'Reset link sent — check your inbox.';
    } catch (_) {
      msg.textContent = 'Could not send reset email.';
    } finally {
      btn.disabled = false;
    }
  });

  // Sign in (logged-out state)
  document.getElementById('sdSignInBtn')?.addEventListener('click', showAuthModal);

  // Logout
  document.getElementById('sdLogoutBtn')?.addEventListener('click', logout);
}
