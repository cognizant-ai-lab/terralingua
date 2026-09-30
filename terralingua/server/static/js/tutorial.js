/**
 * tutorial.js — first-run UI tour built on Driver.js (vendored).
 *
 * Behavior:
 *   - Shown once per browser identity, persisted in localStorage. Registered
 *     users key by username; anonymous users share a single "anon" key.
 *   - Replay any time via "Replay tutorial" in the settings drawer's Display
 *     section.
 *
 * Theming for the Driver.js popover lives in dashboard.css under the
 * `.tl-driver-popover` scope so the tour matches the dashboard's dark/light
 * look (Driver.js's defaults are light blue / white).
 */
import { state } from './state.js';

const TUTORIAL_STORAGE_KEY = 'ogw-tutorial-seen';

const STEPS = [
  {
    popover: {
      title: 'Welcome to TerraLingua',
      description: "A 30-second tour of the dashboard. Skip anytime — you can replay it from settings.",
    },
  },
  {
    element: '#gridCol',
    popover: {
      title: 'The world',
      description: 'Agents live here. Drag to pan, scroll to zoom. Clicking an agent or artifact focuses it.',
      side: 'right',
    },
  },
  {
    element: '#addAgentBtn',
    popover: {
      title: 'Spawn an agent',
      description: 'Connect an LLM agent (with your API key) or play one yourself as a human agent.',
      side: 'bottom',
    },
  },
  {
    element: '#addArtifactBtn',
    popover: {
      title: 'Place an artifact',
      description: 'Drop notes or objects onto the grid. Agents can read, pick up, and pass them around.',
      side: 'bottom',
    },
  },
  {
    element: '#anthropoBtn',
    popover: {
      title: 'Observation mode',
      description: 'Switch to anthropologist view for live obituaries, field notes, and emergent-behavior analysis.',
      side: 'bottom',
    },
  },
  {
    element: '#rosterSection',
    popover: {
      title: 'Agents',
      description: "Every agent in the simulation. Click one to follow it — you'll see its energy, motivation, traits, family tree, and life analysis.",
      side: 'right',
    },
  },
  {
    element: '#artifactSection',
    popover: {
      title: 'Artifacts',
      description: "Notes and objects in the world, with carriers and positions. Click one to inspect its content, version history, and phylogeny.",
      side: 'left',
    },
  },
  {
    element: '#msgSection',
    popover: {
      title: 'Messages & Field Notes',
      description: "Switch between agent chat and the anthropologist's running observations of the world.",
      side: 'left',
    },
  },
  {
    element: '#userBadge',
    popover: {
      title: 'Settings & feedback',
      description: "Your account, API key, theme, and the 'Replay tutorial' button live here. 'Help shape TerraLingua' opens a feedback issue.",
      side: 'bottom',
    },
  },
  {
    popover: {
      title: 'Have fun!',
      description: "You're ready. Come back to settings anytime to replay this tour.",
    },
  },
];

function _storageKey() {
  return state.currentUser
    ? `${TUTORIAL_STORAGE_KEY}-${state.currentUser.username}`
    : `${TUTORIAL_STORAGE_KEY}-anon`;
}

function _shouldShowTutorial() {
  return !localStorage.getItem(_storageKey());
}

function _markSeen() {
  localStorage.setItem(_storageKey(), '1');
}

let _driverObj    = null;
let _currentTarget = null;  // element currently outlined by the spotlight

// Custom spotlight overlay: a fixed-position DIV with a 2px green border + glow,
// consistently rounded regardless of the target's own shape. Sits in Driver.js's
// stage cutout so it's visible above the dim overlay.
function _ensureSpotlight() {
  let el = document.getElementById('tlSpotlight');
  if (!el) {
    el = document.createElement('div');
    el.id = 'tlSpotlight';
    el.className = 'tl-spotlight tl-spotlight-hidden';
    document.body.appendChild(el);
  }
  return el;
}

function _positionSpotlight(target) {
  const el = _ensureSpotlight();
  const r  = target.getBoundingClientRect();
  const pad = 6;
  el.style.left   = `${r.left - pad}px`;
  el.style.top    = `${r.top - pad}px`;
  el.style.width  = `${r.width + pad * 2}px`;
  el.style.height = `${r.height + pad * 2}px`;
  el.classList.remove('tl-spotlight-hidden');
}

function _hideSpotlight() {
  document.getElementById('tlSpotlight')?.classList.add('tl-spotlight-hidden');
}

function _build() {
  const { driver } = window.driver.js;
  return driver({
    steps: STEPS,
    popoverClass: 'tl-driver-popover',
    showProgress: true,
    progressText: 'Step {{current}} of {{total}}',
    nextBtnText: 'Next',
    prevBtnText: 'Back',
    doneBtnText: 'Done',
    overlayOpacity: 0.55,
    stagePadding: 6,
    stageRadius: 8,
    animate: false,           // no fade/slide between steps
    smoothScroll: false,      // jump to target instead of animating scroll
    onHighlighted: (el) => {
      if (el) { _currentTarget = el; _positionSpotlight(el); }
      else    { _currentTarget = null; _hideSpotlight(); }
    },
    onDeselected: () => { _currentTarget = null; _hideSpotlight(); },
    onDestroyed:  () => { _currentTarget = null; _hideSpotlight(); _markSeen(); },
  });
}

export function startTutorial() {
  if (!_driverObj) _driverObj = _build();
  _driverObj.drive();
}

export function initTutorial() {
  // Driver.js is loaded as an IIFE before this module; bail gracefully if
  // the global is missing (e.g., vendor script failed to load).
  if (!window.driver?.js?.driver) {
    console.warn('Driver.js not loaded; tutorial disabled.');
    return;
  }
  document.getElementById('replayTutorialBtn')?.addEventListener('click', () => {
    document.getElementById('settingsDrawer')?.classList.remove('open');
    startTutorial();
  });
  // Keep the spotlight aligned if the viewport changes mid-tour.
  window.addEventListener('resize', () => {
    if (_currentTarget) _positionSpotlight(_currentTarget);
  });
  if (_shouldShowTutorial()) startTutorial();
}
