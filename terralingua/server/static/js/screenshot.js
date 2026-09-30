import { domToBlob } from '/static/modern-screenshot.mjs';
import { showSnapshotModal } from './snapshot.js';

const _BG = '#0f1117';
const _SCALE = 1;
const _FOOTER_CSS_PX = 36;

export async function captureElement(el) {
  const restore = _fitToContent(el);
  try {
    const blob = await domToBlob(el, {
      backgroundColor: _BG,
      scale: _SCALE,
      filter: node => !(node.nodeType === 1 && (node.classList?.contains('cfg-overlay') || node.classList?.contains('snapshot-exclude'))),
    });
    return _withBranding(blob);
  } finally {
    restore();
  }
}

// Adjust element dimensions for capture:
// - overflow containers that are scrolled: lock to visible height and shift
//   content so the currently-visible portion is what gets rendered
// - overflow containers with content shorter than their box: shrink to remove blank space
// - root: shrink from any flex-inflated height to actual content height
function _fitToContent(root) {
  const restores = [];

  const fix = (el) => {
    if (el.nodeType !== 1) return;
    const cs = window.getComputedStyle(el);
    const oy = cs.overflowY;
    if (oy === 'auto' || oy === 'scroll') {
      const sh = el.scrollHeight, ch = el.clientHeight, st = el.scrollTop;
      const prevH = el.style.height, prevMH = el.style.maxHeight, prevOY = el.style.overflowY;
      el.style.maxHeight = 'none';
      if (sh > ch) {
        // Lock to visible height; shift first child to reproduce current scroll offset
        el.style.height = ch + 'px';
        el.style.overflowY = 'hidden';
        restores.push(() => { el.style.height = prevH; el.style.maxHeight = prevMH; el.style.overflowY = prevOY; });
        if (st > 0) {
          const fc = [...el.children].find(c => window.getComputedStyle(c).display !== 'none');
          if (fc) {
            const prevMT = fc.style.marginTop;
            fc.style.marginTop = `-${st}px`;
            restores.push(() => { fc.style.marginTop = prevMT; });
          }
        }
      } else {
        // Shrink to content height to remove blank space below
        el.style.height = sh + 'px';
        el.style.overflowY = 'visible';
        restores.push(() => { el.style.height = prevH; el.style.maxHeight = prevMH; el.style.overflowY = prevOY; });
      }
    }
    for (const child of el.children) fix(child);
  };

  // Shrink root from flex-inflated height to content height
  const prevH = root.style.height, prevMH = root.style.maxHeight, prevO = root.style.overflow;
  root.style.height = root.scrollHeight + 'px';
  root.style.maxHeight = 'none';
  root.style.overflow = 'hidden';
  restores.push(() => { root.style.height = prevH; root.style.maxHeight = prevMH; root.style.overflow = prevO; });

  for (const child of root.children) fix(child);
  return () => { for (const fn of restores) fn(); };
}

async function _withBranding(blob) {
  const img = await createImageBitmap(blob);
  const out = document.createElement('canvas');
  out.width  = img.width;
  out.height = img.height + _FOOTER_CSS_PX;
  const ctx = out.getContext('2d');
  ctx.drawImage(img, 0, 0);
  ctx.fillStyle = '#0d1117';
  ctx.fillRect(0, img.height, out.width, _FOOTER_CSS_PX);
  ctx.fillStyle = '#4b5563';
  ctx.font = '13px Inter, sans-serif';
  ctx.textAlign = 'center';
  const step = document.getElementById('stepVal')?.textContent ?? '—';
  ctx.fillText(`TerraLingua · step ${step}`, out.width / 2, img.height + _FOOTER_CSS_PX * 0.66);
  return new Promise(resolve => out.toBlob(resolve, 'image/png'));
}

export function addSnapshotBtn(el) {
  el.querySelector('.snapshot-icon-btn')?.remove();
  el.classList.add('snapshotable');
  const btn = document.createElement('button');
  btn.className = 'snapshot-icon-btn';
  btn.title = 'Share';
  btn.textContent = '↗';
  btn.addEventListener('click', e => {
    e.stopPropagation();
    snapshotElement(el, btn);
  });
  el.appendChild(btn);
}

export function snapshotElement(el, btnEl = null) {
  const prevText = btnEl?.textContent;
  if (btnEl) { btnEl.textContent = '…'; btnEl.disabled = true; }

  let resolveCapture;
  const capturePromise = new Promise(r => { resolveCapture = r; });
  showSnapshotModal(capturePromise);

  setTimeout(() => {
    captureElement(el)
      .then(resolveCapture)
      .catch(() => resolveCapture(null))
      .finally(() => {
        if (btnEl) { btnEl.textContent = prevText; btnEl.disabled = false; }
      });
  }, 0);
}
