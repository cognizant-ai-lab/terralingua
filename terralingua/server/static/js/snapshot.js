import { escHtml } from './state.js';

export function showSnapshotModal(blobOrPromise = null) {
  const url = window.location.href;
  const msg = "Check out the live TerraLingua simulation!";
  const encUrl = encodeURIComponent(url);
  const encMsg = encodeURIComponent(msg);
  const hasNative = !!navigator.share;
  const hasCapture = blobOrPromise !== null;
  const isPromise = blobOrPromise instanceof Promise;

  const overlay = document.createElement("div");
  overlay.className = "cfg-overlay";
  overlay.innerHTML = `
    <div class="cfg-box" style="max-width:460px;position:relative;">
      <div class="cfg-header">
        <span>↗ Share</span>
        <button class="cfg-close" id="_snapshotClose">✕</button>
      </div>
      <div class="cfg-body" style="gap:0.85rem;">
        ${hasCapture ? `
        <div id="_snapshotPreviewWrap">
          ${isPromise ? `<div class="snapshot-preview-loading">Capturing screenshot…</div>` : ""}
        </div>
        ` : ""}
        <div class="snapshot-url-row">
          <input class="snapshot-url-input" id="_snapshotUrl" readonly value="${escHtml(url)}" />
        </div>
        <button class="snapshot-copy-btn snapshot-copy-main-btn" id="_snapshotCopy" ${hasCapture ? "disabled" : ""}>Copy screenshot</button>
        <div class="snapshot-social-row">
          <button class="snapshot-social-btn" id="_snapshotTwitter">𝕏 Twitter</button>
          <button class="snapshot-social-btn" id="_snapshotWhatsApp">WhatsApp</button>
          <button class="snapshot-social-btn" id="_snapshotEmail">✉ Email</button>
          ${hasNative ? `<button class="snapshot-social-btn" id="_snapshotNative">More…</button>` : ""}
        </div>
      </div>
    </div>`;
  document.body.appendChild(overlay);

  let _blob = isPromise ? null : blobOrPromise;
  let _previewUrl = null;
  let _closed = false;

  const close = () => {
    _closed = true;
    if (_previewUrl) URL.revokeObjectURL(_previewUrl);
    overlay.remove();
  };
  overlay.querySelector("#_snapshotClose").onclick = close;
  overlay.addEventListener("click", e => { if (e.target === overlay) close(); });

  function _activatePreview(blob) {
    _blob = blob;
    _previewUrl = URL.createObjectURL(blob);
    const wrap = overlay.querySelector("#_snapshotPreviewWrap");
    if (wrap) wrap.innerHTML = `<img src="${_previewUrl}" class="snapshot-preview" />`;
    const btn = overlay.querySelector("#_snapshotCopy");
    if (btn) btn.disabled = false;
  }

  if (isPromise) {
    blobOrPromise.then(blob => {
      if (!_closed) _activatePreview(blob);
    }).catch(() => {
      if (_closed) return;
      overlay.querySelector("#_snapshotPreviewWrap")?.remove();
      const btn = overlay.querySelector("#_snapshotCopy");
      if (btn) { btn.disabled = false; }
    });
  } else if (_blob) {
    _activatePreview(_blob);
  }

  async function _copyScreenshot() {
    if (!_blob) return false;
    try {
      await navigator.clipboard.write([new ClipboardItem({ "image/png": _blob })]);
      return true;
    } catch(e) {
      console.warn("clipboard image write failed:", e);
      return false;
    }
  }

  overlay.querySelector("#_snapshotUrl").onclick = function() { this.select(); };

  const copyBtn = overlay.querySelector("#_snapshotCopy");
  copyBtn.onclick = async () => {
    const ok = await _copyScreenshot();
    copyBtn.textContent = ok ? "✓ Copied!" : "Copy screenshot";
    if (ok) setTimeout(() => { copyBtn.textContent = "Copy screenshot"; }, 2000);
  };

  overlay.querySelector("#_snapshotTwitter").onclick = () =>
    window.open(`https://twitter.com/intent/tweet?text=${encMsg}&url=${encUrl}`, "_blank");

  overlay.querySelector("#_snapshotWhatsApp").onclick = () =>
    window.open(`https://wa.me/?text=${encodeURIComponent(msg + " " + url)}`, "_blank");

  overlay.querySelector("#_snapshotEmail").onclick = () => {
    window.location.href = `mailto:?subject=${encodeURIComponent("TerraLingua Simulation")}&body=${encodeURIComponent(msg + "\n\n" + url)}`;
  };

  if (hasNative) {
    overlay.querySelector("#_snapshotNative").onclick = async () => {
      try { await navigator.share({ title: "TerraLingua", text: msg, url }); } catch(_) {}
    };
  }
}
