// API keys are encrypted with AES-GCM. The non-extractable CryptoKey lives in
// IndexedDB — its raw bytes are never exposed to JS, so XSS can use the key
// but cannot exfiltrate it. Ciphertext and IV live alongside it in the same DB.
// Keys are intentionally ephemeral in private/incognito sessions.

const ANON_ID_KEY  = "ogw-anon-id";
const DB_NAME      = "ogw-keys";
const DB_VERSION   = 1;
const STORE_NAME   = "api-key";
const RECORD_ID    = "main";

// Legacy localStorage keys — kept only for one-time migration.
const LEGACY_PLAIN = "ogw-api-key";
const LEGACY_V2    = "ogw-api-key-v2";

function _openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = e => e.target.result.createObjectStore(STORE_NAME, { keyPath: "id" });
    req.onsuccess = e => resolve(e.target.result);
    req.onerror   = e => reject(e.target.error);
  });
}

function _txGet(db, id) {
  return new Promise((resolve, reject) => {
    const req = db.transaction(STORE_NAME, "readonly").objectStore(STORE_NAME).get(id);
    req.onsuccess = e => resolve(e.target.result ?? null);
    req.onerror   = e => reject(e.target.error);
  });
}

function _txPut(db, record) {
  return new Promise((resolve, reject) => {
    const req = db.transaction(STORE_NAME, "readwrite").objectStore(STORE_NAME).put(record);
    req.onsuccess = () => resolve();
    req.onerror   = e => reject(e.target.error);
  });
}

function _txDelete(db, id) {
  return new Promise((resolve, reject) => {
    const req = db.transaction(STORE_NAME, "readwrite").objectStore(STORE_NAME).delete(id);
    req.onsuccess = () => resolve();
    req.onerror   = e => reject(e.target.error);
  });
}

// Decrypt a legacy v2 localStorage blob so we can migrate it.
async function _decryptLegacyV2(stored) {
  const ciphertext  = Uint8Array.from(atob(stored.ciphertext), c => c.charCodeAt(0));
  const iv          = Uint8Array.from(atob(stored.iv),          c => c.charCodeAt(0));
  const keyMaterial = Uint8Array.from(atob(stored.key_material), c => c.charCodeAt(0));
  const key = await crypto.subtle.importKey(
    "raw", keyMaterial, { name: "AES-GCM" }, false, ["decrypt"],
  );
  return new TextDecoder().decode(
    await crypto.subtle.decrypt({ name: "AES-GCM", iv }, key, ciphertext),
  );
}

export function getAnonId() {
  let id = localStorage.getItem(ANON_ID_KEY);
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem(ANON_ID_KEY, id);
  }
  return id;
}

export async function saveApiKey(plaintext) {
  const db = await _openDB();
  if (!plaintext) {
    await _txDelete(db, RECORD_ID);
    return;
  }
  const key = await crypto.subtle.generateKey(
    { name: "AES-GCM", length: 256 },
    false,                     // non-extractable: raw bytes never leave the JS engine
    ["encrypt", "decrypt"],
  );
  const iv         = crypto.getRandomValues(new Uint8Array(12));
  const ciphertext = await crypto.subtle.encrypt(
    { name: "AES-GCM", iv },
    key,
    new TextEncoder().encode(plaintext),
  );
  await _txPut(db, { id: RECORD_ID, key, iv, ciphertext });
}

export async function loadApiKey() {
  // One-time migration from legacy localStorage formats. A legacy entry is
  // removed only after the new store holds the key. If the browser database
  // is unavailable, the entry stays for the next load and the key is still
  // returned for this session.
  const legacyPlain = localStorage.getItem(LEGACY_PLAIN);
  if (legacyPlain) {
    try {
      await saveApiKey(legacyPlain);
      localStorage.removeItem(LEGACY_PLAIN);
    } catch (_) {}
    return legacyPlain;
  }
  const legacyV2 = localStorage.getItem(LEGACY_V2);
  if (legacyV2) {
    let plaintext = null;
    try {
      plaintext = await _decryptLegacyV2(JSON.parse(legacyV2));
    } catch (_) {
      // Unreadable: it can never migrate, so it must not block the new store.
      localStorage.removeItem(LEGACY_V2);
    }
    if (plaintext !== null) {
      try {
        await saveApiKey(plaintext);
        localStorage.removeItem(LEGACY_V2);
      } catch (_) {}
      return plaintext;
    }
  }

  try {
    const db     = await _openDB();
    const record = await _txGet(db, RECORD_ID);
    if (!record) return "";
    return new TextDecoder().decode(
      await crypto.subtle.decrypt({ name: "AES-GCM", iv: record.iv }, record.key, record.ciphertext),
    );
  } catch (_) {
    return "";
  }
}
