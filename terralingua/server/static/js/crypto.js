// The API key an anonymous visitor types into the dashboard is kept in
// IndexedDB, so the field is filled again on the next visit. It is stored
// AES-GCM encrypted under a non-extractable CryptoKey kept in the same record,
// so the plaintext is not written as such. This is no protection against
// scripts running in this page: any of them can call loadApiKey() and read the
// key. Keys are intentionally ephemeral in private/incognito sessions.

const ANON_ID_KEY  = "ogw-anon-id";
const DB_NAME      = "ogw-keys";
const DB_VERSION   = 1;
const STORE_NAME   = "api-key";
const RECORD_ID    = "main";

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
    false,                     // non-extractable: the raw key bytes cannot be exported
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
