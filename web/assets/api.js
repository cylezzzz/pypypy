// web/assets/api.js
export async function apiGet(path) {
  const res = await fetch(path, { method: 'GET' });
  if (!res.ok) throw new Error(`GET ${path} -> ${res.status}`);
  return await res.json();
}

export async function apiPost(path, body) {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  });
  if (!res.ok) {
    const t = await res.text().catch(() => '');
    throw new Error(`POST ${path} -> ${res.status} ${t}`);
  }
  return await res.json();
}

export async function apiUpload(path, file) {
  const fd = new FormData();
  fd.append('file', file, file.name);
  const res = await fetch(path, { method: 'POST', body: fd });
  if (!res.ok) {
    const t = await res.text().catch(() => '');
    throw new Error(`UPLOAD ${path} -> ${res.status} ${t}`);
  }
  return await res.json();
}

export function connectProgress(jobId, onMsg) {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/api/ws/progress/${jobId}`);
  ws.onmessage = (ev) => {
    try { onMsg(JSON.parse(ev.data)); } catch {}
  };
  ws.onclose = () => {};
  return ws;
}
