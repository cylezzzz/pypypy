export async function apiGet(path){ const r=await fetch(path); if(!r.ok) throw new Error(`${r.status} ${path}`); return r.json(); }
export async function apiPost(path, body){ const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})}); if(!r.ok){const t=await r.text().catch(()=> ''); throw new Error(`${r.status} ${path} ${t}`);} return r.json(); }
export function connectProgress(jobId, onMsg){
  const proto = location.protocol==='https:'?'wss':'ws';
  const s = new WebSocket(`${proto}://${location.host}/api/ws/progress/${jobId}`);
  s.onmessage = e => { try{ onMsg(JSON.parse(e.data)); }catch{} };
  return s;
}