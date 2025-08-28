
const API = {
  async ping(){ return (await fetch('/api/ping')).json() },
  async models(){ return (await fetch('/api/models')).json() },
  async outputs(kind='all'){ return (await fetch('/api/outputs?kind='+encodeURIComponent(kind))).json() },
  async upload(file, target='workspace'){
    const fd = new FormData(); fd.append('file', file); fd.append('target', target);
    return (await fetch('/api/upload', {method:'POST', body:fd})).json();
  },
  async job(payload){
    const res = await fetch('/api/jobs', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
    return res.json();
  },
  async jobs(){ return (await fetch('/api/jobs')).json() }
};
function toast(msg, ms=2500){ let t=document.querySelector('.toast'); if(!t){t=document.createElement('div');t.className='toast';document.body.appendChild(t);} t.textContent=msg; t.classList.add('show'); setTimeout(()=>t.classList.remove('show'), ms); }
class ProgressManager{
  constructor(root){ this.root=root||document.body; this.ws=null; this.startTs=0; }
  mount(){ const h=this.root.querySelector('#progressHost'); if(!h){return} ; const panel=document.createElement('div'); panel.className='progress-panel'; panel.innerHTML=`<div class="progress" style="flex:1"><div class="bar"></div><div class="label">Bereit</div></div><div class="timer" id="eta">—:—</div>`; h.replaceChildren(panel); this.panel=panel; this.bar=panel.querySelector('.bar'); this.label=panel.querySelector('.label'); this.eta=panel.querySelector('#eta'); }
  connect(jobId){ if(!this.panel) this.mount(); this.startTs=Date.now(); this.set(0,'Warte…'); const proto=location.protocol==='https:'?'wss':'ws'; this.ws=new WebSocket(`${proto}://${location.host}/ws/jobs/${jobId}`); this.ws.onmessage=(e)=>{ try{ const msg=JSON.parse(e.data); this.onProgress(msg);}catch{} }; }
  onProgress(msg){ const pct=Math.max(0,Math.min(100,Math.round(msg.percent ?? (msg.step&&msg.total?(msg.step/msg.total*100):0)))); const eta=msg.eta_ms??null; this.set(pct,msg.text||msg.status||'…',eta); if(msg.status==='completed'){ toast('Fertig: '+(msg.job_id||'')); } }
  set(pct,text,eta_ms){ if(!this.bar){this.mount();} this.bar.style.width=pct+'%'; this.label.textContent=`${pct}% – ${text}`; if(eta_ms!=null){ const sec=Math.max(0,Math.round(eta_ms/1000)); const mm=String(Math.floor(sec/60)).padStart(2,'0'); const ss=String(sec%60).padStart(2,'0'); this.eta.textContent=`${mm}:${ss}`; } else { const elapsed=Math.round((Date.now()-this.startTs)/1000); const mm=String(Math.floor(elapsed/60)).padStart(2,'0'); const ss=String(elapsed%60).padStart(2,'0'); this.eta.textContent=`⏱ ${mm}:${ss}`; } }
}
