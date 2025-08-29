// assets/create.page.js
import { apiGet, apiPost, apiUpload, ws } from './api.js';
let currentUpload = null;
function el(id){ return document.getElementById(id); }
function setStatus(txt){ const n=document.getElementById('modelInfo'); if(n) n.textContent = txt; }
async function ping(){ try{ await apiGet('/api/ping'); el('systemStatus')?.querySelector('span')&&(el('systemStatus').querySelector('span').textContent='Online'); setStatus('Backend OK'); }catch{ setStatus('Backend not reachable'); } }
ping();

const fileInput = document.getElementById('fileInput');
if(fileInput){
  fileInput.addEventListener('change', async ()=>{
    const f=fileInput.files?.[0]; if(!f) return;
    const res=await apiUpload('/api/upload', f);
    currentUpload = res;
    const p=document.getElementById('uploadPreview'); if(p){ p.innerHTML = `<img src="${res.path.replace(/^.*workspace\//,'/workspace/')}" style="max-width:100%;border-radius:12px;" />`; }
  });
}
window.triggerUpload = ()=> fileInput?.click();

window.setDimensions = (w,h)=>{ const wi=el('widthInput'); const hi=el('heightInput'); if(wi) wi.value=w; if(hi) hi.value=h; };
window.selectQuality = (q)=>{};

window.generateImage = async ()=>{
  const prompt = (el('mainPrompt')?.value||'').trim();
  if(!prompt){ alert('Please enter a prompt'); return; }
  const payload = {
    prompt,
    negative_prompt: el('negativePrompt')?.value||'',
    width: parseInt(el('widthInput')?.value||'768',10),
    height: parseInt(el('heightInput')?.value||'768',10),
    steps: parseInt(el('stepsSlider')?.value||'20',10),
    guidance: parseFloat(el('guidanceSlider')?.value||'7.5'),
    num_images: 1,
    batch_size: 1,
    seed: el('seedInput')?.value? parseInt(el('seedInput').value,10): null
  };
  const start = await apiPost('/api/txt2img', payload);
  const socket = ws(start.job_id, async (msg)=>{
    if(msg.status==='done'){
      const res = await apiGet(`/api/jobs/${start.job_id}`);
      const grid = document.getElementById('resultsGrid'); const section=document.getElementById('resultsSection');
      if(res?.results?.results && grid && section){
        section.style.display='block';
        grid.innerHTML = res.results.results.map(p=>{
          const local = p.replace(/^.*outputs\//,'/outputs/');
          return `<a class="thumb" href="player.html?src=${encodeURIComponent(local)}"><img src="${local}" style="width:100%;border-radius:12px"/></a>`;
        }).join('');
      }
    }
  });
};