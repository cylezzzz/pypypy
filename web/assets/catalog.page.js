// assets/catalog.page.js
import { apiGet } from './api.js';
async function loadInstalled(){
  const box = document.getElementById('installedModels');
  try{
    const data = await apiGet('/api/models');
    const cur = data.txt2img_current || 'runwayml/stable-diffusion-v1-5';
    const items = (data.discovered||[]).map(m=>`<div class="card"><div><b>${m.name}</b><div class="badge">${m.modality}</div></div><div class="muted" style="font-size:12px">${(m.size_bytes/1e9).toFixed(2)} GB</div><div class="muted" style="font-size:12px">${m.path}</div></div>`).join('');
    box.innerHTML = `<div class="card"><div>Current txt2img model: <b>${cur}</b></div></div>` + items;
  }catch(e){
    box.innerHTML = `<div class="error">Backend not reachable</div>`;
  }
}
loadInstalled();
