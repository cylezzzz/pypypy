// assets/gallery.page.js
import { apiGet } from './api.js';
async function load(){
  const grid = document.getElementById('galleryGrid');
  try{
    const data = await apiGet('/api/outputs');
    const items = [...(data.images||[]).map(p=>({path:p,type:'image'})), ...(data.videos||[]).map(p=>({path:p,type:'video'}))];
    if(items.length===0){ grid.innerHTML = '<div class="muted">No files yet.</div>'; return; }
    grid.classList.remove('loading');
    grid.innerHTML = items.map(x=>{
      const href = `/outputs/${x.path}`;
      const thumb = x.type==='video' ? `<video src="${href}" muted style="width:100%;border-radius:12px"></video>`
                                     : `<img src="${href}" style="width:100%;border-radius:12px"/>`;
      return `<a class="thumb" href="player.html?src=${encodeURIComponent(href)}">${thumb}</a>`;
    }).join('');
  }catch(e){
    grid.innerHTML = `<div class="error">Cannot list outputs</div>`;
  }
}
load();
