// web/assets/wardrobe.page.js
import { apiUpload, apiPost, apiGet } from './api.js';
import { bindDrop } from './uploader.js';
import { MaskCanvas } from './mask-canvas.js';

let uploaded = null;
const drop = document.getElementById('imagePreview');
const brush = document.getElementById('brushSize');
const modeButtons = document.querySelectorAll('[data-mode-btn]');
const applyBtn = document.getElementById('applyBtn');
const resultGrid = document.getElementById('resultGrid');
const desc = document.getElementById('newClothingDesc');

// Mask canvas
let mask = null;
if(document.getElementById('maskHost')){
  mask = new MaskCanvas('maskHost');
  if(brush) brush.addEventListener('input', ()=> mask.setBrush(parseInt(brush.value,10)||28));
  modeButtons.forEach(b=> b.addEventListener('click', ()=>{
    modeButtons.forEach(x=>x.classList.remove('active'));
    b.classList.add('active');
    mask.setMode(b.dataset.modeBtn);
  }));
}

// dropzone
if(drop){
  bindDrop(drop, (res)=>{
    uploaded = res;
    const img = res.path.replace(/^.*workspace\\//,'/workspace/');
    drop.innerHTML = `<img class="image-preview" src="${img}" />`;
  });
}

// preset chips
document.querySelectorAll('[data-chip]').forEach(chip=>{
  chip.addEventListener('click', ()=>{
    const v = chip.getAttribute('data-chip');
    desc.value = (desc.value ? desc.value + ', ' : '') + v;
  });
});

function buildPrompt(){
  const txt = (desc?.value||'').trim();
  const operation = document.querySelector('[name="op"]:checked')?.value || 'change';
  if(operation==='remove') return `remove selected clothing, keep realistic body, natural shading`;
  if(operation==='material') return `change selected clothing material to: ${txt}`;
  return `change selected clothing to: ${txt}`;
}

async function run(){
  if(!uploaded){ alert('Bitte zuerst ein Bild hochladen.'); return; }
  const payload = {
    image_path: uploaded.path,
    mask_b64: mask ? mask.exportPNG() : null,
    prompt: buildPrompt(),
    negative_prompt: "blurry, artifacts, extra limbs, text, watermark",
    steps: 30,
    guidance: 7.0,
    strength: 0.85
  };
  const start = await apiPost('/api/image/inpaint', payload);
  // poll simple (separate router doesn't use ws)
  const poll = setInterval(async ()=>{
    const stat = await apiGet(`/api/image/inpaint/${start.job_id}`);
    if(stat.status === 'done'){
      clearInterval(poll);
      const imgs = stat?.results?.images || [];
      resultGrid.innerHTML = imgs.map(p=>{
        const local = p.replace(/^.*outputs\\//,'/outputs/');
        return `<div class="result-item"><img class="result-image" src="${local}"/></div>`;
      }).join('');
    }
    if(stat.status === 'error'){
      clearInterval(poll);
      alert('Fehler: '+ (stat.error||'unknown'));
    }
  }, 800);
}
applyBtn?.addEventListener('click', run);
