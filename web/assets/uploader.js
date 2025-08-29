// web/assets/uploader.js

export async function uploadFile(file){
  const fd = new FormData();
  fd.append('file', file, file.name);
  const r = await fetch('/api/upload', { method: 'POST', body: fd });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export function bindDrop(el, cb){
  const hi = (e,on)=>{ e.preventDefault(); e.stopPropagation(); el.classList.toggle('ring', !!on); };
  ['dragenter','dragover'].forEach(ev=>el.addEventListener(ev, e=>hi(e,true)));
  ['dragleave','drop'].forEach(ev=>el.addEventListener(ev, e=>hi(e,false)));
  el.addEventListener('drop', async e=>{
    const f = e.dataTransfer?.files?.[0]; if(!f) return;
    const resp = await uploadFile(f);
    cb && cb(resp);
  });
  el.addEventListener('click', ()=>{
    const i = document.createElement('input');
    i.type = 'file'; i.accept = '.jpg,.jpeg,.png,.webp';
    i.onchange = async ()=>{
      const f = i.files?.[0]; if(!f) return;
      const resp = await uploadFile(f);
      cb && cb(resp);
    };
    i.click();
  });
}

// --- Wichtig: alias, damit import { bindDropzone } ... funktioniert
export { bindDrop as bindDropzone };
