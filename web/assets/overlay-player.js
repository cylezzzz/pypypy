// /assets/overlay-player.js
// Globaler Overlay-Player für Bilder/Videos
(function(){
  const tpl = `
  <div id="global-overlay" class="overlay" role="dialog" aria-modal="true" aria-label="Preview" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:9999;align-items:center;justify-content:center">
    <div class="frame" style="width:min(92vw,1200px);max-height:88vh;display:flex;flex-direction:column;overflow:hidden;background:var(--panel);border:1px solid var(--bd);border-radius:14px;box-shadow:0 10px 36px var(--glow)">
      <header style="display:flex;align-items:center;gap:10px;padding:10px 12px;background:rgba(255,255,255,.9);border-bottom:1px solid var(--bd)">
        <strong id="ov2Title">Preview</strong>
        <span class="chips" id="ov2Badges" style="display:flex;gap:6px;flex-wrap:wrap;margin-left:6px"></span>
        <span class="muted" id="ov2Info" style="margin-left:auto;margin-right:8px;opacity:.75;font-size:12px"></span>
        <a id="ov2Download" class="btn primary" download style="height:36px;border-radius:10px;padding:0 10px;display:inline-flex;align-items:center;gap:8px;background:linear-gradient(135deg,var(--pri1),var(--pri2));color:#fff;border:0">Download</a>
        <button id="ov2Close" class="btn" style="height:36px;border-radius:10px;padding:0 10px;border:1px solid #e8e8e8;background:#fff">Schließen</button>
      </header>
      <div class="content" id="ov2Content" style="background:#000;display:flex;align-items:center;justify-content:center;max-height:74vh"></div>
      <footer style="display:flex;align-items:center;gap:10px;padding:10px 12px;background:rgba(255,255,255,.9);border-top:1px solid var(--bd);justify-content:space-between">
        <div class="row">
          <button id="ov2Prev" class="btn ghost" style="height:36px;border-radius:10px;padding:0 10px;background:rgba(102,126,234,.08);color:var(--pri1);border:1px solid rgba(102,126,234,.25)">◀ Vorheriges</button>
          <button id="ov2Next" class="btn ghost" style="height:36px;border-radius:10px;padding:0 10px;background:rgba(102,126,234,.08);color:var(--pri1);border:1px solid rgba(102,126,234,.25)">Nächstes ▶</button>
        </div>
        <div class="row">
          <a href="catalog.html" class="btn" style="height:36px;border-radius:10px;padding:0 10px;border:1px solid #e8e8e8;background:#fff">Im Store öffnen</a>
        </div>
      </footer>
    </div>
  </div>`;
  document.addEventListener('DOMContentLoaded', ()=>{
    document.body.insertAdjacentHTML('beforeend', tpl);
  });

  const $ = (q,r=document)=>r.querySelector(q);

  const state = { items:[], idx:null };

  function chip(text){
    const s=document.createElement('span');
    s.textContent=text;
    s.style.border='1px solid rgba(102,126,234,.25)';
    s.style.background='rgba(102,126,234,.08)';
    s.style.color='var(--pri1)';
    s.style.padding='2px 8px';
    s.style.borderRadius='999px';
    s.style.fontSize='11px';
    s.style.fontWeight='700';
    return s;
  }

  function render(){
    const root = $('#global-overlay');
    const title = $('#ov2Title'), badges = $('#ov2Badges'), info = $('#ov2Info');
    const content = $('#ov2Content'), aDl = $('#ov2Download');
    const it = state.items[state.idx]; if(!it) return;
    title.textContent = it.name || 'Preview';
    badges.innerHTML=''; badges.append(chip(it.kind==='video'?'Video':'Bild'));
    if(it.format) badges.append(chip(it.format));
    badges.append(chip(it.nsfw ? 'NSFW' : 'SFW'));
    if(it.gen) badges.append(chip(it.gen));
    info.textContent = it.date ? new Date(it.date).toLocaleString() : '';
    aDl.href = it.src;
    content.innerHTML='';
    if(it.kind==='video'){
      const v=document.createElement('video'); v.controls=true; v.autoplay=true; v.src=it.src;
      v.style.maxWidth='100%'; v.style.maxHeight='74vh'; content.append(v);
    } else {
      const img=document.createElement('img'); img.src=it.src; img.alt=it.name||'';
      img.style.maxWidth='100%'; img.style.maxHeight='74vh'; content.append(img);
    }
  }

  function openOverlayFromIndex(i, list){
    state.items = list; state.idx = i;
    render();
    const root = document.getElementById('global-overlay');
    root.style.display='flex';
  }

  // Öffentliche API
  window.openOverlay = function(itemOrList, startIndex=0){
    if(Array.isArray(itemOrList)){
      openOverlayFromIndex(startIndex, itemOrList);
    } else {
      openOverlayFromIndex(0, [itemOrList]);
    }
  };

  // Delegation für data-Attribute
  document.addEventListener('click', (e)=>{
    const t = e.target.closest('[data-overlay-src]');
    if(!t) return;
    e.preventDefault();
    const it = {
      src: t.dataset.overlaySrc,
      kind: t.dataset.overlayKind || (/\.(mp4|webm|mov|mkv)$/i.test(t.dataset.overlaySrc)?'video':'image'),
      name: t.dataset.overlayName || t.getAttribute('title') || 'Preview',
      date: t.dataset.overlayDate || null,
      format: t.dataset.overlayFormat || null,
      nsfw: (t.dataset.overlayNsfw === 'true'),
      gen: t.dataset.overlayGen || null
    };
    window.openOverlay(it);
  });

  // Controls
  document.addEventListener('click', (e)=>{
    const root = document.getElementById('global-overlay');
    if(!root) return;
    if(e.target.id==='ov2Close' || e.target===root){
      root.style.display='none';
    }
    if(e.target && e.target.id==='ov2Prev'){
      if(state.items.length>1){ state.idx=(state.idx-1+state.items.length)%state.items.length; render(); }
    }
    if(e.target && e.target.id==='ov2Next'){
      if(state.items.length>1){ state.idx=(state.idx+1)%state.items.length; render(); }
    }
  });
  document.addEventListener('keydown', (e)=>{
    const root = document.getElementById('global-overlay');
    if(!root || root.style.display!=='flex') return;
    if(e.key==='Escape') root.style.display='none';
    if(e.key==='ArrowRight') document.getElementById('ov2Next').click();
    if(e.key==='ArrowLeft')  document.getElementById('ov2Prev').click();
  });
})();