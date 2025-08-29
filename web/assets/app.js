// AndioMediaStudio · Nav + StatusDock ("kleines Gehirn") + Glossy Polish
;(() => {
  const $  = (sel, el=document) => el.querySelector(sel);
  const $$ = (sel, el=document) => Array.from(el.querySelectorAll(sel));

  /* =================== NAV =================== */
  // Nutzt relative Links (robust für Hosting und file://). Wenn du absolute Windows-Pfade willst,
  // stell BASE_PATH z.B. auf "file:///X:/pypygennew/web/" um.
  const BASE_PATH = ""; // z.B. "file:///X:/pypygennew/web/"  (leer lassen = relative Links)
  const LINKS = [
  ["Übersicht", "index.html"],
  ["Image Studio", "images.html"],
  ["Video Studio", "video-gen.html"],
  ["Motion", "motion.html"],
  ["Pro Editor", "editor.html"],
  ["Store", "catalog.html"],
  ["Gallery", "gallery.html"],
];

  function linkHref(file){ return BASE_PATH ? (BASE_PATH + file) : file; }

  function injectNavbar(active=''){
    const header = document.createElement('div');
    header.className = 'header';
    header.innerHTML = `
      <nav>
        <div class="brand">AndioMediaStudio</div>
        <div class="links">
          ${LINKS.map(([label,file]) =>
            `<a href="${linkHref(file)}" class="${active===file.split('.')[0]?'active':''}">${label}</a>`
          ).join('')}
        </div>
        <div class="spacer"></div>
        <div id="status-dock" class="status-dock" title="Task Status">
          <div class="brain" aria-hidden="true">
            <svg viewBox="0 0 64 64" width="22" height="22">
              <defs>
                <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
                  <stop offset="0" stop-color="#66a3ff"/>
                  <stop offset="1" stop-color="#b48cff"/>
                </linearGradient>
              </defs>
              <path d="M20 28c-6 0-10-4-10-9 0-6 5-11 11-11 3 0 5 1 7 3 2-2 4-3 7-3 6 0 11 5 11 11 0 5-4 9-10 9"
                    fill="none" stroke="url(#g)" stroke-width="3" stroke-linecap="round"/>
              <path d="M20 28v14a6 6 0 0 1-6 6M40 28v14a6 6 0 0 0 6 6"
                    fill="none" stroke="url(#g)" stroke-width="3" stroke-linecap="round"/>
            </svg>
          </div>
          <div class="sd-info">
            <div class="sd-top">
              <span class="sd-task">Bereit</span>
              <span class="sd-eta">—</span>
            </div>
            <div class="sd-bar"><div></div></div>
          </div>
        </div>
      </nav>
    `;
    document.body.prepend(header);
  }

  /* =================== TOAST =================== */
  const Toast = {
    el:null,
    show(msg, ms=1800){
      this.hide();
      this.el = document.createElement('div');
      this.el.className = 'toast reveal in';
      this.el.textContent = msg;
      document.body.appendChild(this.el);
      setTimeout(()=>this.hide(), ms);
    },
    hide(){ if(this.el){ this.el.remove(); this.el=null; } }
  };

  /* =================== PROGRESS =================== */
  class Progress {
    constructor(el){ this.el=el; this.bar=el?.querySelector('.progress > div'); }
    set(p){ if(this.bar) this.bar.style.width = Math.max(0,Math.min(100,p)) + '%'; }
    async fake(ms=2000, cb){
      let p=0, step=ms/40;
      Status.start(Status._defaultTaskLabel(), { estMs: ms });
      const t0 = performance.now();
      while(p<100){
        await new Promise(r=>setTimeout(r, step));
        p += 2 + Math.random()*6;
        this.set(p);
        Status.progress(p, { elapsed: performance.now()-t0, estMs: ms });
        cb?.(p);
      }
      this.set(100);
      Status.done('Fertig');
    }
  }

  /* =================== STATUS DOCK (kleines Gehirn) =================== */
  const Status = {
    _el: null, _bar: null, _task: null, _eta: null, _tick: null,
    _startedAt: 0, _estMs: 0, _running: false,
    mount(){
      this._el = $('#status-dock');
      this._bar = this._el?.querySelector('.sd-bar > div');
      this._task = this._el?.querySelector('.sd-task');
      this._eta  = this._el?.querySelector('.sd-eta');
    },
    _fmtETA(ms){
      if(!ms || ms<0) return '—';
      const s = Math.ceil(ms/1000);
      return s>=60 ? `${Math.floor(s/60)}m ${s%60}s` : `${s}s`;
    },
    _defaultTaskLabel(){
      // Titel ohne Suffix
      const t = document.title || 'Task';
      return t.replace(/·.*$/,'');
    },
    start(task='Task', {estMs=3000}={}){
      if(!this._el) this.mount();
      this._startedAt = performance.now();
      this._estMs = estMs;
      this._running = true;
      if(this._task) this._task.textContent = task;
      if(this._eta)  this._eta.textContent  = this._fmtETA(estMs);
      if(this._bar)  this._bar.style.width  = '2%';
      // subtle pulse
      this._el?.classList.add('active');
      // ticking ETA fallback
      clearInterval(this._tick);
      this._tick = setInterval(()=>{
        if(!this._running) return;
        const elapsed = performance.now() - this._startedAt;
        const remain  = Math.max(0, this._estMs - elapsed);
        if(this._eta) this._eta.textContent = this._fmtETA(remain);
      }, 500);
      return { cancel: ()=>this.fail('Abgebrochen') };
    },
    progress(p, {elapsed, estMs}={}){
      if(!this._el) this.mount();
      if(typeof p==='number' && this._bar) this._bar.style.width = Math.max(0,Math.min(100,p))+'%';
      if(typeof elapsed==='number' && typeof estMs==='number' && this._eta){
        const remain = Math.max(0, estMs - elapsed);
        this._eta.textContent = this._fmtETA(remain);
      }
    },
    done(msg='Fertig'){
      this._running=false;
      clearInterval(this._tick);
      if(this._task) this._task.textContent = msg;
      if(this._eta)  this._eta.textContent  = '0s';
      if(this._bar)  this._bar.style.width  = '100%';
      setTimeout(()=> this._el?.classList.remove('active'), 700);
    },
    fail(msg='Fehler'){
      this._running=false;
      clearInterval(this._tick);
      if(this._task) this._task.textContent = msg;
      if(this._eta)  this._eta.textContent  = '—';
      if(this._bar)  this._bar.style.width  = '0%';
      this._el?.classList.remove('active');
    }
  };

  /* =================== POLISH (Optik/Interaktion) =================== */
  function addTilt(el, max=6){
    let rect, raf;
    function onMove(e){
      rect = rect || el.getBoundingClientRect();
      const x = (e.clientX - rect.left)/rect.width - .5;
      const y = (e.clientY - rect.top)/rect.height - .5;
      const rx = (+max * y).toFixed(2);
      const ry = (-max * x).toFixed(2);
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(()=>{
        el.style.transform = `perspective(800px) rotateX(${rx}deg) rotateY(${ry}deg) translateY(-4px)`;
      });
    }
    function reset(){ el.style.transform=''; rect=null; }
    el.addEventListener('mousemove', onMove);
    el.addEventListener('mouseleave', reset);
  }

  function addRipple(el){
    el.style.overflow='hidden';
    el.addEventListener('click', (e)=>{
      const r = el.getBoundingClientRect();
      const d = Math.max(r.width, r.height);
      const span = document.createElement('span');
      span.style.position='absolute';
      span.style.width=span.style.height=d+'px';
      span.style.left=(e.clientX-r.left-d/2)+'px';
      span.style.top =(e.clientY-r.top -d/2)+'px';
      span.style.borderRadius='50%';
      span.style.background='radial-gradient(circle, rgba(255,255,255,.35), rgba(255,255,255,0) 60%)';
      span.style.pointerEvents='none';
      span.style.transform='scale(0)';
      span.style.transition='transform .6s ease, opacity .8s ease';
      el.appendChild(span);
      requestAnimationFrame(()=>{ span.style.transform='scale(1)'; span.style.opacity='0'; });
      setTimeout(()=> span.remove(), 800);
    });
  }

  function enableReveal(){
    const io = new IntersectionObserver((entries)=>{
      for(const en of entries){
        if(en.isIntersecting){ en.target.classList.add('in'); io.unobserve(en.target); }
      }
    }, {threshold:.12});
    $$('.card, .thumb, .toolbar, .preview').forEach(el=>{
      el.classList.add('reveal'); io.observe(el);
    });
  }

  function polish(){
    document.documentElement.style.scrollBehavior='smooth';
    document.body.style.opacity='0'; document.body.style.transition='opacity .45s ease';
    requestAnimationFrame(()=> document.body.style.opacity='1');
    $$('.card').forEach(el => addTilt(el, 5));
    $$('.btn').forEach(el => { addTilt(el, 4); addRipple(el); });
    enableReveal();
  }

  // Public API
  window.Andio = { $, $$, Toast, Progress, injectNavbar, polish, Status };

})();


// ===== Global Overlay Player =====
window.Andio = window.Andio || {};
(function(NS){
  let root = null;

  function ensureRoot(){
    if (root) return root;
    root = document.createElement('div');
    root.id = 'andio-player-root';
    root.innerHTML = `
      <div class="andio-modal is-hidden" role="dialog" aria-modal="true" aria-label="Player">
        <div class="andio-modal__backdrop" data-close></div>
        <div class="andio-modal__dialog">
          <header class="andio-modal__header">
            <div class="andio-modal__title">Player</div>
            <div class="andio-modal__spacer"></div>
            <a class="btn" id="andio-player-download" download>Download</a>
            <button class="btn danger" id="andio-player-close" aria-label="Schließen">✕</button>
          </header>
          <div class="andio-modal__body">
            <div class="andio-player__media" id="andio-player-media"></div>
            <aside class="andio-player__meta">
              <div class="badge">Metadaten</div>
              <pre id="andio-player-meta" class="andio-player__pre">Lade Metadaten…</pre>
            </aside>
          </div>
        </div>
      </div>`;
    document.body.appendChild(root);
    root.querySelector('#andio-player-close').addEventListener('click', close);
    root.querySelector('.andio-modal__backdrop').addEventListener('click', close);
    document.addEventListener('keydown', (e)=>{ if(e.key === 'Escape') close(); });
    return root;
  }

  function open(src){
    ensureRoot();
    const modal = root.querySelector('.andio-modal');
    const mediaBox = root.querySelector('#andio-player-media');
    const metaBox  = root.querySelector('#andio-player-meta');
    const dlLink   = root.querySelector('#andio-player-download');
    mediaBox.innerHTML='';
    metaBox.textContent='Lade Metadaten…';
    dlLink.href = src;
    dlLink.download = (src.split('/').pop() || 'download');

    // Render media
    if (/\.(mp4|webm|mkv)$/i.test(src)) {
      const v = document.createElement('video');
      v.src = src; v.controls = true; v.autoplay = true; v.playsInline = true;
      mediaBox.appendChild(v);
    } else {
      const img = new Image();
      img.src = src; img.alt = 'Media';
      mediaBox.appendChild(img);
    }

    // Fetch metadata (skip for data: URLs)
    if (!/^data:/.test(src)) {
      fetch(`/api/player/meta?url=${encodeURIComponent(src)}`)
        .then(r => r.ok ? r.json() : Promise.reject(r))
        .then(j => {
          if (j && j.ok) metaBox.textContent = JSON.stringify(j.meta || {}, null, 2);
          else metaBox.textContent = JSON.stringify({error: j?.message || 'Keine Metadaten'}, null, 2);
        })
        .catch(()=> metaBox.textContent = JSON.stringify({error:'Meta-Request fehlgeschlagen'}, null, 2));
    } else {
      metaBox.textContent = JSON.stringify({note:'Lokale Vorschau (data URL), keine Server-Metadaten vorhanden.'}, null, 2);
    }

    modal.classList.remove('is-hidden');
    document.documentElement.classList.add('andio-modal-open');
  }

  function close(){
    if (!root) return;
    const modal = root.querySelector('.andio-modal');
    const mediaBox = root.querySelector('#andio-player-media');
    mediaBox.querySelectorAll('video').forEach(v => { try{ v.pause(); v.src=''; }catch{} });
    modal.classList.add('is-hidden');
    document.documentElement.classList.remove('andio-modal-open');
  }

  // Helper: attach click/dblclick handlers and auto-annotate dynamic previews
  function bindGlobal(){
    // Click to open
    document.addEventListener('click', (e)=>{
      const el = e.target.closest('[data-player-src], .open-in-player, a[href], .preview img, .thumb img, .thumb video');
      if (!el) return;
      let src = el.getAttribute('data-player-src');

      if (!src && (el.matches('.preview img, .thumb img, .thumb video'))) {
        // Use media src directly
        src = el.currentSrc || el.src;
      }
      if (!src && el.tagName === 'A') {
        const href = el.getAttribute('href') || '';
        if (/^\/?outputs\//i.test(href)) src = href;
      }
      if (!src) return;

      // Prevent navigation for links
      if (el.tagName === 'A') e.preventDefault();
      // If it's a video thumb with a play button in parent, allow opening
      open(src);
    });

    // Double-click on any media to open
    document.addEventListener('dblclick', (e)=>{
      const el = e.target.closest('[data-player-src], .preview img, .thumb img, .thumb video');
      if (!el) return;
      let src = el.getAttribute('data-player-src') || el.currentSrc || el.src;
      if (!src) return;
      e.preventDefault();
      open(src);
    });

    // Observe dynamic media in .preview/.thumb containers and auto-annotate
    const obs = new MutationObserver((muts)=>{
      muts.forEach(m=>{
        m.addedNodes.forEach(node=>{
          if (!(node instanceof HTMLElement)) return;
          node.querySelectorAll?.('.preview img, .thumb img, .thumb video').forEach(media=>{
            if (!media.getAttribute('data-player-src')) {
              const s = media.currentSrc || media.src || '';
              if (s) media.setAttribute('data-player-src', s);
            }
          });
        });
      });
    });
    obs.observe(document.body, { childList: true, subtree: true });
  }

  // expose
  NS.Player = { open, close, bindGlobal };

  // Bind at ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bindGlobal);
  } else {
    bindGlobal();
  }
})(window.Andio);


// ===== Andio Helpers Export =====
(function(NS){
  try{
    if (!NS.$)  NS.$  = (sel, el=document) => el.querySelector(sel);
    if (!NS.$$) NS.$$ = (sel, el=document) => Array.from(el.querySelectorAll(sel));
    if (!NS.Progress) {
      NS.Progress = class {
        constructor(el){ this.el=el; this.bar=el?.querySelector('.progress > div'); }
        set(p){ if(this.bar) this.bar.style.width = Math.max(0,Math.min(100,p)) + '%'; }
        async fake(ms=2000){
          let p=0, step=Math.max(40, ms/40);
          while(p<100){
            await new Promise(r=>setTimeout(r, step));
            p += 3 + Math.random()*5;
            this.set(p);
          }
          this.set(100);
          setTimeout(()=> this.set(0), 500);
        }
      }
    }
  }catch(e){ /* noop */ }
})(window.Andio = window.Andio || {});

