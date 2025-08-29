// web/assets/app.js
// =====================================================================
// Andio Media Studio - UI Helpers (Navbar, Polish, JobHUD, Tools, Random, API)
// =====================================================================

window.Andio = window.Andio || {};

(function () {
  // ---------- Navbar ----------
  Andio.navbar = function (active) {
    const nav = document.createElement('div');
    nav.className = 'andio-navbar';
    nav.innerHTML = `
      <a data-link="index" href="index.html">Übersicht</a>
      <a data-link="images" href="images.html">Image Studio</a>
      <a data-link="video-gen" href="video-gen.html">Video Studio</a>
      <a data-link="editor" href="editor.html">Editor</a>
      <a data-link="wardrobe" href="wandrobe.html">Wardrobe</a>
      <a data-link="motion" href="motion.html">Motion</a>
      <span class="spacer"></span>
      <span class="brand">AndioMediaStudio</span>
    `;
    document.body.prepend(nav);
    if (active) {
      nav.querySelectorAll('a').forEach(a => {
        if (a.dataset.link === active) a.classList.add('active');
      });
    }
  };

  // ---------- Polish ----------
  Andio.polish = function () {
    // kleine UI-Verbesserungen, falls nötig
  };

  // =====================================================================
  // RANDOM PROMPTS / IDEEN pro Seite & Modus
  // =====================================================================
  const RANDOM_BANK = {
    editor: {
      generate: [
        "hochwertige Lederjacke, studio light, realistische Textur",
        "Goldkette dezent, kleines Anhängerchen, soft highlights",
        "Make-up subtle, skin unify, cinematic rim light",
        "Sticker entfernen, Kanten sauber, Lichtausgleich"
      ],
      remove: [
        "Entferne BH-Träger rechts, Hauttöne angleichen",
        "Entferne Logo auf T-Shirt, Stoffstruktur erhalten",
        "Entferne Sensorfleck, Farbton lokal anpassen",
        "Entferne Objekte im Hintergrund, weiche Bokeh-Rekonstruktion"
      ],
      filter: [
        "Farblook: warm, leichte Vignette, Körnung 15%",
        "Kontrast +10, Klarheit +8, Glanz reduzieren",
        "Filmlook 90s, leichter Cyan-Shift in Schatten",
        "High-key Portrait, Shadow Lift, weiches Skin Smoothing"
      ],
      pose: [
        "Kopf minimal nach links, Blick leicht nach oben",
        "Rechter Arm Bisschen heben, Hand offen, natürlich",
        "Becken 2° rotieren, Standbein rechts",
        "Schulterblätter entspannen, Nacken verlängern"
      ]
    },
    wardrobe: {
      sfw: [
        "Casual Jeans + weiße Bluse (Seide), weiche Falten",
        "Business: Blazer schwarz, Hose mit Bügelfalte",
        "Street: Oversized Hoodie, Cargo-Pants, Sneaker",
        "Glamour: Kleid aus Satin, subtile Lichtreflexe"
      ],
      nsfw: [
        "Lingerie (Satin/Spitze), sanfte Highlights, natürliche Hauttöne",
        "Body (schwarz), weiches Studiolicht, dezenter Schmuck",
        "Slip/Tanga ton-in-ton, saubere Kantenretusche",
        "Transparente Akzente, detailreiche Stofftextur"
      ],
      nude: [
        "Nahtlose Hautrekonstruktion, realistische Übergänge",
        "Feine Poren, gleichmäßige Tonwerte, korrektes Schattenspiel",
        "Kanten sauber, Pose beibehalten, Artefakte vermeiden",
        "Subtle Specular Highlights, keine Überglättung"
      ]
    },
    motion: {
      action: [
        "Idle subtle sway, Atembewegung weich",
        "Winken leicht, natürliches Timing",
        "Walk in place slow, Schwerpunkt stabil",
        "Dance groove minimal, Hüfte soft"
      ],
      pose: [
        "Pose anmutig, Kopf 5° neigen, Arme entspannt",
        "Power-Pose, Schultern zurück, Standbein links",
        "Sitzpose locker, Blick zur Kamera, rechter Arm stützt",
        "Dynamik: diagonale Linie, Schwerpunkt vorverlagert"
      ],
      lipsync: [
        "Klarer Lippenkontakt, leichte Kieferbewegung",
        "Natürliche Blinzler, Mikromimik zufällig",
        "Kopf-Nicken minimal zum Rhythmus",
        "Gesichts-Deformationen subtil halten"
      ],
      camera: [
        "Parallax slow, sanfte Ease-in/out",
        "Dolly-in minimal, Fokus auf Augen",
        "Orbit leicht, Schulterhöhe",
        "Crane up, 10% speed, Ende auf Brusthöhe"
      ]
    }
  };

  Andio.randomPrompt = function (page, modeOrKind) {
    const bank = RANDOM_BANK[page] || {};
    const bucket = bank[modeOrKind] || bank['sfw'] || [];
    if (!bucket.length) return "";
    return bucket[Math.floor(Math.random() * bucket.length)];
  };

  // =====================================================================
  // JOB HUD – Fortschritt, Stage, ETA
  // =====================================================================
  class JobHUD {
    constructor() {
      this.root = document.createElement('div');
      this.root.className = 'andio-jobhud';
      this.root.innerHTML = `
        <div class="hud-card">
          <div class="hud-row">
            <span class="hud-title">KI-Aufgabe</span>
            <button class="hud-close" title="ausblenden">×</button>
          </div>
          <div class="hud-line"><b>Aktion:</b> <span data-k="action">–</span></div>
          <div class="hud-line"><b>Stage:</b> <span data-k="stage">–</span></div>
          <div class="hud-line"><b>Fortschritt:</b> <span data-k="pct">0%</span></div>
          <div class="hud-line"><b>ETA:</b> <span data-k="eta">—</span></div>
          <div class="hud-bar"><div class="hud-bar-fill" style="width:0%"></div></div>
        </div>
      `;
      document.body.appendChild(this.root);
      this.root.querySelector('.hud-close').addEventListener('click', () => {
        this.root.classList.add('hidden');
      });
      this.timer = null;
      this.jobId = null;
    }
    show(action) {
      this.root.classList.remove('hidden');
      this.set('action', action || '–');
      this.set('stage', 'initialisieren…');
      this.setPct(0);
      this.set('eta', '—');
    }
    set(key, val) {
      const el = this.root.querySelector(`[data-k="${key}"]`);
      if (el) el.textContent = val;
    }
    setPct(pct) {
      pct = Math.max(0, Math.min(100, Math.round(pct)));
      this.set('pct', pct + '%');
      const bar = this.root.querySelector('.hud-bar-fill');
      if (bar) bar.style.width = pct + '%';
    }
    attach(jobId, pollFn) {
      this.jobId = jobId;
      if (this.timer) clearInterval(this.timer);
      if (!pollFn) return;
      this.timer = setInterval(async () => {
        try {
          const s = await pollFn(jobId);
          if (s?.stage) this.set('stage', s.stage);
          if (typeof s?.progress === 'number') this.setPct(s.progress * 100);
          if (typeof s?.etaSec === 'number') {
            const m = Math.floor(s.etaSec / 60);
            const sec = Math.max(0, Math.round(s.etaSec % 60));
            this.set('eta', (m ? m + 'm ' : '') + sec + 's');
          }
          if (s?.status === 'done' || (typeof s?.progress === 'number' && s.progress >= 1)) {
            this.set('stage', 'fertig');
            this.setPct(100);
            clearInterval(this.timer);
          }
        } catch (e) {
          console.warn('JobHUD poll error', e);
        }
      }, 700);
    }
  }
  Andio.JobHUD = JobHUD;

  // =====================================================================
  // Canvas-Annotator: Pinsel / Lasso / Rechteck / Ellipse
  // =====================================================================
  Andio.tools = Andio.tools || {};
  Andio.tools.attachAnnotator = function (canvas, maskCanvas, onChange) {
    const state = { tool: 'rect', selections: [], imgLoaded: false, start: null, dragging: false };
    const mctx = maskCanvas.getContext('2d');

    function draw() {
      if (!state.imgLoaded) return;
      mctx.clearRect(0,0,maskCanvas.width,maskCanvas.height);
      mctx.lineWidth = 2; mctx.strokeStyle = '#66f'; mctx.setLineDash([6,4]);
      state.selections.forEach(s=>{
        if (s.type==='rect') {
          mctx.strokeRect(s.x, s.y, s.w, s.h);
        } else if (s.type==='ellipse') {
          mctx.beginPath();
          mctx.ellipse(s.x+s.w/2, s.y+s.h/2, Math.abs(s.w/2), Math.abs(s.h/2), 0, 0, 2*Math.PI);
          mctx.stroke();
        } else {
          mctx.beginPath();
          s.points.forEach((p,i)=> i? mctx.lineTo(p.x,p.y): mctx.moveTo(p.x,p.y));
          if (s.type==='lasso') mctx.closePath();
          mctx.stroke();
        }
        // Nummer
        mctx.setLineDash([]);
        mctx.fillStyle='#000a';
        mctx.fillRect(s.x, s.y-18, 24,16);
        mctx.fillStyle='#fff';
        mctx.font='12px system-ui';
        mctx.fillText('#'+s.id, s.x+4, s.y-6);
        mctx.setLineDash([6,4]);
      });
    }

    function setTool(t){ state.tool = t; }
    function reset(){ state.selections=[]; draw(); onChange && onChange(state.selections); }

    function canvasPoint(e) {
      const r = canvas.getBoundingClientRect();
      return {
        x: (e.clientX - r.left) * canvas.width / r.width,
        y: (e.clientY - r.top) * canvas.height / r.height
      };
    }

    canvas.addEventListener('mousedown', e=>{
      if (!state.imgLoaded) return;
      state.dragging = true;
      state.start = canvasPoint(e);
      if (state.tool==='lasso' || state.tool==='brush') {
        state.selections.push({id:state.selections.length+1, type:state.tool, points:[state.start]});
        onChange && onChange(state.selections);
      }
    });

    canvas.addEventListener('mousemove', e=>{
      if (!state.dragging) return;
      const cur = canvasPoint(e);
      const last = state.selections[state.selections.length-1];
      if (state.tool==='rect' || state.tool==='ellipse') {
        if (!last || last.final) {
          const sel = {id:state.selections.length+1, type:state.tool, x:state.start.x, y:state.start.y, w:0, h:0, final:false};
          state.selections.push(sel); onChange && onChange(state.selections);
        }
        last.x = Math.min(state.start.x, cur.x); last.y = Math.min(state.start.y, cur.y);
        last.w = Math.abs(cur.x - state.start.x); last.h = Math.abs(cur.y - state.start.y);
      } else {
        last.points.push(cur);
      }
      draw();
    });

    window.addEventListener('mouseup', ()=>{
      if (!state.dragging) return;
      state.dragging = false;
      const last = state.selections[state.selections.length-1];
      if (last) last.final = true;
      draw();
      onChange && onChange(state.selections);
    });

    return {
      setTool, reset, draw, state,
      markImageLoaded(){ state.imgLoaded = true; },
    };
  };

  // =====================================================================
  // API-Bindings: FastAPI-Backend + FormData Uploads + Live Previews
  // =====================================================================
  Andio.api = Andio.api || {};

  async function jfetch(path, opts = {}) {
    const res = await fetch(path, { ...opts });
    if (!res.ok) throw new Error(`${path} ${res.status}`);
    return await res.json();
  }

  function toFormData(payload = {}, files = {}) {
    const fd = new FormData();
    fd.append('json', new Blob([JSON.stringify(payload)], { type: 'application/json' }));
    if (files.image) fd.append('image', files.image);
    if (files.mask) fd.append('mask', files.mask);
    if (files.audio) fd.append('audio', files.audio);
    if (Array.isArray(files.storyboard)) files.storyboard.forEach((f,i)=> fd.append('storyboard', f, f.name || `frame_${i}.png`));
    if (files.extras && typeof files.extras === 'object') {
      Object.entries(files.extras).forEach(([k,v])=> { if (v!=null) fd.append(k, v); });
    }
    return fd;
  }

  Andio.api.jobStatus = async (jobId) => jfetch(`/api/jobs/${encodeURIComponent(jobId)}`);

  // Models
  Andio.api.models = {
    download: (body) => jfetch('/api/models/download', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }),
    install:  (body) => jfetch('/api/models/install',  { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }),
    setDefault: (modelId) => jfetch('/api/models/set-default', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ modelId }) }),
  };

  // Image
  Andio.api.image = {
    start: (payload, files) => {
      if (files && (files.image || files.mask)) {
        const fd = toFormData(payload, files);
        return jfetch('/api/image/start', { method: 'POST', body: fd });
      }
      return jfetch('/api/image/start', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    },
    finish: (jobId) => jfetch(`/api/image/finish${jobId ? ('?jobId='+encodeURIComponent(jobId)) : ''}`),
    preview: (payload, files) => {
      const fd = toFormData(payload, files || {});
      return jfetch('/api/editor/preview', { method: 'POST', body: fd });
    },
  };

  // Video
  Andio.api.video = {
    start: (payload, files) => {
      if (files && (files.image || files.mask || files.audio || (files.storyboard && files.storyboard.length))) {
        const fd = toFormData(payload, files);
        return jfetch('/api/video/start', { method: 'POST', body: fd });
      }
      return jfetch('/api/video/start', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    },
    finish: (jobId) => jfetch(`/api/video/finish${jobId ? ('?jobId='+encodeURIComponent(jobId)) : ''}`),
  };

  // Editor
  Andio.api.editorRunStart  = (payload, files) => {
    if (files && (files.image || files.mask)) {
      const fd = toFormData(payload, files);
      return jfetch('/api/editor/run/start', { method: 'POST', body: fd });
    }
    return jfetch('/api/editor/run/start', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
  };
  Andio.api.editorRunFinish = (jobId) => jfetch(`/api/editor/run/finish${jobId ? ('?jobId='+encodeURIComponent(jobId)) : ''}`);
  Andio.api.editorPreview   = (payload, files) => {
    const fd = toFormData(payload, files || {});
    return jfetch('/api/editor/preview', { method: 'POST', body: fd });
  };

  // Wardrobe
  Andio.api.wardrobeDetectStart  = () => jfetch('/api/wardrobe/detect/start', { method: 'POST' });
  Andio.api.wardrobeDetectFinish = () => jfetch('/api/wardrobe/detect/finish');
  Andio.api.wardrobeApplyStart   = (payload, files) => {
    if (files && (files.image || files.mask)) {
      const fd = toFormData(payload, files);
      return jfetch('/api/wardrobe/apply/start', { method: 'POST', body: fd });
    }
    return jfetch('/api/wardrobe/apply/start', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
  };
  Andio.api.wardrobeApplyFinish  = (jobId) => jfetch(`/api/wardrobe/apply/finish${jobId ? ('?jobId='+encodeURIComponent(jobId)) : ''}`);
  Andio.api.wardrobePreview      = (payload, files) => {
    const fd = toFormData(payload, files || {});
    return jfetch('/api/wardrobe/preview', { method: 'POST', body: fd });
  };

  // Motion
  Andio.api.motionStart   = (payload) => jfetch('/api/motion/start',   { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
  Andio.api.motionPreview = (payload) => jfetch('/api/motion/preview', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
  Andio.api.motionFinish  = (jobId) => jfetch(`/api/motion/finish${jobId ? ('?jobId='+encodeURIComponent(jobId)) : ''}`);

  // Job-Helfer
  Andio.startJobWithHUD = async function (actionLabel, startFn /* -> {jobId} */) {
    const hud = new Andio.JobHUD();
    hud.show(actionLabel);
    let jobId = null;
    try {
      const started = await startFn();
      jobId = started?.jobId || ('demo_' + Date.now());
      hud.attach(jobId, Andio.api.jobStatus);
      return { hud, jobId, started };
    } catch (e) {
      console.error(e);
      hud.set('stage', 'Fehler'); hud.setPct(0); hud.set('eta', '—');
      throw e;
    }
  };

  // =====================================================================
  // Global Media Overlay (Bilder/Videos überall im Overlay öffnen)
  // =====================================================================
  (function(){
    if (document.getElementById('andio-media-overlay')) return;
    const root = document.createElement('div');
    root.id = 'andio-media-overlay';
    root.style.cssText = 'position:fixed;inset:0;display:none;place-items:center;background:#000a;z-index:99999;';
    root.innerHTML = `
      <div style="width:min(92vw,1080px);max-height:92vh;background:#111;border:1px solid #333;border-radius:14px;overflow:hidden;display:grid;grid-template-rows:auto 1fr auto">
        <header style="padding:10px 12px;border-bottom:1px solid #222;display:flex;justify-content:space-between;align-items:center">
          <div id="ameta" style="color:#ddd">–</div>
          <button id="aclose" style="background:#222;border:1px solid #444;border-radius:8px;padding:6px 10px;cursor:pointer">Schließen</button>
        </header>
        <div id="astage" style="background:#000;display:grid;place-items:center"></div>
        <footer style="padding:10px 12px;border-top:1px solid #222;display:flex;gap:8px;justify-content:flex-end">
          <button id="afit" style="background:#222;border:1px solid #444;border-radius:8px;padding:6px 10px;cursor:pointer">Fit</button>
          <button id="a1"   style="background:#222;border:1px solid #444;border-radius:8px;padding:6px 10px;cursor:pointer">1:1</button>
        </footer>
      </div>`;
    document.body.appendChild(root);
    const stage = root.querySelector('#astage');
    const meta  = root.querySelector('#ameta');
    root.querySelector('#aclose').onclick = ()=> root.style.display='none';
    root.querySelector('#afit').onclick = ()=> { const el=stage.querySelector('img,video'); if(!el) return; el.style.maxWidth='100%'; el.style.maxHeight='100%'; };
    root.querySelector('#a1').onclick   = ()=> { const el=stage.querySelector('img,video'); if(!el) return; el.style.maxWidth='none'; el.style.maxHeight='none'; };

    Andio.openMedia = function(url, type='image', name='Output'){
      stage.innerHTML='';
      const el = type==='video' ? document.createElement('video') : document.createElement('img');
      if (type==='video'){ el.src = url; el.controls = true; el.muted = true; el.playsInline = true; el.autoplay = true; }
      else { el.src = url; el.alt = name; }
      el.style.maxWidth='100%'; el.style.maxHeight='100%';
      stage.appendChild(el);
      meta.textContent = name;
      root.style.display='grid';
    };

    // Auto: jedes generierte Medium mit data-generated="true" öffnet Overlay
    document.addEventListener('click', (e)=>{
      const m = e.target;
      if (!m) return;
      if ((m.tagName === 'IMG' || m.tagName === 'VIDEO') && m.dataset.generated === 'true') {
        const t = m.tagName === 'VIDEO' ? 'video' : 'image';
        const name = m.getAttribute('alt') || m.closest('[data-name]')?.dataset?.name || 'Output';
        Andio.openMedia(m.currentSrc || m.src, t, name);
      }
    }, true);
  })();

})();
