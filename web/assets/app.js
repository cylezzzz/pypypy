/*! Andio – global utilities & overlay (additive) */
/* File: web/assets/app.js */

(function () {
  "use strict";

  // ---- Safe define/augment ---------------------------------------------------
  const Andio = (function initAndioNamespace(root) {
    const ns = root.Andio || {};
    ns.version = ns.version || "1.0.0";
    ns.state = ns.state || {
      nsfw: localStorage.getItem("andio.nsfw") === "1",
      models: null,
      overlayOpen: false,
      lastOpen: null,
    };
    ns.ui = ns.ui || {};
    ns.utils = ns.utils || {};
    ns.player = ns.player || {};
    ns.models = ns.models || {};
    ns.llm = ns.llm || {};
    ns.suggest = ns.suggest || {};
    return ns;
  })(window);

  // ---- Utils -----------------------------------------------------------------
  if (!Andio.utils.qs) {
    Andio.utils.qs = (sel, el) => (el || document).querySelector(sel);
  }
  if (!Andio.utils.qsa) {
    Andio.utils.qsa = (sel, el) => Array.from((el || document).querySelectorAll(sel));
  }
  if (!Andio.utils.on) {
    Andio.utils.on = function (el, ev, fn, opts) {
      if (!el) return;
      el.addEventListener(ev, fn, opts || false);
    };
  }
  if (!Andio.utils.cssInjectOnce) {
    Andio.utils.cssInjectOnce = function (id, cssText) {
      if (document.getElementById(id)) return;
      const style = document.createElement("style");
      style.id = id;
      style.type = "text/css";
      style.appendChild(document.createTextNode(cssText));
      document.head.appendChild(style);
    };
  }
  if (!Andio.utils.createEl) {
    Andio.utils.createEl = function (tag, attrs, children) {
      const el = document.createElement(tag);
      if (attrs) {
        Object.keys(attrs).forEach((k) => {
          if (k === "style" && typeof attrs[k] === "object") {
            Object.assign(el.style, attrs[k]);
          } else if (k in el) {
            el[k] = attrs[k];
          } else {
            el.setAttribute(k, attrs[k]);
          }
        });
      }
      (children || []).forEach((c) => el.appendChild(typeof c === "string" ? document.createTextNode(c) : c));
      return el;
    };
  }
  if (!Andio.utils.human) {
    Andio.utils.human = {
      bytes(n) {
        if (n == null || isNaN(n)) return "-";
        const u = ["B", "KB", "MB", "GB", "TB"];
        let i = 0;
        while (n >= 1024 && i < u.length - 1) {
          n /= 1024; i++;
        }
        return `${n.toFixed(n < 10 && i > 0 ? 1 : 0)} ${u[i]}`;
      },
    };
  }

  // ---- Overlay CSS (injected, minimal) ---------------------------------------
  Andio.utils.cssInjectOnce("andio-overlay-css", `
    #andio-overlay-backdrop{position:fixed;inset:0;display:none;align-items:center;justify-content:center;background:rgba(0,0,0,.8);z-index:9999}
    #andio-overlay-backdrop[aria-hidden="false"]{display:flex}
    .andio-modal{position:relative;max-width:92vw;max-height:88vh;background:#111;border:1px solid #333;border-radius:16px;box-shadow:0 10px 40px rgba(0,0,0,.5);overflow:hidden}
    .andio-modal header{display:flex;align-items:center;justify-content:space-between;padding:10px 14px;border-bottom:1px solid #333;background:#0e0e0e;color:#eee;font-weight:600}
    .andio-modal .andio-body{display:flex;gap:12px;padding:12px}
    .andio-modal .andio-media{display:flex;align-items:center;justify-content:center;max-width:70vw;max-height:70vh;background:#000;border-radius:12px;overflow:hidden}
    .andio-modal .andio-media img,.andio-modal .andio-media video{max-width:100%;max-height:70vh;display:block}
    .andio-modal .andio-meta{min-width:260px;max-width:28vw;color:#ddd;font-size:13px}
    .andio-close{appearance:none;border:0;background:#222;color:#ddd;border-radius:10px;padding:6px 10px;cursor:pointer}
    .andio-close:hover{background:#2d2d2d}
    .andio-badge{display:inline-block;background:#1f2937;border:1px solid #374151;color:#e5e7eb;border-radius:999px;font-size:11px;padding:2px 8px;margin-left:8px}
    .andio-meta dl{display:grid;grid-template-columns:90px 1fr;gap:6px;padding:8px 0;margin:0}
    .andio-meta dt{color:#aaa}
    .andio-meta dd{margin:0;color:#eee;overflow-wrap:anywhere}
    .andio-actions{display:flex;gap:8px;margin-top:8px}
    .andio-btn{appearance:none;border:1px solid #374151;background:#111;color:#e5e7eb;border-radius:10px;padding:7px 10px;cursor:pointer}
    .andio-btn:hover{background:#1a1a1a}
    .andio-hidden{display:none!important}
    .andio-toast{position:fixed;right:16px;bottom:16px;background:#111;color:#eee;border:1px solid #333;border-radius:10px;padding:10px 12px;z-index:10000}
    .andio-nsfw-toggle{display:inline-flex;align-items:center;gap:6px;cursor:pointer}
    .andio-nsfw-toggle input{accent-color:#6b7280}
  `);

  // ---- Overlay creation (idempotent) -----------------------------------------
  function ensureOverlay() {
    if (document.getElementById("andio-overlay-backdrop")) return;

    const closeBtn = Andio.utils.createEl("button", { className: "andio-close", type: "button", innerText: "Schließen (Esc)" });
    const titleSpan = Andio.utils.createEl("span", { innerText: "Player" });
    const badge = Andio.utils.createEl("span", { className: "andio-badge", innerText: "Overlay" });

    const img = Andio.utils.createEl("img", { id: "andio-overlay-img", className: "andio-hidden", alt: "preview" });
    const vid = Andio.utils.createEl("video", { id: "andio-overlay-vid", className: "andio-hidden", controls: true });

    const mediaBox = Andio.utils.createEl("div", { className: "andio-media" }, [img, vid]);

    const metaBox = Andio.utils.createEl("div", { className: "andio-meta" });
    const metaList = Andio.utils.createEl("dl", { id: "andio-overlay-meta" });
    const actions = Andio.utils.createEl("div", { className: "andio-actions" }, [
      Andio.utils.createEl("button", { className: "andio-btn", type: "button", innerText: "Download", onclick: () => {
        const src = Andio.state?.lastOpen?.src;
        if (!src) return;
        const a = document.createElement("a");
        a.href = src; a.download = ""; a.click();
      }}),
      Andio.utils.createEl("button", { className: "andio-btn", type: "button", innerText: "In neuem Tab", onclick: () => {
        const src = Andio.state?.lastOpen?.src;
        if (src) window.open(src, "_blank");
      }}),
    ]);
    metaBox.appendChild(metaList);
    metaBox.appendChild(actions);

    const header = Andio.utils.createEl("header", null, [
      Andio.utils.createEl("div", null, [titleSpan, badge]),
      closeBtn
    ]);

    const body = Andio.utils.createEl("div", { className: "andio-body" }, [mediaBox, metaBox]);
    const modal = Andio.utils.createEl("div", { className: "andio-modal", role: "dialog", "aria-modal": "true", "aria-labelledby": "andio-overlay-title" }, [header, body]);

    const backdrop = Andio.utils.createEl("div", { id: "andio-overlay-backdrop", "aria-hidden": "true" }, [modal]);

    document.body.appendChild(backdrop);

    // Events
    Andio.utils.on(closeBtn, "click", Andio.closePlayer);
    Andio.utils.on(backdrop, "click", (e) => { if (e.target === backdrop) Andio.closePlayer(); });
    Andio.utils.on(document, "keydown", (e) => { if (e.key === "Escape") Andio.closePlayer(); });
  }

  // ---- Player controls --------------------------------------------------------
  if (!Andio.openPlayer) {
    Andio.openPlayer = function openPlayer(opts) {
      ensureOverlay();
      const o = Object.assign({ type: "image", src: "", meta: null, autoplay: true }, opts || {});
      const backdrop = document.getElementById("andio-overlay-backdrop");
      const img = document.getElementById("andio-overlay-img");
      const vid = document.getElementById("andio-overlay-vid");
      const meta = document.getElementById("andio-overlay-meta");
      if (!backdrop || !img || !vid || !meta) return;

      // Reset
      img.classList.add("andio-hidden");
      vid.classList.add("andio-hidden");
      try { vid.pause(); vid.currentTime = 0; } catch (_) {}

      // Media
      if (o.type === "video") {
        vid.src = o.src || "";
        vid.classList.remove("andio-hidden");
        if (o.autoplay) {
          setTimeout(() => { try { vid.play(); } catch (_) {} }, 50);
        }
      } else {
        img.src = o.src || "";
        img.classList.remove("andio-hidden");
      }

      // Meta
      meta.innerHTML = "";
      const entries = [];
      if (o.meta && typeof o.meta === "object") {
        for (const k of Object.keys(o.meta)) {
          entries.push([k, String(o.meta[k])]);
        }
      } else if (o.src) {
        // Fallback meta
        try {
          const url = new URL(o.src, window.location.href);
          entries.push(["Datei", url.pathname.split("/").pop()]);
          entries.push(["Pfad", url.pathname]);
        } catch {
          entries.push(["Quelle", o.src]);
        }
      }
      if (entries.length === 0) entries.push(["Info", "—"]);

      for (const [k, v] of entries) {
        const dt = Andio.utils.createEl("dt", { innerText: k });
        const dd = Andio.utils.createEl("dd", { innerText: v });
        meta.appendChild(dt); meta.appendChild(dd);
      }

      Andio.state.lastOpen = { type: o.type, src: o.src, meta: o.meta };
      backdrop.setAttribute("aria-hidden", "false");
      Andio.state.overlayOpen = true;
    };
  }

  if (!Andio.closePlayer) {
    Andio.closePlayer = function closePlayer() {
      const backdrop = document.getElementById("andio-overlay-backdrop");
      const vid = document.getElementById("andio-overlay-vid");
      if (!backdrop) return;
      if (vid && !vid.classList.contains("andio-hidden")) {
        try { vid.pause(); } catch (_) {}
      }
      backdrop.setAttribute("aria-hidden", "true");
      Andio.state.overlayOpen = false;
    };
  }

  // ---- Navbar mark active -----------------------------------------------------
  if (!Andio.navbar) {
    Andio.navbar = function navbar(active) {
      // Sucht Links mit data-nav oder href, markiert .active
      const links = Andio.utils.qsa('[data-nav], nav a, .navbar a');
      links.forEach((a) => a.classList.remove("active"));
      if (!active) return;
      links.forEach((a) => {
        const key = a.getAttribute("data-nav") || a.id || a.textContent?.trim()?.toLowerCase();
        if (key && String(key).toLowerCase() === String(active).toLowerCase()) {
          a.classList.add("active");
        } else if (a.getAttribute("href")) {
          const href = a.getAttribute("href").toLowerCase();
          if (href.includes(`${active}.html`)) a.classList.add("active");
        }
      });
    };
  }

  // ---- NSFW toggle ------------------------------------------------------------
  if (!Andio.toggleNSFW) {
    Andio.toggleNSFW = function toggleNSFW(on) {
      const newVal = !!on;
      Andio.state.nsfw = newVal;
      localStorage.setItem("andio.nsfw", newVal ? "1" : "0");
      // UI: Schalter syncen (optional)
      const inputs = Andio.utils.qsa('#nsfw-toggle, [data-toggle="nsfw"]');
      inputs.forEach((el) => {
        if ("checked" in el) el.checked = newVal;
        el.setAttribute("aria-pressed", String(newVal));
      });
      Andio.toast(`NSFW ${newVal ? "aktiv" : "deaktiviert"}`);
    };
  }

  // ---- Toast helper -----------------------------------------------------------
  if (!Andio.toast) {
    Andio.toast = function toast(msg, ms) {
      const div = document.createElement("div");
      div.className = "andio-toast";
      div.textContent = msg || "";
      document.body.appendChild(div);
      setTimeout(() => div.remove(), ms || 2200);
    };
  }

  // ---- Polish (auto hooks) ----------------------------------------------------
  if (!Andio.polish) {
    Andio.polish = function polish() {
      ensureOverlay();

      // Doppelklick auf Bilder/Videos in .card oder [data-open-player]
      document.addEventListener("dblclick", (e) => {
        const t = e.target;
        if (!(t instanceof HTMLElement)) return;
        // Direct <img>/<video>
        if (t.tagName === "IMG" || t.tagName === "VIDEO") {
          const src = (t).getAttribute("src");
          if (src) {
            Andio.openPlayer({ type: t.tagName === "VIDEO" ? "video" : "image", src });
          }
          return;
        }
        // Click on a wrapper element that declares data-open-player/src/type
        const opener = t.closest("[data-open-player]");
        if (opener) {
          const src = opener.getAttribute("data-src") || opener.querySelector("img,video")?.getAttribute("src");
          const type = opener.getAttribute("data-type") || (opener.querySelector("video") ? "video" : "image");
          if (src) Andio.openPlayer({ type, src });
        }
      });

      // NSFW UI Hook (optional)
      const nsfwInputs = Andio.utils.qsa('#nsfw-toggle, [data-toggle="nsfw"]');
      nsfwInputs.forEach((el) => {
        if ("checked" in el) el.checked = Andio.state.nsfw;
        el.setAttribute("aria-pressed", String(Andio.state.nsfw));
        el.addEventListener("change", (ev) => {
          const val = "checked" in ev.target ? !!ev.target.checked : ev.target.getAttribute("aria-pressed") !== "true";
          Andio.toggleNSFW(val);
        });
        el.addEventListener("click", (ev) => {
          if (!("checked" in ev.target)) {
            const newVal = ev.target.getAttribute("aria-pressed") !== "true";
            Andio.toggleNSFW(newVal);
          }
        });
      });

      // Basic drag&drop (optional)
      document.addEventListener("dragover", (e) => { e.preventDefault(); });
      document.addEventListener("drop", (e) => {
        if (!e.dataTransfer || !e.dataTransfer.files || e.dataTransfer.files.length === 0) return;
        e.preventDefault();
        Andio.toast(`${e.dataTransfer.files.length} Datei(en) gedroppt`);
      });

      // Auto Navbar markieren anhand aktueller Seite
      try {
        const page = (location.pathname.split("/").pop() || "").replace(".html", "");
        if (page) Andio.navbar(page);
      } catch {}
    };
  }

  // ---- Models API -------------------------------------------------------------
  if (!Andio.models.load) {
    Andio.models.load = async function loadModels() {
      try {
        const res = await fetch("/api/models");
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        Andio.state.models = data;
        // Optional: simple render if container exists
        const host = document.getElementById("catalog-list");
        if (host && Array.isArray(data?.items)) {
          host.innerHTML = "";
          data.items.forEach((m) => {
            const card = document.createElement("div");
            card.className = "card";
            card.innerHTML = `
              <h4>${m.name || "Model"}</h4>
              <p>Kategorie: ${m.category || "-"}</p>
              <p>Pfad: <code>${m.path || "-"}</code></p>
            `;
            host.appendChild(card);
          });
        }
        return data;
      } catch (err) {
        Andio.toast("Modelle konnten nicht geladen werden");
        console.error("Andio.models.load:", err);
        return null;
      }
    };
  }

  // ---- LLM proxy --------------------------------------------------------------
  if (!Andio.llm.ask) {
    Andio.llm.ask = async function ask(prompt, options) {
      try {
        const res = await fetch("/api/llm", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt: String(prompt || ""), options: options || {} }),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json().catch(() => ({}));
        return data;
      } catch (err) {
        console.error("Andio.llm.ask:", err);
        Andio.toast("LLM nicht erreichbar");
        return null;
      }
    };
  }

  // ---- Suggest prompts --------------------------------------------------------
  if (!Andio.suggest.prompt) {
    Andio.suggest.prompt = function prompt(kind) {
      const nsfw = Andio.state.nsfw;
      const pools = {
        image: [
          { t: "cinematic portrait, shallow depth of field, soft light" },
          { t: "studio still life, high contrast, rim light" },
          { t: "macro photo of a flower with dewdrops, bokeh" },
          { t: "[NSFW] tasteful artistic nude, chiaroscuro lighting" },
        ],
        video: [
          { t: "slow camera push-in on a city skyline at sunset" },
          { t: "looping particle animation, hypnotic flow" },
          { t: "[NSFW] sensual silhouette dance, low key lighting" },
        ],
        editor: [
          { t: "inpainting: add a red scarf, soft wool texture" },
          { t: "remove: unwanted object on the right, blend background" },
          { t: "[NSFW] bikini to lingerie, lace texture, soft shadows" },
        ],
        motion: [
          { t: "pose transfer: T-pose to relaxed stance, subtle sway" },
          { t: "action: waving hand, slow rhythm" },
          { t: "lip-sync: natural speech with gentle head motion" },
        ],
      };
      const list = pools[kind] || pools.image;
      const filtered = nsfw ? list : list.filter((x) => !/\[NSFW\]/i.test(x.t));
      if (filtered.length === 0) return "—";
      const pick = filtered[Math.floor(Math.random() * filtered.length)];
      return pick.t.replace(/\[NSFW\]\s*/i, "");
    };
  }

  // ---- Expose back ------------------------------------------------------------
  window.Andio = Andio;

  // Auto-polish on DOM ready (non-breaking)
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => {
      try { Andio.polish(); } catch (e) { console.error(e); }
    });
  } else {
    try { Andio.polish(); } catch (e) { console.error(e); }
  }
})();

/* === Andio Full-KI additions === */
(function(){
  const A = window.Andio || {};

  // Simple API wrapper
  if (!A.api) A.api = {};
  if (!A.api.get) {
    A.api.get = async function get(url){
      try {
        const r = await fetch(url);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return await r.json();
      } catch (e) {
        console.warn("GET failed", url, e);
        return null;
      }
    };
  }

  // Outputs loader with fallbacks
  if (!A.outputs) A.outputs = {};
  if (!A.outputs.load) {
    A.outputs.load = async function load(){
      // try unified
      let data = await A.api.get('/api/outputs');
      if (data && (data.images || data.videos)) return data;
      // fallbacks
      const images = await A.api.get('/api/outputs/images') || [];
      const videos = await A.api.get('/api/outputs/videos') || [];
      return { images, videos };
    };
  }

  // Card element factory
  function mediaCard(path){
    const ext = (path||'').split('.').pop().toLowerCase();
    const isVideo = ['mp4','webm','mkv','avi','mov'].includes(ext);
    const card = A.utils.createEl('div', { className:'media-card', tabIndex:0 });
    const media = A.utils.createEl(isVideo ? 'video':'img', { className:'thumb', src:path });
    if (isVideo) { media.muted = true; media.playsInline = true; media.loop = true; }
    const meta = A.utils.createEl('div', { className:'meta' });
    const row = A.utils.createEl('div', { className:'row' });
    const name = A.utils.createEl('span', { className:'muted' }, [ (path.split('/').pop() || 'media') ]);
    const badge = A.utils.createEl('span', { className:'badge sfw' }, [ 'KI' ]);
    row.appendChild(name); row.appendChild(badge);
    meta.appendChild(row);
    card.appendChild(media); card.appendChild(meta);
    const open = () => A.openPlayer({ type: isVideo ? 'video':'image', src: path });
    card.addEventListener('dblclick', open);
    card.addEventListener('keydown', (e)=>{ if (e.key === 'Enter') open(); });
    return card;
  }

  // Gallery render
  if (!A.gallery) A.gallery = {};
  if (!A.gallery.render) {
    A.gallery.render = async function render(selector){
      const host = document.querySelector(selector || '#gallery') || document.querySelector('.grid.gallery');
      if (!host) return;
      host.classList.add('is-loading');
      host.innerHTML = '';
      const data = await A.outputs.load();
      const items = [];
      (data.images || []).forEach(p => items.push(p));
      (data.videos || []).forEach(p => items.push(p));
      host.classList.remove('is-loading');
      if (!items.length){
        host.innerHTML = '<div class="muted">Noch keine generierten Inhalte.</div>';
        return;
      }
      const frag = document.createDocumentFragment();
      items.forEach(p => frag.appendChild(mediaCard(p)));
      host.appendChild(frag);
    };
  }

  // Catalog render (models)
  if (!A.catalog) A.catalog = {};
  if (!A.catalog.render) {
    A.catalog.render = async function render(selector){
      const host = document.querySelector(selector || '#catalog-list') || document.querySelector('.grid');
      if (!host) return;
      host.classList.add('is-loading');
      host.innerHTML = '';
      let data = await A.api.get('/api/catalog');
      if (!data || !Array.isArray(data.models)) {
        // fallback to /api/models shape
        const alt = await A.api.get('/api/models');
        const list = Array.isArray(alt?.items) ? alt.items : [];
        data = { models: list };
      }
      host.classList.remove('is-loading');
      if (!data || !Array.isArray(data.models) || data.models.length === 0){
        host.innerHTML = '<div class="muted">Keine Modelle gefunden.</div>';
        return;
      }
      const frag = document.createDocumentFragment();
      data.models.forEach(m => {
        const card = A.utils.createEl('div', { className:'card reveal in' });
        const title = (m.name || m.model_id || 'Model');
        const badgeCls = m.nsfw ? 'badge nsfw' : 'badge sfw';
        card.innerHTML = `
          <h3>${title} <span class="${badgeCls}">${m.nsfw?'NSFW':'SFW'}</span></h3>
          <div class="muted">${m.family || ''}</div>
          <div class="chips">${(m.tags||[]).slice(0,8).map(t=>`<span class="badge">${t}</span>`).join(' ')}</div>
          <div class="toolbar" style="margin-top:10px;display:flex;gap:8px;align-items:center">
            <button class="btn primary" data-action="select">Auswählen</button>
            ${m.size_gb ? `<span class="badge">${m.size_gb} GB</span>`:''}
            ${m.path ? `<span class="badge">${(m.path.split('/').pop())}</span>`:''}
          </div>`;
        const btn = card.querySelector('[data-action="select"]');
        if (btn) btn.addEventListener('click', ()=>{
          A.toast(`${title} ausgewählt`);
          // Hook: hier könnte man A.state.selectedModel setzen
          A.state.selectedModel = (m.model_id || m.name || title);
        });
        frag.appendChild(card);
      });
      host.appendChild(frag);
    };
  }

  // Page init dispatcher
  if (!A.page) A.page = {};
  if (!A.page.init) {
    A.page.init = function init(){
      const file = (location.pathname.split('/').pop() || 'index.html').toLowerCase();
      const page = (document.body.getAttribute('data-page') || file.replace('.html','')).toLowerCase();
      const key = page === 'wandrobe' ? 'wardrobe' : page; // normalize
      if (key === 'gallery') A.gallery.render('#gallery');
      if (key === 'catalog') A.catalog.render('#catalog-list');
      if (key === 'index') {
        // optional quick counters
        A.outputs.load().then(d=>{
          const imgCount = (d.images||[]).length;
          const vidCount = (d.videos||[]).length;
          const el = document.getElementById('andio-counters');
          if (el) el.innerHTML = `<span class="badge">Bilder: ${imgCount}</span> <span class="badge">Videos: ${vidCount}</span>`;
        });
      }
    };
  }

  // Auto-run init once DOM is ready (in addition to Andio.polish)
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => { try { A.page.init(); } catch(e){ console.warn(e); } });
  } else {
    try { A.page.init(); } catch(e){ console.warn(e); }
  }

  window.Andio = A;
})();