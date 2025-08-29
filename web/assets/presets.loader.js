// web/assets/presets.loader.js
// Erkennt unterschiedliche "presets.de.js"-Formate, normalisiert sie, und stellt eine einheitliche API bereit:
// Andio.presets.getAll(), getCategories(), getModelsByCategory(id), getTags(), random({category, nsfw, sfw}), search(text)

window.Andio = window.Andio || {};
(function () {
  // ---- 1) Quelle suchen: verschiedene mögliche Formen unterstützen ----
  // Erwartete Kandidaten (eine davon reicht):
  //  - window.PRESETS_DE = { models:[...], categories:[...], ... }
  //  - window.PRESETS = { ... }
  //  - window.presets = { ... }
  //  - oder ein globales Array von Modellen/Kategorien

  function detectRaw() {
    if (window.PRESETS_DE) return window.PRESETS_DE;
    if (window.PRESETS) return window.PRESETS;
    if (window.presets) return window.presets;
    if (window.MODELS) return { models: window.MODELS };
    if (Array.isArray(window.models)) return { models: window.models };
    return null;
  }

  // ---- 2) Normalisieren in kanonisches Schema ----
  // Zielschema:
  // {
  //   models: [{id, name, desc, img, tags:[], categories:[], base, files:{ckpt,vae,lora:[]}, nsfw?:bool}],
  //   categories: [{id, name, thumb, nsfw?:bool, genres?:[], prompt?:{pos,neg}}],
  //   tags: [ ... ]
  // }
  function normalize(raw) {
    if (!raw) return { models: [], categories: [], tags: [] };

    const out = { models: [], categories: [], tags: [] };

    // Kategorien
    const rawCats = raw.categories || raw.category || raw.kategorien || [];
    rawCats.forEach((c, idx) => {
      const id = c.id || c.slug || (c.name ? c.name.toLowerCase().replace(/\s+/g, '-') : 'cat-' + idx);
      out.categories.push({
        id,
        name: c.name || c.title || `Kategorie ${idx + 1}`,
        thumb: c.thumb || c.image || c.img || null,
        nsfw: typeof c.nsfw === 'boolean' ? c.nsfw : undefined,
        genres: c.genres || c.genre || [],
        prompt: c.prompt || c.prompts || c.default || undefined
      });
    });

    // Modelle
    const rawModels = raw.models || raw.modelle || raw.items || [];
    rawModels.forEach((m, idx) => {
      const id = m.id || m.key || (m.name ? m.name.toLowerCase().replace(/\s+/g, '-') : 'model-' + idx);
      const categories = m.categories || m.category || m.kategorie || [];
      const tags = m.tags || m.tag || [];
      const img = m.img || m.image || m.cover || (m.examples && m.examples[0]) || null;
      const files = m.files || {
        ckpt: m.ckpt || m.path || undefined,
        vae: m.vae || undefined,
        lora: m.loras || m.lora || []
      };
      out.models.push({
        id,
        name: m.name || m.title || `Modell ${idx + 1}`,
        desc: m.desc || m.description || '',
        img,
        tags: Array.isArray(tags) ? tags : (tags ? [tags] : []),
        categories: Array.isArray(categories) ? categories : (categories ? [categories] : []),
        base: m.base || m.family || m.sdxl ? 'sdxl' : (m.sd15 ? 'sd15' : m.base),
        files,
        nsfw: typeof m.nsfw === 'boolean' ? m.nsfw : undefined,
        example: m.example || undefined,
        prompt: m.prompt || m.prompts || undefined
      });
      // Tags sammeln
      (Array.isArray(tags) ? tags : (tags ? [tags] : [])).forEach(t => {
        if (t && !out.tags.includes(t)) out.tags.push(t);
      });
    });

    // Falls die Rohdatei "tags" separat liefert
    if (Array.isArray(raw.tags)) {
      raw.tags.forEach(t => { if (!out.tags.includes(t)) out.tags.push(t); });
    }

    return out;
  }

  // ---- 3) Öffentliche API ----
  const raw = detectRaw();
  const data = normalize(raw);

  function getAll() { return data; }
  function getCategories() { return data.categories; }
  function getTags() { return data.tags; }
  function getModels() { return data.models; }
  function getModelsByCategory(categoryIdOrName) {
    if (!categoryIdOrName) return data.models;
    const needle = ('' + categoryIdOrName).toLowerCase();
    return data.models.filter(m => (m.categories || []).some(c => ('' + c).toLowerCase() === needle));
  }
  function filter({ category, tag, nsfw, sfw, base } = {}) {
    let list = data.models.slice();
    if (category) {
      const n = ('' + category).toLowerCase();
      list = list.filter(m => (m.categories || []).some(c => ('' + c).toLowerCase() === n));
    }
    if (tag) {
      const t = ('' + tag).toLowerCase();
      list = list.filter(m => (m.tags || []).some(x => ('' + x).toLowerCase() === t));
    }
    if (typeof nsfw === 'boolean') list = list.filter(m => m.nsfw === nsfw || (nsfw && (m.tags||[]).includes('nsfw')));
    if (typeof sfw === 'boolean' && sfw) list = list.filter(m => m.nsfw !== true && !(m.tags||[]).includes('nsfw'));
    if (base) {
      const b = ('' + base).toLowerCase();
      list = list.filter(m => (m.base || '').toLowerCase().includes(b));
    }
    return list;
  }
  function random(opts = {}) {
    const list = filter(opts);
    if (!list.length) return null;
    return list[Math.floor(Math.random() * list.length)];
  }
  function search(text) {
    const q = (text || '').trim().toLowerCase();
    if (!q) return data.models.slice();
    return data.models.filter(m =>
      (m.name || '').toLowerCase().includes(q) ||
      (m.desc || '').toLowerCase().includes(q) ||
      (m.tags || []).some(t => ('' + t).toLowerCase().includes(q)) ||
      (m.categories || []).some(c => ('' + c).toLowerCase().includes(q))
    );
  }

  Andio.presets = { getAll, getCategories, getTags, getModels, getModelsByCategory, filter, random, search };
})();
