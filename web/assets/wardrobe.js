/* Andio Media Studio — Wardrobe (ohne Demo)
 * - Verwendet echte Endpoints: /api/wardrobe/remove, /api/wardrobe/change
 * - Persistenter SFW/NSFW-Toggle (localStorage 'andio.mode' = 'sfw'|'nsfw')
 * - Avatar-Pfad als relativer Projektpfad (z.B. workspace/uploads/avatar.png)
 */

(function(){
  'use strict';

  const API_BASE = '/api/wardrobe';

  // ---------- Elements ----------
  const modeToggle  = document.getElementById('modeToggle');
  const modeLabel   = document.getElementById('modeLabel');
  const modeBadge   = document.getElementById('modeBadge');
  const avatarPath  = document.getElementById('avatarPath');
  const btnLoad     = document.getElementById('btnLoadAvatar');
  const avatarImg   = document.getElementById('avatarImg');
  const itemsList   = document.getElementById('itemsList');

  const categorySel = document.getElementById('category');
  const clothTypeSel= document.getElementById('clothType');

  const optSeamBlend  = document.getElementById('optSeamBlend');
  const optShadowSynth= document.getElementById('optShadowSynth');
  const optHairRefine = document.getElementById('optHairRefine');
  const optSuperRes   = document.getElementById('optSuperRes');

  const fitSel      = document.getElementById('fit');
  const materialSel = document.getElementById('material');
  const colorInput  = document.getElementById('color');

  const btnChange   = document.getElementById('btnChange');
  const btnFit      = document.getElementById('btnFit');
  const btnMaterial = document.getElementById('btnMaterial');
  const btnRemove   = document.getElementById('btnRemove');

  const promptTA    = document.getElementById('prompt');
  const styleSel    = document.getElementById('style');

  const statusBox   = document.getElementById('status');

  // ---------- State ----------
  const State = {
    mode: 'sfw',                 // 'sfw' | 'nsfw'
    avatarPath: '',              // relativer Projektpfad
    clothType: 'shirt',          // aktueller Slot
    category: 'tops',            // UI-Kategorie links
    selectedItem: null           // future use (falls Katalog)
  };

  // ---------- Init ----------
  initMode();
  initEvents();
  refreshNSFWVisibility();
  renderItems(); // leerer, aber echter renderer (keine Demo-Inhalte)

  // ---------- Funktionen ----------
  function initMode(){
    const saved = localStorage.getItem('andio.mode');
    State.mode = (saved === 'nsfw') ? 'nsfw' : 'sfw';
    document.documentElement.setAttribute('data-mode', State.mode);
    const isNSFW = (State.mode === 'nsfw');
    modeToggle.checked = isNSFW;
    modeLabel.textContent = isNSFW ? 'NSFW' : 'SFW';
    modeBadge.textContent = isNSFW ? 'NSFW' : 'SFW';
  }

  function setMode(nsfw){
    State.mode = nsfw ? 'nsfw' : 'sfw';
    localStorage.setItem('andio.mode', State.mode);
    document.documentElement.setAttribute('data-mode', State.mode);
    modeLabel.textContent = nsfw ? 'NSFW' : 'SFW';
    modeBadge.textContent = nsfw ? 'NSFW' : 'SFW';
    refreshNSFWVisibility();
  }

  function refreshNSFWVisibility(){
    // Blende „NAKED“ nur in NSFW ein
    const nakedOption = [...categorySel.options].find(o => o.value === 'naked');
    if (nakedOption){
      nakedOption.hidden = (State.mode !== 'nsfw');
    }
  }

  function initEvents(){
    modeToggle.addEventListener('change', () => setMode(modeToggle.checked));
    btnLoad.addEventListener('click', onLoadAvatar);

    categorySel.addEventListener('change', () => {
      State.category = categorySel.value;
      renderItems();
    });

    clothTypeSel.addEventListener('change', () => {
      State.clothType = clothTypeSel.value;
    });

    colorInput.addEventListener('input', () => {
      // Nur visuelles Overlay — beeinflusst Backend nicht
      avatarImg.style.mixBlendMode = 'screen';
      avatarImg.style.filter = 'saturate(1.0)';
      avatarImg.style.boxShadow = `inset 0 0 0 9999px ${hexToRGBA(colorInput.value, 0.0)}`;
    });

    btnChange.addEventListener('click', onChangeClothing);
    btnRemove.addEventListener('click', onRemoveClothing);
    btnFit.addEventListener('click', onAdjustFit);
    btnMaterial.addEventListener('click', onChangeMaterial);
  }

  function renderItems(){
    // Kein Demo-Content. Wir zeigen nur UI-Slots/Kategorien an.
    itemsList.innerHTML = '';
    const info = document.createElement('div');
    info.className = 'hint small';
    info.innerHTML = `
      <b>Kategorie:</b> ${escapeHTML(State.category)} &nbsp;&middot;&nbsp;
      <b>Slot:</b> ${escapeHTML(State.clothType)}<br/>
      Wähle links die Kategorie &amp; den Slot. Lege dann <i>Avatar-Pfad</i> fest und nutze die Aktionen rechts.
    `;
    itemsList.appendChild(info);
  }

  async function onLoadAvatar(){
    const val = (avatarPath.value || '').trim();
    if (!val){
      return setStatus('Bitte einen gültigen relativen Pfad eingeben (z. B. workspace/uploads/avatar.png)', 'error');
    }
    State.avatarPath = val;
    // Wir zeigen das Bild über eine Browser-URL an, wenn es vom Webserver erreichbar ist.
    // Normalerweise wird /workspace statisch bedient. Ansonsten bleibt der Platzhalter leer.
    try {
      avatarImg.src = val;
      avatarImg.onload = () => setStatus('Avatar geladen.', 'ok');
      avatarImg.onerror = () => setStatus('Konnte Bild nicht laden (Pfad erreichbar?)', 'error');
    } catch {
      setStatus('Konnte Bild nicht laden (Pfad erreichbar?)', 'error');
    }
  }

  async function onRemoveClothing(){
    if (!ensureReady()) return;
    setStatus('Kleidung wird entfernt…');
    try {
      const res = await fetchJSON(`${API_BASE}/remove`, {
        image: State.avatarPath,
        clothing_type: State.clothType,
        preserve_anatomy: true
      });
      // Ausgabe nutzen: erstes passendes Ergebnis
      const dataForType = res.results?.[State.clothType] || Object.values(res.results || {})[0];
      if (dataForType?.output_path){
        avatarImg.src = dataForType.output_path;
        setStatus(`Fertig. Ergebnis: ${dataForType.output_path}`, 'ok');
      } else {
        setStatus('Keine Ausgabe erhalten.', 'error');
      }
    } catch (e){
      setStatus(`Fehler: ${e}`, 'error');
    }
  }

  async function onChangeClothing(){
    if (!ensureReady()) return;
    const prompt = (promptTA.value || '').trim();
    const style  = styleSel.value;
    if (!prompt){
      return setStatus('Bitte Prompt eingeben (Beschreibung der neuen Kleidung).', 'error');
    }
    setStatus('Kleidung wird generiert/ersetzt…');
    try {
      const res = await fetchJSON(`${API_BASE}/change`, {
        image: State.avatarPath,
        clothing_type: State.clothType,
        prompt,
        style
      });
      if (res.output_path){
        avatarImg.src = res.output_path;
        setStatus(`Fertig. Ergebnis: ${res.output_path}`, 'ok');
      } else {
        setStatus('Keine Ausgabe erhalten.', 'error');
      }
    } catch (e){
      setStatus(`Fehler: ${e}`, 'error');
    }
  }

  async function onAdjustFit(){
    if (!ensureReady()) return;
    const fit = fitSel.value;
    // Mapping wird serverseitig in clothing_editor.change_clothing_fit gehandhabt.
    setStatus(`Passform („${fit}“) wird angepasst…`);
    try {
      // Wir nutzen dafür die generische change-Route mit passendem Prompt:
      const prompt = fitPrompt(State.clothType, fit);
      const res = await fetchJSON(`${API_BASE}/change`, {
        image: State.avatarPath,
        clothing_type: State.clothType,
        prompt,
        style: 'realistic'
      });
      if (res.output_path){
        avatarImg.src = res.output_path;
        setStatus(`Fertig. Ergebnis: ${res.output_path}`, 'ok');
      } else {
        setStatus('Keine Ausgabe erhalten.', 'error');
      }
    } catch (e){
      setStatus(`Fehler: ${e}`, 'error');
    }
  }

  async function onChangeMaterial(){
    if (!ensureReady()) return;
    const material = materialSel.value;
    setStatus(`Material wird auf „${material}“ geändert…`);
    try {
      // Material-Prompt analog zu server/pipelines/clothing_editor.py
      const prompt = materialPrompt(material);
      const res = await fetchJSON(`${API_BASE}/change`, {
        image: State.avatarPath,
        clothing_type: State.clothType,
        prompt,
        style: 'realistic'
      });
      if (res.output_path){
        avatarImg.src = res.output_path;
        setStatus(`Fertig. Ergebnis: ${res.output_path}`, 'ok');
      } else {
        setStatus('Keine Ausgabe erhalten.', 'error');
      }
    } catch (e){
      setStatus(`Fehler: ${e}`, 'error');
    }
  }

  // ---------- Helpers ----------
  function ensureReady(){
    if (!State.avatarPath){
      setStatus('Bitte zuerst Avatar-Pfad laden.', 'error');
      return false;
    }
    if (!State.clothType){
      setStatus('Bitte einen Kleidungs-Slot wählen.', 'error');
      return false;
    }
    return true;
  }

  async function fetchJSON(url, body){
    const res = await fetch(url, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(body)
    });
    if (!res.ok){
      const text = await res.text();
      throw new Error(`${res.status} ${res.statusText} — ${text}`);
    }
    return res.json();
  }

  function setStatus(msg, type){
    statusBox.textContent = msg;
    statusBox.classList.remove('ok','danger');
    if (type === 'ok') statusBox.classList.add('ok');
    if (type === 'error') statusBox.classList.add('danger');
  }

  function escapeHTML(s){ return (s||'').replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m])); }

  function hexToRGBA(hex, a){
    const h = hex.replace('#','');
    const bigint = parseInt(h,16);
    const r = (bigint >> 16) & 255;
    const g = (bigint >> 8) & 255;
    const b = bigint & 255;
    return `rgba(${r},${g},${b},${a})`;
  }

  function fitPrompt(type, fit){
    const map = {
      tighter: "form-fitting, tight-fitting, body-hugging",
      looser: "loose-fitting, baggy, oversized, relaxed fit",
      fitted: "well-fitted, tailored, perfect fit",
      baggy: "very loose, oversized, baggy style"
    };
    const descr = map[fit] || "well-fitted";
    return `same ${type} but ${descr}, keep same color and material`;
  }

  function materialPrompt(material){
    const mats = {
      leather: "high quality black leather texture, realistic leather material, glossy",
      silk:    "smooth silk fabric, elegant silk texture, flowing silk material",
      denim:   "blue denim texture, jeans material, cotton denim fabric",
      cotton:  "soft cotton fabric, natural cotton texture, comfortable material",
      lace:    "delicate lace pattern, intricate lace texture, elegant lace fabric",
      velvet:  "luxurious velvet texture, soft velvet material, rich velvet fabric",
      satin:   "glossy satin fabric, smooth satin texture, elegant satin material"
    };
    const m = mats[material] || mats.cotton;
    return `same clothing item but made of ${m}, keep the same cut and style`;
  }
})();
