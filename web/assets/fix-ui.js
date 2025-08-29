// assets/fix-ui.js
(function(){
  // Fallbacks, falls Andio/app.js noch nicht da ist
  window.Andio = window.Andio || {};
  const $  = (sel, root=document) => root.querySelector(sel);
  const $$ = (sel, root=document) => Array.from(root.querySelectorAll(sel));
  // Simple Progress-Fallback: füllt die Leiste sichtbar
  Andio.Progress = Andio.Progress || function(el){
    return {
      fake: async (ms=1200)=>{
        const bar = el?.querySelector?.('.progress > div');
        if(!bar) return;
        bar.style.transition = 'width .25s ease';
        bar.style.width = '85%';
        await new Promise(r=>setTimeout(r, ms));
        bar.style.width = '100%';
        setTimeout(()=>{ bar.style.width = '0%'; }, 300);
      }
    };
  };

  // ——— Wippschalter: überall klickbar + ARIA & Keyboard
  function setupSwitches(){
    $$('.switch').forEach(sw=>{
      const input = sw.querySelector('input[type="checkbox"]');
      if(!input || sw.dataset.wired) return;
      sw.dataset.wired = '1';
      sw.setAttribute('role','switch');
      sw.setAttribute('tabindex','0');
      const sync = ()=> sw.setAttribute('aria-checked', input.checked ? 'true' : 'false');
      sync();

      sw.addEventListener('click', (e)=>{
        // Klick direkt auf Input? Dann native Änderung nutzen.
        if(e.target === input) { sync(); return; }
        input.checked = !input.checked;
        input.dispatchEvent(new Event('change', {bubbles:true}));
        sync();
      });
      sw.addEventListener('keydown', (e)=>{
        if(e.key === ' ' || e.key === 'Enter'){
          e.preventDefault();
          input.checked = !input.checked;
          input.dispatchEvent(new Event('change', {bubbles:true}));
          sync();
        }
      });
      input.addEventListener('change', sync);
    });
  }

  // ——— Mode-Buttons: visuelles Active-State Fallback
  function setupModeButtons(){
    $$('.btn.mode').forEach(btn=>{
      if(btn.dataset.wired) return;
      btn.dataset.wired = '1';
      btn.addEventListener('click', ()=>{
        const g = btn.closest('.toolbar') || document;
        $$('.btn.mode', g).forEach(b=>b.classList.toggle('primary', b===btn));
      });
    });
  }

  // ——— Progress-Leisten initialisieren
  function setupProgress(){
    $$('.progress > div').forEach(bar=>{
      if(bar.dataset.init) return;
      bar.dataset.init = '1';
      bar.style.width = '0%';
    });
  }

  // ——— Helfer überall
  document.addEventListener('DOMContentLoaded', ()=>{
    setupSwitches();
    setupModeButtons();
    setupProgress();
  });

  // Falls SPA/Hot-Reload: auch später erneut verdrahten
  window.AndioFixUI = { setupSwitches, setupModeButtons, setupProgress };
})();
