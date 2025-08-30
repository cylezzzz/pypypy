// /assets/include.js
// Lädt die Navbar-Partial und markiert den aktiven Link
(async function(){
  const host = document.querySelector('[data-include="navbar"]');
  if(!host) return;
  try{
    const res = await fetch('partials/navbar.html', {cache:'no-cache'});
    host.innerHTML = await res.text();

    const map = {
      'index.html':'index','images.html':'images','video-gen.html':'video-gen',
      'wandrobe.html':'wardrobe','motion.html':'motion','gallery.html':'gallery',
      'catalog.html':'catalog','editor.html':'editor','avatar.html':null
    };
    const path = location.pathname.split('/').pop() || 'index.html';
    const key = map[path];
    if(key){
      const a = document.querySelector(`.nav-links a[data-link="${key}"]`);
      if(a){ a.classList.add('active'); a.setAttribute('aria-current','page'); }
    }
  }catch(e){ console.error('Navbar konnte nicht geladen werden', e); }
})();