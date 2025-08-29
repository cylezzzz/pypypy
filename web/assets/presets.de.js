<script>
/*!
 * Andio Presets (DE) · SFW=Ausdruck/Charakter · NSFW=Akt/Posen
 * Exposed als: window.AndioPresets
 */
(function(){
  const A = {};

  /* ---------- Utils ---------- */
  const rand = (n)=>Math.floor(Math.random()*n);
  A.rand = rand;
  A.shuffle = arr => {
    const a = arr.slice();
    for(let i=a.length-1;i>0;i--){ const j=rand(i+1); [a[i],a[j]]=[a[j],a[i]]; }
    return a;
  };
  A.sample = (arr, n)=> A.shuffle(arr).slice(0, Math.max(0, Math.min(n, arr.length)));
  A.pickOne = arr => arr[rand(arr.length)];
  const uniq = arr => [...new Set(arr)];

  /* ---------- SFW: Ausdruck / Charakter / Stimmung ---------- */
  const EXPRESSIONS = [
    'natürliches Lächeln','breites Lachen','ernster Blick','melancholisch','nachdenklich','verträumt','souverän','selbstbewusst',
    'schelmisch','verspielt','neugierig','überrascht','fokussiert','entspannt','gelassen','energisch','leidenschaftlich','cool',
    'mysteriös','dramatisch','zart','warmherzig','elegant','freudig','entschlossen','stolz','introvertiert','extrovertiert',
    '„smize“ (mit den Augen lächeln)','„fierce“ (modisch-stark)','sanftes Grinsen','subtiles Lächeln','ernsthafte Ruhe','Nostalgie',
    'hoffnungsvoll','verwundert','frech','gelöst','optimistisch','distanziert','wärmend','kontemplativ'
  ];

  const CHARACTER_MOODS = [
    'heroisch','geheimnisvoll','romantisch','cool & urban','nostalgisch','poetisch','episch','minimalistisch',
    'documentary-real','editorial-polished','luxuriös','handgemacht','verspielt-bunt','ruhig-monochrom','winterlich','sommerlich',
    'regnerisch-reflektierend','neon-future','retro 70s','analog-filmisch','cinematic-drama','timelapse-dynamisch'
  ];

  /* ---------- NSFW: Künstlerischer Akt (Posen/Setups/Licht) ---------- */
  // — bewusst kunstbezogene, nicht-explizite Begriffe —
  const AKT_POSES = [
    'Kontrapost stehend','Stehend Gewicht auf einem Bein','Halbakt mit Draperie','Sitzend Rücken aufrecht',
    'Sitzend Rücken leicht gekrümmt','Sitzend angezogene Knie','Liegend auf Seite (Relaxed)',
    'Liegend auf Rücken (Arme über Kopf)','Liegend Bauch (Kopf zur Seite)','Rückenakt im Profil',
    'Drehung aus der Hüfte (Torsion)','Arme hinter Kopf (Streckung)','Arme vor Brust (verdeckend)',
    'Hände an Taille','Hände am Nacken','Über die Schulter blicken','Kreuzbein-Kurve betont',
    'Gewichtsstütze an Wand','Sitzend auf Hocker (klassisch)','Aufrechte Kniepose (Skulptural)',
    'Kauernde Pose (kompakt)','Diagonal liegend (S-Kurve)','Bodyscape (Ausschnitt, Lichtbahnen)',
    'Silhouette im Gegenlicht','Implied Nude mit Stoff','Draperie um Hüfte','Tuch halbtransparent',
    'Schulter frei, Kopf geneigt','Hände an Rücken gebunden (nur Pose, Stoff)','Standbein Spielbein Kontrast',
    'Beuge aus der Hüfte (sanft)','Lang ausgestrecktes Bein (Ballerina-Anmutung)'
  ];
  const AKT_SETUPS = [
    'Klassischer Studio-Akt','Boudoir am Fenster','Chiaroscuro-Akt (Hell-Dunkel)','Low-Key Akt (dunkler Fond)',
    'High-Key Akt (heller Fond)','Silhouetten-Vorhang','Spiegelreflexion','Stuhl/Hocker-Pose','Draperie auf Boden',
    'Bettlaken-Faltenwurf','Fensterlicht am Morgen','Vorhänge & Schattenmuster','Alte Meister Referenz',
    'Schwarzweiß-Akt','Körniger Filmlook','Weiche Vignette','Weitwinkel Raumwirkung','Tele Verdichtung',
    'Feiner Nebel/Steam (verhüllend)','Rücklicht mit Halo'
  ];
  const AKT_LIGHTS = [
    'Fensterlicht weich','Rembrandt-Licht','Butterfly-Licht','Split-Licht','Edge-Light/Seitenlicht','Rim-Light',
    'Low-Key Kontrast','High-Key diffus','Gegenlicht Silhouette','Goldene Stunde','Blaue Stunde','Kerzenlicht',
    'Softbox großflächig','Hartes Punktlicht','Gobo-Schattenmuster'
  ];
  const AKT_STYLES = [
    'Klassizistisch','Skulptural','Minimalistisch','Vintage Boudoir','Modern Editorial','Fine-Art',
    'Schwarzweiß edel','Sepia warm','Film Grain subtil','Pastellweich','Moody Tiefen','Matte Highlights',
    'Warmton Porträt','Kühler Studiolook','Naturtexturen (Leinen/Holz)'
  ];

  /* ---------- Video (SFW Genres + expressive Varianten) ---------- */
  const VIDEO_GENRES_SFW_BASE = [
    'Reisevlog Küste','City-Hyperlapse Nacht','Drohnenflug über Berge','Natur-Makro Wassertropfen',
    'Wildlife-Doku Savanne','Landschafts-Timelapse Sonnenaufgang','Architektur-Rundgang',
    'Interior-Design Kamerafahrt','Produkt B-Roll Macro','Tech-Reveal auf Tisch',
    'Auto Rolling Shot','Motorrad Kurve','Fashion-Runway','Beauty Close-up',
    'Food-Prep Top-Down','Kaffee-Bar B-Roll','Interview Talking-Head','Podcast Split-Screen',
    'Sport Slow-Motion','Gym Montage','Street-Timelapse Regen','Abstrakte Loop-Formen',
    'Cinematic Trailer','Anime AMV','Game Montage','Car-Commercial Studio','Hands-on Unboxing',
    'Reportage Doku','Konferenz Recap','Hochzeits-Highlights','Festival Aftermovie',
    'Haustier Shorts','DIY Tutorial Overhead','Kochen Rezept Steps'
  ];
  const VIDEO_GENRES_SFW_EXPRESSIVE = [
    'Porträtfilm „Freude“','Porträtfilm „Melancholie“','Porträtfilm „Entschlossenheit“','Charakterstudie „Geheimnisvoll“',
    'Mood-Reel „Hoffnung“','Mood-Reel „Nostalgie“','Editorial-Porträt „Fierce“','Street-Porträt „Cool & Urban“',
    'Cinematic-Close-Up „Verträumt“','Authentic-Docu „Lachen“','Silent-Portrait „Kontemplativ“','Portrait-Montage „Optimistisch“'
  ];

  /* ---------- Video (NSFW künstlerisch/akt-orientiert) ---------- */
  const VIDEO_GENRES_NSFW = [
    'Künstlerischer Akt Studio','Boudoir Schlafzimmer','Silhouette im Gegenlicht',
    'Bodypaint-Studie','Retro Pin-Up','Glamour Editorial','Underwear-Commercial',
    'Figurenstudie Chiaroscuro','Boudoir Schwarzweiß','Spiegel-Glamour','Draperie & Faltenwurf',
    'Fensterlicht-Akt','Low-Key Rückenakt','High-Key Halbakt','Bodyscape Lichtstreifen'
  ];

  /* ---------- Image (SFW motiv+ausdruck) ---------- */
  const IMAGE_SUBJECTS_SFW = [
    'Porträt natürlich','Mode Editorial','Produktfotografie Studio','Landschaft Alpen','Stadt Skyline',
    'Essen Food-Styling','Makro Blüte','Architektur Minimal','Straßenfotografie Nacht',
    'Tierportrait Hund','Interior Cozy','Stillleben Holz','Automotive Studio','Konzept-Collage',
    'Analog-Film Look','Cinematic Still','Sport Action','Natur Waldpfad','Regen Spiegelung',
    // Ausdrucksorientierte Porträts:
    'Porträt: breites Lachen','Porträt: ernst & ruhig','Porträt: schelmisches Grinsen',
    'Porträt: verträumter Blick','Porträt: selbstbewusst','Porträt: geheimnisvoll',
    'Porträt: extrovertiert','Porträt: introvertiert','Porträt: optimistisch'
  ];

  /* ---------- Image (NSFW akt-orientiert) ---------- */
  const IMAGE_SUBJECTS_NSFW = [
    'Aktstudie klassisch','Boudoir weiches Licht','Implied Nude Silhouette','Bodypaint Muster',
    'Pin-Up Vintage','Glamour Editorial','Rückenakt Low-Key','Halbakt mit Draperie',
    'Bodyscape Akzente','Fensterlicht-Akt','Schwarzweiß-Akt','Chiaroscuro-Akt'
  ];

  /* ---------- Allgemeine Listen für Video & Image ---------- */
  const VIDEO_CAMERA   = ['Handheld','Gimbal','Dolly','Stativ/Tripod','Drone'];
  const VIDEO_MOTION   = ['Slow cinematic','Natürlich','Fast kinetic','Timelapse','Hyperlapse'];
  const VIDEO_TRANS    = ['Cut','Dissolve','Whip Pan','Zoom','Match Cut'];
  const VIDEO_GRADES   = ['Neutral','Film LUT','Teal-Orange','Kontrastreich','BW','Pastell','Moody Green','Warm Sunset'];
  const VIDEO_LOOPS    = ['none','hard','soft'];

  const IMAGE_STYLES   = ['Realistisch','Cinematic','Analog Film','Digital Art','Illustration','Anime','Cyberpunk','Minimal','Surreal','Isometrisch'];
  const IMAGE_LIGHTS   = ['Weiches Fensterlicht','Hartes Seitenlicht','Gegenlicht','Goldene Stunde','Blaue Stunde','Neon','Kerzenlicht','Studio Softbox','Ringlicht','Butterfly Lighting'];
  const IMAGE_COMPO    = ['Rule of Thirds','Zentral','Symmetrie','Leading Lines','Framing','Negative Space','Dutch Angle','Close-Crop'];
  const IMAGE_PALETTES = ['Monochrom Blau','Teal-Orange','Pastell Frühling','Moody Grün','Warm Sand','Schwarzweiß','Erde-Töne','Retro 70s'];

  /* ---------- Modell-Empfehlungen (IDs Platzhalter) ---------- */
  const MODELS = {
    general: [
      {id:'andio-video-xl', name:'Andio-VideoXL (General)', hint:'Allround, realistisch'},
      {id:'andio-dynamix',  name:'Andio-DynamiX (Action)',  hint:'Kinetik, schnelle Cuts'},
      {id:'andio-story',    name:'Andio-Story (Narrativ)',  hint:'Szenenfluss'}
    ],
    cinematic: [
      {id:'andio-cinegen',  name:'Andio-CineGen (Cinematic)', hint:'Filmischer Look'},
      {id:'andio-video-xl', name:'Andio-VideoXL (General)',   hint:'Allround'}
    ],
    anime: [
      {id:'andio-anime-v',  name:'Andio-AnimeV',             hint:'Anime/AMV'},
      {id:'andio-video-xl', name:'Andio-VideoXL (General)',  hint:'Fallback'}
    ],
    product: [
      {id:'andio-product',  name:'Andio-Product B-Roll',     hint:'Produkte, Makro'},
      {id:'andio-video-xl', name:'Andio-VideoXL (General)',  hint:'Fallback'}
    ],
    faces: [
      {id:'andio-face-v',   name:'Andio-FaceV (Beauty)',     hint:'Close-ups, Hauttöne'},
      {id:'andio-cinegen',  name:'Andio-CineGen (Cinematic)',hint:'Weiche Haut, LUTs'}
    ],
    landscape: [
      {id:'andio-land',     name:'Andio-Landscape',          hint:'Weite Panoramen'},
      {id:'andio-drone',    name:'Andio-Drone',              hint:'Aerial Shots'}
    ],
    loop: [
      {id:'andio-loop',     name:'Andio-Loops',              hint:'Nahtlose Loops'}
    ],
    nsfw: [
      {id:'andio-nsfw-art', name:'Andio-NSFW Art',           hint:'Künstlerisch, Soft/Artistic'},
      {id:'andio-boudoir',  name:'Andio-Boudoir Lite',       hint:'Boudoir/Glamour'}
    ]
  };

  function genreBucket(genre){
    const g = (genre||'').toLowerCase();
    if(g.includes('anime')) return 'anime';
    if(g.includes('produkt')||g.includes('product')||g.includes('b-roll')||g.includes('unboxing')||g.includes('tech')) return 'product';
    if(g.includes('beauty')||g.includes('close')) return 'faces';
    if(g.includes('interior')||g.includes('architektur')||g.includes('architecture')) return 'landscape';
    if(g.includes('landschaft')||g.includes('wildlife')||g.includes('natur')||g.includes('drohnen')) return 'landscape';
    if(g.includes('loop')) return 'loop';
    if(g.includes('trailer')||g.includes('cinematic')) return 'cinematic';
    if(g.includes('akt')||g.includes('boudoir')||g.includes('glamour')||g.includes('underwear')||g.includes('silhouette')||g.includes('bodypaint')||g.includes('pin')) return 'nsfw';
    return 'general';
  }

  /* ---------- API ---------- */
  // Chips mit starker Ausrichtung je Modus:
  // - SFW: Ausdruck + Genres
  // - NSFW: Akt-Posen + Setups/Licht + künstlerische Genres
  A.chipsFor = function({page='video', nsfw=false, n=14}={}){
    let pool = [];
    if(page==='video'){
      if(nsfw){
        pool = [
          ...A.sample(AKT_POSES, Math.ceil(n*0.5)),
          ...A.sample(AKT_SETUPS, Math.ceil(n*0.3)),
          ...A.sample(VIDEO_GENRES_NSFW, n)  // ggf. überschüssig, wird am Ende gekürzt
        ];
      } else {
        pool = [
          ...A.sample(EXPRESSIONS, Math.ceil(n*0.5)),
          ...A.sample(VIDEO_GENRES_SFW_EXPRESSIVE, Math.ceil(n*0.3)),
          ...A.sample(VIDEO_GENRES_SFW_BASE, n)
        ];
      }
    } else { // page==='image'
      if(nsfw){
        pool = [
          ...A.sample(AKT_POSES, Math.ceil(n*0.5)),
          ...A.sample(AKT_LIGHTS, Math.ceil(n*0.3)),
          ...A.sample(IMAGE_SUBJECTS_NSFW, n)
        ];
      } else {
        pool = [
          ...A.sample(EXPRESSIONS, Math.ceil(n*0.5)),
          ...A.sample(CHARACTER_MOODS, Math.ceil(n*0.3)),
          ...A.sample(IMAGE_SUBJECTS_SFW, n)
        ];
      }
    }
    return uniq(pool).slice(0, n);
  };

  A.randomFeatured = function({page='video', nsfw=false, n=14}={}){
    const chips = A.chipsFor({page, nsfw, n});
    return { chips, featured: A.pickOne(chips) };
  };

  // Kompakte Genre-Liste fürs <select> (seitenübergreifend nutzbar)
  A.genreListForSelect = function({page='video'}={}){
    if(page==='video'){
      const base = [
        'Montage','Travel Vlog','City Hyperlapse','Aerial Drone','Nature Macro','Wildlife Documentary',
        'Landscape','Architecture','Interior Walkthrough','Product B-Roll','Tech Reveal','Car Commercial',
        'Fashion Walk','Beauty Close-up','Food Prep','Interview','Podcast','Sports Highlights','Street Timelapse',
        'Abstract Loops','Cinematic Trailer','Anime AMV','Game Montage',
        // künstlerisch
        'Künstlerischer Akt','Boudoir','Implied Silhouette','Bodypaint','Pin-Up','Glamour','Underwear'
      ];
      return base;
    } else {
      return uniq([...IMAGE_SUBJECTS_SFW, ...IMAGE_SUBJECTS_NSFW]);
    }
  };

  // Modellvorschläge je Genre + NSFW-Flag
  A.modelsFor = function(genre, {nsfw=false}={}){
    const bucket = genreBucket(genre);
    let list = MODELS[bucket] || MODELS.general;
    if(!nsfw) list = list.filter(m => m.id!=='andio-nsfw-art' && m.id!=='andio-boudoir');
    if(bucket!=='nsfw' && nsfw){
      list = [...list, ...MODELS.nsfw.filter(m=>!list.find(x=>x.id===m.id))];
    }
    return list;
  };

  // Export Listen für Seiten
  A.video = {
    camera: VIDEO_CAMERA,
    motion: VIDEO_MOTION,
    transitions: VIDEO_TRANS,
    grades: VIDEO_GRADES,
    loops: VIDEO_LOOPS,
    // Zusatz-Nuancen, falls gebraucht:
    akt: { poses: AKT_POSES, setups: AKT_SETUPS, lights: AKT_LIGHTS, styles: AKT_STYLES },
    expressive: { expressions: EXPRESSIONS, moods: CHARACTER_MOODS }
  };

  A.image = {
    styles: IMAGE_STYLES,
    lights: IMAGE_LIGHTS,
    compositions: IMAGE_COMPO,
    palettes: IMAGE_PALETTES,
    akt: { poses: AKT_POSES, setups: AKT_SETUPS, lights: AKT_LIGHTS, styles: AKT_STYLES },
    expressive: { expressions: EXPRESSIONS, moods: CHARACTER_MOODS }
  };

  /* ---------- Prompt-Templates (DE) ---------- */
  const T2V_TEMPLATES = [
    'Cinematic {genre} in {camera} mit {motion}, Farblook {grade}, {transition}-Cuts, {length}s',
    '{genre}, {fps} FPS, {res}, Tempo {tempo}, stabilisiert {stab}, filmischer Look',
    'Mood-Trailer: {genre}, {grade}, subtile Körnung, leichte Vignette, sanfte {transition}'
  ];
  const T2I_TEMPLATES = [
    '{subject}, Stil {style}, Licht {light}, Komposition {comp}, Palette {palette}',
    'Fotorealistisch: {subject}, {light}, {palette}, klare Details',
    'Editorial: {subject}, {style}, {light}, {comp}'
  ];
  A.template = { t2v(){ return A.pickOne(T2V_TEMPLATES); }, t2i(){ return A.pickOne(T2I_TEMPLATES); } };

  /* ---------- Export ---------- */
  window.AndioPresets = A;
})();
</script>
