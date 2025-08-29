// assets/data/registry.js
/*!
 * AndioMediaStudio - Frontend API Bridge
 * Verbindet Frontend mit Backend-APIs
 * Fallbacks für Demo-Modus ohne Backend
 */

;(function() {
  'use strict';

  // =============== API CLIENT ===============
  
  class AndioAPI {
    constructor(baseURL = '') {
      this.baseURL = baseURL;
      this.timeout = 30000; // 30s timeout
    }

    async request(endpoint, options = {}) {
      const url = `${this.baseURL}/api${endpoint}`;
      const config = {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          ...options.headers
        },
        ...options
      };

      if (config.method !== 'GET' && config.body && typeof config.body === 'object') {
        config.body = JSON.stringify(config.body);
      }

      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.timeout);

        const response = await fetch(url, {
          ...config,
          signal: controller.signal
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        
        // Backend liefert standardisierte APIResponse
        if (data.ok === false) {
          throw new Error(data.message || 'API request failed');
        }

        return data;

      } catch (error) {
        if (error.name === 'AbortError') {
          throw new Error('Request timeout');
        }
        console.warn(`API request failed: ${endpoint}`, error);
        throw error;
      }
    }

    // =============== SPECIFIC API METHODS ===============

    async ping() {
      return this.request('/ping');
    }

    async getModels(params = {}) {
      const query = new URLSearchParams(params).toString();
      return this.request(`/models${query ? '?' + query : ''}`);
    }

    async getImagePresets() {
      return this.request('/presets/image');
    }

    async getVideoPresets() {
      return this.request('/presets/video');
    }

    async generateImage(payload) {
      return this.request('/generate/image', {
        method: 'POST',
        body: payload
      });
    }

    async generateVideo(payload) {
      return this.request('/generate/video', {
        method: 'POST',
        body: payload
      });
    }

    async getGallery(params = {}) {
      const query = new URLSearchParams(params).toString();
      return this.request(`/gallery${query ? '?' + query : ''}`);
    }

    async getCatalog() {
      return this.request('/catalog');
    }

    async getPlayerMeta(url) {
      return this.request(`/player/meta?url=${encodeURIComponent(url)}`);
    }

    async uploadFile(file) {
      const formData = new FormData();
      formData.append('file', file);

      return this.request('/upload', {
        method: 'POST',
        headers: {}, // Let browser set Content-Type for FormData
        body: formData
      });
    }
  }

  // =============== FALLBACK DATA ===============
  
  const FALLBACK_DATA = {
    imagePresets: {
      genres: [
        { id: 'portrait', name: 'Portrait' },
        { id: 'landscape', name: 'Landscape' },
        { id: 'art', name: 'Artistic' },
        { id: 'photography', name: 'Photography' },
        { id: 'anime', name: 'Anime/Manga' },
        { id: 'nsfw_artistic', name: 'NSFW Artistic', nsfw: true },
        { id: 'nsfw_realistic', name: 'NSFW Realistic', nsfw: true }
      ],
      styles: [
        { id: 'realistic', name: 'Realistic' },
        { id: 'cinematic', name: 'Cinematic' },
        { id: 'analog', name: 'Analog Film' },
        { id: 'anime', name: 'Anime Style' },
        { id: 'illustration', name: 'Digital Illustration' }
      ],
      lights: [
        { id: 'natural', name: 'Natural Light' },
        { id: 'golden_hour', name: 'Golden Hour' },
        { id: 'studio', name: 'Studio Lighting' },
        { id: 'dramatic', name: 'Dramatic Lighting' },
        { id: 'soft', name: 'Soft Light' }
      ],
      comps: [
        { id: 'rule_thirds', name: 'Rule of Thirds' },
        { id: 'centered', name: 'Centered' },
        { id: 'close_up', name: 'Close-up' },
        { id: 'wide_shot', name: 'Wide Shot' }
      ],
      palettes: [
        { id: 'natural', name: 'Natural Colors' },
        { id: 'warm', name: 'Warm Tones' },
        { id: 'cool', name: 'Cool Tones' },
        { id: 'monochrome', name: 'Monochrome' },
        { id: 'vibrant', name: 'Vibrant Colors' }
      ],
      negatives: {
        sfw: 'worst quality, low quality, jpeg artifacts, watermark, signature, text, blurry, deformed',
        nsfw: 'worst quality, low quality, jpeg artifacts, watermark, signature, text, blurry, deformed, censored'
      },
      templates: [
        '{subject}, {style}, {light}, {comp}, {palette}',
        'Professional {subject} photo, {style}, {light}',
        '{style} artwork of {subject}, {palette}, {comp}'
      ],
      featured: [
        'Portrait', 'Landscape', 'Artistic', 'Cinematic', 'Realistic',
        'Golden Hour', 'Studio Light', 'Natural Colors', 'Close-up'
      ]
    },

    videoPresets: {
      genres: [
        { id: 'cinematic', name: 'Cinematic' },
        { id: 'nature', name: 'Nature' },
        { id: 'abstract', name: 'Abstract' },
        { id: 'portrait', name: 'Portrait Video' },
        { id: 'action', name: 'Action' },
        { id: 'nsfw_artistic', name: 'NSFW Artistic', nsfw: true }
      ],
      cameras: [
        { id: 'handheld', name: 'Handheld' },
        { id: 'gimbal', name: 'Gimbal Stabilized' },
        { id: 'tripod', name: 'Tripod Static' },
        { id: 'drone', name: 'Drone/Aerial' }
      ],
      motions: [
        { id: 'slow', name: 'Slow Motion' },
        { id: 'natural', name: 'Natural Speed' },
        { id: 'fast', name: 'Fast Motion' },
        { id: 'timelapse', name: 'Timelapse' }
      ],
      grades: [
        { id: 'natural', name: 'Natural' },
        { id: 'cinematic', name: 'Cinematic LUT' },
        { id: 'warm', name: 'Warm Grade' },
        { id: 'cool', name: 'Cool Grade' },
        { id: 'bw', name: 'Black & White' }
      ],
      negatives: {
        sfw: 'artifacts, jitter, compression artifacts, low quality, blurry',
        nsfw: 'artifacts, jitter, compression artifacts, low quality, blurry, censored'
      },
      templates: [
        'Cinematic {genre} video, {camera} shot, {motion}, {grade}',
        '{genre} footage with {camera} movement, {grade} color grading',
        '{motion} {genre} scene, professional {grade} grade'
      ],
      featured: [
        'Cinematic', 'Nature', 'Abstract', 'Slow Motion', 'Drone Shot',
        'Natural Grade', 'Warm Tones', 'Portrait Video'
      ]
    },

    models: [
      { id: 'sd15-realistic', name: 'SD 1.5 Realistic', type: 'image', installed: true, hint: 'Klassisches SD für realistische Bilder' },
      { id: 'sdxl-base', name: 'SDXL Base', type: 'image', installed: false, hint: 'Hochauflösende SDXL-Generierung' },
      { id: 'flux-dev', name: 'FLUX.1 Dev', type: 'image', installed: false, hint: 'Kreative FLUX-Generierung' },
      { id: 'svd-xt', name: 'SVD XT', type: 'video', installed: false, hint: 'Video-Generierung' },
      { id: 'realistic-nsfw', name: 'Realistic NSFW', type: 'image', installed: true, nsfw: true, hint: 'Ohne Content-Filter' }
    ]
  };

  // =============== MAIN API CLASS ===============

  class AndioDataAPI {
    constructor() {
      this.api = new AndioAPI();
      this.cache = new Map();
      this.cacheTimeout = 5 * 60 * 1000; // 5 Minuten
    }

    async _getCached(key, fetcher, fallback) {
      // Cache-Check
      const cached = this.cache.get(key);
      if (cached && (Date.now() - cached.timestamp) < this.cacheTimeout) {
        return cached.data;
      }

      try {
        // Versuche Backend-Request
        const response = await fetcher();
        const data = response.data || response;
        
        // Cache result
        this.cache.set(key, { data, timestamp: Date.now() });
        return data;

      } catch (error) {
        console.warn(`API call failed for ${key}, using fallback:`, error);
        
        // Return fallback data
        const data = fallback();
        this.cache.set(key, { data, timestamp: Date.now() });
        return data;
      }
    }

    // =============== PUBLIC API METHODS ===============

    async loadImagePresets() {
      return this._getCached(
        'imagePresets',
        () => this.api.getImagePresets(),
        () => FALLBACK_DATA.imagePresets
      );
    }

    async loadVideoPresets() {
      return this._getCached(
        'videoPresets',
        () => this.api.getVideoPresets(),
        () => FALLBACK_DATA.videoPresets
      );
    }

    async models(type = 'image', options = {}) {
      const cacheKey = `models_${type}_${JSON.stringify(options)}`;
      
      return this._getCached(
        cacheKey,
        () => this.api.getModels({ type, ...options }),
        () => ({ 
          models: FALLBACK_DATA.models.filter(m => m.type === type),
          smart_pick: FALLBACK_DATA.models.find(m => m.type === type && m.installed)
        })
      );
    }

    // =============== CONVENIENCE METHODS ===============

    smartPickModel(models, genreId) {
      if (!models || !Array.isArray(models)) return null;
      
      // Priorität: installiert + Genre-Match
      let candidates = models.filter(m => m.installed);
      
      if (genreId) {
        const genreMatch = candidates.find(m => 
          m.genres && m.genres.includes(genreId)
        );
        if (genreMatch) return genreMatch;
      }
      
      // Fallback: erstes installiertes Modell
      return candidates[0] || models[0] || null;
    }

    featuredChips(featured, options = {}) {
      const { nsfw = false, n = 12, genreId = null } = options;
      
      if (!featured || !Array.isArray(featured)) {
        featured = nsfw ? 
          FALLBACK_DATA.imagePresets.featured.concat(['Artistic Nude', 'Boudoir', 'Implied']) :
          FALLBACK_DATA.imagePresets.featured;
      }
      
      let chips = [...featured];
      
      // NSFW-Filter (einfach)
      if (nsfw) {
        chips = chips.concat(['Artistic', 'Boudoir', 'Soft Light', 'Intimate']);
      }
      
      // Genre-spezifische Chips
      if (genreId) {
        const genreChips = {
          'portrait': ['Close-up', 'Natural Light', 'Studio', 'Beauty'],
          'landscape': ['Golden Hour', 'Wide Shot', 'Natural', 'Panorama'],
          'art': ['Creative', 'Abstract', 'Experimental', 'Artistic'],
          'anime': ['Manga Style', 'Character', 'Colorful', 'Stylized']
        };
        
        if (genreChips[genreId]) {
          chips = genreChips[genreId].concat(chips);
        }
      }
      
      // Unique + Limit
      chips = [...new Set(chips)].slice(0, n);
      
      return chips;
    }

    randomImagePrompt(presets, options = {}) {
      const { genreId, nsfw = false } = options;
      
      if (!presets) presets = FALLBACK_DATA.imagePresets;
      
      const templates = presets.templates || FALLBACK_DATA.imagePresets.templates;
      const template = this._pickRandom(templates);
      
      // Genre-basiertes Subject
      const subjects = genreId ? this._getSubjectsForGenre(genreId, nsfw) : ['beautiful scene'];
      const subject = this._pickRandom(subjects);
      
      const style = this._pickRandom(presets.styles || [])?.name || 'realistic';
      const light = this._pickRandom(presets.lights || [])?.name || 'natural light';
      const comp = this._pickRandom(presets.comps || [])?.name || 'rule of thirds';
      const palette = this._pickRandom(presets.palettes || [])?.name || 'natural colors';
      
      return template
        .replace('{subject}', subject)
        .replace('{style}', style)
        .replace('{light}', light)
        .replace('{comp}', comp)
        .replace('{palette}', palette);
    }

    randomVideoPrompt(presets, options = {}) {
      const { genreId, nsfw = false } = options;
      
      if (!presets) presets = FALLBACK_DATA.videoPresets;
      
      const templates = presets.templates || FALLBACK_DATA.videoPresets.templates;
      const template = this._pickRandom(templates);
      
      const genre = genreId || this._pickRandom(presets.genres || [])?.id || 'cinematic';
      const camera = this._pickRandom(presets.cameras || [])?.id || 'handheld';
      const motion = this._pickRandom(presets.motions || [])?.id || 'natural';
      const grade = this._pickRandom(presets.grades || [])?.id || 'natural';
      
      let prompt = template
        .replace('{genre}', genre)
        .replace('{camera}', camera)
        .replace('{motion}', motion)
        .replace('{grade}', grade);
      
      // NSFW-Zusätze
      if (nsfw && genreId && genreId.includes('nsfw')) {
        prompt += ', artistic nude, soft lighting, tasteful composition';
      }
      
      return prompt;
    }

    _getSubjectsForGenre(genreId, nsfw) {
      const subjects = {
        'portrait': nsfw ? 
          ['artistic nude portrait', 'intimate portrait', 'boudoir portrait'] :
          ['professional portrait', 'natural portrait', 'beauty portrait'],
        'landscape': ['beautiful landscape', 'scenic nature', 'dramatic vista'],
        'art': nsfw ?
          ['artistic nude study', 'figure drawing', 'creative nude art'] :
          ['creative artwork', 'abstract composition', 'artistic scene'],
        'photography': ['professional photo', 'documentary shot', 'lifestyle photo'],
        'anime': ['anime character', 'manga illustration', 'stylized character'],
        'nsfw_artistic': ['artistic nude', 'figure study', 'intimate art'],
        'nsfw_realistic': ['realistic nude', 'sensual portrait', 'intimate photography']
      };
      
      return subjects[genreId] || ['beautiful scene'];
    }

    _pickRandom(array) {
      if (!array || array.length === 0) return null;
      return array[Math.floor(Math.random() * array.length)];
    }

    // =============== GENERATION API BRIDGES ===============

    async imageGenerate(payload) {
      try {
        const response = await this.api.generateImage(payload);
        return response.url || response.data?.url || null;
      } catch (error) {
        console.error('Image generation failed:', error);
        throw error;
      }
    }

    async videoGenerate(payload) {
      try {
        const response = await this.api.generateVideo(payload);
        return response.url || response.data?.url || null;
      } catch (error) {
        console.error('Video generation failed:', error);
        throw error;
      }
    }

    // =============== UPLOAD & FILES ===============

    async uploadFile(file) {
      try {
        const response = await this.api.uploadFile(file);
        return response.data || response;
      } catch (error) {
        console.error('Upload failed:', error);
        throw error;
      }
    }

    // =============== GALLERY & CATALOG ===============

    async getGallery(params = {}) {
      try {
        const response = await this.api.getGallery(params);
        return response.data || response;
      } catch (error) {
        console.warn('Gallery API failed, returning empty result:', error);
        return { images: [], videos: [] };
      }
    }

    async getCatalog() {
      try {
        const response = await this.api.getCatalog();
        return response.data || response;
      } catch (error) {
        console.warn('Catalog API failed, using fallback:', error);
        return {
          models: FALLBACK_DATA.models,
          statistics: { total_models: FALLBACK_DATA.models.length }
        };
      }
    }

    // =============== STATUS & HEALTH ===============

    async isBackendAvailable() {
      try {
        await this.api.ping();
        return true;
      } catch (error) {
        return false;
      }
    }

    // =============== CACHE MANAGEMENT ===============

    clearCache(key = null) {
      if (key) {
        this.cache.delete(key);
      } else {
        this.cache.clear();
      }
    }

    getCacheStats() {
      return {
        entries: this.cache.size,
        keys: Array.from(this.cache.keys()),
        timeout: this.cacheTimeout
      };
    }
  }

  // =============== GLOBAL EXPORT ===============

  // Create global instance
  window.AndioData = new AndioDataAPI();

  // Convenience aliases for backward compatibility
  window.AndioAPI = AndioDataAPI;
  window.AndioBridge = window.AndioData; // Alternative name

  // Status indicator
  window.AndioData.isBackendAvailable().then(available => {
    if (available) {
      console.info('🟢 AndioMediaStudio Backend: Online');
    } else {
      console.warn('🟡 AndioMediaStudio Backend: Offline (Fallback-Modus)');
    }
  });

  // Debug helpers (only in development)
  if (typeof window !== 'undefined' && window.location.hostname === 'localhost') {
    window.AndioDebug = {
      api: window.AndioData.api,
      cache: window.AndioData.cache,
      fallbacks: FALLBACK_DATA,
      clearCache: () => window.AndioData.clearCache(),
      cacheStats: () => window.AndioData.getCacheStats()
    };
  }

})();