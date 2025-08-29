// assets/talking.page.js
import { apiGet } from './api.js';
async function init(){ try{ await apiGet('/api/ping'); const s=document.getElementById('talkingModelStatus'); if(s) s.textContent='Backend Ready'; }catch{} }
init();
