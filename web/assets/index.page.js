// assets/index.page.js
import { apiGet } from './api.js';
async function ping(){
  const dot = document.querySelector('.dot');
  try{ await apiGet('/api/ping'); if(dot){dot.style.background='linear-gradient(135deg,#22c55e,#16a34a)';} }
  catch{ if(dot){dot.style.background='linear-gradient(135deg,#ef4444,#b91c1c)';} }
}
ping();
