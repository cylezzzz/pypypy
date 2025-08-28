const API = {
  async ping(){ return (await fetch('/api/ping')).json() },
  async models(){ return (await fetch('/api/models')).json() },
  async outputs(kind='all'){ return (await fetch('/api/outputs?kind='+encodeURIComponent(kind))).json() },
  async upload(file, target='workspace'){
    const fd = new FormData(); fd.append('file', file); fd.append('target', target);
    return (await fetch('/api/upload', {method:'POST', body:fd})).json();
  },
  async job(payload){
    return (await fetch('/api/jobs', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)})).json();
  }
};
