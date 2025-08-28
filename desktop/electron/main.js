const { app, BrowserWindow } = require('electron');
const { spawn } = require('child_process');
let py, win;
function createWindow(){
  win = new BrowserWindow({ width:1280, height:800, webPreferences:{ nodeIntegration:false }});
  win.loadURL('http://localhost:3000');
}
app.whenReady().then(()=>{
  py = spawn(process.platform === 'win32' ? 'python' : 'python3', ['start.py'], { cwd: __dirname + '/../../' });
  py.stdout.on('data', d => process.stdout.write(d));
  py.stderr.on('data', d => process.stderr.write(d));
  setTimeout(createWindow, 1500);
});
app.on('window-all-closed', ()=>{ if (process.platform !== 'darwin') app.quit(); });
app.on('quit', ()=>{ if(py) py.kill('SIGTERM'); });
