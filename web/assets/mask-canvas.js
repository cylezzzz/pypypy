// web/assets/mask-canvas.js
// Minimal masking canvas: white strokes = to-be-inpainted, black = keep
export class MaskCanvas {
  constructor(containerId){
    this.container = document.getElementById(containerId);
    this.canvas = document.createElement('canvas');
    this.canvas.width = this.container.clientWidth || 640;
    this.canvas.height = this.container.clientHeight || 480;
    this.ctx = this.canvas.getContext('2d');
    this.ctx.fillStyle = 'black';       // default keep
    this.ctx.fillRect(0,0,this.canvas.width,this.canvas.height);
    this.brushColor = 'white';          // paint area to change
    this.brushSize = 28;
    this.isDown = false;
    this._bind();
    this.container.appendChild(this.canvas);
  }
  _bind(){
    const c=this.canvas;
    const pos = (e)=>{
      const r=c.getBoundingClientRect();
      const x=(e.touches?e.touches[0].clientX:e.clientX)-r.left;
      const y=(e.touches?e.touches[0].clientY:e.clientY)-r.top;
      return {x,y};
    };
    const draw = (e)=>{
      if(!this.isDown) return;
      const {x,y}=pos(e);
      this.ctx.lineCap='round';
      this.ctx.lineJoin='round';
      this.ctx.strokeStyle=this.brushColor;
      this.ctx.lineWidth=this.brushSize;
      this.ctx.beginPath();
      this.ctx.moveTo(this._lastX ?? x, this._lastY ?? y);
      this.ctx.lineTo(x,y);
      this.ctx.stroke();
      this._lastX=x; this._lastY=y;
      e.preventDefault();
    };
    c.addEventListener('mousedown', e=>{this.isDown=true; this._lastX=this._lastY=null; draw(e);});
    c.addEventListener('mousemove', draw);
    c.addEventListener('mouseup', ()=>{this.isDown=false;});
    c.addEventListener('mouseleave', ()=>{this.isDown=false;});
    c.addEventListener('touchstart', e=>{this.isDown=true; this._lastX=this._lastY=null; draw(e);}, {passive:false});
    c.addEventListener('touchmove', draw, {passive:false});
    c.addEventListener('touchend', ()=>{this.isDown=false;});
  }
  setBrush(size){ this.brushSize = size; }
  setMode(mode){ this.brushColor = (mode==='erase' ? 'black' : 'white'); }
  clear(){ this.ctx.fillStyle='black'; this.ctx.fillRect(0,0,this.canvas.width,this.canvas.height); }
  exportPNG(){ return this.canvas.toDataURL('image/png'); } // data URL
}