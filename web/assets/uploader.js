// web/assets/uploader.js
import { apiUpload } from './api.js';

export function bindDropzone(el, onUploaded) {
  const highlight = (e, on) => {
    e.preventDefault();
    e.stopPropagation();
    if (on) el.classList.add('ring', 'ring-blue-500'); else el.classList.remove('ring', 'ring-blue-500');
  };
  ['dragenter','dragover'].forEach(ev => el.addEventListener(ev, e => highlight(e, true)));
  ['dragleave','drop'].forEach(ev => el.addEventListener(ev, e => highlight(e, false)));
  el.addEventListener('drop', async (e) => {
    const file = e.dataTransfer.files[0];
    if (file) {
      const res = await apiUpload('/api/upload', file);
      onUploaded(res);
    }
  });
  el.addEventListener('click', async () => {
    const inp = document.createElement('input');
    inp.type = 'file';
    inp.accept = '.jpg,.jpeg,.png,.webp';
    inp.onchange = async () => {
      const file = inp.files[0];
      if (file) {
        const res = await apiUpload('/api/upload', file);
        onUploaded(res);
      }
    };
    inp.click();
  });
}
