const fileInput = document.getElementById('fileInput');
const predictBtn = document.getElementById('predictBtn');
const preview = document.getElementById('preview');
const result = document.getElementById('result');

fileInput.addEventListener('change', () => {
  const f = fileInput.files[0];
  if (!f) { predictBtn.disabled = true; preview.style.display='none'; return; }
  predictBtn.disabled = false;
  const reader = new FileReader();
  reader.onload = e => { preview.src = e.target.result; preview.style.display='block'; }
  reader.readAsDataURL(f);
});

predictBtn.addEventListener('click', async () => {
  const f = fileInput.files[0];
  if (!f) return;
  predictBtn.disabled = true;
  result.innerText = 'Predicting...';
  const fd = new FormData();
  fd.append('file', f);
  try {
    const res = await fetch('/predict', { method: 'POST', body: fd });
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(txt || 'Request failed');
    }
    const j = await res.json();
    result.innerHTML = `<strong>${j.predicted_class}</strong> — ${(j.probability*100).toFixed(1)}%<br>` +
      j.topk.map(t => `${t.class}: ${(t.probability*100).toFixed(1)}%`).join('<br>');
  } catch (err) {
    result.innerText = 'Error: ' + err.message;
  }
  predictBtn.disabled = false;
});
