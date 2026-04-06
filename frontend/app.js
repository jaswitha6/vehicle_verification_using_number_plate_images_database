/* ── Config ───────────────────────────────────────────────── */
const API = '/api';

/* ── State ────────────────────────────────────────────────── */
let selectedFile = null;

/* ── DOM refs ─────────────────────────────────────────────── */
const uploadZone       = document.getElementById('uploadZone');
const fileInput        = document.getElementById('fileInput');
const previewContainer = document.getElementById('previewContainer');
const previewImage     = document.getElementById('previewImage');
const clearBtn         = document.getElementById('clearBtn');
const verifyBtn        = document.getElementById('verifyBtn');
const manualPlate      = document.getElementById('manualPlate');
const manualBtn        = document.getElementById('manualBtn');

const resultCard        = document.getElementById('resultCard');
const resultPlaceholder = document.getElementById('resultPlaceholder');
const resultContent     = document.getElementById('resultContent');
const loadingState      = document.getElementById('loadingState');
const loadingText       = document.getElementById('loadingText');
const loadingSteps      = document.getElementById('loadingSteps');

const decisionBanner  = document.getElementById('decisionBanner');
const decisionIcon    = document.getElementById('decisionIcon');
const decisionLabel   = document.getElementById('decisionLabel');
const decisionSub     = document.getElementById('decisionSub');
const confidenceBadge = document.getElementById('confidenceBadge');
const plateDisplay    = document.getElementById('plateDisplay');
const infoGrid        = document.getElementById('infoGrid');
const dipStepsEl      = document.getElementById('dipSteps');
const nlpStepsEl      = document.getElementById('nlpSteps');
const stageImagesEl   = document.getElementById('stageImages');

/* ── Move loadingState out of resultContent so tabs don't show it ── */
if (loadingState && resultCard) {
  resultCard.appendChild(loadingState);
}

/* ── Upload / Preview ─────────────────────────────────────── */
uploadZone.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f) setFile(f);
});
fileInput.addEventListener('change', () => { if (fileInput.files[0]) setFile(fileInput.files[0]); });
clearBtn.addEventListener('click', clearFile);

function setFile(f) {
  selectedFile = f;
  previewImage.src = URL.createObjectURL(f);
  previewContainer.classList.remove('hidden');
  uploadZone.classList.add('hidden');
  verifyBtn.disabled = false;
}

function clearFile() {
  selectedFile = null;
  fileInput.value = '';
  previewContainer.classList.add('hidden');
  uploadZone.classList.remove('hidden');
  verifyBtn.disabled = true;
  showPlaceholder();
}

/* ── Loading UI ───────────────────────────────────────────── */
const LOADING_MSGS = [
  'Preprocessing image…',
  'Running DIP pipeline…',
  'Extracting plate region…',
  'Running OCR…',
  'Analysing with NLP…',
  'Querying database…',
  'Finalizing result…'
];

function showLoading() {
  resultPlaceholder.classList.add('hidden');
  resultContent.classList.add('hidden');
  resultContent.style.display = 'none';
  loadingState.classList.remove('hidden');
  loadingState.style.display = '';
  let idx = 0;
  loadingText.textContent = LOADING_MSGS[0];
  loadingSteps.innerHTML = '';
  const iv = setInterval(() => {
    idx = Math.min(idx + 1, LOADING_MSGS.length - 1);
    loadingText.textContent = LOADING_MSGS[idx];
  }, 1800);
  loadingState._iv = iv;
}

function stopLoading() {
  if (loadingState._iv) { clearInterval(loadingState._iv); loadingState._iv = null; }
  loadingState.classList.add('hidden');
  loadingState.style.display = 'none';
}

function showPlaceholder() {
  stopLoading();
  resultContent.classList.add('hidden');
  resultContent.style.display = 'none';
  resultPlaceholder.classList.remove('hidden');
}

function showResultPanel() {
  stopLoading();
  resultPlaceholder.classList.add('hidden');
  resultContent.classList.remove('hidden');
  resultContent.style.display = '';
}

/* ── Tab switching ────────────────────────────────────────── */
document.querySelectorAll('.ptab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.ptab').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.ptab-panel').forEach(p => {
      p.classList.add('hidden');
      p.style.display = 'none';
    });
    btn.classList.add('active');
    const panel = document.getElementById('ptab-' + btn.dataset.ptab);
    panel.classList.remove('hidden');
    panel.style.display = '';
  });
});

/* ── Verify image ─────────────────────────────────────────── */
verifyBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  showLoading();
  verifyBtn.disabled = true;
  try {
    const fd = new FormData();
    fd.append('image', selectedFile);
    const res  = await fetch(`${API}/verify`, { method: 'POST', body: fd });
    const data = await res.json();
    renderResult(data);
  } catch (err) {
    stopLoading(); showPlaceholder(); alert('Error: ' + err.message);
  } finally {
    verifyBtn.disabled = false;
  }
});

/* ── Manual verify ────────────────────────────────────────── */
manualBtn.addEventListener('click', doManual);
manualPlate.addEventListener('keydown', e => { if (e.key === 'Enter') doManual(); });

async function doManual() {
  const plate = manualPlate.value.trim().toUpperCase();
  if (!plate) return;
  showLoading();
  manualBtn.disabled = true;
  try {
    const res  = await fetch(`${API}/manual-verify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ plate_number: plate })
    });
    const data = await res.json();
    renderResult(data);
  } catch (err) {
    stopLoading(); showPlaceholder(); alert('Error: ' + err.message);
  } finally {
    manualBtn.disabled = false;
  }
}

/* ── Render result ────────────────────────────────────────── */
function renderResult(d) {
  const allowed = d.decision === 'ALLOW';
  decisionBanner.className  = 'decision-banner ' + (allowed ? 'allow' : 'deny');
  decisionIcon.textContent  = allowed ? '✓' : '✕';
  decisionLabel.textContent = allowed ? 'Access Granted' : 'Access Denied';
  decisionSub.textContent   = allowed ? 'Vehicle is registered and authorised.' : (d.reason || 'Vehicle not found in database.');

  const confRaw = d.final_confidence != null ? d.final_confidence : (d.confidence != null ? d.confidence : null);
  const conf    = confRaw != null ? Math.round(confRaw) : null;
  confidenceBadge.textContent = conf != null ? conf + '% confidence' : '';
  confidenceBadge.className   = 'confidence-badge ' + (conf >= 70 ? 'high' : conf >= 40 ? 'mid' : 'low');

  plateDisplay.innerHTML = '<span class="plate-text">' + (d.plate_number || d.extracted_plate || '—') + '</span>';


  var dips = d.dip_steps || [];
  dipStepsEl.innerHTML  = dips.length  ? dips.map(renderStep).join('')  : '<p class="no-steps">No DIP steps recorded.</p>';

  var nlps = d.nlp_steps || [];
  nlpStepsEl.innerHTML  = nlps.length  ? nlps.map(renderStep).join('')  : '<p class="no-steps">No NLP steps recorded.</p>';

  var imgs = (d.stage_images || []).filter(function(img){ return (img.label||"").toLowerCase() !== "plate crop"; });
  stageImagesEl.innerHTML = imgs.length
    ? imgs.map(function(img) {
        var src = img.data ? 'data:image/jpeg;base64,' + img.data : (img.url || '');
        return '<div class="stage-img-wrap"><img src="' + src + '" alt="' + (img.label||'') + '" loading="lazy"/><span class="stage-label">' + (img.label||'') + '</span></div>';
      }).join('')
    : '<p class="no-steps">No stage images available.</p>';

  /* reset to DIP tab */
  document.querySelectorAll('.ptab').forEach(function(b, i) { b.classList.toggle('active', i === 0); });
  document.querySelectorAll('.ptab-panel').forEach(function(p, i) {
    p.classList.toggle('hidden', i !== 0);
    p.style.display = i === 0 ? '' : 'none';
  });

  showResultPanel();
}

function renderStep(s) {
  var status = s.status || 'ok';
  var icon = status === 'ok' ? '✓' : status === 'error' ? '✕' : '◎';
  var cls  = status === 'ok' ? 'step-ok' : status === 'error' ? 'step-err' : 'step-warn';
  return '<div class="pipeline-step ' + cls + '"><span class="step-icon">' + icon + '</span><div class="step-body"><span class="step-name">' + (s.name || s.step || '') + '</span>' + (s.detail ? '<span class="step-detail">' + s.detail + '</span>' : '') + '</div></div>';
}