// INDIGO frontend — vanilla JS, no framework

const $  = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ============================================================================
// localStorage shims
// ============================================================================
const LS = {
  get(k, fallback)   { try { const v = localStorage.getItem(k); return v == null ? fallback : JSON.parse(v); } catch { return fallback; } },
  set(k, v)          { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
};

// ============================================================================
// Status + bootstrap
// ============================================================================
let _MMAX = 32;

async function loadStatus() {
  try {
    const r = await fetch('/api/status'); if (!r.ok) return;
    const s = await r.json();
    if (s.m_max)             _MMAX = s.m_max;
    if (s.device)            $('#statusDevice').querySelector('span').textContent = s.device;
    if (s.pool_size != null) $('#statusPool').querySelector('span').textContent   = `${s.pool_size} mats (cap ${_MMAX})`;
    if (s.model_tag)         $('#statusModel').querySelector('span').textContent  = s.model_tag.slice(0, 36) + (s.model_tag.length > 36 ? '…' : '');
    $('#csvMmax').textContent = String(_MMAX);
    if (!s.openai_key_present && !LS.get('openai_api_key')) {
      $('#promptHint').textContent = 'No server-side OPENAI_API_KEY. Paste yours in Advanced settings, or set INDIGO_PARSE_BACKEND=mock.';
      $('#promptHint').classList.remove('text-slate-500');
      $('#promptHint').classList.add('text-amber-300/80');
    }
  } catch (e) { console.error('status', e); }
}

// ============================================================================
// Tab switching
// ============================================================================
function bindTabs() {
  $$('.tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      $$('.tab').forEach((t) => t.classList.remove('tab-active'));
      tab.classList.add('tab-active');
      const target = tab.dataset.tab;
      $$('[data-tab-pane]').forEach((p) => p.classList.toggle('hidden', p.dataset.tabPane !== target));
    });
  });
}

// ============================================================================
// Lab ↔ sRGB
// ============================================================================
function labToSrgbObj(L, a, b) {
  const fy = (L + 16) / 116;
  const fx = a / 500 + fy;
  const fz = fy - b / 200;
  const eps3 = (6/29)**3;
  const f = (t) => (t**3 > eps3 ? t**3 : (t - 4/29) * 3 * (6/29)**2);
  const X = 95.047  * f(fx) / 100;
  const Y = 100.0   * f(fy) / 100;
  const Z = 108.883 * f(fz) / 100;
  const rl =  3.2404542*X + -1.5371385*Y + -0.4985314*Z;
  const gl = -0.9692660*X +  1.8760108*Y +  0.0415560*Z;
  const bl =  0.0556434*X + -0.2040259*Y +  1.0572252*Z;
  const enc = (c) => { c = Math.max(0, Math.min(1, c)); return c <= 0.0031308 ? 12.92*c : 1.055*Math.pow(c, 1/2.4) - 0.055; };
  return { r: enc(rl), g: enc(gl), b: enc(bl) };
}
function labToSrgb(L,a,b) { const {r,g,b:bb} = labToSrgbObj(L,a,b); return `rgb(${Math.round(r*255)}, ${Math.round(g*255)}, ${Math.round(bb*255)})`; }

function srgbHexToLab(hex) {
  const m = hex.replace('#','').match(/^([0-9a-f]{6})$/i);
  if (!m) return null;
  const r = parseInt(m[1].slice(0,2),16)/255, g = parseInt(m[1].slice(2,4),16)/255, b = parseInt(m[1].slice(4,6),16)/255;
  const lin = (c) => (c <= 0.04045 ? c/12.92 : Math.pow((c+0.055)/1.055, 2.4));
  const rl=lin(r), gl=lin(g), bl=lin(b);
  const X = (0.4124564*rl + 0.3575761*gl + 0.1804375*bl) * 100;
  const Y = (0.2126729*rl + 0.7151522*gl + 0.0721750*bl) * 100;
  const Z = (0.0193339*rl + 0.1191920*gl + 0.9503041*bl) * 100;
  const Xn=95.047, Yn=100.0, Zn=108.883;
  const delta = 6/29;
  const f = (t) => (t > delta**3 ? Math.cbrt(t) : t/(3*delta*delta) + 4/29);
  const fx=f(X/Xn), fy=f(Y/Yn), fz=f(Z/Zn);
  return { L: 116*fy - 16, a: 500*(fx - fy), b: 200*(fy - fz) };
}

function bindColorInputs() {
  const sync = (src) => {
    const L = parseFloat($('#labL').value);
    const a = parseFloat($('#labA').value);
    const b = parseFloat($('#labB').value);
    if (![L,a,b].every(Number.isFinite)) return;
    $('#labPreview').style.backgroundColor = labToSrgb(L,a,b);
    if (src !== 'rgb') {
      const { r, g, b:bb } = labToSrgbObj(L,a,b);
      const hex = '#' + [r,g,bb].map((v) => Math.round(v*255).toString(16).padStart(2,'0')).join('');
      $('#rgbPicker').value = hex;
      $('#rgbReadout').textContent = hex;
    }
  };
  ['labL','labA','labB'].forEach((id) => $(`#${id}`).addEventListener('input', () => sync('lab')));
  $('#rgbPicker').addEventListener('input', (e) => {
    const hex = e.target.value; $('#rgbReadout').textContent = hex;
    const lab = srgbHexToLab(hex); if (!lab) return;
    $('#labL').value = lab.L.toFixed(1);
    $('#labA').value = lab.a.toFixed(1);
    $('#labB').value = lab.b.toFixed(1);
    sync('rgb');
  });
  sync('lab');
}

// ============================================================================
// Constraint chips (cheat-sheet → appends to textarea)
// ============================================================================
function bindConstraintChips() {
  $$('.cs-btn').forEach((b) => {
    b.addEventListener('click', () => {
      const ta = $('#constraintsInput');
      let cur = []; const txt = ta.value.trim();
      if (txt) {
        try { cur = JSON.parse(txt); if (!Array.isArray(cur)) cur = [cur]; }
        catch { cur = []; }
      }
      try { cur.push(JSON.parse(b.dataset.cs)); } catch {}
      ta.value = JSON.stringify(cur, null, 2);
    });
  });
}

// ============================================================================
// Material picker
// ============================================================================
let MATERIAL_LIST = [];           // [{canonical_name, source}]
const SELECTED   = new Set();     // selected canonical_name set
const CUSTOM     = new Map();     // canonical_name -> {n,k,source}

async function loadPool() {
  const r = await fetch('/api/pool'); if (!r.ok) return;
  const data = await r.json();
  MATERIAL_LIST = data.materials || [];
  _MMAX = data.m_max || _MMAX;
  // Server's default subset becomes our initial selection unless localStorage has one.
  const stored = LS.get('pool_subset', null);
  const initial = stored ?? (data.default_subset || MATERIAL_LIST.slice(0, _MMAX).map((m) => m.canonical_name));
  SELECTED.clear(); initial.forEach((n) => SELECTED.add(n));
  renderPool();
}

function renderPool() {
  const filter = ($('#poolSearch').value || '').toLowerCase();
  const list = $('#poolList'); list.innerHTML = '';

  const allMats = [
    ...[...CUSTOM.values()].map((m) => ({ ...m, custom: true })),
    ...MATERIAL_LIST,
  ];

  allMats
    .filter((m) => !filter || m.canonical_name.toLowerCase().includes(filter))
    .forEach((m) => {
      const row = document.createElement('label');
      row.className = 'pool-row';
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = SELECTED.has(m.canonical_name);
      cb.addEventListener('change', () => {
        if (cb.checked) {
          if (SELECTED.size >= _MMAX) {
            cb.checked = false;
            flashError(`Pool capped at ${_MMAX}. Deselect another first.`);
            return;
          }
          SELECTED.add(m.canonical_name);
        } else {
          SELECTED.delete(m.canonical_name);
        }
        LS.set('pool_subset', [...SELECTED]);
        updatePoolCount();
      });
      const label = document.createElement('span');
      label.className = 'truncate';
      label.textContent = m.canonical_name;
      const badge = document.createElement('span');
      badge.className = 'pool-badge';
      badge.textContent = m.custom ? 'custom' : (m.source === 'jaxlayerlumos' ? 'JLL' : m.source);
      row.appendChild(cb); row.appendChild(label); row.appendChild(badge);
      list.appendChild(row);
    });
  updatePoolCount();
}

function updatePoolCount() {
  $('#poolCount').textContent = `${SELECTED.size} / ${_MMAX}`;
  $('#poolCount').classList.toggle('text-rose-300', SELECTED.size > _MMAX);
}

function bindPoolControls() {
  $('#poolSearch').addEventListener('input', renderPool);
  $('#poolSelectAll').addEventListener('click', () => {
    SELECTED.clear();
    const allNames = [...CUSTOM.keys(), ...MATERIAL_LIST.map((m) => m.canonical_name)];
    for (const n of allNames) { if (SELECTED.size >= _MMAX) break; SELECTED.add(n); }
    LS.set('pool_subset', [...SELECTED]); renderPool();
  });
  $('#poolSelectNone').addEventListener('click', () => {
    SELECTED.clear(); LS.set('pool_subset', []); renderPool();
  });
  $('#poolUploadBtn').addEventListener('click', () => openCsvModal());
}

// ============================================================================
// CSV upload modal
// ============================================================================
function openCsvModal() {
  $('#csvModal').classList.remove('hidden');
  $('#csvName').value = ''; $('#csvFile').value = '';
  $('#csvError').classList.add('hidden');
}
function closeCsvModal() { $('#csvModal').classList.add('hidden'); }

async function submitCsvUpload() {
  const name = ($('#csvName').value || '').trim();
  const file = $('#csvFile').files[0];
  const err  = $('#csvError');
  err.classList.add('hidden');
  if (!name)  { err.textContent = 'Canonical name is required.'; err.classList.remove('hidden'); return; }
  if (!file)  { err.textContent = 'Choose a CSV file.';            err.classList.remove('hidden'); return; }

  const btn = $('#csvSubmitBtn');
  btn.disabled = true; btn.querySelector('.spinner').classList.remove('hidden');
  btn.querySelector('.btn-label').textContent = 'Uploading…';

  const fd = new FormData(); fd.append('name', name); fd.append('file', file);
  try {
    const r = await fetch('/api/material_from_csv', { method: 'POST', body: fd });
    if (!r.ok) { const txt = await r.text(); throw new Error(`${r.status}: ${txt}`); }
    const mat = await r.json();
    CUSTOM.set(mat.canonical_name, mat);
    if (SELECTED.size < _MMAX) SELECTED.add(mat.canonical_name);
    LS.set('pool_subset', [...SELECTED]);
    renderPool();
    closeCsvModal();
  } catch (e) {
    err.textContent = e.message; err.classList.remove('hidden');
  } finally {
    btn.disabled = false; btn.querySelector('.spinner').classList.add('hidden');
    btn.querySelector('.btn-label').textContent = 'Upload & add to pool';
  }
}

function bindCsvModal() {
  $('#csvCloseBtn').addEventListener('click', closeCsvModal);
  $('#csvCancelBtn').addEventListener('click', closeCsvModal);
  $('#csvSubmitBtn').addEventListener('click', submitCsvUpload);
  $('#csvModal').addEventListener('click', (e) => { if (e.target === $('#csvModal')) closeCsvModal(); });
}

// ============================================================================
// Progress strip + elapsed timer
//
// Two sources feed the strip:
//   1) Server-Sent Events from /api/solve_stream — REAL milestones (phase
//      changes, per-iter ticks during refinement and MC). When they arrive,
//      `setRealProgress` overrides the simulated bar and pins the phase label.
//   2) A wall-clock timer animating the bar between milestones so the user
//      always sees motion, even during a slow JAX call.
// ============================================================================

// Real-progress checkpoints. The bar is monotonic; once a higher pct is
// reached, the timer can only push it further along (up to the next
// checkpoint). Each entry: stage -> {pctAtStart, pctAtEnd, label}.
const STAGE_BANDS = {
  start:                  { start: 0.00, end: 0.05, label: 'encoding pool' },
  generate:               { start: 0.05, end: 0.18, label: 'sampling ensemble' },
  generated:              { start: 0.18, end: 0.20, label: 'deduplicating candidates' },
  select_start:           { start: 0.20, end: 0.22, label: 'simulating candidates' },
  simulate:               { start: 0.22, end: 0.45, label: 'simulating candidates' },
  refine_start:           { start: 0.45, end: 0.47, label: 'refining top-k' },
  refine_candidate_start: { start: 0.47, end: 0.48, label: 'refining candidate' },
  refine_iter:            { start: 0.47, end: 0.85, label: 'refining candidate' },
  refine_candidate_end:   { start: 0.85, end: 0.86, label: 'refining candidate' },
  mc_start:               { start: 0.86, end: 0.87, label: 'Monte-Carlo robustness' },
  mc:                     { start: 0.87, end: 0.96, label: 'Monte-Carlo robustness' },
  finalising:             { start: 0.96, end: 0.99, label: 'finalising — re-ranking' },
  done:                   { start: 1.00, end: 1.00, label: 'done' },
};

let _progressTimer = null;
let _progressStart = 0;
let _progressEnd  = 0;       // estimated wall-clock end (for animation only)
let _realPct      = 0;       // highest pct reported by the server
let _realLabel    = '';      // current phase label (from server, if any)
let _bandMax      = 0.05;    // upper bound the simulated timer may push toward
let _detailText   = '';      // sub-step text: "iter 12/25 — ΔE 4.71" etc.

function startProgress(estSeconds) {
  _progressStart = performance.now();
  _progressEnd   = _progressStart + estSeconds * 1000;
  _realPct = 0; _realLabel = ''; _detailText = ''; _bandMax = 0.05;
  $('#progressStrip').classList.remove('hidden');
  $('#errorBanner').classList.add('hidden');
  $('#elapsedBadge').classList.remove('hidden');
  $('#progressBar').style.width = '0%';
  $('#progressPhase').textContent = 'queued';
  const tick = () => {
    const now = performance.now();
    const elapsedS = (now - _progressStart) / 1000;
    // Estimated pct from wall clock, slowed near the top.
    let estPct = ((now - _progressStart) / (_progressEnd - _progressStart));
    if (estPct >= 0.95) estPct = 0.95 + (1 - Math.exp(-(elapsedS - estSeconds) / 20)) * 0.049;
    // Real pct + slow drift toward the current band's upper edge.
    let pct = Math.max(_realPct, Math.min(_bandMax, estPct)) * 100;
    pct = Math.max(0, Math.min(99.5, pct));
    $('#progressBar').style.width = pct.toFixed(1) + '%';
    $('#progressTime').textContent = elapsedS.toFixed(1) + 's';
    $('#elapsedBadge').textContent = elapsedS.toFixed(1) + 's';
    const phaseTxt = _realLabel || 'preparing';
    $('#progressPhase').textContent = _detailText
      ? `${phaseTxt} — ${_detailText}` : phaseTxt;
  };
  tick();
  _progressTimer = setInterval(tick, 100);
}

function setRealProgress(stage, current, total, info) {
  const band = STAGE_BANDS[stage];
  if (!band) return;
  let frac;
  if (total > 0) {
    frac = Math.max(0, Math.min(1, current / total));
  } else {
    frac = 1.0;
  }
  // Map progress into the band [start, end].
  const pct = band.start + (band.end - band.start) * frac;
  _realPct = Math.max(_realPct, pct);
  _bandMax = Math.max(_bandMax, band.end);
  _realLabel = band.label;
  _detailText = buildDetail(stage, current, total, info || {});
}

function buildDetail(stage, current, total, info) {
  switch (stage) {
    case 'generate':       return '';
    case 'generated':      return `${info.unique ?? current} unique`;
    case 'simulate':       return `${current}/${total}`;
    case 'refine_candidate_start':
    case 'refine_candidate_end':
      return `${current}/${total} candidate${total > 1 ? 's' : ''}`;
    case 'refine_iter': {
      const cand = info.candidate, candTotal = info.candidates_total;
      const candPart = (cand && candTotal && candTotal > 1)
        ? `cand ${cand}/${candTotal} · ` : '';
      const dePart = (typeof info.de === 'number')
        ? ` · ΔE ${info.de.toFixed(2)}` : '';
      return `${candPart}iter ${current}/${total}${dePart}`;
    }
    case 'mc': {
      const cand = info.candidate, candTotal = info.candidates_total;
      const candPart = (cand && candTotal && candTotal > 1)
        ? `cand ${cand}/${candTotal} · ` : '';
      return `${candPart}draw ${current}/${total}`;
    }
    case 'finalising': return 're-ranking + provenance';
    default: return '';
  }
}

function stopProgress(success) {
  if (_progressTimer) { clearInterval(_progressTimer); _progressTimer = null; }
  $('#progressBar').style.width = success ? '100%' : '0%';
  setTimeout(() => {
    $('#progressStrip').classList.add('hidden');
    $('#elapsedBadge').classList.add('hidden');
    $('#progressBar').style.width = '0%';
  }, success ? 500 : 0);
}

// ============================================================================
// SSE consumer — fetch + ReadableStream (EventSource is GET-only).
// ============================================================================

// Most-recent stream diagnostics; surfaced in the error banner when a solve
// ends without dispatching a result. Don't trust server logs alone — when
// the page is opened from another machine the user doesn't see them.
let _lastSseDiag = null;

async function streamSolve(body, onProgress, onResult, onError) {
  const r = await fetch('/api/solve_stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const txt = await r.text();
    throw new Error(`HTTP ${r.status}: ${txt}`);
  }
  if (!r.body) throw new Error('Streaming not supported in this browser.');
  const reader = r.body.getReader();
  const dec = new TextDecoder('utf-8');
  let buf = '';
  let raw = '';        // append-only copy of every decoded chunk (for diag)
  let nEvents = 0;
  let nChunks = 0;
  let nFrames = 0;
  let totalBytes = 0;
  const counts = { progress: 0, result: 0, error: 0, message: 0, hello: 0 };
  const parseErrors = [];
  const seenEventLines = [];          // every `event: …` line we observed

  const drainFrames = () => {
    while (true) {
      const m = buf.match(/\r\n\r\n|\n\n|\r\r/);
      if (!m) break;
      const idx = m.index;
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + m[0].length);
      nFrames++;
      let event = 'message'; const dataLines = [];
      frame.split(/\r\n|\n|\r/).forEach((line) => {
        if (line.startsWith(':')) return;
        if (line.startsWith('event:')) {
          event = line.slice(6).trim();
          seenEventLines.push(event);
        }
        else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim());
      });
      counts[event] = (counts[event] || 0) + 1;
      if (dataLines.length === 0) continue;
      let payload;
      try { payload = JSON.parse(dataLines.join('\n')); }
      catch (e) {
        parseErrors.push({ event, err: String(e),
                           sample: dataLines.join('\n').slice(0, 240) });
        console.warn('SSE: bad JSON', e, 'event=', event);
        continue;
      }
      nEvents++;
      if (event === 'result' || event === 'error') {
        console.log('SSE: dispatching', event, 'frame');
      }
      if      (event === 'progress') onProgress(payload);
      else if (event === 'result')   onResult(payload);
      else if (event === 'error')    onError(payload);
    }
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    nChunks++; totalBytes += value.length;
    const piece = dec.decode(value, { stream: true });
    buf += piece; raw += piece;
    drainFrames();
  }
  const tail = dec.decode();
  buf += tail; raw += tail;
  if (buf.length > 0 && !/(\r\n\r\n|\n\n|\r\r)$/.test(buf)) buf += '\n\n';
  drainFrames();

  _lastSseDiag = {
    chunks: nChunks, bytes: totalBytes, frames: nFrames, events: nEvents,
    counts, parseErrors,
    seenEventLines,
    bufLeftLen: buf.length,
    bufLeftSample: buf.slice(0, 500),
    rawLen: raw.length,
    rawTail: raw.slice(-1200),
    rawContainsResult: raw.includes('event: result'),
  };
  console.log('SSE done:', _lastSseDiag);
}

// ============================================================================
// Submit
// ============================================================================
function readKnobs() {
  return {
    ensemble_N:      parseInt($('#knobN').value, 10),
    top_k:           parseInt($('#knobK').value, 10),
    refine_top_n:    parseInt($('#knobRefineN').value, 10),
    temperature:     parseFloat($('#knobT').value),
    tolerance_pct:   parseFloat($('#knobTol').value),
    weight_lambda:   parseFloat($('#knobLambda').value),
    mc_samples:      parseInt($('#knobMc').value, 10),
    refine_max_iters:parseInt($('#knobRefine').value, 10),
    seed:            parseInt($('#knobSeed').value, 10),
  };
}

const PRESETS = {
  fast:     { knobN:100, knobK:1, knobRefineN:1, knobRefine:15, knobT:1.0, knobTol:0,  knobLambda:1.0, knobMc:0 },
  balanced: { knobN:150, knobK:3, knobRefineN:1, knobRefine:25, knobT:1.0, knobTol:0,  knobLambda:1.0, knobMc:0 },
  best:     { knobN:500, knobK:5, knobRefineN:3, knobRefine:80, knobT:1.0, knobTol:5,  knobLambda:1.0, knobMc:16 },
};
function bindPresets() {
  $$('.preset-btn').forEach((b) => {
    b.addEventListener('click', () => {
      const preset = PRESETS[b.dataset.preset]; if (!preset) return;
      for (const [id, v] of Object.entries(preset)) {
        const el = $(`#${id}`); if (el) el.value = v;
      }
    });
  });
}

function bindHowItWorks() {
  const open = () => $('#howModal').classList.remove('hidden');
  const close = () => $('#howModal').classList.add('hidden');
  $('#howItWorksBtn').addEventListener('click', (e) => { e.preventDefault(); open(); });
  $('#howCloseBtn').addEventListener('click', close);
  $('#howOkBtn').addEventListener('click', close);
  $('#howModal').addEventListener('click', (e) => { if (e.target === $('#howModal')) close(); });
}

function buildBody() {
  const activeTab = $('.tab-active').dataset.tab;
  const knobs = readKnobs();
  const openai_api_key = ($('#openaiKey').value || LS.get('openai_api_key') || '').trim() || undefined;
  const allow_custom_constraints = !!$('#knobAllowCustom').checked;

  // Pool: subset of JLL by canonical name + any custom materials currently selected
  const jllNames = new Set(MATERIAL_LIST.map((m) => m.canonical_name));
  const pool_subset       = [...SELECTED].filter((n) =>  jllNames.has(n));
  const custom_materials  = [...SELECTED]
    .filter((n) => !jllNames.has(n))
    .map((n) => CUSTOM.get(n))
    .filter(Boolean);

  if (activeTab === 'prompt') {
    const prompt = $('#promptInput').value.trim();
    if (!prompt) throw new Error('Prompt is empty.');
    return { prompt, knobs, pool_subset, custom_materials, openai_api_key,
             allow_custom_constraints };
  }
  const L = parseFloat($('#labL').value);
  const a = parseFloat($('#labA').value);
  const b = parseFloat($('#labB').value);
  if (![L,a,b].every(Number.isFinite)) throw new Error('Lab values must all be numbers.');
  let constraints = null;
  const ct = $('#constraintsInput').value.trim();
  if (ct) {
    try { constraints = JSON.parse(ct); }
    catch (e) { throw new Error('Constraints JSON invalid: ' + e.message); }
    if (!Array.isArray(constraints)) throw new Error('Constraints must be a JSON array.');
  }
  return { target_lab: [L,a,b], constraints, knobs, pool_subset, custom_materials,
           allow_custom_constraints };
}

function estimateSolveSeconds(knobs) {
  // Crude CPU model. Ensemble simulation is sequential per candidate;
  // refinement is per-(refine_top_n × refine_iters). Refinement dominates
  // when those are non-tiny, so weight it more aggressively than before.
  const base = 4;
  const refineN = knobs.refine_top_n > 0 ? knobs.refine_top_n : knobs.top_k;
  return base
       + (knobs.ensemble_N * 0.04)
       + (refineN * knobs.refine_max_iters * 0.25)
       + (knobs.mc_samples * refineN * 0.15);
}

async function runSolve() {
  const btn = $('#runBtn');
  const spinner = btn.querySelector('.spinner');
  const lbl = btn.querySelector('.btn-label');
  const err = $('#errorBanner');
  err.classList.add('hidden');

  let body; try { body = buildBody(); } catch (e) { err.textContent = e.message; err.classList.remove('hidden'); return; }

  // Persist OpenAI key whenever the user kicks off a run.
  if (body.openai_api_key) LS.set('openai_api_key', body.openai_api_key);

  btn.disabled = true; spinner.classList.remove('hidden'); lbl.textContent = 'Generating';
  startProgress(estimateSolveSeconds(body.knobs));

  let finished = false; let renderedError = null; let result = null;
  try {
    await streamSolve(
      body,
      (p) => setRealProgress(p.stage, p.current, p.total, p.info),
      (r) => { result = r; finished = true; },
      (e) => { renderedError = e.message || 'solve error'; finished = true; },
    );
  } catch (e) {
    stopProgress(false);
    console.error(e); err.textContent = e.message; err.classList.remove('hidden');
    btn.disabled = false; spinner.classList.add('hidden'); lbl.textContent = 'Generate';
    return;
  }
  if (renderedError) {
    stopProgress(false);
    err.innerHTML = ''; err.appendChild(buildErrorBlock(renderedError));
    err.classList.remove('hidden');
  } else if (result) {
    stopProgress(true);
    renderResult(result);
  } else if (!finished) {
    stopProgress(false);
    err.innerHTML = '';
    err.appendChild(buildErrorBlock(
      'Connection closed before a result arrived.',
      _lastSseDiag,
    ));
    err.classList.remove('hidden');
  }
  btn.disabled = false; spinner.classList.add('hidden'); lbl.textContent = 'Generate';
}

// ============================================================================
// Visible error block (with collapsible SSE diagnostics).
// ============================================================================
function buildErrorBlock(message, diag) {
  const wrap = document.createElement('div');
  const top = document.createElement('div');
  top.className = 'font-medium';
  top.textContent = message;
  wrap.appendChild(top);
  if (!diag) return wrap;

  const det = document.createElement('details');
  det.className = 'mt-2 text-[11px] text-slate-300/80';
  const sum = document.createElement('summary');
  sum.className = 'cursor-pointer text-slate-400 hover:text-slate-200';
  sum.textContent = 'SSE diagnostics';
  det.appendChild(sum);

  const summaryRow = document.createElement('div');
  summaryRow.className = 'mt-2 font-mono';
  summaryRow.textContent =
    `chunks=${diag.chunks} bytes=${diag.bytes} `
    + `frames=${diag.frames} events=${diag.events} `
    + `bufLeft=${diag.bufLeftLen} `
    + `rawContainsResult=${diag.rawContainsResult}`;
  det.appendChild(summaryRow);

  const countsRow = document.createElement('div');
  countsRow.className = 'font-mono';
  countsRow.textContent = 'counts: ' + JSON.stringify(diag.counts);
  det.appendChild(countsRow);

  const seenRow = document.createElement('div');
  seenRow.className = 'font-mono';
  const uniq = [...new Set(diag.seenEventLines)];
  seenRow.textContent = 'event lines seen: ' + JSON.stringify(uniq);
  det.appendChild(seenRow);

  if (diag.parseErrors && diag.parseErrors.length) {
    const pe = document.createElement('pre');
    pe.className = 'mt-2 whitespace-pre-wrap font-mono text-rose-300';
    pe.textContent = 'parse errors:\n' +
      diag.parseErrors.slice(0, 4).map((e, i) =>
        `[${i}] event=${e.event}\n    err=${e.err}\n    sample=${e.sample}`
      ).join('\n');
    det.appendChild(pe);
  }

  if (diag.bufLeftLen > 0) {
    const lb = document.createElement('pre');
    lb.className = 'mt-2 whitespace-pre-wrap font-mono';
    lb.textContent = 'leftover buffer (first 500 chars):\n' + diag.bufLeftSample;
    det.appendChild(lb);
  }

  const rt = document.createElement('pre');
  rt.className = 'mt-2 whitespace-pre-wrap font-mono opacity-80';
  rt.textContent = `raw tail (last ${Math.min(1200, diag.rawLen)} chars):\n`
    + (diag.rawTail || '');
  det.appendChild(rt);

  wrap.appendChild(det);
  return wrap;
}

// ============================================================================
// Result rendering
// ============================================================================
function deltaELabel(de) {
  if (de < 1)  return 'imperceptible';
  if (de < 2)  return 'just perceptible';
  if (de < 5)  return 'noticeable';
  if (de < 10) return 'clearly different';
  return 'far off';
}
const fmt = (x, d=2) => Number(x).toFixed(d);
const PALETTE = ['#3b82f6','#f59e0b','#10b981','#ef4444','#8b5cf6','#06b6d4','#ec4899','#84cc16','#a855f7','#f97316'];

// Most-recent result + which candidate is currently shown in the main view.
// idx 0 = chosen, idx 1..N = alternatives. Click on an alt card swaps the
// main view (swatches / ΔE / reflectance / layers / stats) to that candidate.
let _currentResult = null;
let _activeIdx = 0;

function allCandidates(result) {
  if (!result || !result.chosen) return [];
  return [result.chosen, ...(result.alternatives || [])];
}

function renderResult(result) {
  $('#placeholder').classList.add('hidden');
  const body = $('#resultBody');
  body.classList.remove('hidden');
  body.classList.remove('show'); void body.offsetWidth; body.classList.add('show');

  _currentResult = result;
  _activeIdx = 0;

  if (!result.chosen) {
    $('#deltaE').textContent = 'fail';
    $('#deltaELabel').textContent = (result.errors || []).join(' | ') || 'no candidate';
    $('#targetSwatch').style.backgroundColor = '#1a1c2a';
    $('#achievedSwatch').style.backgroundColor = '#1a1c2a';
    return;
  }

  renderCustomConstraintNotice(result);
  buildAltGrid();
  renderCandidateView(0);

  $('#provJSON').textContent = JSON.stringify({
    spec_echo: result.spec_echo,
    ensemble_stats: result.ensemble_stats,
    provenance: result.provenance,
  }, null, 2);
}

// ============================================================================
// Custom-constraint disclosure
//
// The parser fires an extra LLM call when the 8 standard constraint kinds
// can't express the user's request, generates Python source, runs it in a
// sandbox, and attaches the result to spec_echo.constraints[*]. The full
// source code is in the Result envelope, but we don't want users to have to
// dig through the Details JSON to see it — surface it as the FIRST thing in
// the result panel whenever it happens.
// ============================================================================
function renderCustomConstraintNotice(result) {
  const host = $('#customConstraintNotice');
  if (!host) return;
  host.innerHTML = '';

  const constraints = (result.spec_echo && result.spec_echo.constraints) || [];
  const customs = constraints.filter(
    (c) => (c.kind === 'custom') || (typeof c.source_code === 'string' && c.source_code.length > 0)
  );

  // Also surface the "requested but SKIPPED" case from the disclaimer so
  // users see when the parser TRIED to write code but bailed out (gate
  // rejection, frozen-dataclass repair miss, etc.).
  const disclaimer = (result.spec_echo && result.spec_echo.parsed_disclaimer) || '';
  const skippedMatch = disclaimer.match(
    /CUSTOM CONSTRAINT REQUESTED BUT SKIPPED[^\n]*/
  );

  if (customs.length === 0 && !skippedMatch) {
    host.classList.add('hidden');
    return;
  }
  host.classList.remove('hidden');

  customs.forEach((c) => host.appendChild(buildCustomConstraintCard(c)));
  if (skippedMatch) {
    host.appendChild(buildCustomConstraintSkipped(skippedMatch[0], disclaimer));
  }
}

function buildCustomConstraintCard(c) {
  const box = document.createElement('div');
  box.className = 'cc-notice';

  const header = document.createElement('div');
  header.className = 'cc-header';
  header.innerHTML = '<span class="cc-warn">⚠</span>'
    + '<span>LLM-authored constraint code ran on this request</span>';
  box.appendChild(header);

  const grid = document.createElement('div');
  grid.className = 'cc-row';

  const addRow = (label, value, valueClass) => {
    const l = document.createElement('div');
    l.className = 'cc-label'; l.textContent = label;
    const v = document.createElement('div');
    v.className = 'cc-value' + (valueClass ? ' ' + valueClass : '');
    v.textContent = value;
    grid.appendChild(l); grid.appendChild(v);
  };

  addRow('request',    c.description || '—', 'cc-desc');
  addRow('class name', c.class_name || c.params?.class_name || '—');
  addRow('kind',       c.kind || 'custom');
  addRow('source',     `${(c.source_code || '').length} chars (sandboxed exec)`);
  box.appendChild(grid);

  const det = document.createElement('details');
  det.className = 'cc-code';
  const sum = document.createElement('summary');
  sum.textContent = 'show generated Python';
  const pre = document.createElement('pre');
  pre.textContent = c.source_code || '(no source captured)';
  det.appendChild(sum); det.appendChild(pre);
  box.appendChild(det);

  return box;
}

function buildCustomConstraintSkipped(messageLine, fullDisclaimer) {
  const box = document.createElement('div');
  box.className = 'cc-notice cc-skipped';

  const header = document.createElement('div');
  header.className = 'cc-header';
  header.innerHTML = '<span class="cc-warn">⚠</span>'
    + '<span>LLM tried to author custom code, but it was rejected</span>';
  box.appendChild(header);

  const grid = document.createElement('div');
  grid.className = 'cc-row';
  const l = document.createElement('div');
  l.className = 'cc-label'; l.textContent = 'reason';
  const v = document.createElement('div');
  v.className = 'cc-value'; v.textContent = messageLine;
  grid.appendChild(l); grid.appendChild(v);
  box.appendChild(grid);

  const tail = document.createElement('div');
  tail.className = 'cc-row';
  const tl = document.createElement('div');
  tl.className = 'cc-label'; tl.textContent = 'fallback';
  const tv = document.createElement('div');
  tv.className = 'cc-value cc-desc';
  tv.textContent = 'Solve continued with the 8 standard constraint kinds the first LLM call produced.';
  tail.appendChild(tl); tail.appendChild(tv);
  box.appendChild(tail);

  // parse.py appends the generated source to the disclaimer when codegen
  // fails. Pull it out and render it in a disclosure so the user can see
  // exactly what the LLM produced (and we can debug sandbox issues).
  if (fullDisclaimer) {
    const srcMatch = fullDisclaimer.match(
      /Generated source \(first \d+ chars\):\s*\n([\s\S]+?)(?:\n\n|$)/
    );
    if (srcMatch && srcMatch[1]) {
      const det = document.createElement('details');
      det.className = 'cc-code';
      const sum = document.createElement('summary');
      sum.textContent = 'show generated Python (rejected)';
      const pre = document.createElement('pre');
      pre.textContent = srcMatch[1].trim();
      det.appendChild(sum); det.appendChild(pre);
      box.appendChild(det);
    }
  }

  return box;
}

function renderCandidateView(idx) {
  if (!_currentResult) return;
  const all = allCandidates(_currentResult);
  if (idx < 0 || idx >= all.length) return;
  _activeIdx = idx;
  const c = all[idx];
  const tgt = _currentResult.spec_echo.target_lab_raw;

  $('#targetSwatch').style.backgroundColor   = labToSrgb(tgt[0], tgt[1], tgt[2]);
  $('#achievedSwatch').style.backgroundColor = labToSrgb(c.achieved_lab[0], c.achieved_lab[1], c.achieved_lab[2]);
  $('#targetLab').textContent   = `L* ${fmt(tgt[0],1)}  a* ${fmt(tgt[1],1)}  b* ${fmt(tgt[2],1)}`;
  $('#achievedLab').textContent = `L* ${fmt(c.achieved_lab[0],1)}  a* ${fmt(c.achieved_lab[1],1)}  b* ${fmt(c.achieved_lab[2],1)}`;
  $('#deltaE').textContent = fmt(c.delta_e, 3);
  $('#deltaELabel').textContent = deltaELabel(c.delta_e)
    + (idx === 0 ? '' : ` · alt #${idx}`);

  // The other candidates become the faint background lines in the chart.
  const others = all.filter((_, i) => i !== idx);
  drawReflectance(c.reflectance, others);
  drawLayerBars(c.material_names, c.thicknesses_nm);

  $('#statR').textContent       = fmt(c.robustness.grad_l2_shift, 3);
  $('#statJ').textContent       = fmt(c.objective, 3);
  $('#statMc').textContent      = c.robustness.mc_samples > 0 ? fmt(c.robustness.mc_p95, 2) : '—';
  $('#statRefined').textContent = c.refined ? 'yes' : 'no';

  // Highlight the active card (or clear all if showing the chosen).
  $$('.alt-card').forEach((card) => {
    const cardIdx = parseInt(card.dataset.idx, 10);
    card.classList.toggle('active', cardIdx === idx);
  });
}

function buildAltGrid() {
  const grid = $('#altGrid'); grid.innerHTML = '';
  const all = allCandidates(_currentResult);
  if (all.length <= 1) {
    grid.innerHTML = '<div class="text-xs text-slate-500 col-span-full">No alternatives — top_k = 1.</div>';
    return;
  }
  // Build a card per candidate: chosen first, then alternatives. Clicking
  // any card swaps the main view to that candidate.
  all.slice(0, 9).forEach((cand, idx) => {
    const card = document.createElement('div');
    card.className = 'alt-card';
    card.dataset.idx = String(idx);
    if (idx === 0) {
      const pill = document.createElement('span'); pill.className = 'alt-pill';
      pill.textContent = 'chosen'; card.appendChild(pill);
    }
    const sw = document.createElement('div'); sw.className = 'alt-swatch';
    sw.style.backgroundColor = labToSrgb(cand.achieved_lab[0], cand.achieved_lab[1], cand.achieved_lab[2]);
    const meta = document.createElement('div'); meta.className = 'alt-meta';
    meta.innerHTML = `<span class="alt-de">ΔE ${fmt(cand.delta_e, 2)}</span> · ${cand.slot_indices.length}L`;
    card.appendChild(sw); card.appendChild(meta);
    card.addEventListener('click', () => renderCandidateView(idx));
    grid.appendChild(card);
  });
}

function drawReflectance(refl, alts) {
  const svg = $('#reflChart');
  const W=600, H=200, padL=32, padR=8, padT=12, padB=20;
  const innerW = W - padL - padR, innerH = H - padT - padB;
  const N = refl.length, lamMin = 300, lamMax = 900;
  const altsArr = alts.flatMap((a) => a.reflectance || []);
  const yMax = Math.max(1.0, Math.max(...refl) * 1.1, ...altsArr) || 1.0;
  const yScale = (v) => padT + innerH - (v / yMax) * innerH;
  const xScale = (i) => padL + (i / (N - 1)) * innerW;
  const pathFor = (arr) => { let d=''; for (let i=0;i<arr.length;i++) d += (i===0?'M':'L') + xScale(i).toFixed(1) + ' ' + yScale(arr[i]).toFixed(1) + ' '; return d.trim(); };

  let s = '';
  s += `<defs><linearGradient id="mainGradient" x1="0" x2="1" y1="0" y2="0"><stop offset="0%"  stop-color="#7c5cff"/><stop offset="100%" stop-color="#f0abfc"/></linearGradient></defs>`;
  const visX0 = padL + ((380 - lamMin)/(lamMax - lamMin)) * innerW;
  const visX1 = padL + ((780 - lamMin)/(lamMax - lamMin)) * innerW;
  s += `<rect class="vis-band" x="${visX0}" y="${padT}" width="${visX1 - visX0}" height="${innerH}"/>`;
  [0,0.25,0.5,0.75,1.0].forEach((v) => {
    if (v > yMax) return;
    const y = yScale(v);
    s += `<line class="grid-line" x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}"/>`;
    s += `<text class="axis-label" x="${padL - 4}" y="${y + 3}" text-anchor="end">${v}</text>`;
  });
  [400,500,600,700,800].forEach((nm) => {
    const x = padL + ((nm - lamMin)/(lamMax - lamMin)) * innerW;
    s += `<text class="axis-label" x="${x}" y="${H - 6}" text-anchor="middle">${nm}</text>`;
  });
  alts.slice(0, 4).forEach((a) => {
    if (Array.isArray(a.reflectance) && a.reflectance.length === N)
      s += `<path class="alt-line" d="${pathFor(a.reflectance)}"/>`;
  });
  s += `<path class="main-line" d="${pathFor(refl)}"/>`;
  svg.innerHTML = s;
}

function drawLayerBars(names, thicks) {
  const total = thicks.reduce((a,b) => a + b, 0) || 1;
  const cm = new Map();
  const assign = (n) => { if (!cm.has(n)) cm.set(n, PALETTE[cm.size % PALETTE.length]); return cm.get(n); };
  const bars = $('#layerBars'); bars.innerHTML = '';
  for (let i=0;i<names.length;i++) {
    const w = (thicks[i]/total) * 100;
    const bar = document.createElement('div'); bar.className = 'layer-bar';
    bar.style.flex = `0 0 ${w}%`; bar.style.backgroundColor = assign(names[i]);
    bar.title = `${names[i]}  ·  ${thicks[i].toFixed(1)} nm`;
    if (w > 8) bar.innerHTML = `<span class="lbl">${names[i].split('-')[0]}<br>${thicks[i].toFixed(0)} nm</span>`;
    bars.appendChild(bar);
  }
  const legend = $('#layerLegend'); legend.innerHTML = '';
  [...cm.entries()].forEach(([name, color]) => {
    const row = document.createElement('div'); row.className = 'flex items-center gap-2';
    row.innerHTML = `<span class="inline-block w-3 h-3 rounded-sm" style="background:${color}"></span><span class="truncate">${name}</span>`;
    legend.appendChild(row);
  });
}

// ============================================================================
// Misc
// ============================================================================
function flashError(msg) {
  const err = $('#errorBanner');
  err.textContent = msg; err.classList.remove('hidden');
  setTimeout(() => err.classList.add('hidden'), 4000);
}

function bindOpenAIKey() {
  const saved = LS.get('openai_api_key', '');
  if (saved) $('#openaiKey').value = saved;
  $('#openaiKey').addEventListener('change', (e) => {
    const v = e.target.value.trim();
    if (v) LS.set('openai_api_key', v); else localStorage.removeItem('openai_api_key');
  });
}

function bindAllowCustomConstraints() {
  // Default OFF — only the user can opt in to LLM-authored sandboxed code.
  const cb = $('#knobAllowCustom');
  if (!cb) return;
  cb.checked = !!LS.get('allow_custom_constraints', false);
  cb.addEventListener('change', () => {
    LS.set('allow_custom_constraints', !!cb.checked);
  });
}

document.addEventListener('DOMContentLoaded', () => {
  bindTabs();
  bindColorInputs();
  bindConstraintChips();
  bindOpenAIKey();
  bindAllowCustomConstraints();
  bindPoolControls();
  bindCsvModal();
  bindPresets();
  bindHowItWorks();
  $('#runBtn').addEventListener('click', runSolve);
  loadStatus();
  loadPool();
});
