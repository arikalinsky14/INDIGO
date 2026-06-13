// INDIGO frontend — vanilla JS, no framework
// Layout:
//   - tab switching (prompt vs structured)
//   - Lab live preview
//   - submit → POST /api/solve → render result
//   - SVG reflectance chart, layer bars, alternatives grid

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ---------- status ----------------------------------------------------------
async function loadStatus() {
  try {
    const r = await fetch('/api/status');
    if (!r.ok) return;
    const s = await r.json();
    if (s.device)            $('#statusDevice').querySelector('span').textContent = s.device;
    if (s.pool_size != null) $('#statusPool').querySelector('span').textContent   = `${s.pool_size} materials`;
    if (s.model_tag)         $('#statusModel').querySelector('span').textContent  = s.model_tag.slice(0, 36) + (s.model_tag.length > 36 ? '…' : '');
    if (!s.openai_key_present) {
      $('#promptHint').textContent = 'No OPENAI_API_KEY — set INDIGO_PARSE_BACKEND=mock to test with keyword routing.';
      $('#promptHint').classList.remove('text-slate-500');
      $('#promptHint').classList.add('text-amber-300/80');
    }
  } catch (e) { console.error('status', e); }
}

// ---------- tabs ------------------------------------------------------------
function bindTabs() {
  $$('.tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      $$('.tab').forEach((t) => t.classList.remove('tab-active'));
      tab.classList.add('tab-active');
      const target = tab.dataset.tab;
      $$('[data-tab-pane]').forEach((pane) => {
        pane.classList.toggle('hidden', pane.dataset.tabPane !== target);
      });
    });
  });
}

// ---------- Lab live preview ------------------------------------------------
function labToSrgb(L, a, b) {
  // CIE Lab D65 → linear sRGB → sRGB-gamma. Clamped to [0,1].
  const fy = (L + 16) / 116;
  const fx = a / 500 + fy;
  const fz = fy - b / 200;
  const eps3 = (6 / 29) ** 3;
  const f = (t) => (t ** 3 > eps3 ? t ** 3 : (t - 4 / 29) * 3 * (6 / 29) ** 2);
  const X = 95.047 * f(fx) / 100;
  const Y = 100.0  * f(fy) / 100;
  const Z = 108.883* f(fz) / 100;
  const rl =  3.2404542 * X + -1.5371385 * Y + -0.4985314 * Z;
  const gl = -0.9692660 * X +  1.8760108 * Y +  0.0415560 * Z;
  const bl =  0.0556434 * X + -0.2040259 * Y +  1.0572252 * Z;
  const enc = (c) => {
    c = Math.max(0, Math.min(1, c));
    return c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
  };
  const r = enc(rl), g = enc(gl), bb = enc(bl);
  return `rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(bb * 255)})`;
}

function bindLabPreview() {
  const upd = () => {
    const L = parseFloat($('#labL').value);
    const a = parseFloat($('#labA').value);
    const b = parseFloat($('#labB').value);
    if (Number.isFinite(L) && Number.isFinite(a) && Number.isFinite(b)) {
      $('#labPreview').style.backgroundColor = labToSrgb(L, a, b);
    }
  };
  ['labL', 'labA', 'labB'].forEach((id) => $(`#${id}`).addEventListener('input', upd));
  upd();
}

// ---------- submit ----------------------------------------------------------
function readKnobs() {
  return {
    ensemble_N: parseInt($('#knobN').value, 10),
    top_k:      parseInt($('#knobK').value, 10),
    temperature:    parseFloat($('#knobT').value),
    tolerance_pct:  parseFloat($('#knobTol').value),
    weight_lambda:  parseFloat($('#knobLambda').value),
    seed:           parseInt($('#knobSeed').value, 10),
  };
}

function buildBody() {
  const activeTab = $('.tab-active').dataset.tab;
  if (activeTab === 'prompt') {
    const prompt = $('#promptInput').value.trim();
    if (!prompt) throw new Error('prompt is empty');
    return { prompt, knobs: readKnobs() };
  }
  const L = parseFloat($('#labL').value);
  const a = parseFloat($('#labA').value);
  const b = parseFloat($('#labB').value);
  if (![L, a, b].every(Number.isFinite))
    throw new Error('Lab values must all be numbers');
  let constraints = null;
  const ct = $('#constraintsInput').value.trim();
  if (ct) {
    try { constraints = JSON.parse(ct); }
    catch (e) { throw new Error('constraints JSON is not valid: ' + e.message); }
    if (!Array.isArray(constraints)) throw new Error('constraints must be an array');
  }
  return { target_lab: [L, a, b], constraints, knobs: readKnobs() };
}

async function runSolve() {
  const btn = $('#runBtn');
  const spinner = btn.querySelector('.spinner');
  const lbl = btn.querySelector('.btn-label');
  const err = $('#errorBanner');
  err.classList.add('hidden');

  let body;
  try { body = buildBody(); }
  catch (e) { err.textContent = e.message; err.classList.remove('hidden'); return; }

  btn.disabled = true;
  spinner.classList.remove('hidden');
  lbl.textContent = 'Generating…';

  try {
    const r = await fetch('/api/solve', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const txt = await r.text();
      throw new Error(`HTTP ${r.status}: ${txt}`);
    }
    const result = await r.json();
    renderResult(result);
  } catch (e) {
    console.error(e);
    err.textContent = e.message;
    err.classList.remove('hidden');
  } finally {
    btn.disabled = false;
    spinner.classList.add('hidden');
    lbl.textContent = 'Generate';
  }
}

// ---------- result rendering ------------------------------------------------
function deltaELabel(de) {
  if (de < 1)  return 'imperceptible';
  if (de < 2)  return 'just perceptible';
  if (de < 5)  return 'noticeable';
  if (de < 10) return 'clearly different';
  return 'far off';
}

function fmt(x, d = 2) { return Number(x).toFixed(d); }

const PALETTE = ['#3b82f6', '#f59e0b', '#10b981', '#ef4444', '#8b5cf6', '#06b6d4', '#ec4899', '#84cc16', '#a855f7', '#f97316'];

function renderResult(result) {
  $('#placeholder').classList.add('hidden');
  const body = $('#resultBody');
  body.classList.remove('hidden');
  body.classList.remove('show'); void body.offsetWidth; body.classList.add('show');

  if (!result.chosen) {
    $('#deltaE').textContent = 'fail';
    $('#deltaELabel').textContent = (result.errors || []).join(' | ') || 'no candidate';
    $('#targetSwatch').style.backgroundColor = '#1a1c2a';
    $('#achievedSwatch').style.backgroundColor = '#1a1c2a';
    return;
  }

  const c = result.chosen;
  const tgt = result.spec_echo.target_lab_raw;

  // Swatches + labels
  $('#targetSwatch').style.backgroundColor   = labToSrgb(tgt[0], tgt[1], tgt[2]);
  $('#achievedSwatch').style.backgroundColor = labToSrgb(c.achieved_lab[0], c.achieved_lab[1], c.achieved_lab[2]);
  $('#targetLab').textContent   = `L* ${fmt(tgt[0], 1)}  a* ${fmt(tgt[1], 1)}  b* ${fmt(tgt[2], 1)}`;
  $('#achievedLab').textContent = `L* ${fmt(c.achieved_lab[0], 1)}  a* ${fmt(c.achieved_lab[1], 1)}  b* ${fmt(c.achieved_lab[2], 1)}`;
  $('#deltaE').textContent = fmt(c.delta_e, 3);
  $('#deltaELabel').textContent = deltaELabel(c.delta_e);

  // Reflectance chart
  drawReflectance(c.reflectance, result.alternatives || []);

  // Layer bars
  drawLayerBars(c.material_names, c.thicknesses_nm);

  // Stats
  $('#statR').textContent       = fmt(c.robustness.grad_l2_shift, 3);
  $('#statJ').textContent       = fmt(c.objective, 3);
  $('#statMc').textContent      = c.robustness.mc_samples > 0 ? fmt(c.robustness.mc_p95, 2) : '—';
  $('#statRefined').textContent = c.refined ? 'yes' : 'no';

  // Alternatives
  const grid = $('#altGrid');
  grid.innerHTML = '';
  (result.alternatives || []).slice(0, 8).forEach((alt) => {
    const card = document.createElement('div'); card.className = 'alt-card';
    const sw = document.createElement('div');   sw.className = 'alt-swatch';
    sw.style.backgroundColor = labToSrgb(alt.achieved_lab[0], alt.achieved_lab[1], alt.achieved_lab[2]);
    const meta = document.createElement('div'); meta.className = 'alt-meta';
    meta.innerHTML = `<span class="alt-de">ΔE ${fmt(alt.delta_e, 2)}</span> · ${alt.slot_indices.length}L`;
    card.appendChild(sw); card.appendChild(meta);
    grid.appendChild(card);
  });
  if (!result.alternatives || result.alternatives.length === 0) {
    grid.innerHTML = '<div class="text-xs text-slate-500 col-span-full">No alternatives — top_k = 1.</div>';
  }

  // Provenance
  $('#provJSON').textContent = JSON.stringify({
    spec_echo: result.spec_echo,
    ensemble_stats: result.ensemble_stats,
    provenance: result.provenance,
  }, null, 2);
}

function drawReflectance(refl, alts) {
  const svg = $('#reflChart');
  const W = 600, H = 200;
  const padL = 32, padR = 8, padT = 12, padB = 20;
  const innerW = W - padL - padR, innerH = H - padT - padB;

  // canonical wavelength grid: 300-900 nm, 128 samples (matches CANONICAL_LAMBDA_NM)
  const N = refl.length;
  const lamMin = 300, lamMax = 900;
  const yMax = Math.max(1.0, Math.max(...refl) * 1.1, ...alts.flatMap((a) => a.reflectance || []).map(Math.abs).concat([0])) || 1.0;
  const yScale = (v) => padT + innerH - (v / yMax) * innerH;
  const xScale = (i) => padL + (i / (N - 1)) * innerW;
  const pathFor = (arr) => {
    let d = '';
    for (let i = 0; i < arr.length; i++) {
      d += (i === 0 ? 'M' : 'L') + xScale(i).toFixed(1) + ' ' + yScale(arr[i]).toFixed(1) + ' ';
    }
    return d.trim();
  };

  // Build SVG
  let svgInner = '';
  svgInner += `<defs><linearGradient id="mainGradient" x1="0" x2="1" y1="0" y2="0">
    <stop offset="0%"  stop-color="#7c5cff"/>
    <stop offset="100%" stop-color="#f0abfc"/>
  </linearGradient></defs>`;

  // Visible-band shading (380-780 nm).
  const visX0 = padL + ((380 - lamMin) / (lamMax - lamMin)) * innerW;
  const visX1 = padL + ((780 - lamMin) / (lamMax - lamMin)) * innerW;
  svgInner += `<rect class="vis-band" x="${visX0}" y="${padT}" width="${visX1 - visX0}" height="${innerH}"/>`;

  // Y-grid lines.
  [0, 0.25, 0.5, 0.75, 1.0].forEach((v) => {
    if (v > yMax) return;
    const y = yScale(v);
    svgInner += `<line class="grid-line" x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}"/>`;
    svgInner += `<text class="axis-label" x="${padL - 4}" y="${y + 3}" text-anchor="end">${v}</text>`;
  });
  // X axis labels.
  [400, 500, 600, 700, 800].forEach((nm) => {
    const x = padL + ((nm - lamMin) / (lamMax - lamMin)) * innerW;
    svgInner += `<text class="axis-label" x="${x}" y="${H - 6}" text-anchor="middle">${nm}</text>`;
  });

  // Alternatives faint.
  alts.slice(0, 4).forEach((a) => {
    if (Array.isArray(a.reflectance) && a.reflectance.length === N) {
      svgInner += `<path class="alt-line" d="${pathFor(a.reflectance)}"/>`;
    }
  });

  // Main line.
  svgInner += `<path class="main-line" d="${pathFor(refl)}"/>`;
  svg.innerHTML = svgInner;
}

function drawLayerBars(names, thicks) {
  const total = thicks.reduce((a, b) => a + b, 0) || 1;
  const colorMap = new Map();
  const assign = (name) => {
    if (!colorMap.has(name)) colorMap.set(name, PALETTE[colorMap.size % PALETTE.length]);
    return colorMap.get(name);
  };
  const bars = $('#layerBars'); bars.innerHTML = '';
  for (let i = 0; i < names.length; i++) {
    const w = (thicks[i] / total) * 100;
    const bar = document.createElement('div');
    bar.className = 'layer-bar';
    bar.style.flex = `0 0 ${w}%`;
    bar.style.backgroundColor = assign(names[i]);
    bar.title = `${names[i]}  ·  ${thicks[i].toFixed(1)} nm`;
    if (w > 8) {
      bar.innerHTML = `<span class="lbl">${names[i].split('-')[0]}<br>${thicks[i].toFixed(0)} nm</span>`;
    }
    bars.appendChild(bar);
  }

  // legend (unique materials)
  const legend = $('#layerLegend'); legend.innerHTML = '';
  [...colorMap.entries()].forEach(([name, color]) => {
    const row = document.createElement('div'); row.className = 'flex items-center gap-2';
    row.innerHTML = `<span class="inline-block w-3 h-3 rounded-sm" style="background:${color}"></span><span class="truncate">${name}</span>`;
    legend.appendChild(row);
  });
}

// ---------- init ------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  bindTabs();
  bindLabPreview();
  $('#runBtn').addEventListener('click', runSolve);
  loadStatus();
});
