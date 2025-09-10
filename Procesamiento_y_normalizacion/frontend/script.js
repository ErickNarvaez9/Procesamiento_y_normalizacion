// ==== Detectar base de API ====
// Si la página NO está en 5000 (por ej. Live Server 5500), usa 5000 explícito.
const API_BASE = (location.port === "5000") ? "" : "http://127.0.0.1:5000";

// DOM
const elSel  = document.getElementById("selLetra");
const elPca  = document.getElementById("pcaCanvas");
const ctx    = elPca.getContext("2d");
const elJSON = document.getElementById("coordsJson");
const elBox  = document.getElementById("analysisBox");

// ---------- utils ----------
function setVal(sel, v){
  const e = document.querySelector(sel);
  if(!e) return;
  e.textContent = (v == null || Number.isNaN(v))
    ? "—"
    : (typeof v === "number" ? v.toFixed(3) : v);
}

// Sanitiza y asegura puntos visibles (corrige letras como e, k, l)
function sanitizePoints(arr){
  if(!Array.isArray(arr)) return [];
  const clamp01 = (n)=> Number.isFinite(n) ? Math.max(0, Math.min(1, n)) : null;
  const out = [];
  for(const p of arr){
    const x = clamp01(p?.x);
    const y = clamp01(p?.y);
    if(x === null || y === null) continue;
    out.push({x, y});
  }
  return out;
}

function drawPoints(points, radius = 6){
  const pts = sanitizePoints(points);        // <-- corrección necesaria
  const W = elPca.width, H = elPca.height;
  ctx.clearRect(0,0,W,H);

  // grid suave
  ctx.strokeStyle = "rgba(0,0,0,.07)";
  for(let i=1;i<5;i++){
    const x = (W/5)*i, y = (H/5)*i;
    ctx.beginPath(); ctx.moveTo(x, 10); ctx.lineTo(x, H-10); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(10, y); ctx.lineTo(W-10, y); ctx.stroke();
  }

  if(!pts.length) return; // si no hay datos válidos, deja solo la grilla

  ctx.fillStyle = "#2B6FF9";
  ctx.strokeStyle = "rgba(5,28,63,.25)";
  for(const p of pts){
    const x = p.x * (W - 20) + 10;
    const y = (1 - p.y) * (H - 20) + 10;
    ctx.beginPath(); ctx.arc(x, y, radius, 0, Math.PI*2); ctx.fill(); ctx.stroke();
  }
}

async function jget(path){
  const r = await fetch(API_BASE + path);
  if(!r.ok){
    const t = await r.text().catch(()=>r.statusText);
    throw new Error(`${r.status} ${r.statusText} -> ${path}\n${t}`);
  }
  return r.json();
}

// ---------- cargas ----------
async function loadLetters(){
  const { letras } = await jget("/letras");
  const uniq = [...new Set(letras)].sort();
  elSel.innerHTML = uniq.map(l => `<option value="${l}">${String(l).toUpperCase()}</option>`).join("");
}


async function loadPCA(letter){
  const { points } = await jget(`/pca?letter=${encodeURIComponent(letter)}`);
  drawPoints(points, 6);
}

async function loadMetrics(letter){
  const m = await jget(`/metrics?letter=${encodeURIComponent(letter)}`);
  setVal("#m_acc",    m.accuracy);
  setVal("#m_prec",   m.precision);
  setVal("#m_rec",    m.recall);
  setVal("#m_f1",     m.f1);
  setVal("#m_sil",    (m.silhouette?.letter ?? m.silhouette?.global));
  setVal("#m_fisher", m.fisher);
  setVal("#m_p50",    m.p50);
  setVal("#m_p95",    m.p95);
}

async function loadCoords(letter){
  const { points, analysis } = await jget(`/coords?letter=${encodeURIComponent(letter)}`);

  // guardamos las coords ya saneadas (para evitar NaN/valores fuera de [0,1])
  const ptsOK = sanitizePoints(points);      // <-- corrección necesaria
  elJSON.value = JSON.stringify(ptsOK, null, 2);

  if(!analysis || !analysis.centroid){
    elBox.innerHTML = `<div class="card-mini"><span class="muted">Sin datos para la letra seleccionada.</span></div>`;
    return;
  }
  const { centroid, radial, svd, points:totalPts, inside01 } = analysis;
  elBox.innerHTML = `
    <div class="card-mini">
      <div><b>Centroide:</b> (${centroid[0].toFixed(3)}, ${centroid[1].toFixed(3)})</div>
      <div><b>Radial</b> μ=${radial.mean.toFixed(3)}, σ=${radial.std.toFixed(3)}, min=${radial.min.toFixed(3)}, max=${radial.max.toFixed(3)}</div>
      <div><b>SVD forma</b> s0=${svd.s0.toFixed(3)}, s1=${svd.s1.toFixed(3)}, anisotropía=${svd.anisotropy.toFixed(3)}</div>
      <div><b>Puntos:</b> ${totalPts}</div>
      <div><b>Validación:</b> <span class="badge">${inside01 ? "Dentro de [0,1]" : "Fuera de rango"}</span></div>
    </div>`;
}

// ---------- init ----------
async function init(){
  try{
    await loadLetters();           // llena el select
    const letter = elSel.value;    // primera letra
    await Promise.all([
      loadPCA(letter),
      loadMetrics(letter),
      loadCoords(letter)
    ]);
  }catch(e){
    console.error(e);
    alert("Error inicializando la app. Revisa la consola.");
  }
}

elSel.addEventListener("change", async e=>{
  const letter = e.target.value;
  try{
    await Promise.all([
      loadPCA(letter),
      loadMetrics(letter),
      loadCoords(letter)
    ]);
  }catch(err){
    console.error(err);
    alert("Error al actualizar. Revisa la consola.");
  }
});

window.addEventListener("DOMContentLoaded", init);
