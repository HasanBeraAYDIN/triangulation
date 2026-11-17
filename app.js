/* İki fotoğraf mesafe ölçer — telefonda canlı kamera.
 *
 * Akış (robottaki karşılığı parantez içinde):
 *   1. Her karede ChArUco tahtasından kameranın konumu bulunur   (kol kontrolcüsü)
 *   2. "Uzak fotoğraf": kare, konum ve birleşim noktası saklanır  (segmentasyon noktası)
 *   3. Sonraki her kare "yakın fotoğraf" adayıdır: nokta ORB eşleştirmeyle bulunur,
 *      iki görüşle üçgenlenir, yakın kameradan mesafe hesaplanır.
 *   Gerçek mesafe, noktanın tahtadaki bilinen yerinden gelir; tahmin bunu kullanmaz.
 */
"use strict";

const OCV_URL = "https://cdn.jsdelivr.net/npm/@techstark/opencv-js@5.0.0-release.1/dist/opencv.js";
const SHEET = { nx: 7, ny: 5, sqDesign: 22.0, mkDesign: 15.4, junction: [-2.992561983471074, 2.506611570247934],
                originMm: [136.0, 50.0] };
const CAL_VIEWS = 15;
const SIGMA_PX = 1.5;
const MIN_PARALLAX = 2.0;      // derece; altında tahmin gösterilmez
const REF_RADIUS = 130;        // uzak fotoğrafta noktanın çevresi (px, 1080p)
const SEARCH_RADIUS = 520;     // yakın fotoğrafta arama yarıçapı (px)
const params = new URLSearchParams(location.search);
const SIM = params.has("sim");

let cv = null;
const $ = (id) => document.getElementById(id);
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
  del(k) { try { localStorage.removeItem(k); } catch {} },
};

/* ---------- küçük lineer cebir ---------- */
const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add3 = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul3 = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm3 = (a) => Math.hypot(a[0], a[1], a[2]);
const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const unit3 = (a) => mul3(a, 1 / norm3(a));
const matVec = (R, v) => [dot3(R[0], v), dot3(R[1], v), dot3(R[2], v)];
const matTVec = (R, v) => [R[0][0] * v[0] + R[1][0] * v[1] + R[2][0] * v[2],
                           R[0][1] * v[0] + R[1][1] * v[1] + R[2][1] * v[2],
                           R[0][2] * v[0] + R[1][2] * v[1] + R[2][2] * v[2]];
function solve3(A, b) {           // Cramer
  const d = det3(A);
  if (Math.abs(d) < 1e-300) return null;
  const r = [];
  for (let i = 0; i < 3; i++) { const M = A.map((row, k) => row.map((x, j) => (j === i ? b[k] : x))); r.push(det3(M) / d); }
  return r;
}
function det3(A) {
  return A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
         A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
}
function inv3(A) {
  const d = det3(A), c = (i, j) => {
    const m = A.filter((_, r) => r !== i).map((row) => row.filter((_, k) => k !== j));
    return ((i + j) % 2 ? -1 : 1) * (m[0][0] * m[1][1] - m[0][1] * m[1][0]);
  };
  return [0, 1, 2].map((i) => [0, 1, 2].map((j) => c(j, i) / d));
}
function rodrigues(r) {
  const th = Math.hypot(r[0], r[1], r[2]);
  if (th < 1e-12) return [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const [x, y, z] = r.map((v) => v / th), c = Math.cos(th), s = Math.sin(th), C = 1 - c;
  return [[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
          [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
          [z * x * C - y * s, z * y * C + x * s, c + z * z * C]];
}

/* ---------- kamera modeli ---------- */
function undistort(u, v, cal) {          // piksel -> normalize (OpenCV ile aynı yineleme)
  const [fx, fy, cx, cy] = [cal.K[0], cal.K[4], cal.K[2], cal.K[5]];
  const [k1 = 0, k2 = 0, p1 = 0, p2 = 0, k3 = 0] = cal.dist;
  const x0 = (u - cx) / fx, y0 = (v - cy) / fy;
  let x = x0, y = y0;
  for (let i = 0; i < 20; i++) {
    const r2 = x * x + y * y, ic = 1 / (1 + ((k3 * r2 + k2) * r2 + k1) * r2);
    const dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x), dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y;
    x = (x0 - dx) * ic; y = (y0 - dy) * ic;
  }
  return [x, y];
}
function project(Xc, cal) {               // kamera koordinatı -> piksel
  const [fx, fy, cx, cy] = [cal.K[0], cal.K[4], cal.K[2], cal.K[5]];
  const [k1 = 0, k2 = 0, p1 = 0, p2 = 0, k3 = 0] = cal.dist;
  const x = Xc[0] / Xc[2], y = Xc[1] / Xc[2], r2 = x * x + y * y, rad = 1 + ((k3 * r2 + k2) * r2 + k1) * r2;
  const xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x), yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y;
  return [fx * xd + cx, fy * yd + cy];
}
const camCenter = (P) => mul3(matTVec(P.R, P.t), -1);
const toCam = (P, X) => add3(matVec(P.R, X), P.t);

/* ---------- iki görüşle üçgenleme ---------- */
function triangulate(obs, sigmaN) {       // obs: [{xn:[x,y], R, t}]
  // Doğrusal başlangıç: (x*r3 - r1)·X = t1 - x*t3
  const AtA = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], Atb = [0, 0, 0];
  for (const { xn, R, t } of obs) {
    for (const [c, k] of [[xn[0], 0], [xn[1], 1]]) {
      const a = sub3(mul3(R[2], c), R[k]), b = t[k] - c * t[2];
      for (let i = 0; i < 3; i++) { Atb[i] += a[i] * b; for (let j = 0; j < 3; j++) AtA[i][j] += a[i] * a[j]; }
    }
  }
  let X = solve3(AtA, Atb);
  if (!X) return null;
  let JtJ;
  for (let it = 0; it < 15; it++) {        // Gauss-Newton: yeniden izdüşüm hatası
    JtJ = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    const Jtr = [0, 0, 0];
    for (const { xn, R, t } of obs) {
      const Xc = add3(matVec(R, X), t), z = Xc[2];
      const r = [Xc[0] / z - xn[0], Xc[1] / z - xn[1]];
      const d = [[1 / z, 0, -Xc[0] / (z * z)], [0, 1 / z, -Xc[1] / (z * z)]];
      for (let k = 0; k < 2; k++) {
        const Jrow = [0, 1, 2].map((j) => d[k][0] * R[0][j] + d[k][1] * R[1][j] + d[k][2] * R[2][j]);
        for (let i = 0; i < 3; i++) { Jtr[i] += Jrow[i] * r[k]; for (let j = 0; j < 3; j++) JtJ[i][j] += Jrow[i] * Jrow[j]; }
      }
    }
    const dx = solve3(JtJ, Jtr.map((v) => -v));
    if (!dx) break;
    X = add3(X, dx);
    if (norm3(dx) < 1e-6) break;
  }
  const cov = inv3(JtJ).map((row) => row.map((v) => v * sigmaN * sigmaN));
  return { X, cov };
}
function distanceFrom(X, cov, P) {
  const Xc = toCam(P, X), d = norm3(Xc), g = matTVec(P.R, mul3(Xc, 1 / d));
  const s2 = dot3(g, [dot3(cov[0], g), dot3(cov[1], g), dot3(cov[2], g)]);
  return { d, sigma: Math.sqrt(Math.max(0, s2)) };
}
function parallaxDeg(X, P0, P1) {
  const a = unit3(sub3(X, camCenter(P0))), b = unit3(sub3(X, camCenter(P1)));
  return Math.acos(Math.max(-1, Math.min(1, dot3(a, b)))) * 180 / Math.PI;
}

/* ---------- uygulama durumu ---------- */
const S = {
  W: 0, H: 0, calKey: "", cal: null, sq: SHEET.sqDesign,
  mode: "idle",            // idle | live | calib
  tapPoint: null,          // dokunularak seçilen nokta (görüntü pikseli)
  ref: null,               // uzak fotoğraf: {gray, P, p0, Xtrue, kp, des, dTrue}
  last: null,              // son karenin sonucu
  lastGuess: null,
  calViews: [], calLastT: 0,
  rows: store.get("tri_rows", []),
  frameMs: 0,
};

let video, procCanvas, procCtx, screen, sctx;
let dict, board, detector, orbRef, orbCur, matcher;

/* ---------- başlatma ---------- */
async function main() {
  screen = $("screen"); sctx = screen.getContext("2d");
  procCanvas = document.createElement("canvas"); procCtx = procCanvas.getContext("2d", { willReadFrequently: true });
  S.sq = store.get("tri_sq", SHEET.sqDesign);
  $("iSq").value = S.sq;
  $("iSq").addEventListener("change", () => {
    const v = parseFloat(String($("iSq").value).replace(",", "."));
    if (v > 0) { S.sq = v; store.set("tri_sq", v); resetRef(); }
  });
  wireButtons();
  renderTable();

  try { await loadOpenCV(); } catch (e) { return fatal("OpenCV yüklenemedi. İnternet bağlantısını kontrol edip sayfayı yenile. (" + e.message + ")"); }
  setupDetectors();

  if (SIM) await startSim(); else await startCamera();
  $("overlay").hidden = true;
  requestAnimationFrame(loop);
}

function loadOpenCV() {
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = OCV_URL; s.async = true;
    s.onerror = () => reject(new Error("indirme hatası"));
    s.onload = () => {
      const t0 = Date.now();
      (function check() {
        const c = window.cv;
        if (c && typeof c.then === "function" && !c.Mat) {
          c.then((m) => { try { delete m.then; } catch {} cv = m; resolve(); });
          return;
        }
        if (c && c.Mat && c.aruco_CharucoDetector) { cv = c; resolve(); return; }
        if (Date.now() - t0 > 60000) { reject(new Error("zaman aşımı")); return; }
        setTimeout(check, 100);
      })();
    };
    document.head.appendChild(s);
  });
}

function setupDetectors() {
  dict = cv.getPredefinedDictionary(cv.DICT_4X4_50);
  board = new cv.aruco_CharucoBoard(new cv.Size(SHEET.nx, SHEET.ny), SHEET.sqDesign, SHEET.mkDesign, dict, new cv.Mat());
  try {
    detector = new cv.aruco_CharucoDetector(board, new cv.aruco_CharucoParameters(), new cv.aruco_DetectorParameters(),
                                            new cv.aruco_RefineParameters(10, 3, true));
  } catch { detector = new cv.aruco_CharucoDetector(board); }
  orbRef = new cv.ORB(600);
  orbCur = new cv.ORB(3000);
  matcher = new cv.BFMatcher(cv.NORM_HAMMING, false);
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) return fatal("Bu tarayıcı kameraya erişim vermiyor. Sayfayı Safari'de https bağlantısıyla aç.");
  video = document.createElement("video");
  video.setAttribute("playsinline", ""); video.muted = true;
  const camId = store.get("tri_cam", null);
  await openStream(camId);
  try { await navigator.wakeLock?.request("screen"); } catch {}
  await listCameras();
  $("iCam").addEventListener("change", async () => { store.set("tri_cam", $("iCam").value); await openStream($("iCam").value); });
}
async function openStream(deviceId) {
  if (video.srcObject) video.srcObject.getTracks().forEach((t) => t.stop());
  const v = { width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { ideal: 30 } };
  if (deviceId) v.deviceId = { exact: deviceId }; else v.facingMode = { ideal: "environment" };
  let stream;
  try { stream = await navigator.mediaDevices.getUserMedia({ video: v, audio: false }); }
  catch (e) {
    if (deviceId) { store.del("tri_cam"); return openStream(null); }
    return fatal("Kamera açılamadı: " + e.message + ". Safari ayarlarından bu siteye kamera izni ver.");
  }
  video.srcObject = stream;
  await video.play();
  const track = stream.getVideoTracks()[0];
  S.camLabel = track.label || "kamera";
  try { track.applyConstraints({ advanced: [{ focusMode: "continuous" }] }); } catch {}
  resetRef();
}
async function listCameras() {
  try {
    const devs = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
    const cur = video.srcObject?.getVideoTracks()[0]?.getSettings().deviceId;
    $("iCam").innerHTML = "";
    devs.forEach((d, i) => {
      const o = document.createElement("option");
      o.value = d.deviceId; o.textContent = d.label || "Kamera " + (i + 1);
      if (d.deviceId === cur) o.selected = true;
      $("iCam").appendChild(o);
    });
  } catch {}
}

function fatal(msg) { $("overlay").hidden = false; $("overlay").textContent = msg; $("hint").textContent = msg; }

/* ---------- kalibrasyon kaydı (çözünürlük + kamera başına) ---------- */
function syncCalibration() {
  const key = "tri_cal_" + (SIM ? "sim" : (S.camLabel || "")) + "_" + S.W + "x" + S.H;
  if (key !== S.calKey) {
    S.calKey = key;
    S.cal = SIM ? simCal() : store.get(key, null);
    resetRef();
  }
}

/* ---------- ana döngü ---------- */
async function loop() {
  try { frame(); } catch (e) { console.error(e); $("hint").textContent = "Hata: " + (e.message || e); }
  requestAnimationFrame(loop);
}

function grabGray() {
  if (SIM) return simFrame();
  if (!video || video.readyState < 2 || !video.videoWidth) return null;
  const W = video.videoWidth, H = video.videoHeight;
  if (procCanvas.width !== W || procCanvas.height !== H) { procCanvas.width = W; procCanvas.height = H; }
  procCtx.drawImage(video, 0, 0, W, H);
  const rgba = cv.matFromImageData(procCtx.getImageData(0, 0, W, H));
  const gray = new cv.Mat();
  cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);
  rgba.delete();
  return gray;
}

function frame() {
  const t0 = performance.now();
  const gray = grabGray();
  if (!gray) return;
  if (gray.cols !== S.W || gray.rows !== S.H) { S.W = gray.cols; S.H = gray.rows; S.lastGuess = null; }
  syncCalibration();

  const det = detectBoard(gray);
  let P = null;
  if (det && S.cal && det.n >= 6) P = boardPose(det);

  let res = null;
  if (S.mode === "calib") calibStep(det);
  else if (S.mode === "live" && S.ref && P) res = measure(gray, P);
  S.last = res ? { ...res, P } : (S.mode === "live" ? { P } : null);
  S.curP = P;

  draw(gray, det, P, res);
  updateUI(det, P, res);
  if (S.pendingFar) { S.pendingFar = false; takeFar(gray, P); }
  if (gray !== S.ref?.gray) gray.delete();
  S.frameMs = 0.8 * S.frameMs + 0.2 * (performance.now() - t0);
  $("perf").textContent = `${S.W}×${S.H} · ${Math.round(S.frameMs)} ms/kare`;
  if (SIM) simAdvance(res);
}

/* ---------- tahta ve konum ---------- */
function detectBoard(gray) {
  const corners = new cv.Mat(), ids = new cv.Mat(), mCorners = new cv.MatVector(), mIds = new cv.Mat();
  try {
    detector.detectBoard(gray, corners, ids, mCorners, mIds);
    const n = ids.empty() ? 0 : ids.rows * ids.cols;   // 1xN ya da Nx1 gelebilir
    if (n < 4) return null;
    const pts = [], idArr = [];
    for (let i = 0; i < n; i++) { pts.push([corners.data32F[2 * i], corners.data32F[2 * i + 1]]); idArr.push(ids.data32S[i]); }
    return { n, pts, ids: idArr };
  } finally { corners.delete(); ids.delete(); mCorners.delete(); mIds.delete(); }
}
function cornerObj(id, sq) {           // ChArUco iç köşesi -> tahta koordinatı (mm)
  const c = SHEET.nx - 1;
  return [((id % c) + 1) * sq, (Math.floor(id / c) + 1) * sq, 0];
}
function boardPose(det) {
  const obj = cv.matFromArray(det.n, 1, cv.CV_32FC3, det.ids.flatMap((id) => cornerObj(id, S.sq)));
  const img = cv.matFromArray(det.n, 1, cv.CV_32FC2, det.pts.flat());
  const K = cv.matFromArray(3, 3, cv.CV_64F, S.cal.K), D = cv.matFromArray(1, S.cal.dist.length, cv.CV_64F, S.cal.dist);
  const rv = new cv.Mat(), tv = new cv.Mat();
  let P = null;
  try {
    const ok = cv.solvePnP(obj, img, K, D, rv, tv, false, cv.SOLVEPNP_ITERATIVE);
    if (ok) {
      const R = rodrigues(Array.from(rv.data64F)), t = Array.from(tv.data64F);
      // Yeniden izdüşüm hatası: tahta düzgün bulunmuş mu?
      let e2 = 0;
      det.ids.forEach((id, i) => { const q = project(toCam({ R, t }, cornerObj(id, S.sq)), S.cal); e2 += (q[0] - det.pts[i][0]) ** 2 + (q[1] - det.pts[i][1]) ** 2; });
      const rms = Math.sqrt(e2 / det.n);
      if (t[2] > 0 && rms < 4) P = { R, t, rms };
    }
  } finally { obj.delete(); img.delete(); K.delete(); D.delete(); rv.delete(); tv.delete(); }
  return P;
}

/* ---------- uzak fotoğraf ---------- */
function takeFar(gray, P) {
  if (!P) { $("hint").textContent = "Tahta görünmüyor, uzak fotoğraf alınamadı."; return; }
  let p0, Xtrue;
  const sq = S.sq;
  if (S.tapPoint) {
    // Dokunulan noktanın gerçek yeri: ışının kâğıt düzlemiyle (z=0) kesişimi
    p0 = S.tapPoint;
    const xn = undistort(p0[0], p0[1], S.cal), C = camCenter(P), dir = matTVec(P.R, [xn[0], xn[1], 1]);
    const s = -C[2] / dir[2];
    Xtrue = add3(C, mul3(dir, s));
  } else {
    Xtrue = [SHEET.junction[0] * sq, SHEET.junction[1] * sq, 0];
    p0 = project(toCam(P, Xtrue), S.cal);
  }
  if (p0[0] < 0 || p0[1] < 0 || p0[0] >= S.W || p0[1] >= S.H) { $("hint").textContent = "Birleşim noktası kadrajın dışında. Kâğıdın tamamını kadraja al."; return; }
  const mask = cv.Mat.zeros(S.H, S.W, cv.CV_8UC1);
  cv.circle(mask, new cv.Point(Math.round(p0[0]), Math.round(p0[1])), REF_RADIUS, new cv.Scalar(255), -1);
  const kp = new cv.KeyPointVector(), des = new cv.Mat();
  orbRef.detectAndCompute(gray, mask, kp, des, false);
  mask.delete();
  if (kp.size() < 12) { kp.delete(); des.delete(); $("hint").textContent = "Noktanın çevresinde yeterli doku yok. Biraz yaklaş ya da ışığı artır."; return; }
  const kpts = []; for (let i = 0; i < kp.size(); i++) { const k = kp.get(i); kpts.push([k.pt.x, k.pt.y]); }
  kp.delete();
  clearRef();
  S.ref = { gray: gray.clone(), P, p0, Xtrue, kpts, des, dTrue: norm3(toCam(P, Xtrue)) };
  S.mode = "live"; S.lastGuess = p0;
}
function clearRef() {
  if (S.ref) { S.ref.gray.delete(); S.ref.des.delete(); }
  S.ref = null;
}
function resetRef() { clearRef(); if (S.mode !== "calib") S.mode = "idle"; S.last = null; S.lastGuess = null; }

/* ---------- yakın fotoğraf: nokta eşleştirme + üçgenleme ---------- */
function matchPoint(gray, P) {
  const R = S.ref;
  const mask = cv.Mat.zeros(S.H, S.W, cv.CV_8UC1);
  if (S.lastGuess) cv.circle(mask, new cv.Point(Math.round(S.lastGuess[0]), Math.round(S.lastGuess[1])), SEARCH_RADIUS, new cv.Scalar(255), -1);
  else mask.setTo(new cv.Scalar(255));
  // Tahtanın kendisini aramadan çıkar (kare köşeleri eşleşmeyi bulandırmasın)
  const quad = [[0, 0], [SHEET.nx, 0], [SHEET.nx, SHEET.ny], [0, SHEET.ny]].map(([x, y]) => {
    const pad = 0.15;
    const X = [(x === 0 ? -pad : x + pad) * S.sq, (y === 0 ? -pad : y + pad) * S.sq, 0];
    return project(toCam(P, X), S.cal);
  });
  const qm = cv.matFromArray(4, 1, cv.CV_32SC2, quad.flat().map(Math.round));
  try { cv.fillConvexPoly(mask, qm, new cv.Scalar(0)); } catch {}
  qm.delete();

  const kp = new cv.KeyPointVector(), des = new cv.Mat();
  orbCur.detectAndCompute(gray, mask, kp, des, false);
  mask.delete();
  const out = { n: 0 };
  if (kp.size() < 12 || des.rows === 0) { kp.delete(); des.delete(); return out; }
  const mm = new cv.DMatchVectorVector();
  matcher.knnMatch(R.des, des, mm, 2);
  const a = [], b = [];
  for (let i = 0; i < mm.size(); i++) {
    const m = mm.get(i);
    if (m.size() < 2) continue;
    const m0 = m.get(0), m1 = m.get(1);
    if (m0.distance < 0.8 * m1.distance) { a.push(R.kpts[m0.queryIdx]); const k = kp.get(m0.trainIdx); b.push([k.pt.x, k.pt.y]); }
  }
  mm.delete(); kp.delete(); des.delete();
  out.n = a.length;
  if (a.length < 10) return out;
  const A0 = cv.matFromArray(a.length, 1, cv.CV_32FC2, a.flat()), B0 = cv.matFromArray(b.length, 1, cv.CV_32FC2, b.flat());
  const inl = new cv.Mat();
  const Am = cv.estimateAffine2D(A0, B0, inl, cv.RANSAC, 3.0, 2000, 0.995, 10);
  A0.delete(); B0.delete();
  if (Am.empty()) { Am.delete(); inl.delete(); return out; }
  let A = Array.from(Am.data64F);
  const ia = [], ib = [];
  for (let i = 0; i < a.length; i++) if (inl.data[i]) { ia.push(a[i]); ib.push(b[i]); }
  Am.delete(); inl.delete();
  out.inliers = ia.length;
  if (ia.length < 10) return out;
  // Noktaya yakın eşleşmelere ağırlık veren yerel afin düzeltme
  const sg = REF_RADIUS / 2, p0 = R.p0;
  const M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], bu = [0, 0, 0], bv = [0, 0, 0];
  let wsum = 0;
  ia.forEach((q, i) => {
    const w = Math.exp(-((q[0] - p0[0]) ** 2 + (q[1] - p0[1]) ** 2) / (2 * sg * sg));
    if (w < 0.05) return;
    wsum += w;
    const v = [q[0], q[1], 1];
    for (let r = 0; r < 3; r++) { bu[r] += w * v[r] * ib[i][0]; bv[r] += w * v[r] * ib[i][1]; for (let c = 0; c < 3; c++) M[r][c] += w * v[r] * v[c]; }
  });
  if (wsum > 4) {
    const ru = solve3(M, bu), rv = solve3(M, bv);
    if (ru && rv) A = [...ru, ...rv];
  }
  out.p = [A[0] * p0[0] + A[1] * p0[1] + A[2], A[3] * p0[0] + A[4] * p0[1] + A[5]];
  out.scale = Math.sqrt(Math.abs(A[0] * A[4] - A[1] * A[3]));
  return out;
}

function measure(gray, P) {
  const R = S.ref;
  const m = matchPoint(gray, P);
  const dTrue = norm3(toCam(P, R.Xtrue));
  const res = { matches: m.n, inliers: m.inliers || 0, dTrue };
  // Hareket: uzak fotoğraftan bu yana ileri ve yana
  const C0 = camCenter(R.P), C1 = camCenter(P), ray = unit3(sub3(R.Xtrue, C0)), mv = sub3(C1, C0);
  res.fwd = dot3(mv, ray); res.lat = norm3(sub3(mv, mul3(ray, res.fwd)));
  if (!m.p) { S.lastGuess = null; return res; }
  S.lastGuess = m.p;
  res.p = m.p;
  const sigmaN = SIGMA_PX / S.cal.K[0];
  const tri = triangulate([{ xn: undistort(R.p0[0], R.p0[1], S.cal), R: R.P.R, t: R.P.t },
                           { xn: undistort(m.p[0], m.p[1], S.cal), R: P.R, t: P.t }], sigmaN);
  if (!tri) return res;
  res.parallax = parallaxDeg(tri.X, R.P, P);
  const q = project(toCam(P, R.Xtrue), S.cal);
  res.pxErr = Math.hypot(q[0] - m.p[0], q[1] - m.p[1]);
  if (res.parallax >= MIN_PARALLAX) {
    const { d, sigma } = distanceFrom(tri.X, tri.cov, P);
    res.dEst = d; res.sigma = sigma; res.err = d - dTrue;
  }
  return res;
}

/* ---------- kalibrasyon ---------- */
function calibStep(det) {
  const need = CAL_VIEWS;
  if (!det || det.n < 12) return;
  const now = performance.now();
  if (now - S.calLastT < 700) return;
  // Yeni görüntü öncekilerden yeterince farklı mı? (ortak köşelerin ortalama kayması)
  const diff = S.calViews.length === 0 ? 1 : Math.min(...S.calViews.map((v) => {
    let s = 0, c = 0;
    det.ids.forEach((id, i) => { const j = v.ids.indexOf(id); if (j >= 0) { s += Math.hypot(det.pts[i][0] - v.pts[j][0], det.pts[i][1] - v.pts[j][1]); c++; } });
    return c ? s / c / S.W : 1;
  }));
  if (diff < 0.06) return;
  S.calViews.push({ ids: det.ids.slice(), pts: det.pts.map((p) => p.slice()) });
  S.calLastT = now;
  if (S.calViews.length >= need) runCalibration();
}
function runCalibration() {
  const objs = new cv.MatVector(), imgs = new cv.MatVector();
  for (const v of S.calViews) {
    const o = cv.matFromArray(v.ids.length, 1, cv.CV_32FC3, v.ids.flatMap((id) => cornerObj(id, S.sq)));
    const im = cv.matFromArray(v.ids.length, 1, cv.CV_32FC2, v.pts.flat());
    objs.push_back(o); imgs.push_back(im); o.delete(); im.delete();
  }
  const K = new cv.Mat(), D = new cv.Mat(), rv = new cv.MatVector(), tv = new cv.MatVector();
  const sI = new cv.Mat(), sE = new cv.Mat(), pv = new cv.Mat();
  let rms = NaN;
  try {
    const flags = cv.CALIB_FIX_K3 | cv.CALIB_ZERO_TANGENT_DIST;
    const crit = new cv.TermCriteria(cv.TermCriteria_COUNT + cv.TermCriteria_EPS, 60, 1e-9);
    rms = cv.calibrateCameraExtended(objs, imgs, new cv.Size(S.W, S.H), K, D, rv, tv, sI, sE, pv, flags, crit);
    const cal = { K: Array.from(K.data64F), dist: Array.from(D.data64F).slice(0, 5), rms, size: [S.W, S.H], when: new Date().toISOString(), cam: S.camLabel };
    S.cal = cal; store.set(S.calKey, cal);
    $("hint").textContent = `Kalibrasyon tamam: fx=${cal.K[0].toFixed(1)} px, hata ${rms.toFixed(2)} px.`;
  } catch (e) {
    $("hint").textContent = "Kalibrasyon hesaplanamadı (" + (e.message || e) + "). Tekrar dene.";
  } finally {
    [objs, imgs, K, D, rv, tv, sI, sE, pv].forEach((m) => m.delete());
    S.calViews = []; S.mode = "idle"; resetRef();
    $("bCal").textContent = "Kalibrasyonu başlat";
  }
}

/* ---------- çizim ---------- */
function draw(gray, det, P, res) {
  const vw = screen.clientWidth || 360, dpr = Math.min(window.devicePixelRatio || 1, 2);
  const cw = Math.round(vw * dpr), ch = Math.round(cw * S.H / S.W);
  if (screen.width !== cw || screen.height !== ch) { screen.width = cw; screen.height = ch; }
  if (SIM) {
    const rgba = new cv.Mat(); cv.cvtColor(gray, rgba, cv.COLOR_GRAY2RGBA);
    const id = new ImageData(new Uint8ClampedArray(rgba.data), S.W, S.H); rgba.delete();
    if (procCanvas.width !== S.W) { procCanvas.width = S.W; procCanvas.height = S.H; }
    procCtx.putImageData(id, 0, 0);
  }
  sctx.drawImage(procCanvas, 0, 0, cw, ch);
  const k = cw / S.W;
  const dot = (p, r, col) => { sctx.beginPath(); sctx.arc(p[0] * k, p[1] * k, r, 0, 7); sctx.fillStyle = col; sctx.fill(); };
  const cross = (p, r, col, w = 2.5) => {
    sctx.strokeStyle = col; sctx.lineWidth = w * dpr; sctx.beginPath();
    sctx.moveTo(p[0] * k - r, p[1] * k); sctx.lineTo(p[0] * k + r, p[1] * k);
    sctx.moveTo(p[0] * k, p[1] * k - r); sctx.lineTo(p[0] * k, p[1] * k + r); sctx.stroke();
  };
  if (det) det.pts.forEach((p) => dot(p, 2.5 * dpr, P ? "#3ddc6a" : "#f0b429"));
  if (S.mode === "calib") S.calViews.forEach((v) => v.pts.forEach((p) => dot(p, 1.5 * dpr, "rgba(80,170,255,.7)")));
  if (S.mode === "idle" && P && S.cal) {
    const p = S.tapPoint || project(toCam(P, [SHEET.junction[0] * S.sq, SHEET.junction[1] * S.sq, 0]), S.cal);
    cross(p, 14 * dpr, "#ff3b30");
    sctx.strokeStyle = "rgba(255,59,48,.8)"; sctx.lineWidth = 1.5 * dpr; sctx.beginPath(); sctx.arc(p[0] * k, p[1] * k, REF_RADIUS * k, 0, 7); sctx.stroke();
  }
  if (S.mode === "live" && res?.p) cross(res.p, 16 * dpr, "#ff3b30", 3);
  if (S.tapPoint && S.mode === "idle" && !P) cross(S.tapPoint, 14 * dpr, "#ff3b30");
}

/* ---------- arayüz ---------- */
const fmtCm = (mm, sign = false) => (mm == null || !isFinite(mm)) ? "–" : (sign && mm >= 0 ? "+" : "") + (mm / 10).toFixed(2);
function updateUI(det, P, res) {
  const chips = [];
  chips.push(`<span class="chip ${S.cal ? "ok" : "no"}">Kalibrasyon ${S.cal ? `<b>fx ${S.cal.K[0].toFixed(0)}</b>` : "<b>yok</b>"}</span>`);
  chips.push(`<span class="chip ${det && det.n >= 6 ? "ok" : "no"}">Tahta <b>${det ? det.n : 0}/24</b> köşe</span>`);
  if (P) chips.push(`<span class="chip">Kamera–kâğıt <b>${(norm3(P.t) / 10).toFixed(1)} cm</b></span>`);
  if (S.mode === "live" && res) chips.push(`<span class="chip ${res.p ? "ok" : "no"}">Eşleşme <b>${res.inliers}</b></span>`);
  $("chips").innerHTML = chips.join("");

  const hint = $("hint"); hint.classList.remove("warn");
  if (S.mode === "calib") hint.textContent = `Kalibrasyon: ${S.calViews.length}/${CAL_VIEWS} görüntü. Tahtayı farklı açılardan, görüntünün farklı yerlerinden göster.`;
  else if (!S.cal) { hint.textContent = "Önce kamerayı kalibre et (aşağıda “Kamera kalibrasyonu”)."; hint.classList.add("warn"); }
  else if (S.mode === "idle") hint.textContent = P ? "Kâğıda 50–70 cm uzaktan bak, sonra “Uzak fotoğrafı çek”e bas. Kırmızı artı ölçülecek nokta." : "Test kâğıdını kadraja al; tahta bulununca köşeler yeşil yanar.";
  else if (S.mode === "live") {
    if (!P) { hint.textContent = "Tahta görünmüyor (robotta bunu kol kontrolcüsü verir). Kâğıdı kadrajda tut."; hint.classList.add("warn"); }
    else if (!res?.p) { hint.textContent = "Nokta bu karede bulunamadı. Yavaş hareket et, noktayı kadrajda tut."; hint.classList.add("warn"); }
    else if (res.dEst == null) { hint.textContent = `Yaklaş ve biraz yana kay. Paralaks ${res.parallax?.toFixed(1)}°, en az ${MIN_PARALLAX}° gerekli.`; hint.classList.add("warn"); }
    else hint.textContent = "Her kare yakın fotoğraf adayı. Beğendiğin anda “Yakın fotoğrafı kaydet”e bas.";
  }

  const est = $("vEst"), tru = $("vTrue"), err = $("vErr");
  if (S.mode === "live" && res) {
    est.innerHTML = res.dEst != null ? `${fmtCm(res.dEst)}<small> cm</small>` : "–";
    tru.innerHTML = `${fmtCm(res.dTrue)}<small> cm</small>`;
    err.innerHTML = res.err != null ? `${fmtCm(res.err, true)}<small> cm</small>` : "–";
    err.className = "val " + (res.err == null ? "" : Math.abs(res.err) <= 10 ? "good" : Math.abs(res.err) <= 20 ? "warn" : "bad");
    $("vDetail").textContent = `uzak ${fmtCm(S.ref.dTrue)} cm · ileri ${fmtCm(res.fwd)} · yana ${fmtCm(res.lat)} cm · paralaks ${res.parallax != null ? res.parallax.toFixed(1) : "–"}°` +
      (res.sigma != null ? ` · ±2σ ${fmtCm(2 * res.sigma)} cm` : "") + (res.pxErr != null ? ` · nokta kayması ${res.pxErr.toFixed(1)} px` : "");
  } else if (S.mode !== "live") {
    est.textContent = "–"; err.textContent = "–"; err.className = "val";
    tru.innerHTML = P && S.cal ? `${fmtCm(norm3(toCam(P, S.tapPoint ? [0, 0, 0] : [SHEET.junction[0] * S.sq, SHEET.junction[1] * S.sq, 0])))}<small> cm</small>` : "–";
    if (S.tapPoint) tru.textContent = "–";
    $("vDetail").innerHTML = "&nbsp;";
  }
  $("bFar").disabled = !(S.cal && P && S.mode !== "calib");
  $("bFar").textContent = S.mode === "live" ? "1 · Uzak fotoğrafı yeniden çek" : "1 · Uzak fotoğrafı çek";
  $("bNear").disabled = !(S.mode === "live" && res && res.dEst != null);
  $("bReset").disabled = S.mode !== "live";
  $("calInfo").textContent = S.cal ? `fx ${S.cal.K[0].toFixed(1)}, fy ${S.cal.K[4].toFixed(1)}, cx ${S.cal.K[2].toFixed(1)}, cy ${S.cal.K[5].toFixed(1)} px · hata ${S.cal.rms?.toFixed(2) ?? "–"} px · ${S.W}×${S.H}` : `Bu kamera ve çözünürlük (${S.W}×${S.H}) için kalibrasyon yok.`;
}

function wireButtons() {
  $("bFar").onclick = () => { S.pendingFar = true; };
  $("bNear").onclick = () => {
    const r = S.last; if (!r || r.dEst == null) return;
    S.rows.push({ far: S.ref.dTrue, near: r.dTrue, est: r.dEst, err: r.err, lat: r.lat, fwd: r.fwd, par: r.parallax, sig: r.sigma, t: new Date().toISOString() });
    store.set("tri_rows", S.rows); renderTable();
    resetRef();
    $("hint").textContent = "Kaydedildi. Yeni ölçüm için uzaklaşıp tekrar “Uzak fotoğrafı çek”e bas.";
  };
  $("bReset").onclick = () => { resetRef(); };
  $("bCal").onclick = () => {
    if (S.mode === "calib") { if (S.calViews.length >= 8) runCalibration(); else { S.mode = "idle"; S.calViews = []; $("bCal").textContent = "Kalibrasyonu başlat"; } return; }
    resetRef(); S.mode = "calib"; S.calViews = []; $("bCal").textContent = "Şimdi hesapla (en az 8)";
  };
  $("bCalDel").onclick = () => { store.del(S.calKey); S.cal = SIM ? simCal() : null; resetRef(); };
  $("bAuto").onclick = () => { S.tapPoint = null; };
  $("bCopy").onclick = async () => {
    const txt = tableText();
    try { await navigator.clipboard.writeText(txt); $("bCopy").textContent = "Kopyalandı"; }
    catch { window.prompt?.("Kopyala:", txt); }
    setTimeout(() => ($("bCopy").textContent = "Sonuçları kopyala"), 1500);
  };
  $("bClear").onclick = () => {
    if ($("bClear").dataset.sure) { S.rows = []; store.set("tri_rows", []); renderTable(); delete $("bClear").dataset.sure; $("bClear").textContent = "Tabloyu temizle"; }
    else { $("bClear").dataset.sure = "1"; $("bClear").textContent = "Emin misin? Tekrar bas"; setTimeout(() => { delete $("bClear").dataset.sure; $("bClear").textContent = "Tabloyu temizle"; }, 3000); }
  };
  $("screen").addEventListener("click", (e) => {
    if (S.mode !== "idle" || !S.W) return;
    const r = $("screen").getBoundingClientRect();
    S.tapPoint = [(e.clientX - r.left) / r.width * S.W, (e.clientY - r.top) / r.height * S.H];
  });
}

function renderTable() {
  const tb = $("tbl").querySelector("tbody");
  tb.innerHTML = S.rows.map((r, i) => `<tr><td>${i + 1}</td><td>${fmtCm(r.far)}</td><td>${fmtCm(r.near)}</td><td>${fmtCm(r.est)}</td>` +
    `<td style="color:var(--${Math.abs(r.err) <= 10 ? "good" : Math.abs(r.err) <= 20 ? "warn" : "bad"})">${fmtCm(r.err, true)}</td><td>${fmtCm(r.lat)}</td><td>${r.par.toFixed(1)}°</td></tr>`).join("");
  if (!S.rows.length) { $("summary").textContent = "Henüz kayıtlı ölçüm yok."; return; }
  const e = S.rows.map((r) => Math.abs(r.err) / 10);
  const mean = e.reduce((a, b) => a + b, 0) / e.length, max = Math.max(...e), ok = e.filter((v) => v <= 1).length;
  $("summary").innerHTML = `<b>${S.rows.length}</b> ölçüm · ortalama hata <b>${mean.toFixed(2)} cm</b> · en kötü <b>${max.toFixed(2)} cm</b> · ≤1 cm: <b>${ok}/${S.rows.length}</b> (cm)`;
}
function tableText() {
  const c = S.cal ? `fx=${S.cal.K[0].toFixed(1)} rms=${S.cal.rms?.toFixed(2)} ${S.W}x${S.H} kare=${S.sq}mm kamera=${S.camLabel || ""}` : "";
  return ["İki fotoğraf ölçümleri " + c, "#\tuzak_cm\tyakin_cm\ttahmin_cm\thata_cm\tyana_cm\tileri_cm\tparalaks_deg"]
    .concat(S.rows.map((r, i) => [i + 1, fmtCm(r.far), fmtCm(r.near), fmtCm(r.est), fmtCm(r.err, true), fmtCm(r.lat), fmtCm(r.fwd), r.par.toFixed(2)].join("\t"))).join("\n");
}

/* ---------- simülasyon modu (?sim): kâğıt sanal bir kamerayla görüntülenir ---------- */
let simSheet = null, simT = 0, simLog = [];
function simCal() { return { K: [1500, 0, 960, 0, 1500, 540, 0, 0, 1], dist: [0, 0, 0, 0, 0], rms: 0, size: [1920, 1080] }; }
async function startSim() {
  const img = new Image(); img.src = "kagit.jpg"; await img.decode();
  const c = document.createElement("canvas"); c.width = img.width; c.height = img.height;
  const x = c.getContext("2d"); x.drawImage(img, 0, 0);
  const rgba = cv.matFromImageData(x.getImageData(0, 0, img.width, img.height));
  simSheet = new cv.Mat(); cv.cvtColor(rgba, simSheet, cv.COLOR_RGBA2GRAY); rgba.delete();
  S.sq = SHEET.sqDesign; $("iSq").value = S.sq;
  window.__sim = { log: simLog };
}
function simPose(s) {
  // Kamera 60 cm'den ~30 cm'ye yaklaşıyor, bu sırada 12 cm sağa kayıyor; hedef kâğıdın ortası
  const target = [5, 55, 0];
  const C = [-60 + 120 * s, 20 - 20 * s, -(560 - 260 * s)];
  const f = unit3(sub3(target, C)), x = unit3(cross3([0, 1, 0], f)), y = cross3(f, x);
  const R = [x, y, f];
  return { R, t: mul3(matVec(R, C), -1) };
}
function simFrame() {
  if (!simSheet) return null;
  const s = Math.min(1, simT);
  const P = simPose(s), cal = simCal(), Kk = cal.K;
  // Kâğıt pikseli (10 px/mm) -> tahta mm -> görüntü
  const T = [[0.1, 0, -SHEET.originMm[0]], [0, 0.1, -SHEET.originMm[1]], [0, 0, 1]];
  const Rt = [[P.R[0][0], P.R[0][1], P.t[0]], [P.R[1][0], P.R[1][1], P.t[1]], [P.R[2][0], P.R[2][1], P.t[2]]];
  const Km = [[Kk[0], 0, Kk[2]], [0, Kk[4], Kk[5]], [0, 0, 1]];
  const mm = (A, B) => A.map((r) => [0, 1, 2].map((j) => r[0] * B[0][j] + r[1] * B[1][j] + r[2] * B[2][j]));
  const H = mm(mm(Km, Rt), T);
  const Hm = cv.matFromArray(3, 3, cv.CV_64F, H.flat());
  const out = new cv.Mat();
  cv.warpPerspective(simSheet, out, Hm, new cv.Size(1920, 1080), cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar(120));
  Hm.delete();
  const noise = new cv.Mat(1080, 1920, cv.CV_8UC1);
  const mu = cv.matFromArray(1, 1, cv.CV_64F, [0]), sd = cv.matFromArray(1, 1, cv.CV_64F, [3]);
  try { cv.randn(noise, mu, sd); cv.add(out, noise, out); } catch {}
  mu.delete(); sd.delete(); noise.delete();
  S.simTrue = P;
  return out;
}
function simAdvance(res) {
  if (params.has("auto")) {
    if (S.mode === "idle" && S.cal && S.curP && !S.ref && simT === 0) S.pendingFar = true;
    if (S.mode === "live" && res) simLog.push({ s: simT, dTrue: res.dTrue, dEst: res.dEst, err: res.err, par: res.parallax, inl: res.inliers, pxErr: res.pxErr });
    if (S.mode === "live") simT = Math.min(1, simT + 0.04);
  } else if (S.mode === "live") simT = Math.min(1, simT + 0.01);
}

main();
