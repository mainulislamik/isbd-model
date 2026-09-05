"""
ISBD v1.00 — Designer Pair Training Panel (Studio)
Upload before/after Photoshop pairs -> converted in RAM to 64px tensors
(images NEVER touch disk) -> fine-tunes the live model via realfinetune.py.
Run: .venv/bin/python -m uvicorn isbd.panel:app --host 0.0.0.0 --port 8077
"""
import hashlib
import io
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
NPZ = DATA / "pairs.npz"
HASHES = DATA / "hashes.json"
FT_LOG = DATA / "ft.log"
FT_STATE = DATA / "ft_state.json"
IMG = 64
MAX_FILE = 25 * 1024 * 1024

app = FastAPI(title="ISBD Studio")


# ── helpers ──────────────────────────────────────────────────────────────────
def _hashes():
    try:
        return json.loads(HASHES.read_text())
    except Exception:
        return []


def _save_hashes(h):
    HASHES.write_text(json.dumps(h))


def _n_pairs():
    if not NPZ.exists():
        return 0
    try:
        with np.load(NPZ) as d:
            return len(d["X"])
    except Exception:
        return 0


def _live():
    """Parse the latest step/loss from the 24/7 trainer's journalctl."""
    try:
        out = subprocess.run(
            ["journalctl", "--user", "-u", "isbd-train", "--no-pager", "-n", "20", "-o", "cat"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        last = {"step": 0, "loss": 0.0}
        for line in out.splitlines():
            if line.strip().startswith("step"):
                p = line.replace("|", " ").split()
                try:
                    last = {"step": int(p[1]), "loss": float(p[3])}
                except Exception:
                    pass
        return last
    except Exception:
        return {"step": 0, "loss": 0.0}


def _lock_busy():
    import fcntl
    path = ROOT / "checkpoints" / "train.lock"
    if not path.exists():
        return False
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    except OSError:
        return True
    finally:
        os.close(fd)


def _ft_state():
    try:
        st = json.loads(FT_STATE.read_text())
    except Exception:
        st = {"running": False}
    if st.get("running") and st.get("pid"):
        try:
            os.kill(st["pid"], 0)
        except OSError:
            st["running"] = False
            st["stale"] = True
            FT_STATE.write_text(json.dumps(st))
    return st


def _ft_log_tail(n=80):
    try:
        return "\n".join(FT_LOG.read_text().splitlines()[-n:])
    except Exception:
        return ""


# ── API ──────────────────────────────────────────────────────────────────────
@app.post("/api/pair")
async def add_pair(before: UploadFile = File(...), after: UploadFile = File(...)):
    b, a = await before.read(), await after.read()
    if not b or not a:
        raise HTTPException(400, "ফাইল খালি")
    if len(b) > MAX_FILE or len(a) > MAX_FILE:
        raise HTTPException(413, "ফাইল 25MB-এর বেশি")
    h = hashlib.sha256(b + b"|" + a).hexdigest()
    hs = _hashes()
    if h in hs:
        return {"ok": True, "dup": True, "pairs": len(hs)}

    def prep(bs):
        img = Image.open(io.BytesIO(bs)).convert("RGB")
        arr = np.asarray(img.resize((IMG, IMG), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
        return np.ascontiguousarray(arr.transpose(2, 0, 1))

    try:
        x, y = prep(b), prep(a)
    except Exception:
        raise HTTPException(400, "ছবি পড়া যায়নি — JPG/PNG/WebP দিন")

    if NPZ.exists():
        with np.load(NPZ) as d:
            X, Y = d["X"], d["Y"]
        X, Y = np.concatenate([X, x[None]]), np.concatenate([Y, y[None]])
    else:
        X, Y = x[None], y[None]
    np.savez(NPZ, X=X.astype(np.float32), Y=Y.astype(np.float32))
    hs.append(h)
    _save_hashes(hs)
    return {"ok": True, "dup": False, "pairs": len(hs)}


@app.post("/api/train")
async def train(steps: int = 300):
    steps = max(20, min(steps, 2000))
    if _n_pairs() == 0:
        raise HTTPException(400, "আগে অন্তত একটি before/after পেয়ার আপলোড করুন")
    if _ft_state().get("running"):
        raise HTTPException(409, "ফাইন-টিউন ইতিমধ্যে চলছে — শেষ হতে দিন")

    def worker():
        FT_STATE.write_text(json.dumps({"running": True, "pid": 0, "started": time.time()}))
        with open(FT_LOG, "w") as log:
            p = subprocess.Popen(
                [str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "isbd" / "realfinetune.py"),
                 "--steps", str(steps), "--wait-lock", "900"],
                cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
            FT_STATE.write_text(json.dumps({"running": True, "pid": p.pid, "started": time.time()}))
            p.wait()
            FT_STATE.write_text(json.dumps({"running": False, "pid": p.pid, "code": p.returncode,
                                            "ended": time.time()}))

    threading.Thread(target=worker, daemon=True).start()
    return {"queued": True, "steps": steps, "pairs": _n_pairs()}


@app.post("/api/purge")
async def purge():
    if _ft_state().get("running"):
        raise HTTPException(409, "ফাইন-টিউন চলাকালীন ডেটা মুছতে পারবেন না")
    NPZ.unlink(missing_ok=True)
    HASHES.unlink(missing_ok=True)
    return {"ok": True, "pairs": 0}


@app.get("/api/status")
async def status():
    return {
        "live": _live(),
        "lock_busy": _lock_busy(),
        "pairs": _n_pairs(),
        "ft": _ft_state(),
        "log": _ft_log_tail(),
    }


# ── Modern UI ────────────────────────────────────────────────────────────────
HTML = """<!doctype html>
<html lang="bn">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ISBD Studio</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🎨</text></svg>">
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root {
  --bg: #0b0d13; --bg2: #111420; --card: rgba(20,24,38,0.7);
  --glass: rgba(255,255,255,0.04); --glass-border: rgba(255,255,255,0.08);
  --tx: #e8ecf4; --dim: #7b8499; --muted: #4a5268;
  --acc: #6c8cff; --acc2: #4cc2ff; --acc-glow: rgba(108,140,255,0.15);
  --ok: #4ade80; --ok-bg: rgba(74,222,128,0.1);
  --warn: #fbbf24; --warn-bg: rgba(251,191,36,0.1);
  --err: #f87171; --err-bg: rgba(248,113,113,0.1);
  --radius: 14px; --radius-sm: 10px;
}
* { box-sizing: border-box; margin: 0; }
body {
  font-family: 'Inter', 'Noto Sans Bengali', 'Hind Siliguri', system-ui, sans-serif;
  background: var(--bg); color: var(--tx); min-height: 100vh;
  background-image:
    radial-gradient(ellipse 80% 50% at 50% -20%, rgba(108,140,255,0.08), transparent),
    radial-gradient(ellipse 60% 40% at 80% 100%, rgba(76,194,255,0.05), transparent);
}
.container { max-width: 960px; margin: 0 auto; padding: 24px 20px 60px; }

/* ─ header ─ */
header { text-align: center; padding: 32px 0 24px; }
header h1 { font-size: 28px; font-weight: 700; letter-spacing: -0.5px;
  background: linear-gradient(135deg, var(--acc), var(--acc2));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
header p { color: var(--dim); font-size: 14px; margin-top: 6px; }

/* ─ stat bar ─ */
.stats {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px; margin: 20px 0 28px;
}
.stat {
  background: var(--card); border: 1px solid var(--glass-border);
  backdrop-filter: blur(12px); border-radius: var(--radius);
  padding: 16px; text-align: center; transition: border-color 0.3s;
}
.stat:hover { border-color: rgba(255,255,255,0.15); }
.stat .icon { font-size: 22px; margin-bottom: 6px; }
.stat .label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
.stat .value { font-size: 22px; font-weight: 700; font-variant-numeric: tabular-nums; }
.stat .sub { font-size: 11px; color: var(--dim); margin-top: 2px; }
.stat.active { border-color: var(--acc); box-shadow: 0 0 20px var(--acc-glow); }
.stat.ok { border-color: var(--ok); box-shadow: 0 0 12px rgba(74,222,128,0.1); }
.stat.warn { border-color: var(--warn); box-shadow: 0 0 12px rgba(251,191,36,0.1); }

/* ─ sections ─ */
.section {
  background: var(--card); border: 1px solid var(--glass-border);
  backdrop-filter: blur(12px); border-radius: var(--radius);
  padding: 24px; margin-bottom: 20px;
}
.section h2 {
  font-size: 16px; font-weight: 600; margin-bottom: 16px;
  display: flex; align-items: center; gap: 8px;
}
.section h2 .num {
  display: inline-flex; align-items: center; justify-content: center;
  width: 26px; height: 26px; border-radius: 8px; font-size: 13px; font-weight: 700;
  background: linear-gradient(135deg, var(--acc), var(--acc2)); color: #0b0d13;
}

/* ─ upload zone ─ */
.upload-area {
  border: 2px dashed var(--glass-border); border-radius: var(--radius-sm);
  padding: 28px; text-align: center; transition: all 0.3s; cursor: pointer;
  position: relative; overflow: hidden;
}
.upload-area:hover, .upload-area.dragover {
  border-color: var(--acc); background: var(--acc-glow);
}
.upload-area .icon { font-size: 36px; margin-bottom: 8px; opacity: 0.7; }
.upload-area p { color: var(--dim); font-size: 13px; }
.upload-area p b { color: var(--tx); }

/* ─ pair rows ─ */
.pair-grid { display: flex; flex-direction: column; gap: 10px; margin: 16px 0; }
.pair-row {
  display: grid; grid-template-columns: 1fr 1fr auto; gap: 12px;
  background: var(--glass); border: 1px solid var(--glass-border);
  border-radius: var(--radius-sm); padding: 12px 14px; align-items: center;
  transition: border-color 0.3s;
}
.pair-row:hover { border-color: rgba(255,255,255,0.12); }
.pair-col label {
  font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px;
  display: block; margin-bottom: 6px;
}
.pair-col input[type=file] {
  width: 100%; font-size: 12px; color: var(--dim); padding: 6px 0;
}
.pair-col input[type=file]::file-selector-button {
  background: var(--glass); border: 1px solid var(--glass-border); color: var(--tx);
  border-radius: 6px; padding: 5px 12px; font-size: 11px; cursor: pointer; margin-right: 8px;
  transition: background 0.2s;
}
.pair-col input[type=file]::file-selector-button:hover { background: rgba(255,255,255,0.08); }
.preview-row { display: flex; gap: 6px; margin-top: 8px; }
.preview-row img {
  width: 56px; height: 56px; object-fit: cover; border-radius: 8px;
  border: 1px solid var(--glass-border); display: none;
}
.pair-remove {
  background: transparent; border: 1px solid var(--glass-border); color: var(--err);
  border-radius: 8px; width: 32px; height: 32px; cursor: pointer; font-size: 14px;
  display: flex; align-items: center; justify-content: center; transition: all 0.2s;
}
.pair-remove:hover { background: var(--err-bg); border-color: var(--err); }
.pair-status { font-size: 11px; margin-top: 6px; }
.pair-status.ok { color: var(--ok); }
.pair-status.dup { color: var(--warn); }
.pair-status.err { color: var(--err); }

/* ─ controls ─ */
.controls {
  display: flex; gap: 12px; flex-wrap: wrap; align-items: center; margin-top: 16px;
}
.btn {
  border: 0; border-radius: var(--radius-sm); padding: 12px 22px;
  font-size: 14px; font-weight: 600; cursor: pointer; transition: all 0.2s;
  display: inline-flex; align-items: center; gap: 6px;
}
.btn-primary {
  background: linear-gradient(135deg, var(--acc), var(--acc2)); color: #0b0d13;
  box-shadow: 0 4px 16px var(--acc-glow);
}
.btn-primary:hover { transform: translateY(-1px); box-shadow: 0 6px 24px var(--acc-glow); }
.btn-primary:disabled { opacity: 0.4; transform: none; cursor: not-allowed; box-shadow: none; }
.btn-ghost {
  background: var(--glass); border: 1px solid var(--glass-border); color: var(--tx);
}
.btn-ghost:hover { background: rgba(255,255,255,0.06); border-color: rgba(255,255,255,0.15); }
.btn-danger { background: var(--err-bg); border: 1px solid rgba(248,113,113,0.2); color: var(--err); }
.btn-danger:hover { background: rgba(248,113,113,0.15); }
.btn-add {
  background: transparent; border: 2px dashed var(--glass-border); color: var(--dim);
  border-radius: var(--radius-sm); padding: 10px; width: 100%; cursor: pointer;
  font-size: 13px; transition: all 0.2s;
}
.btn-add:hover { border-color: var(--acc); color: var(--acc); background: var(--acc-glow); }

select.ctrl {
  background: var(--glass); border: 1px solid var(--glass-border); color: var(--tx);
  border-radius: var(--radius-sm); padding: 11px 14px; font-size: 13px; cursor: pointer;
}

/* ─ info card ─ */
.info-card {
  background: var(--glass); border: 1px solid var(--glass-border);
  border-radius: var(--radius-sm); padding: 14px 16px; font-size: 12.5px;
  color: var(--dim); line-height: 1.8; margin-top: 16px;
}
.info-card b { color: var(--tx); }
.info-card .row { display: flex; gap: 6px; align-items: flex-start; margin-bottom: 4px; }
.info-card .row .ic { flex-shrink: 0; width: 18px; text-align: center; }

/* ─ log ─ */
.log-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
.log-header h3 { font-size: 14px; font-weight: 600; }
.log-badge {
  font-size: 11px; padding: 3px 10px; border-radius: 20px; font-weight: 600;
}
.log-badge.running { background: var(--ok-bg); color: var(--ok); animation: pulse 2s infinite; }
.log-badge.idle { background: var(--glass); color: var(--dim); }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.6} }
pre#log {
  background: var(--bg); border: 1px solid var(--glass-border); border-radius: var(--radius-sm);
  padding: 14px; font-size: 12px; font-family: 'JetBrains Mono', 'Fira Code', monospace;
  max-height: 320px; overflow: auto; white-space: pre-wrap; color: #a8d8a8; min-height: 60px;
  line-height: 1.6;
}
pre#log::-webkit-scrollbar { width: 6px; }
pre#log::-webkit-scrollbar-track { background: transparent; }
pre#log::-webkit-scrollbar-thumb { background: var(--glass-border); border-radius: 3px; }

/* ─ toast ─ */
.toast-container { position: fixed; top: 20px; right: 20px; z-index: 9999; display: flex; flex-direction: column; gap: 8px; }
.toast {
  padding: 12px 18px; border-radius: var(--radius-sm); font-size: 13px; font-weight: 500;
  backdrop-filter: blur(12px); border: 1px solid var(--glass-border);
  animation: slideIn 0.3s ease, fadeOut 0.3s ease 4s forwards;
  box-shadow: 0 8px 32px rgba(0,0,0,0.3); max-width: 340px;
}
.toast.ok { background: rgba(74,222,128,0.15); border-color: rgba(74,222,128,0.3); color: var(--ok); }
.toast.warn { background: rgba(251,191,36,0.15); border-color: rgba(251,191,36,0.3); color: var(--warn); }
.toast.err { background: rgba(248,113,113,0.15); border-color: rgba(248,113,113,0.3); color: var(--err); }
.toast.info { background: rgba(108,140,255,0.15); border-color: rgba(108,140,255,0.3); color: var(--acc2); }
@keyframes slideIn { from { transform: translateX(100%); opacity: 0; } to { transform: translateX(0); opacity: 1; } }
@keyframes fadeOut { to { opacity: 0; transform: translateY(-10px); } }

/* ─ progress bar ─ */
.progress-wrap { margin-top: 14px; display: none; }
.progress-wrap.show { display: block; }
.progress-bar {
  height: 6px; background: var(--glass); border-radius: 3px; overflow: hidden;
}
.progress-fill {
  height: 100%; border-radius: 3px; width: 0%;
  background: linear-gradient(90deg, var(--acc), var(--acc2));
  transition: width 0.4s ease;
}
.progress-text { font-size: 11px; color: var(--dim); margin-top: 6px; }

/* ─ responsive ─ */
@media (max-width: 600px) {
  .stats { grid-template-columns: repeat(2, 1fr); }
  .pair-row { grid-template-columns: 1fr; }
  .pair-remove { position: absolute; top: 8px; right: 8px; }
  .controls { flex-direction: column; }
  .controls .btn, .controls select { width: 100%; justify-content: center; }
  header h1 { font-size: 22px; }
}
</style>
</head>
<body>

<div class="toast-container" id="toasts"></div>

<div class="container">

<header>
  <h1>🎨 ISBD Studio</h1>
  <p>ডিজাইনারদের before/after পেয়ার দিয়ে ISBD v1.00 ফাইন-টিউন প্যানেল</p>
</header>

<!-- ─ stats dashboard ─ -->
<div class="stats">
  <div class="stat" id="stat-step">
    <div class="icon">🔥</div>
    <div class="label">24/7 ট্রেনার</div>
    <div class="value" id="v-step">—</div>
    <div class="sub">ধাপ</div>
  </div>
  <div class="stat" id="stat-loss">
    <div class="icon">📉</div>
    <div class="label">লস</div>
    <div class="value" id="v-loss">—</div>
    <div class="sub">বর্তমান</div>
  </div>
  <div class="stat" id="stat-pairs">
    <div class="icon">📦</div>
    <div class="label">পেয়ার</div>
    <div class="value" id="v-pairs">—</div>
    <div class="sub">আপলোডকৃত</div>
  </div>
  <div class="stat" id="stat-lock">
    <div class="icon">🔒</div>
    <div class="label">ট্রেনার লক</div>
    <div class="value" id="v-lock">—</div>
    <div class="sub">অবস্থা</div>
  </div>
  <div class="stat" id="stat-ft">
    <div class="icon">⚙️</div>
    <div class="label">ফাইন-টিউন</div>
    <div class="value" id="v-ft">—</div>
    <div class="sub">স্ট্যাটাস</div>
  </div>
</div>

<!-- ─ section 1: upload ─ -->
<div class="section">
  <h2><span class="num">১</span> পেয়ার আপলোড করুন</h2>
  <div class="upload-area" id="dropzone" onclick="addRow()">
    <div class="icon">📁</div>
    <p><b>ক্লিক করুন</b> বা ছবি টেনে আনুন</p>
    <p>একটি পেয়ার = একটি "আগে" + একটি "পরে/ফাইনাল" ছবি</p>
  </div>
  <div class="pair-grid" id="rows"></div>
</div>

<!-- ─ section 2: train ─ -->
<div class="section">
  <h2><span class="num">২</span> ফাইন-টিউন শুরু করুন</h2>
  <div class="controls">
    <select class="ctrl" id="steps">
      <option value="100">১০০ ধাপ (দ্রুত পরীক্ষা)</option>
      <option value="300" selected>৩০০ ধাপ (স্বাভাবিক)</option>
      <option value="600">৬০০ ধাপ</option>
      <option value="1200">১২০০ ধাপ (গভীর ট্রেনিং)</option>
    </select>
    <button class="btn btn-primary" id="tr" onclick="doTrain()">▶ প্রসেস ও ফাইন-টিউন</button>
    <button class="btn btn-danger" onclick="doPurge()">🗑 সব ডেটা মুছুন</button>
  </div>
  <div class="progress-wrap" id="prog-wrap">
    <div class="progress-bar"><div class="progress-fill" id="prog-fill"></div></div>
    <div class="progress-text" id="prog-text">আপলোড হচ্ছে…</div>
  </div>
  <div class="info-card">
    <div class="row"><span class="ic">🔒</span><span><b>আপনার ছবি কোথাও সংরক্ষিত হয় না।</b> আপলোডের সাথে সাথেই মেমরিতে 64px টেনসরে রূপান্তর — ডিস্কে কোনো ছবি লেখা হয় না।</span></div>
    <div class="row"><span class="ic">⚙️</span><span>ফাইন-টিউন 24/7 ট্রেনারের lock মুক্ত হলে <b>নিজে থেকেই</b> শুরু হয়। নিচের লগে অপেক্ষার অবস্থা দেখা যায়।</span></div>
    <div class="row"><span class="ic">🧠</span><span>ফাইন-টিউন 40% আসল পেয়ার + 60% synthetic মিক্সে ট্রেন করে — আগের শেখা ভুলে যায় না।</span></div>
  </div>
</div>

<!-- ─ section 3: log ─ -->
<div class="section">
  <div class="log-header">
    <h2 style="margin:0"><span class="num">৩</span> ট্রেনিং লগ</h2>
    <span class="log-badge idle" id="log-badge">idle</span>
  </div>
  <pre id="log">প্যানেল প্রস্তুত — পেয়ার আপলোড করুন…</pre>
</div>

</div><!-- /container -->

<script>
/* ─ toast ─ */
function toast(msg, type='info') {
  const c = document.getElementById('toasts');
  const t = document.createElement('div');
  t.className = 'toast ' + type;
  t.textContent = msg;
  c.appendChild(t);
  setTimeout(() => t.remove(), 4500);
}

/* ─ drag & drop ─ */
const dz = document.getElementById('dropzone');
['dragenter','dragover'].forEach(e => dz.addEventListener(e, ev => { ev.preventDefault(); dz.classList.add('dragover'); }));
['dragleave','drop'].forEach(e => dz.addEventListener(e, ev => { ev.preventDefault(); dz.classList.remove('dragover'); }));
dz.addEventListener('drop', ev => {
  const files = ev.dataTransfer.files;
  if (files.length >= 2) {
    addRow(files[0], files[1]);
    toast('পেয়ার যোগ হয়েছে (drag & drop)', 'ok');
  } else if (files.length === 1) {
    toast('দুটি ছবি দিন — আগে ও পরে', 'warn');
  }
});

/* ─ pair management ─ */
let pairCount = 0;
function addRow(fileBefore, fileAfter) {
  pairCount++;
  const r = document.createElement('div');
  r.className = 'pair-row';
  r.innerHTML = `
    <div class="pair-col">
      <label>আগে — মূল ছবি</label>
      <input type="file" accept="image/*" onchange="preview(this)">
      <div class="preview-row"><img class="th"></div>
    </div>
    <div class="pair-col">
      <label>পরে — ডিজাইনারের এডিট</label>
      <input type="file" accept="image/*" onchange="preview(this)">
      <div class="preview-row"><img class="th"></div>
    </div>
    <button class="pair-remove" onclick="this.closest('.pair-row').remove()" title="সরান">✕</button>`;
  document.getElementById('rows').appendChild(r);
  // If files provided (drag-drop), set them
  if (fileBefore) { const dt = new DataTransfer(); dt.items.add(fileBefore); r.querySelectorAll('input')[0].files = dt.files; preview(r.querySelectorAll('input')[0]); }
  if (fileAfter) { const dt = new DataTransfer(); dt.items.add(fileAfter); r.querySelectorAll('input')[1].files = dt.files; preview(r.querySelectorAll('input')[1]); }
}
function preview(inp) {
  const f = inp.files[0]; if (!f) return;
  const img = inp.closest('.pair-col').querySelector('.th');
  img.src = URL.createObjectURL(f); img.style.display = 'block';
}

/* ─ upload + train ─ */
async function doTrain() {
  const rows = [...document.querySelectorAll('.pair-row')];
  if (!rows.length) { toast('আগে অন্তত একটি পেয়ার যোগ করুন', 'warn'); return; }
  const tr = document.getElementById('tr');
  tr.disabled = true;
  const pw = document.getElementById('prog-wrap');
  const pf = document.getElementById('prog-fill');
  const pt = document.getElementById('prog-text');
  pw.classList.add('show'); pf.style.width = '0%';
  
  let ok = 0, total = rows.length;
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i];
    const [bf, af] = r.querySelectorAll('input[type=file]');
    // Clear old status
    r.querySelectorAll('.pair-status').forEach(s => s.remove());
    const st = document.createElement('div'); st.className = 'pair-status'; r.appendChild(st);
    
    if (!bf.files[0] || !af.files[0]) {
      st.className = 'pair-status err'; st.textContent = '✗ দুটো ছবি দিন';
      continue;
    }
    pt.textContent = `আপলোড ${i+1}/${total}…`;
    pf.style.width = ((i+1)/total*60) + '%';
    
    const fd = new FormData(); fd.append('before', bf.files[0]); fd.append('after', af.files[0]);
    try {
      const res = await fetch('/api/pair', {method:'POST', body:fd});
      const j = await res.json();
      if (res.ok && !j.dup) { ok++; st.className = 'pair-status ok'; st.textContent = '✓ যোগ হয়েছে (মোট ' + j.pairs + ')'; }
      else if (res.ok && j.dup) { st.className = 'pair-status dup'; st.textContent = '⚠ ডুপ্লিকেট'; }
      else { st.className = 'pair-status err'; st.textContent = '✗ ' + (j.detail || 'ব্যর্থ'); }
    } catch(e) { st.className = 'pair-status err'; st.textContent = '✗ নেটওয়ার্ক সমস্যা'; }
  }
  
  if (ok === 0) {
    toast('নতুন পেয়ার যোগ হয়নি', 'warn');
    tr.disabled = false; pw.classList.remove('show'); return;
  }
  
  pf.style.width = '70%'; pt.textContent = 'ফাইন-টিউন শুরু হচ্ছে…';
  const steps = document.getElementById('steps').value;
  try {
    const res = await fetch('/api/train?steps=' + steps, {method:'POST'});
    const j = await res.json();
    if (res.ok) {
      toast('ফাইন-টিউন শুরু: ' + steps + ' ধাপ, ' + j.pairs + ' পেয়ার', 'ok');
      pf.style.width = '100%'; pt.textContent = '✓ কিউড — lock মুক্ত হলেই চলবে';
    } else { toast(j.detail || 'ব্যর্থ', 'err'); }
  } catch(e) { toast('শুরু করা যায়নি', 'err'); }
  tr.disabled = false;
}

function doPurge() {
  if (!confirm('সব টেনসর-ডেটা মুছে ফেলবেন? এটা পূর্বাবস্থায় ফেরানো যাবে না।')) return;
  fetch('/api/purge', {method:'POST'}).then(r => r.json()).then(j => {
    toast('সব ডেটা মুছে ফেলা হয়েছে', 'ok');
    document.getElementById('rows').innerHTML = '';
    poll();
  }).catch(() => toast('মুছতে ব্যর্থ', 'err'));
}

/* ─ live poll ─ */
async function poll() {
  try {
    const j = await (await fetch('/api/status')).json();
    // Step
    document.getElementById('v-step').textContent = j.live.step ? j.live.step.toLocaleString() : '—';
    // Loss
    document.getElementById('v-loss').textContent = j.live.loss || '—';
    // Pairs
    document.getElementById('v-pairs').textContent = j.pairs;
    const sp = document.getElementById('stat-pairs');
    sp.className = 'stat' + (j.pairs > 0 ? ' ok' : '');
    // Lock
    const lockEl = document.getElementById('v-lock');
    lockEl.textContent = j.lock_busy ? 'busy' : 'free';
    document.getElementById('stat-lock').className = 'stat' + (j.lock_busy ? ' warn' : ' ok');
    // FT
    const ftEl = document.getElementById('v-ft');
    const sft = document.getElementById('stat-ft');
    if (j.ft.running) {
      ftEl.textContent = 'চলছে…';
      sft.className = 'stat active';
      document.getElementById('log-badge').className = 'log-badge running';
      document.getElementById('log-badge').textContent = 'চলছে';
    } else {
      ftEl.textContent = 'idle';
      sft.className = 'stat';
      document.getElementById('log-badge').className = 'log-badge idle';
      document.getElementById('log-badge').textContent = 'idle';
    }
    // Log
    if (j.log) document.getElementById('log').textContent = j.log;
  } catch(e) {}
}
setInterval(poll, 3000);
poll();
addRow();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML