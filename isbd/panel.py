"""
ISBD v1.00 — Designer Pair Training Panel & Studio Pro
Features:
- Live Dashboard (24/7 step, loss, pairs, lock, fine-tune progress, metrics)
- Pair Upload (Drag & drop, multi-pair, live before/after image preview)
- AI Object Detection & Recognition (YOLOv8 Vision Scanner with Bengali labels)
- Interactive Live Inference Playground (Upload any photo, instant AI restore, comparison slider)
- Hyperparameter controls (Steps, Learning Rate, Batch Size, Real-pair Ratio)
- Model Architecture & Training Insights / Loss graph
- Memory-safe, non-blocking asynchronous training with lock-safety
"""
import base64
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
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
NPZ = DATA / "pairs.npz"
HASHES = DATA / "hashes.json"
FT_LOG = DATA / "ft.log"
FT_STATE = DATA / "ft_state.json"
CKPT = ROOT / "checkpoints"
HIST = CKPT / "history.json"
IMG = 64
MAX_FILE = 25 * 1024 * 1024

app = FastAPI(title="ISBD Studio Pro Vision")


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
            ["journalctl", "--user", "-u", "isbd-train", "--no-pager", "-n", "30", "-o", "cat"],
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


def _ft_log_tail(n=100):
    try:
        return "\n".join(FT_LOG.read_text().splitlines()[-n:])
    except Exception:
        return ""


def _loss_history(limit=50):
    """Return recent loss points for real-time sparkline/graph."""
    try:
        if HIST.exists():
            h = json.loads(HIST.read_text())
            steps = h.get("steps", [])[-limit:]
            losses = h.get("losses", [])[-limit:]
            return [{"step": s, "loss": l} for s, l in zip(steps, losses)]
    except Exception:
        pass
    return []


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
async def train(
    steps: int = Query(300, ge=20, le=3000),
    lr: float = Query(0.0001, ge=0.00001, le=0.01),
    batch: int = Query(8, ge=2, le=32)
):
    if _n_pairs() == 0:
        raise HTTPException(400, "আগে অন্তত একটি before/after পেয়ার আপলোড করুন")
    if _ft_state().get("running"):
        raise HTTPException(409, "ফাইন-টিউন ইতিমধ্যে চলছে — শেষ হতে দিন")

    def worker():
        FT_STATE.write_text(json.dumps({"running": True, "pid": 0, "started": time.time(), "target_steps": steps}))
        with open(FT_LOG, "w") as log:
            cmd = [
                str(ROOT / ".venv" / "bin" / "python"),
                str(ROOT / "isbd" / "realfinetune.py"),
                "--steps", str(steps),
                "--lr", str(lr),
                "--batch", str(batch),
                "--wait-lock", "900"
            ]
            p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
            FT_STATE.write_text(json.dumps({
                "running": True, "pid": p.pid, "started": time.time(),
                "target_steps": steps, "lr": lr, "batch": batch
            }))
            p.wait()
            FT_STATE.write_text(json.dumps({
                "running": False, "pid": p.pid, "code": p.returncode, "ended": time.time()
            }))

    threading.Thread(target=worker, daemon=True).start()
    return {"queued": True, "steps": steps, "lr": lr, "batch": batch, "pairs": _n_pairs()}


@app.post("/api/detect")
async def detect_api(image: UploadFile = File(...), conf: float = Query(0.25, ge=0.1, le=0.9)):
    """Detect and classify all objects in the image with Bengali descriptions and Bounding Boxes."""
    raw = await image.read()
    if not raw:
        raise HTTPException(400, "ছবি পাওয়া যায়নি")
    try:
        from isbd.detector import detect_objects_in_image
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        annotated_img, detections, summary = detect_objects_in_image(img, conf_threshold=conf)

        # Convert annotated image to base64
        buf = io.BytesIO()
        annotated_img.save(buf, format="JPEG", quality=85)
        buf.seek(0)
        img_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.read()).decode("utf-8")

        return {
            "ok": True,
            "total_objects": len(detections),
            "summary": summary,
            "detections": detections,
            "annotated_image": img_b64
        }
    except Exception as e:
        raise HTTPException(500, f"অবজেক্ট ডিটেকশন ত্রুটি: {str(e)}")


@app.post("/api/infer")
async def infer_image(image: UploadFile = File(...)):
    """Live AI Image Restoration Playground via current trained model."""
    raw = await image.read()
    if not raw:
        raise HTTPException(400, "ফাইল পাওয়া যায়নি")
    try:
        from isbd.model import TinyUNet
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        orig_size = img.size
        small = img.resize((IMG, IMG), Image.Resampling.LANCZOS)
        x = torch.from_numpy(np.asarray(small, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]

        model = TinyUNet()
        best_ckpt = CKPT / "best.pt"
        last_ckpt = CKPT / "last.pt"
        ckpt_path = best_ckpt if best_ckpt.exists() else last_ckpt
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state["model"])
        model.eval()

        with torch.no_grad():
            out = model(x)[0].clamp(0, 1).numpy().transpose(1, 2, 0)

        restored = Image.fromarray((out * 255).astype(np.uint8)).resize(orig_size, Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        restored.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(500, f"ইনফারেন্স ত্রুটি: {str(e)}")


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
        "history": _loss_history(40),
    }


# ── Modern UI ────────────────────────────────────────────────────────────────
HTML = """<!doctype html>
<html lang="bn">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ISBD Studio Pro — AI Image Engine & Vision</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🚀</text></svg>">
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root {
  --bg: #07090e; --bg-secondary: #0d111a; --card: rgba(18, 23, 37, 0.75);
  --card-border: rgba(255, 255, 255, 0.08); --glass: rgba(255, 255, 255, 0.03);
  --tx-primary: #f1f5f9; --tx-secondary: #94a3b8; --tx-muted: #64748b;
  --accent: #6366f1; --accent-glow: rgba(99, 102, 241, 0.25);
  --cyan: #06b6d4; --cyan-glow: rgba(6, 182, 212, 0.2);
  --success: #10b981; --success-bg: rgba(16, 185, 129, 0.12);
  --warning: #f59e0b; --warning-bg: rgba(245, 158, 11, 0.12);
  --danger: #ef4444; --danger-bg: rgba(239, 68, 68, 0.12);
  --radius-lg: 18px; --radius-md: 12px; --radius-sm: 8px;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Plus Jakarta Sans', 'Noto Sans Bengali', system-ui, sans-serif;
  background-color: var(--bg); color: var(--tx-primary); min-height: 100vh;
  background-image:
    radial-gradient(circle at 15% 15%, rgba(99, 102, 241, 0.12) 0%, transparent 40%),
    radial-gradient(circle at 85% 85%, rgba(6, 182, 212, 0.1) 0%, transparent 40%);
  background-attachment: fixed; line-height: 1.5;
}
.wrapper { max-width: 1160px; margin: 0 auto; padding: 24px 20px 80px; }

/* ─ Navigation / Top Brand ─ */
.navbar {
  display: flex; justify-content: space-between; align-items: center;
  padding: 16px 24px; background: var(--card); border: 1px solid var(--card-border);
  backdrop-filter: blur(20px); border-radius: var(--radius-lg); margin-bottom: 24px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}
.brand { display: flex; align-items: center; gap: 12px; }
.brand-icon {
  width: 44px; height: 44px; border-radius: 12px;
  background: linear-gradient(135deg, var(--accent), var(--cyan));
  display: flex; align-items: center; justify-content: center; font-size: 22px;
  box-shadow: 0 0 20px var(--accent-glow);
}
.brand-text h1 { font-size: 20px; font-weight: 800; letter-spacing: -0.5px;
  background: linear-gradient(135deg, #fff, #94a3b8);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.brand-text span { font-size: 11px; color: var(--cyan); font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
.top-pill {
  display: inline-flex; align-items: center; gap: 8px; font-size: 12px;
  padding: 6px 14px; border-radius: 20px; background: var(--glass);
  border: 1px solid var(--card-border); color: var(--tx-secondary);
}
.pulse-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--success); box-shadow: 0 0 10px var(--success); animation: pulse 2s infinite; }

/* ─ Stat Dashboard Cards ─ */
.stats-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 16px; margin-bottom: 24px;
}
.stat-card {
  background: var(--card); border: 1px solid var(--card-border);
  backdrop-filter: blur(16px); border-radius: var(--radius-md);
  padding: 16px 18px; position: relative; overflow: hidden;
  transition: transform 0.2s, border-color 0.2s;
}
.stat-card:hover { transform: translateY(-2px); border-color: rgba(255,255,255,0.18); }
.stat-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.stat-title { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; color: var(--tx-muted); }
.stat-icon { font-size: 18px; }
.stat-val { font-size: 22px; font-weight: 800; font-family: 'JetBrains Mono', monospace; }
.stat-footer { font-size: 11px; color: var(--tx-secondary); margin-top: 4px; }
.stat-card.active { border-color: var(--accent); box-shadow: 0 0 20px var(--accent-glow); }
.stat-card.ok { border-color: var(--success); }
.stat-card.warn { border-color: var(--warning); }

/* ─ Vision Detector Spotlight Box ─ */
.detector-banner {
  background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(6, 182, 212, 0.08) 100%);
  border: 1px solid rgba(99, 102, 241, 0.3); border-radius: var(--radius-lg);
  padding: 24px; margin-bottom: 28px; box-shadow: 0 8px 32px rgba(0,0,0,0.2);
}
.det-grid { display: grid; grid-template-columns: 1fr 1.2fr; gap: 24px; align-items: start; }
@media (max-width: 860px) { .det-grid { grid-template-columns: 1fr; } }
.det-controls { display: flex; flex-direction: column; gap: 14px; }
.det-result-img {
  width: 100%; max-height: 380px; object-fit: contain; border-radius: var(--radius-md);
  border: 1px solid var(--card-border); background: #000; display: none;
}
.det-tags { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.det-tag {
  background: rgba(255, 255, 255, 0.06); border: 1px solid var(--card-border);
  padding: 6px 12px; border-radius: 20px; font-size: 12px; font-weight: 600;
  display: inline-flex; align-items: center; gap: 6px;
}
.det-tag .count { background: var(--cyan); color: #000; border-radius: 10px; padding: 2px 6px; font-size: 10px; }

/* ─ Main Layout Grid ─ */
.grid-main {
  display: grid; grid-template-columns: 1.15fr 0.85fr; gap: 24px; margin-bottom: 24px;
}
@media (max-width: 920px) { .grid-main { grid-template-columns: 1fr; } }

/* ─ Section Styling ─ */
.panel-box {
  background: var(--card); border: 1px solid var(--card-border);
  backdrop-filter: blur(16px); border-radius: var(--radius-lg);
  padding: 22px; margin-bottom: 24px;
}
.panel-header {
  display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;
}
.panel-header h2 {
  font-size: 15px; font-weight: 700; display: flex; align-items: center; gap: 10px;
}
.step-badge {
  width: 24px; height: 24px; border-radius: 7px; font-size: 11px; font-weight: 800;
  background: linear-gradient(135deg, var(--accent), var(--cyan)); color: #fff;
  display: inline-flex; align-items: center; justify-content: center;
}

/* ─ Drag & Drop Upload ─ */
.dropzone {
  border: 2px dashed var(--card-border); border-radius: var(--radius-md);
  padding: 26px 16px; text-align: center; cursor: pointer; transition: all 0.3s;
  background: var(--glass);
}
.dropzone:hover, .dropzone.dragover {
  border-color: var(--accent); background: var(--accent-glow);
}
.dz-icon { font-size: 32px; margin-bottom: 6px; opacity: 0.8; }
.dz-text { font-size: 13px; font-weight: 600; color: var(--tx-primary); }
.dz-sub { font-size: 11px; color: var(--tx-muted); margin-top: 2px; }

/* ─ Pair Rows List ─ */
.pairs-container { display: flex; flex-direction: column; gap: 10px; margin-top: 14px; }
.pair-item {
  background: rgba(255, 255, 255, 0.02); border: 1px solid var(--card-border);
  border-radius: var(--radius-md); padding: 12px; display: grid;
  grid-template-columns: 1fr 1fr auto; gap: 12px; align-items: center;
}
.pair-side label {
  font-size: 10.5px; font-weight: 600; text-transform: uppercase; color: var(--tx-muted);
  display: block; margin-bottom: 4px;
}
.pair-side input[type=file] { width: 100%; font-size: 11px; color: var(--tx-secondary); }
.pair-side input[type=file]::file-selector-button {
  background: rgba(255, 255, 255, 0.08); border: 1px solid var(--card-border);
  color: var(--tx-primary); border-radius: 6px; padding: 5px 10px; font-size: 11px;
  cursor: pointer; margin-right: 6px; transition: background 0.2s;
}
.thumb-preview { width: 44px; height: 44px; border-radius: 6px; object-fit: cover; margin-top: 4px; display: none; border: 1px solid var(--card-border); }
.btn-del-pair {
  background: transparent; border: 1px solid var(--card-border); color: var(--danger);
  border-radius: 8px; width: 32px; height: 32px; cursor: pointer; display: flex;
  align-items: center; justify-content: center; font-size: 13px; transition: all 0.2s;
}
.btn-del-pair:hover { background: var(--danger-bg); border-color: var(--danger); }

/* ─ Hyperparameter Controls ─ */
.param-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 14px; }
.param-field label { font-size: 10.5px; font-weight: 600; text-transform: uppercase; color: var(--tx-muted); display: block; margin-bottom: 4px; }
.param-input, .param-select {
  width: 100%; background: rgba(255, 255, 255, 0.04); border: 1px solid var(--card-border);
  color: var(--tx-primary); padding: 8px 12px; border-radius: var(--radius-sm); font-size: 12.5px;
  font-family: inherit; outline: none; transition: border-color 0.2s;
}
.param-input:focus, .param-select:focus { border-color: var(--accent); }

/* ─ Action Buttons ─ */
.btn-group { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 14px; }
.btn {
  padding: 10px 20px; border-radius: var(--radius-sm); font-size: 12.5px; font-weight: 700;
  cursor: pointer; border: 0; display: inline-flex; align-items: center; gap: 6px;
  transition: all 0.2s;
}
.btn-main {
  background: linear-gradient(135deg, var(--accent), var(--cyan)); color: #fff;
  box-shadow: 0 4px 16px var(--accent-glow); flex: 1; justify-content: center;
}
.btn-main:hover { transform: translateY(-1px); box-shadow: 0 6px 24px var(--accent-glow); }
.btn-main:disabled { opacity: 0.4; cursor: not-allowed; transform: none; box-shadow: none; }
.btn-outline { background: var(--glass); border: 1px solid var(--card-border); color: var(--tx-primary); }
.btn-outline:hover { background: rgba(255,255,255,0.08); }
.btn-danger-light { background: var(--danger-bg); border: 1px solid rgba(239,68,68,0.3); color: var(--danger); }
.btn-danger-light:hover { background: rgba(239,68,68,0.25); }

/* ─ Playground (Interactive AI Inference) ─ */
.playground-box {
  background: linear-gradient(180deg, rgba(99, 102, 241, 0.04) 0%, rgba(6, 182, 212, 0.02) 100%);
  border: 1px solid rgba(99, 102, 241, 0.2); border-radius: var(--radius-lg); padding: 20px;
}
.compare-container {
  position: relative; width: 100%; height: 230px; border-radius: var(--radius-md);
  overflow: hidden; background: #000; border: 1px solid var(--card-border); margin: 14px 0;
  display: flex; align-items: center; justify-content: center;
}
.compare-img { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain; }
.compare-overlay {
  position: absolute; top: 0; left: 0; width: 50%; height: 100%; overflow: hidden;
  border-right: 2px solid #fff; box-shadow: 2px 0 10px rgba(0,0,0,0.5);
}
.compare-overlay img { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain; }
.slider-handle {
  position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
  width: 28px; height: 28px; border-radius: 50%; background: #fff; color: #000;
  display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 800;
  pointer-events: none; box-shadow: 0 0 15px rgba(0,0,0,0.8);
}
.compare-range {
  position: absolute; top: 0; left: 0; width: 100%; height: 100%; opacity: 0; cursor: ew-resize; z-index: 10;
}
.play-placeholder { text-align: center; color: var(--tx-muted); font-size: 12px; }

/* ─ Terminal Log & Chart ─ */
.terminal-area {
  background: #04060a; border: 1px solid var(--card-border); border-radius: var(--radius-md);
  padding: 14px; font-family: 'JetBrains Mono', monospace; font-size: 11.5px; color: #38bdf8;
  max-height: 260px; overflow-y: auto; white-space: pre-wrap; line-height: 1.5;
}
.terminal-area::-webkit-scrollbar { width: 6px; }
.terminal-area::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }

/* ─ Toast Notifications ─ */
.toasts { position: fixed; top: 20px; right: 20px; z-index: 9999; display: flex; flex-direction: column; gap: 8px; }
.toast {
  padding: 10px 18px; border-radius: var(--radius-sm); font-size: 12.5px; font-weight: 600;
  backdrop-filter: blur(16px); border: 1px solid var(--card-border);
  box-shadow: 0 10px 30px rgba(0,0,0,0.5); animation: toastIn 0.3s ease;
}
.toast.ok { background: var(--success-bg); border-color: rgba(16,185,129,0.4); color: #34d399; }
.toast.warn { background: var(--warning-bg); border-color: rgba(245,158,11,0.4); color: #fbbf24; }
.toast.err { background: var(--danger-bg); border-color: rgba(239,68,68,0.4); color: #f87171; }
@keyframes toastIn { from { transform: translateX(50px); opacity: 0; } to { transform: translateX(0); opacity: 1; } }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
</style>
</head>
<body>

<div class="toasts" id="toasts"></div>

<div class="wrapper">

  <!-- ─ Navbar ─ -->
  <div class="navbar">
    <div class="brand">
      <div class="brand-icon">🚀</div>
      <div class="brand-text">
        <h1>ISBD Studio Pro Vision</h1>
        <span>AI Image Restoration & Object Recognition Engine</span>
      </div>
    </div>
    <div class="top-pill">
      <span class="pulse-dot"></span>
      <span id="nav-status">Vision & Trainer Live</span>
    </div>
  </div>

  <!-- ─ Stats Row ─ -->
  <div class="stats-grid">
    <div class="stat-card" id="card-step">
      <div class="stat-top">
        <span class="stat-title">24/7 Continuous Step</span>
        <span class="stat-icon">🔥</span>
      </div>
      <div class="stat-val" id="v-step">—</div>
      <div class="stat-footer">মোট ট্রেইন্ড আইটারেশন</div>
    </div>

    <div class="stat-card" id="card-loss">
      <div class="stat-top">
        <span class="stat-title">Training Loss</span>
        <span class="stat-icon">📉</span>
      </div>
      <div class="stat-val" id="v-loss">—</div>
      <div class="stat-footer">L1 + MSE কম্পোজিট লস</div>
    </div>

    <div class="stat-card" id="card-pairs">
      <div class="stat-top">
        <span class="stat-title">Custom Pairs</span>
        <span class="stat-icon">📦</span>
      </div>
      <div class="stat-val" id="v-pairs">0</div>
      <div class="stat-footer">ডিজাইনার রিয়েল পেয়ার</div>
    </div>

    <div class="stat-card" id="card-lock">
      <div class="stat-top">
        <span class="stat-title">Engine Lock</span>
        <span class="stat-icon">🔒</span>
      </div>
      <div class="stat-val" id="v-lock">Free</div>
      <div class="stat-footer" id="v-lock-sub">ট্রেনিং সেফটি কন্ট্রোল</div>
    </div>
  </div>

  <!-- ─ NEW FEATURE: Object Recognition & Visual Tracking Spotlight ─ -->
  <div class="detector-banner">
    <div class="panel-header">
      <h2><span class="step-badge">👁️</span> এআই অবজেক্ট ট্র্যাকিং ও আইডেন্টিফায়ার (Object Recognition)</h2>
      <span style="font-size: 11px; color: var(--cyan); font-weight: 600;">COCO 80+ Classes (বাংলা নাম সহ)</span>
    </div>
    <p style="font-size: 12.5px; color: var(--tx-secondary); margin-bottom: 16px;">
      যেকোনো ছবি আপলোড করুন — এআই নিজে নিজে ছবিতে থাকা মানুষ, গাড়ি, পশু-পাখি, ফোন, ল্যাপটপ ইত্যাদি সব অবজেক্ট চিনে বাউন্ডিং বক্স সহ বাংলায় চিহ্নিত করবে।
    </p>

    <div class="det-grid">
      <div class="det-controls">
        <input type="file" id="det-file" accept="image/*" style="display:none" onchange="runObjectDetection(this)">
        <button class="btn btn-main" style="width: 100%;" onclick="document.getElementById('det-file').click()">
          📷 ছবি আপলোড করে অবজেক্ট স্ক্যান করুন
        </button>
        <div style="display: flex; justify-content: space-between; align-items: center; font-size: 11px; color: var(--tx-muted);">
          <span>Confidence থ্রেশহোল্ড:</span>
          <select class="param-select" id="det-conf" style="width: 120px; padding: 4px 8px;" onchange="reScanDet()">
            <option value="0.15">15% (বেশি অবজেক্ট)</option>
            <option value="0.25" selected>25% (স্ট্যান্ডার্ড)</option>
            <option value="0.45">45% (হাই অ্যাকুরেসি)</option>
          </select>
        </div>
        <div id="det-summary-box" style="display:none;">
          <span style="font-size: 12px; font-weight: 700; color: var(--tx-primary);">শনাক্তকৃত অবজেক্ট সমূহ:</span>
          <div class="det-tags" id="det-tags"></div>
        </div>
      </div>

      <div>
        <div id="det-placeholder" style="border: 2px dashed var(--card-border); border-radius: var(--radius-md); padding: 36px; text-align: center; color: var(--tx-muted); font-size: 12px;">
          ছবি স্ক্যান করার পর এখানে বাউন্ডিং বক্স সহ ভিজ্যুয়ালাইজেশন দেখতে পাবেন
        </div>
        <img id="det-result" class="det-result-img">
      </div>
    </div>
  </div>

  <!-- ─ Main Section: Upload & Controls ─ -->
  <div class="grid-main">
    
    <!-- Left Column: Fine-tune & Pairs -->
    <div>
      <div class="panel-box">
        <div class="panel-header">
          <h2><span class="step-badge">1</span> ডিজাইনার পেয়ার আপলোড</h2>
          <button class="btn btn-outline" style="padding: 6px 14px; font-size: 11.5px;" onclick="addPairRow()">+ নতুন পেয়ার</button>
        </div>

        <div class="dropzone" id="dz" onclick="addPairRow()">
          <div class="dz-icon">📂</div>
          <div class="dz-text">ক্লিক করুন বা ছবি ড্র্যাগ করে আনুন</div>
          <div class="dz-sub">ফটোশপের মূল ছবি (Before) এবং এডিটেড ছবি (After) দিন</div>
        </div>

        <div class="pairs-container" id="pairs-list"></div>
      </div>

      <div class="panel-box">
        <div class="panel-header">
          <h2><span class="step-badge">2</span> অ্যাডভান্সড ফাইন-টিউন কন্ট্রোল</h2>
        </div>

        <div class="param-grid">
          <div class="param-field">
            <label>ট্রেনিং ধাপ (Steps)</label>
            <select class="param-select" id="p-steps">
              <option value="100">১০০ ধাপ (কুইক ড্রাফট)</option>
              <option value="300" selected>৩০০ ধাপ (স্ট্যান্ডার্ড)</option>
              <option value="600">৬০০ ধাপ (গভীর টিউনিং)</option>
              <option value="1200">১২০০ ধাপ (ম্যাক্সিমাম পারফেকশন)</option>
            </select>
          </div>
          <div class="param-field">
            <label>লার্নিং রেট (Learning Rate)</label>
            <select class="param-select" id="p-lr">
              <option value="0.00005">0.00005 (খুব সূক্ষ্ম)</option>
              <option value="0.0001" selected>0.0001 (ব্যালেন্সড)</option>
              <option value="0.0003">0.0003 (দ্রুত অ্যাডাপ্টেশন)</option>
            </select>
          </div>
          <div class="param-field">
            <label>ব্যাচ সাইজ (Batch Size)</label>
            <select class="param-select" id="p-batch">
              <option value="4">4 (লাইটওয়েট)</option>
              <option value="8" selected>8 (সুপারফাস্ট 4-থ্রেড)</option>
              <option value="16">16 (হাই থ্রুপুট)</option>
            </select>
          </div>
          <div class="param-field">
            <label>রিয়েল ডেটা রেশিও</label>
            <input class="param-input" type="text" value="40% Real + 60% Synthetic" disabled>
          </div>
        </div>

        <div class="btn-group">
          <button class="btn btn-main" id="btn-train" onclick="startTraining()">⚡ প্রসেস ও ফাইন-টিউন শুরু</button>
          <button class="btn btn-danger-light" onclick="purgeData()">🗑 ডেটা মুছুন</button>
        </div>
      </div>
    </div>

    <!-- Right Column: Interactive Inference Playground & Specs -->
    <div>
      <div class="panel-box playground-box">
        <div class="panel-header">
          <h2><span class="step-badge">⚡</span> লাইভ এআই রেস্টোরেশন ল্যাব</h2>
        </div>
        <p style="font-size: 11.5px; color: var(--tx-secondary); margin-bottom: 10px;">
          যেকোনো সাধারণ বা নষ্ট ছবি দিন — বর্তমান ট্রেন হওয়া এআই মডেল রিয়েল-টাইমে রিস্টোর করবে।
        </p>

        <input type="file" id="play-input" accept="image/*" style="display:none" onchange="runInference(this)">
        <button class="btn btn-outline" style="width: 100%; justify-content: center;" onclick="document.getElementById('play-input').click()">
          📸 ছবি সিলেক্ট করে টেস্ট করুন
        </button>

        <div class="compare-container" id="comp-box">
          <div class="play-placeholder" id="play-placeholder">ছবি আপলোড করলে এখানে Before/After স্লাইডার দেখতে পাবেন</div>
          <img id="comp-after" class="compare-img" style="display:none">
          <div class="compare-overlay" id="comp-overlay" style="display:none">
            <img id="comp-before">
          </div>
          <div class="slider-handle" id="comp-handle" style="display:none">⬍</div>
          <input type="range" class="compare-range" id="comp-range" min="0" max="100" value="50" style="display:none" oninput="slideCompare(this.value)">
        </div>
        <div style="display:flex; justify-content:space-between; font-size:11px; color:var(--tx-muted);" id="comp-labels" style="display:none">
          <span>◀ Original (Before)</span>
          <span>AI Restored (After) ▶</span>
        </div>
      </div>

      <!-- Engine Logs -->
      <div class="panel-box">
        <div class="panel-header">
          <h2><span class="step-badge">📊</span> লাইভ ইঞ্জিন লগ</h2>
          <span id="badge-ft" style="font-size: 11px; padding: 2px 8px; border-radius: 12px; background: rgba(255,255,255,0.05);">Idle</span>
        </div>
        <div class="terminal-area" id="term-log">ইঞ্জিন প্রস্তুত — পেয়ার আপলোড বা ফাইন-টিউন ট্র্যাকিং…</div>
      </div>
    </div>

  </div>

</div>

<script>
let lastUploadedDetFile = null;

function toast(msg, type='ok') {
  const c = document.getElementById('toasts');
  const t = document.createElement('div');
  t.className = 'toast ' + type;
  t.textContent = msg;
  c.appendChild(t);
  setTimeout(() => t.remove(), 4000);
}

// ─ Object Detection Scanner ─
async function runObjectDetection(input) {
  const file = input.files ? input.files[0] : input;
  if (!file) return;
  lastUploadedDetFile = file;

  toast('এআই অবজেক্ট স্ক্যান করছে…', 'ok');
  const fd = new FormData();
  fd.append('image', file);
  const conf = document.getElementById('det-conf').value;

  try {
    const res = await fetch(`/api/detect?conf=${conf}`, { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'ডিটেকশন ফেইল্ড');

    // Display Annotated Image
    const resImg = document.getElementById('det-result');
    resImg.src = data.annotated_image;
    resImg.style.display = 'block';
    document.getElementById('det-placeholder').style.display = 'none';

    // Populate Bengali summary tags
    const tagContainer = document.getElementById('det-tags');
    tagContainer.innerHTML = '';
    for (const [bnName, count] of Object.entries(data.summary)) {
      const tag = document.createElement('div');
      tag.className = 'det-tag';
      tag.innerHTML = `<span>${bnName}</span> <span class="count">${count}</span>`;
      tagContainer.appendChild(tag);
    }
    document.getElementById('det-summary-box').style.display = 'block';
    toast(`মোট ${data.total_objects}টি অবজেক্ট চিহ্নিত হয়েছে!`, 'ok');
  } catch(e) {
    toast('অবজেক্ট স্ক্যানিং ত্রুটি: ' + e.message, 'err');
  }
}

function reScanDet() {
  if (lastUploadedDetFile) {
    runObjectDetection(lastUploadedDetFile);
  }
}

// ─ Drag and drop ─
const dz = document.getElementById('dz');
['dragenter','dragover'].forEach(e => dz.addEventListener(e, ev => { ev.preventDefault(); dz.classList.add('dragover'); }));
['dragleave','drop'].forEach(e => dz.addEventListener(e, ev => { ev.preventDefault(); dz.classList.remove('dragover'); }));
dz.addEventListener('drop', ev => {
  ev.preventDefault();
  const f = ev.dataTransfer.files;
  if (f.length >= 2) {
    addPairRow(f[0], f[1]);
    toast('ড্র্যাগ করা পেয়ার যোগ হয়েছে!', 'ok');
  } else {
    toast('আগে ও পরের দুটি ছবি ড্রপ করুন', 'warn');
  }
});

// ─ Pair Rows ─
function addPairRow(bfFile, afFile) {
  const container = document.getElementById('pairs-list');
  const row = document.createElement('div');
  row.className = 'pair-item';
  row.innerHTML = `
    <div class="pair-side">
      <label>আসল ছবি (Before)</label>
      <input type="file" accept="image/*" onchange="previewThumb(this)">
      <img class="thumb-preview">
    </div>
    <div class="pair-side">
      <label>ডিজাইনার এডিট (After)</label>
      <input type="file" accept="image/*" onchange="previewThumb(this)">
      <img class="thumb-preview">
    </div>
    <button class="btn-del-pair" onclick="this.closest('.pair-item').remove()" title="মুছুন">✕</button>
  `;
  container.appendChild(row);

  if (bfFile) {
    const dt = new DataTransfer(); dt.items.add(bfFile);
    row.querySelectorAll('input')[0].files = dt.files;
    previewThumb(row.querySelectorAll('input')[0]);
  }
  if (afFile) {
    const dt = new DataTransfer(); dt.items.add(afFile);
    row.querySelectorAll('input')[1].files = dt.files;
    previewThumb(row.querySelectorAll('input')[1]);
  }
}

function previewThumb(inp) {
  const file = inp.files[0];
  if (!file) return;
  const img = inp.parentNode.querySelector('.thumb-preview');
  img.src = URL.createObjectURL(file);
  img.style.display = 'block';
}

// ─ Fine-tune trigger ─
async function startTraining() {
  const rows = document.querySelectorAll('.pair-item');
  if (!rows.length) { toast('আগে অন্তত একটি পেয়ার যোগ করুন', 'warn'); return; }

  const btn = document.getElementById('btn-train');
  btn.disabled = true;
  btn.textContent = 'আপলোড ও প্রসেসিং হচ্ছে…';

  let successCount = 0;
  for (let r of rows) {
    const [bf, af] = r.querySelectorAll('input[type=file]');
    if (!bf.files[0] || !af.files[0]) continue;

    const fd = new FormData();
    fd.append('before', bf.files[0]);
    fd.append('after', af.files[0]);
    try {
      const res = await fetch('/api/pair', { method: 'POST', body: fd });
      const data = await res.json();
      if (res.ok) successCount++;
    } catch(e) {}
  }

  const steps = document.getElementById('p-steps').value;
  const lr = document.getElementById('p-lr').value;
  const batch = document.getElementById('p-batch').value;

  try {
    const res = await fetch(`/api/train?steps=${steps}&lr=${lr}&batch=${batch}`, { method: 'POST' });
    const j = await res.json();
    if (res.ok) {
      toast(`ফাইন-টিউন কিউড: ${steps} ধাপ, lr: ${lr}`, 'ok');
    } else {
      toast(j.detail || 'ট্রেনিং শুরু করা যায়নি', 'err');
    }
  } catch(e) {
    toast('সার্ভার এরর', 'err');
  }
  btn.disabled = false;
  btn.textContent = '⚡ প্রসেস ও ফাইন-টিউন শুরু';
}

function purgeData() {
  if (!confirm('সব পেয়ার ডেটা মুছে ফেলবেন?')) return;
  fetch('/api/purge', { method: 'POST' }).then(r => r.json()).then(() => {
    toast('সব পেয়ার টেনসর মুছে ফেলা হয়েছে', 'ok');
    document.getElementById('pairs-list').innerHTML = '';
  });
}

// ─ Live Inference Lab ─
async function runInference(input) {
  const file = input.files[0];
  if (!file) return;

  toast('এআই মডেল রিস্টোর করছে…', 'ok');
  const fd = new FormData();
  fd.append('image', file);

  try {
    const res = await fetch('/api/infer', { method: 'POST', body: fd });
    if (!res.ok) throw new Error('Inference error');
    const blob = await res.blob();
    const restoredUrl = URL.createObjectURL(blob);
    const origUrl = URL.createObjectURL(file);

    document.getElementById('play-placeholder').style.display = 'none';
    const aft = document.getElementById('comp-after');
    const bef = document.getElementById('comp-before');
    aft.src = restoredUrl; aft.style.display = 'block';
    bef.src = origUrl;
    document.getElementById('comp-overlay').style.display = 'block';
    document.getElementById('comp-handle').style.display = 'flex';
    document.getElementById('comp-range').style.display = 'block';
    slideCompare(50);
  } catch(e) {
    toast('রিস্টোরেশনে ব্যর্থ', 'err');
  }
}

function slideCompare(val) {
  document.getElementById('comp-overlay').style.width = val + '%';
  document.getElementById('comp-handle').style.left = val + '%';
}

// ─ Polling ─
async function poll() {
  try {
    const res = await fetch('/api/status');
    const j = await res.json();

    document.getElementById('v-step').textContent = j.live.step ? j.live.step.toLocaleString() : '—';
    document.getElementById('v-loss').textContent = j.live.loss ? j.live.loss.toFixed(4) : '—';
    document.getElementById('v-pairs').textContent = j.pairs;

    const lock = document.getElementById('v-lock');
    lock.textContent = j.lock_busy ? 'Busy (Training)' : 'Free';
    document.getElementById('card-lock').className = 'stat-card ' + (j.lock_busy ? 'warn' : 'ok');

    const badge = document.getElementById('badge-ft');
    if (j.ft.running) {
      badge.textContent = 'Running Fine-tune…';
      badge.style.background = 'rgba(99,102,241,0.2)';
      badge.style.color = '#818cf8';
    } else {
      badge.textContent = 'Idle';
      badge.style.background = 'rgba(255,255,255,0.05)';
      badge.style.color = 'var(--tx-muted)';
    }

    if (j.log) {
      document.getElementById('term-log').textContent = j.log;
    }
  } catch(e) {}
}
setInterval(poll, 3000);
poll();
addPairRow();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML
