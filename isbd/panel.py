"""
ISBD v1.00 — Designer Pair Training Panel & Studio Pro Vision Suite
Features:
- Live Dashboard (24/7 step, loss, pairs, lock, fine-tune progress, metrics)
- Pair Upload (Drag & drop, multi-pair, live before/after image preview)
- AI Object Detection & Recognition (YOLOv8 Vision Scanner with Bengali labels)
- OpenCV & Scikit-Image Vision Tool (Canny Edges, CLAHE Contrast, Pencil Sketch, Fast NLM Denoise, Dominant Color Palette)
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

app = FastAPI(title="ISBD Studio Pro Vision Suite")


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


@app.get("/api/services")
async def get_services_api():
    """Get the list of all 11 Commercial Graphic Design & Photo Editing Services."""
    from isbd.studio_sectors import SERVICES_CONFIG
    return {"ok": True, "services": SERVICES_CONFIG}


@app.post("/api/service_process")
async def process_service_api(image: UploadFile = File(...), service_id: str = Query(...)):
    """Execute any of the 11 Commercial Studio Services on the uploaded photo."""
    raw = await image.read()
    if not raw:
        raise HTTPException(400, "ছবি পাওয়া যায়নি")
    try:
        from isbd.studio_sectors import execute_studio_service
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        
        out_img, desc = execute_studio_service(img, service_id)

        buf = io.BytesIO()
        # If RGBA, save as PNG, else JPEG
        fmt = "PNG" if out_img.mode == "RGBA" else "JPEG"
        out_img.save(buf, format=fmt)
        buf.seek(0)
        img_b64 = f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.read()).decode("utf-8")

        return {
            "ok": True,
            "service_id": service_id,
            "description": desc,
            "processed_image": img_b64
        }
    except Exception as e:
        raise HTTPException(500, f"সার্ভিস প্রসেসিং ত্রুটি: {str(e)}")


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


@app.post("/api/cv_filter")
async def cv_filter_api(image: UploadFile = File(...), filter_type: str = Query("canny")):
    """Apply OpenCV & Scikit-Image Computer Vision algorithms & Color Palette."""
    raw = await image.read()
    if not raw:
        raise HTTPException(400, "ছবি পাওয়া যায়নি")
    try:
        from isbd.detector import apply_cv_filter, analyze_image_colors
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        
        filtered_img, desc = apply_cv_filter(img, filter_type)
        palette = analyze_image_colors(img, num_colors=5)

        buf = io.BytesIO()
        filtered_img.save(buf, format="JPEG", quality=85)
        buf.seek(0)
        img_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.read()).decode("utf-8")

        return {
            "ok": True,
            "filter_applied": filter_type,
            "description": desc,
            "palette": palette,
            "filtered_image": img_b64
        }
    except Exception as e:
        raise HTTPException(500, f"ফিল্টার প্রয়োগে ত্রুটি: {str(e)}")


@app.post("/api/infer")
async def infer_image(image: UploadFile = File(...)):
    """Live AI Image Restoration Playground via current trained model."""
    raw = await image.read()
    if not raw:
        raise HTTPException(400, "ফাইল পাওয়া যায়নি")
    try:
        from isbd.model import TinyUNet
        from isbd.self_learner import harvest_and_synthesize_pair
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        
        # Autonomous Self-Supervised Learning trigger: Learn from this image in background
        try:
            harvest_and_synthesize_pair(img)
        except Exception:
            pass

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


@app.post("/api/heatmap")
async def heatmap_api(
    before: UploadFile = File(...),
    after: UploadFile = File(...),
    colormap: str = Query("turbo"),
    blend: float = Query(0.55, ge=0.1, le=0.9),
    sensitivity: float = Query(1.5, ge=0.5, le=5.0)
):
    """Generate precise per-pixel Difference Heatmap between Before and After images."""
    raw_b = await before.read()
    raw_a = await after.read()
    if not raw_b or not raw_a:
        raise HTTPException(400, "Before এবং After দুটি ছবিই প্রদান করুন")
    try:
        from isbd.heatmap import generate_difference_heatmap
        img_b = Image.open(io.BytesIO(raw_b)).convert("RGB")
        img_a = Image.open(io.BytesIO(raw_a)).convert("RGB")

        hm_img, ov_img, stats = generate_difference_heatmap(
            img_b, img_a, colormap_type=colormap, blend_alpha=blend, boost_sensitivity=sensitivity
        )

        buf_hm, buf_ov = io.BytesIO(), io.BytesIO()
        hm_img.save(buf_hm, format="JPEG", quality=85)
        ov_img.save(buf_ov, format="JPEG", quality=85)
        buf_hm.seek(0)
        buf_ov.seek(0)

        return {
            "ok": True,
            "stats": stats,
            "heatmap_image": "data:image/jpeg;base64," + base64.b64encode(buf_hm.read()).decode("utf-8"),
            "overlay_image": "data:image/jpeg;base64," + base64.b64encode(buf_ov.read()).decode("utf-8")
        }
    except Exception as e:
        raise HTTPException(500, f"হিটম্যাপ জেনারেশন ত্রুটি: {str(e)}")


@app.post("/api/inpaint")
async def inpaint_api(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    radius: int = Query(5, ge=1, le=25),
    method: str = Query("telea")
):
    """Erase unwanted objects, blemishes or watermarks using AI Brush Mask."""
    raw_img = await image.read()
    raw_mask = await mask.read()
    if not raw_img or not raw_mask:
        raise HTTPException(400, "ছবি এবং ব্রাশ মাস্ক প্রদান করুন")
    try:
        from isbd.inpaint import inpaint_image
        img = Image.open(io.BytesIO(raw_img)).convert("RGB")
        mask_img = Image.open(io.BytesIO(raw_mask)).convert("L")

        res_img = inpaint_image(img, mask_img, radius=radius, method=method)

        buf = io.BytesIO()
        res_img.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(500, f"ইনপেইন্টিং ত্রুটি: {str(e)}")


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
        "self_learn": __import__("isbd.self_learner", fromlist=["get_self_learn_stats"]).get_self_learn_stats(),
    }


# ── Modern UI ────────────────────────────────────────────────────────────────
HTML = """<!doctype html>
<html lang="bn">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ISBD Studio Pro Vision Suite</title>
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
  padding: 24px; margin-bottom: 24px; box-shadow: 0 8px 32px rgba(0,0,0,0.2);
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

/* ─ Computer Vision Lab Banner ─ */
.cv-lab-banner {
  background: var(--card); border: 1px solid var(--card-border);
  backdrop-filter: blur(16px); border-radius: var(--radius-lg);
  padding: 22px; margin-bottom: 24px;
}
.cv-palette { display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap; }
.palette-chip {
  padding: 6px 10px; border-radius: 6px; font-size: 11px; font-weight: 700;
  display: flex; align-items: center; gap: 6px; border: 1px solid rgba(255,255,255,0.1);
}

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
        <h1>ISBD Studio Pro Vision Suite</h1>
        <span>AI Image Restoration, Object Recognition & Computer Vision Toolkit</span>
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
        <span class="stat-title">Self-Learned & Custom</span>
        <span class="stat-icon">🧠</span>
      </div>
      <div class="stat-val" id="v-pairs">0</div>
      <div class="stat-footer" id="v-self-sub">অটোনোমাস সেলফ-লার্নিং সক্রিয়</div>
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

  <!-- ─ MASTER SECTOR 2: AI Difference Heatmap Visualizer ─ -->
  <div class="cv-lab-banner" style="background: linear-gradient(135deg, rgba(245, 158, 11, 0.08) 0%, rgba(239, 68, 68, 0.06) 100%); border-color: rgba(245, 158, 11, 0.3);">
    <div class="panel-header">
      <h2><span class="step-badge" style="background: linear-gradient(135deg, #f59e0b, #ef4444);">🔥</span> এআই বিফোর/আফটার হিটম্যাপ ভিজ্যুয়ালাইজার (Pixel Difference Heatmap)</h2>
      <span style="font-size: 11px; color: #fbbf24; font-weight: 700;">Thermal Error & Modification Mapping</span>
    </div>
    <p style="font-size: 12.5px; color: var(--tx-secondary); margin-bottom: 16px;">
      অনেক সময় খালি চোখে বিফোর ও আফটার ছবির সূক্ষ্ম পার্থক্য বোঝা যায় না — এআই প্রতিটি পিক্সেলের পার্থক্য থার্মাল হিটম্যাপে (উজ্জ্বল লাল/হলুদ দিয়ে পরিবর্তিত অংশ) স্পষ্ট ফুটিয়ে তোলে।
    </p>

    <div class="det-grid">
      <div class="det-controls">
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
          <div>
            <label style="font-size:11px; font-weight:600; color:var(--tx-muted); display:block; margin-bottom:4px;">আসল ছবি (Before):</label>
            <input type="file" id="hm-before" accept="image/*" class="param-input" style="padding:4px;">
          </div>
          <div>
            <label style="font-size:11px; font-weight:600; color:var(--tx-muted); display:block; margin-bottom:4px;">এডিটেড ছবি (After):</label>
            <input type="file" id="hm-after" accept="image/*" class="param-input" style="padding:4px;">
          </div>
        </div>

        <button class="btn btn-main" style="width: 100%; background: linear-gradient(135deg, #f59e0b, #ef4444);" onclick="generateHeatmap()">
          🔥 হিটম্যাপ জেনারেট করুন
        </button>

        <div style="display: flex; justify-content: space-between; align-items: center; font-size: 11px; color: var(--tx-muted);">
          <span>হিটম্যাপ কালার প্যালেট:</span>
          <select class="param-select" id="hm-cmap" style="width: 130px; padding: 4px 8px;" onchange="reGenHeatmap()">
            <option value="turbo" selected>Turbo (হাই-কনট্রাস্ট)</option>
            <option value="jet">Jet (ক্লাসিক রেইনবো)</option>
            <option value="inferno">Inferno (ডার্ক ফায়ার)</option>
            <option value="magma">Magma (ম্যাগমা)</option>
            <option value="hot">Hot (হোয়াইট-হট)</option>
          </select>
        </div>

        <div id="hm-stats-box" style="display:none; background:rgba(255,255,255,0.03); border:1px solid var(--card-border); padding:10px 14px; border-radius:var(--radius-sm); font-size:11.5px;">
          <div style="color:#fbbf24; font-weight:700; margin-bottom:2px;" id="hm-stat-text"></div>
          <div style="color:var(--tx-secondary); font-size:11px;" id="hm-stat-sub"></div>
        </div>
      </div>

      <div>
        <div id="hm-placeholder" style="border: 2px dashed var(--card-border); border-radius: var(--radius-md); padding: 36px; text-align: center; color: var(--tx-muted); font-size: 12px;">
          Before ও After আপলোড করে হিটম্যাপ রান করলে এখানে থার্মাল এনালাইসিস দেখতে পাবেন
        </div>
        <div id="hm-result-wrap" style="display:none; flex-direction:column; gap:8px;">
          <img id="hm-overlay-img" class="det-result-img" style="display:block;">
          <span style="font-size:11px; color:var(--tx-muted); text-align:center;">🔴 লাল/হলুদ = সর্বোচ্চ পরিবর্তন | 🔵 নীল/কালো = অপরিবর্তিত অংশ</span>
        </div>
      </div>
    </div>
  </div>

  <!-- ─ MASTER SECTOR: 11 Professional Studio Editing Services ─ -->
  <div class="cv-lab-banner" style="background: linear-gradient(135deg, rgba(6, 182, 212, 0.1) 0%, rgba(99, 102, 241, 0.08) 100%); border-color: rgba(6, 182, 212, 0.35);">
    <div class="panel-header">
      <h2><span class="step-badge" style="background: linear-gradient(135deg, var(--cyan), var(--accent));">💎</span> প্রফেশনাল ফটো এডিটিং ও গ্রাফিক সার্ভিসেস (Commercial Services Suite)</h2>
      <span style="font-size: 11px; color: var(--cyan); font-weight: 700;">11 Commercial Studio Services</span>
    </div>
    <p style="font-size: 12.5px; color: var(--tx-secondary); margin-bottom: 16px;">
      ডিজিটাল পোস্ট-প্রোডাকশন ও গ্রাফিক এজেন্সির ১১টি ফুল কমার্শিয়াল সার্ভিস — ক্লিপিং পাথ, নেক জয়েন্ট, শ্যাডো মেকিং, রিফ্লেকশন, মাস্কিং ও রাস্টার টু ভেক্টর এক ক্লিকে এক্সিকিউট করুন।
    </p>

    <div class="det-grid">
      <div class="det-controls">
        <input type="file" id="service-file" accept="image/*" style="display:none" onchange="runStudioService(this)">
        <button class="btn btn-main" style="width: 100%;" onclick="document.getElementById('service-file').click()">
          📸 ছবি সিলেক্ট করে সার্ভিস চালান
        </button>

        <div style="display: flex; flex-direction: column; gap: 6px; font-size: 11px; color: var(--tx-muted);">
          <span>কাঙ্ক্ষিত সার্ভিস নির্বাচন করুন:</span>
          <select class="param-select" id="service-select" style="padding: 8px 12px; font-size: 13px;" onchange="reRunStudioService()">
            <option value="clipping_path" selected>✂️ Clipping Path (ব্যাকগ্রাউন্ড কাটআউট)</option>
            <option value="multi_clipping_path">🎨 Multiple Clipping Path (কালার সেগমেন্টেশন)</option>
            <option value="image_masking">🎭 Image Masking (চুল/পশম মাস্কিং)</option>
            <option value="neck_joint">👔 Neck Joint (গোস্ট ম্যানিকুইন কলার)</option>
            <option value="image_retouching">✨ Image Retouching (স্কিন গ্ল্যামার ও স্মুথ)</option>
            <option value="shadow_making">👥 Shadow Making (ড্রপ ও কন্টাক্ট শ্যাডো)</option>
            <option value="reflection">🪞 Reflection (মিরর গ্লসি রিফ্লেকশন)</option>
            <option value="color_correction">🌈 Color Correction (হোয়াইট ব্যালেন্স ও টোন)</option>
            <option value="image_enhancement">🔮 Image Enhancement (এইচডিআর ও শার্পনেস)</option>
            <option value="image_manipulation">🌌 Image Manipulation (ক্রিয়েটিভ লাইটিং)</option>
            <option value="raster_to_vector">📐 Raster To Vector (ভেক্টর লাইন ট্রেসিং)</option>
          </select>
        </div>
        <div id="service-desc" style="font-size: 12px; color: #38bdf8; font-weight:600; margin-top: 4px;"></div>
      </div>

      <div>
        <div id="service-placeholder" style="border: 2px dashed var(--card-border); border-radius: var(--radius-md); padding: 36px; text-align: center; color: var(--tx-muted); font-size: 12px;">
          ছবি আপলোড করে সার্ভিস সিলেক্ট করলে এখানে প্রসেসড রেজাল্ট দেখতে পাবেন
        </div>
        <img id="service-result" class="det-result-img">
      </div>
    </div>
  </div>

  <!-- ─ NEW FEATURE 1: Object Recognition & Visual Tracking Spotlight ─ -->
  <div class="detector-banner">
    <div class="panel-header">
      <h2><span class="step-badge">👁️</span> এআই অবজেক্ট ট্র্যাকিং ও আইডেন্টিফায়ার (Object Recognition)</h2>
      <span style="font-size: 11px; color: var(--cyan); font-weight: 600;">YOLOv8 Engine (বাংলা নাম সহ)</span>
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

  <!-- ─ NEW FEATURE 2: OpenCV & Scikit-Image Multi-Vision Toolkit ─ -->
  <div class="cv-lab-banner">
    <div class="panel-header">
      <h2><span class="step-badge">🧪</span> কম্পিউটার ভিশন ও কালার প্যালেট ল্যাব (OpenCV & Scikit-Image)</h2>
      <span style="font-size: 11px; color: var(--accent); font-weight: 600;">Edge Detection, CLAHE, Denoising, Palette</span>
    </div>
    <p style="font-size: 12.5px; color: var(--tx-secondary); margin-bottom: 16px;">
      ওপেনসিভি ও সাইকিট-ইমেজ লাইব্রেরির মাধ্যমে ফিল্টারিং, এজ ডিটেকশন, ডিনয়েজ এবং ডমিন্যান্ট কালার প্যালেট এক্সট্রাকশন।
    </p>

    <div class="det-grid">
      <div class="det-controls">
        <input type="file" id="cv-file" accept="image/*" style="display:none" onchange="runCVFilter(this)">
        <button class="btn btn-outline" style="width: 100%; justify-content: center;" onclick="document.getElementById('cv-file').click()">
          🎨 ছবি সিলেক্ট করে প্রসেস করুন
        </button>
        <div style="display: flex; justify-content: space-between; align-items: center; font-size: 11px; color: var(--tx-muted);">
          <span>প্রি-ট্রেইন্ড এডিটিং ইঞ্জিন:</span>
          <select class="param-select" id="cv-type" style="width: 220px; padding: 5px 8px;" onchange="reApplyCV()">
            <option value="hdr" selected>✨ HDR Detail Enhancement</option>
            <option value="wb">🎨 Auto White Balance (Fix Colors)</option>
            <option value="sharpen">🔪 Unsharp Masking (Sharpen)</option>
            <option value="smooth">🫧 Edge-Preserving Smooth</option>
            <option value="stylize">🖌️ Artistic Water-Color</option>
            <option value="sketch">✏️ AI Pencil Sketch</option>
            <option value="clahe">⚡ CLAHE Contrast Boost</option>
            <option value="denoise">🧼 Fast NLM Denoising</option>
            <option value="tv_denoise">🌊 Total Variation Denoise</option>
            <option value="canny">📐 Canny Edge & Contour</option>
          </select>
        </div>
        <div id="cv-desc" style="font-size: 11.5px; color: var(--cyan); margin-top: 4px;"></div>
        <div id="cv-palette-box" style="display:none;">
          <span style="font-size: 11px; font-weight: 700; color: var(--tx-primary);">ছবির প্রধান কালার প্যালেট (Dominant Colors):</span>
          <div class="cv-palette" id="cv-palette"></div>
        </div>
      </div>

      <div>
        <div id="cv-placeholder" style="border: 2px dashed var(--card-border); border-radius: var(--radius-md); padding: 36px; text-align: center; color: var(--tx-muted); font-size: 12px;">
          কম্পিউটার ভিশন ফিল্টার প্রসেসের পর এখানে আউটপুট দেখতে পাবেন
        </div>
        <img id="cv-result" class="det-result-img">
      </div>
    </div>
  </div>

  <!-- ─ NEW FEATURE 3: Magic Eraser & Inpainting Brush Studio ─ -->
  <div class="cv-lab-banner" style="background: linear-gradient(135deg, rgba(236, 72, 153, 0.08) 0%, rgba(99, 102, 241, 0.06) 100%); border-color: rgba(236, 72, 153, 0.25);">
    <div class="panel-header">
      <h2><span class="step-badge" style="background: linear-gradient(135deg, #ec4899, #8b5cf6);">🪄</span> ম্যাজিক ইরেজার ও এআই অবজেক্ট রিমুভার (Inpainting Brush)</h2>
      <span style="font-size: 11px; color: #f472b6; font-weight: 600;">Interactive Mask Drawing Studio</span>
    </div>
    <p style="font-size: 12.5px; color: var(--tx-secondary); margin-bottom: 16px;">
      ছবি আপলোড করে ব্রাশ দিয়ে যে অংশ মুছতে চান (ওয়াটারমার্ক, দাগ বা অনাকাঙ্ক্ষিত অবজেক্ট) তার ওপর আঁকুন — এআই ব্যাকগ্রাউন্ডের সাথে ম্যাচ করে তা অদৃশ্য করে দেবে!
    </p>

    <div class="det-grid">
      <div class="det-controls">
        <input type="file" id="inpaint-file" accept="image/*" style="display:none" onchange="loadInpaintImage(this)">
        <button class="btn btn-main" style="width: 100%; background: linear-gradient(135deg, #ec4899, #8b5cf6);" onclick="document.getElementById('inpaint-file').click()">
          🖼️ ছবি আপলোড করে ব্রাশ শুরু করুন
        </button>
        <div style="display: flex; gap: 10px; align-items: center; font-size: 11px; color: var(--tx-muted);">
          <span>ব্রাশ সাইজ:</span>
          <input type="range" id="brush-size" min="5" max="50" value="20" style="flex:1;">
          <span id="brush-val">20px</span>
        </div>
        <div style="display: flex; gap: 10px;">
          <button class="btn btn-main" style="flex:1;" onclick="applyInpaint()">✨ মুছে ফেলুন (Erase)</button>
          <button class="btn btn-outline" onclick="clearInpaintMask()">রি-সেট ব্রাশ</button>
        </div>
      </div>

      <div>
        <div id="inpaint-canvas-wrap" style="position:relative; display:none; max-width:100%; border:1px solid var(--card-border); border-radius:var(--radius-md); overflow:hidden; background:#000;">
          <canvas id="inpaint-canvas" style="max-width:100%; height:auto; cursor:crosshair; display:block;"></canvas>
        </div>
        <div id="inpaint-placeholder" style="border: 2px dashed var(--card-border); border-radius: var(--radius-md); padding: 36px; text-align: center; color: var(--tx-muted); font-size: 12px;">
          ছবি লোড হলে এখানে ব্রাশ দিয়ে মার্ক করার ক্যানভাস দেখতে পাবেন
        </div>
        <img id="inpaint-result" class="det-result-img" style="margin-top:12px;">
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
let lastUploadedCVFile = null;

function toast(msg, type='ok') {
  const c = document.getElementById('toasts');
  const t = document.createElement('div');
  t.className = 'toast ' + type;
  t.textContent = msg;
  c.appendChild(t);
  setTimeout(() => t.remove(), 4000);
}

// ─ Heatmap Difference Logic ─
async function generateHeatmap() {
  const fileB = document.getElementById('hm-before').files[0];
  const fileA = document.getElementById('hm-after').files[0];

  if (!fileB || !fileA) {
    toast('Before এবং After দুটি ছবিই সিলেক্ট করুন', 'warn');
    return;
  }

  toast('এআই পিক্সেল ডিফারেন্স হিটম্যাপ ক্যালকুলেট করছে…', 'ok');

  const fd = new FormData();
  fd.append('before', fileB);
  fd.append('after', fileA);
  const cmap = document.getElementById('hm-cmap').value;

  try {
    const res = await fetch(`/api/heatmap?colormap=${cmap}`, { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'হিটম্যাপ ফেইল্ড');

    const ovImg = document.getElementById('hm-overlay-img');
    ovImg.src = data.overlay_image;
    document.getElementById('hm-result-wrap').style.display = 'flex';
    document.getElementById('hm-placeholder').style.display = 'none';

    document.getElementById('hm-stat-text').textContent = '✓ ' + data.stats.interpretation;
    document.getElementById('hm-stat-sub').textContent = `Mean Pixel Diff: ${data.stats.mean_diff} | Max Diff: ${data.stats.max_diff} | Modified Area: ${data.stats.altered_pct}%`;
    document.getElementById('hm-stats-box').style.display = 'block';

    toast('হিটম্যাপ তৈরি সম্পন্ন হয়েছে!', 'ok');
  } catch(e) {
    toast('হিটম্যাপ ত্রুটি: ' + e.message, 'err');
  }
}

function reGenHeatmap() {
  const fileB = document.getElementById('hm-before').files[0];
  const fileA = document.getElementById('hm-after').files[0];
  if (fileB && fileA) generateHeatmap();
}

let lastUploadedServiceFile = null;

async function runStudioService(input) {
  const file = input.files ? input.files[0] : input;
  if (!file) return;
  lastUploadedServiceFile = file;

  const serviceId = document.getElementById('service-select').value;
  toast('সার্ভিস প্রসেস করা হচ্ছে…', 'ok');

  const fd = new FormData();
  fd.append('image', file);

  try {
    const res = await fetch(`/api/service_process?service_id=${serviceId}`, { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'সার্ভিস ফেইল্ড');

    const resImg = document.getElementById('service-result');
    resImg.src = data.processed_image;
    resImg.style.display = 'block';
    document.getElementById('service-placeholder').style.display = 'none';
    document.getElementById('service-desc').textContent = '✓ ' + data.description;

    toast('সার্ভিস সফলভাবে সম্পন্ন হয়েছে!', 'ok');
  } catch(e) {
    toast('সার্ভিস ত্রুটি: ' + e.message, 'err');
  }
}

function reRunStudioService() {
  if (lastUploadedServiceFile) runStudioService(lastUploadedServiceFile);
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

    const resImg = document.getElementById('det-result');
    resImg.src = data.annotated_image;
    resImg.style.display = 'block';
    document.getElementById('det-placeholder').style.display = 'none';

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
  if (lastUploadedDetFile) runObjectDetection(lastUploadedDetFile);
}

// ─ Computer Vision & Color Palette Lab ─
async function runCVFilter(input) {
  const file = input.files ? input.files[0] : input;
  if (!file) return;
  lastUploadedCVFile = file;

  toast('কম্পিউটার ভিশন ফিল্টার প্রসেস হচ্ছে…', 'ok');
  const fd = new FormData();
  fd.append('image', file);
  const filterType = document.getElementById('cv-type').value;

  try {
    const res = await fetch(`/api/cv_filter?filter_type=${filterType}`, { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'প্রসেসিং ফেইল্ড');

    const resImg = document.getElementById('cv-result');
    resImg.src = data.filtered_image;
    resImg.style.display = 'block';
    document.getElementById('cv-placeholder').style.display = 'none';
    document.getElementById('cv-desc').textContent = data.description;

    const palContainer = document.getElementById('cv-palette');
    palContainer.innerHTML = '';
    for (const c of data.palette) {
      const chip = document.createElement('div');
      chip.className = 'palette-chip';
      chip.style.backgroundColor = c.hex;
      chip.style.color = (c.rgb[0]*0.299 + c.rgb[1]*0.587 + c.rgb[2]*0.114) > 150 ? '#000' : '#fff';
      chip.innerHTML = `${c.hex} (${c.percent}%)`;
      palContainer.appendChild(chip);
    }
    document.getElementById('cv-palette-box').style.display = 'block';
    toast('ফিল্টার এবং কালার প্যালেট সফলভাবে জেনারেট হয়েছে!', 'ok');
  } catch(e) {
    toast('ভিশন ফিল্টার ত্রুটি: ' + e.message, 'err');
  }
}

function reApplyCV() {
  if (lastUploadedCVFile) runCVFilter(lastUploadedCVFile);
}

// ─ Magic Eraser & Inpainting Canvas Logic ─
let inpaintImgObj = null;
let inpaintCanvas = null;
let inpaintCtx = null;
let maskCanvas = null;
let maskCtx = null;
let isDrawing = false;

document.getElementById('brush-size').addEventListener('input', e => {
  document.getElementById('brush-val').textContent = e.target.value + 'px';
});

function loadInpaintImage(input) {
  const file = input.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = e => {
    inpaintImgObj = new Image();
    inpaintImgObj.onload = () => {
      inpaintCanvas = document.getElementById('inpaint-canvas');
      inpaintCtx = inpaintCanvas.getContext('2d');
      
      inpaintCanvas.width = inpaintImgObj.width;
      inpaintCanvas.height = inpaintImgObj.height;
      inpaintCtx.drawImage(inpaintImgObj, 0, 0);

      // Create separate hidden mask canvas (black background, white strokes)
      maskCanvas = document.createElement('canvas');
      maskCanvas.width = inpaintImgObj.width;
      maskCanvas.height = inpaintImgObj.height;
      maskCtx = maskCanvas.getContext('2d');
      maskCtx.fillStyle = '#000000';
      maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);

      document.getElementById('inpaint-canvas-wrap').style.display = 'block';
      document.getElementById('inpaint-placeholder').style.display = 'none';
      document.getElementById('inpaint-result').style.display = 'none';

      setupCanvasEvents();
      toast('ছবি লোড হয়েছে — অনাকাঙ্ক্ষিত অংশের ওপর ব্রাশ করুন', 'ok');
    };
    inpaintImgObj.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

function setupCanvasEvents() {
  inpaintCanvas.onmousedown = e => { isDrawing = true; drawBrush(e); };
  window.onmouseup = () => { isDrawing = false; };
  inpaintCanvas.onmousemove = drawBrush;
  
  // Touch support for mobile
  inpaintCanvas.ontouchstart = e => { isDrawing = true; drawBrush(e.touches[0]); };
  window.ontouchend = () => { isDrawing = false; };
  inpaintCanvas.ontouchmove = e => { drawBrush(e.touches[0]); };
}

function drawBrush(e) {
  if (!isDrawing || !inpaintCtx) return;
  const rect = inpaintCanvas.getBoundingClientRect();
  const scaleX = inpaintCanvas.width / rect.width;
  const scaleY = inpaintCanvas.height / rect.height;
  const x = (e.clientX - rect.left) * scaleX;
  const y = (e.clientY - rect.top) * scaleY;
  const radius = document.getElementById('brush-size').value;

  // Draw semi-transparent pink on visible canvas
  inpaintCtx.beginPath();
  inpaintCtx.arc(x, y, radius, 0, Math.PI * 2);
  inpaintCtx.fillStyle = 'rgba(236, 72, 153, 0.6)';
  inpaintCtx.fill();

  // Draw white on mask canvas
  maskCtx.beginPath();
  maskCtx.arc(x, y, radius, 0, Math.PI * 2);
  maskCtx.fillStyle = '#ffffff';
  maskCtx.fill();
}

function clearInpaintMask() {
  if (!inpaintImgObj) return;
  inpaintCtx.drawImage(inpaintImgObj, 0, 0);
  maskCtx.fillStyle = '#000000';
  maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
  document.getElementById('inpaint-result').style.display = 'none';
  toast('ব্রাশ রিসেট করা হয়েছে', 'ok');
}

async function applyInpaint() {
  if (!inpaintImgObj) { toast('আগে ছবি আপলোড করুন', 'warn'); return; }
  toast('এআই ব্যাকগ্রাউন্ড ব্লেন্ড করে মুছে ফেলছে…', 'ok');

  const fileInput = document.getElementById('inpaint-file');
  const imgFile = fileInput.files[0];

  maskCanvas.toBlob(async maskBlob => {
    const fd = new FormData();
    fd.append('image', imgFile);
    fd.append('mask', maskBlob, 'mask.png');

    try {
      const res = await fetch('/api/inpaint', { method: 'POST', body: fd });
      if (!res.ok) throw new Error('ইনপেইন্টিং ব্যর্থ');
      const blob = await res.blob();
      const resImg = document.getElementById('inpaint-result');
      resImg.src = URL.createObjectURL(blob);
      resImg.style.display = 'block';
      toast('সফলভাবে মুছে ফেলা হয়েছে!', 'ok');
    } catch(e) {
      toast('ইরেজার ত্রুটি: ' + e.message, 'err');
    }
  }, 'image/png');
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
