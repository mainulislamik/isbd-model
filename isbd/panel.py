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


@app.post("/api/learn-pair")
async def learn_from_real_pair_api(
    before: UploadFile = File(...),
    after:  UploadFile = File(...),
    instruction_file: UploadFile = File(None),
    instruction_text: str = Query("", description="ক্লায়েন্টের সরাসরি টেক্সট নির্দেশনা"),
    label:  str = Query("", description="ট্যাগ — যেমন retouching, color_correction"),
    augment: int = Query(8, ge=1, le=16),
):
    """
    AI Real Pair Learning API with Client Instruction Parser.
    Accepts:
    - Before & After Images
    - Client Instruction Text or Document (PDF, Word DOCX, TXT, MD, JPG/PNG markup)
    """
    raw_b = await before.read()
    raw_a = await after.read()
    if not raw_b or not raw_a:
        raise HTTPException(400, "Before ও After দুটি ছবিই দিন")
    if len(raw_b) > MAX_FILE or len(raw_a) > MAX_FILE:
        raise HTTPException(413, "ছবি 25MB-এর বেশি")
    try:
        img_b = Image.open(io.BytesIO(raw_b)).convert("RGB")
        img_a = Image.open(io.BytesIO(raw_a)).convert("RGB")
    except Exception:
        raise HTTPException(400, "ছবি পড়া যায়নি — JPG/PNG/WebP দিন")

    # Parse client instruction from File / Text
    extracted_instruction = instruction_text.strip() if instruction_text else ""
    client_brief_filename = None
    
    if instruction_file and instruction_file.filename:
        client_brief_filename = instruction_file.filename
        raw_doc = await instruction_file.read()
        fname = instruction_file.filename.lower()
        try:
            if fname.endswith(('.txt', '.md', '.json', '.csv')):
                doc_txt = raw_doc.decode('utf-8', errors='ignore').strip()
                if doc_txt:
                    extracted_instruction = (extracted_instruction + "\n" if extracted_instruction else "") + f"[{fname}]: {doc_txt}"
            elif fname.endswith('.pdf'):
                import pypdf
                pdf_reader = pypdf.PdfReader(io.BytesIO(raw_doc))
                pdf_text = "\n".join([page.extract_text() or "" for page in pdf_reader.pages]).strip()
                if pdf_text:
                    extracted_instruction = (extracted_instruction + "\n" if extracted_instruction else "") + f"[PDF: {fname}]: {pdf_text}"
            elif fname.endswith(('.docx', '.doc')):
                import docx
                doc_obj = docx.Document(io.BytesIO(raw_doc))
                doc_text = "\n".join([p.text for p in doc_obj.paragraphs if p.text.strip()])
                if doc_text:
                    extracted_instruction = (extracted_instruction + "\n" if extracted_instruction else "") + f"[Word: {fname}]: {doc_text}"
            elif fname.endswith(('.png', '.jpg', '.jpeg', '.webp')):
                extracted_instruction = (extracted_instruction + "\n" if extracted_instruction else "") + f"[Visual Markup Ref: {fname}]"
        except Exception as e:
            extracted_instruction = (extracted_instruction + "\n" if extracted_instruction else "") + f"[{fname} Attached]"

    from isbd.self_learner import learn_from_real_pair
    result = learn_from_real_pair(
        before_pil=img_b,
        after_pil=img_a,
        label=label or "client_directed",
        augment=augment,
        trigger_finetune=True,
    )

    # Compute detailed AI Learning Insight & Analysis (Bilingual: English & Bangla)
    diff_stats = {}
    heatmap_preview = None
    learning_insights = {}
    try:
        from isbd.heatmap import generate_difference_heatmap
        hm_img, ov_img, diff_stats = generate_difference_heatmap(img_b, img_a, colormap_type="turbo")
        buf_ov = io.BytesIO()
        ov_img.save(buf_ov, format="JPEG", quality=80)
        buf_ov.seek(0)
        heatmap_preview = "data:image/jpeg;base64," + base64.b64encode(buf_ov.read()).decode("utf-8")

        # Extract precise visual changes
        arr_b = np.asarray(img_b.convert("RGB"), dtype=np.float32)
        arr_a = np.asarray(img_a.convert("RGB"), dtype=np.float32)
        
        # Color tone shift (RGB means)
        mean_b = arr_b.mean(axis=(0,1))
        mean_a = arr_a.mean(axis=(0,1))
        delta_rgb = mean_a - mean_b
        
        # Contrast shift (std)
        std_b = arr_b.std()
        std_a = arr_a.std()
        delta_contrast = std_a - std_b
        
        # Sharpness / Edge strength (Laplacian variance)
        import cv2
        gray_b = cv2.cvtColor(np.asarray(img_b), cv2.COLOR_RGB2GRAY)
        gray_a = cv2.cvtColor(np.asarray(img_a), cv2.COLOR_RGB2GRAY)
        lap_b = cv2.Laplacian(gray_b, cv2.CV_64F).var()
        lap_a = cv2.Laplacian(gray_a, cv2.CV_64F).var()
        delta_sharp = lap_a - lap_b

        learned_en = []
        learned_bn = []

        # 1. Color shift insights
        if abs(delta_rgb[0]) > 3 or abs(delta_rgb[1]) > 3 or abs(delta_rgb[2]) > 3:
            dominant_shift = "Red/Warm" if delta_rgb[0] > delta_rgb[2] else "Blue/Cool"
            bn_shift = "লালচে/উষ্ণ" if delta_rgb[0] > delta_rgb[2] else "নীলচে/শীতল"
            learned_en.append(f"Color Grading: Adapted a {dominant_shift} tone transformation (ΔR:{delta_rgb[0]:+.1f}, ΔG:{delta_rgb[1]:+.1f}, ΔB:{delta_rgb[2]:+.1f})")
            learned_bn.append(f"কালার গ্রেডিং: {bn_shift} টোন শিফট ও ব্যালেন্স সমন্বয় শিখেছে (ΔR:{delta_rgb[0]:+.1f}, ΔG:{delta_rgb[1]:+.1f}, ΔB:{delta_rgb[2]:+.1f})")

        # 2. Exposure & Luminance
        lum_b = (arr_b[:,:,0]*0.299 + arr_b[:,:,1]*0.587 + arr_b[:,:,2]*0.114).mean()
        lum_a = (arr_a[:,:,0]*0.299 + arr_a[:,:,1]*0.587 + arr_a[:,:,2]*0.114).mean()
        delta_lum = lum_a - lum_b
        if abs(delta_lum) > 3:
            lum_dir = "Brightening & shadow recovery" if delta_lum > 0 else "Darkening & highlight dampening"
            bn_lum = "উজ্জ্বলতা বৃদ্ধি ও ডার্ক শ্যাডো রিকভারি" if delta_lum > 0 else "হাইলাইট ও অতিরিক্ত এক্সপোজার ব্যালেন্স"
            learned_en.append(f"Luminance Mapping: Learned {lum_dir} ({delta_lum:+.1f} brightness delta)")
            learned_bn.append(f"আলো ও উজ্জ্বলতা: {bn_lum} আয়ত্ত করেছে ({delta_lum:+.1f} ডেল্টা)")

        # 3. Contrast adjustment
        if abs(delta_contrast) > 2:
            c_dir = "Contrast boosting & dynamic range punch" if delta_contrast > 0 else "Tone smoothing & soft leveling"
            bn_c = "কনট্রাস্ট বুস্ট ও ডায়নামিক রেঞ্জ বৃদ্ধি" if delta_contrast > 0 else "টোন স্মুথিং ও লেভেল ব্লেন্ডিং"
            learned_en.append(f"Dynamic Contrast: {c_dir} (ΔStd: {delta_contrast:+.1f})")
            learned_bn.append(f"কনট্রাস্ট ও ডায়নামিক রেঞ্জ: {bn_c} শিখেছে (ΔStd: {delta_contrast:+.1f})")

        # 4. Sharpness & Texture
        if abs(delta_sharp) > 15:
            s_dir = "High-pass edge sharpening & micro-texture clarity" if delta_sharp > 0 else "Skin/surface smoothing & noise attenuation"
            bn_s = "হাই-পাস এজ শার্পেনিং ও মাইক্রো-টেক্সচার ক্ল্যারিটি" if delta_sharp > 0 else "স্কিন/সারফেস স্মুথিং ও দাগ বিলুপ্তি"
            learned_en.append(f"Spatial Filtering: {s_dir} (ΔLaplacian: {delta_sharp:+.1f})")
            learned_bn.append(f"টেক্সচার ও শার্পনেস: {bn_s} আয়ত্ত করেছে (ΔLaplacian: {delta_sharp:+.1f})")

        # Fallback if subtle
        if not learned_en:
            learned_en.append("Subtle Pixel Refinement: Learned micro-tonal corrections and edge balancing.")
            learned_bn.append("সূক্ষ্ম পিক্সেল রিফাইনমেন্ট: নিখুঁত মাইক্রো-টোনাল কারেকশন ও এজ ব্যালেন্স শিখেছে।")

        learning_insights = {
            "english": learned_en,
            "bangla": learned_bn
        }
    except Exception as e:
        pass

    return {
        "ok"           : True,
        "message"      : f"✅ AI শিখছে! '{label or 'untagged'}' পেয়ার থেকে {result['augmented']}টি অগমেন্টেড স্যাম্পল তৈরি হয়েছে এবং মাইক্রো ফাইন-টিউন শুরু হয়েছে।",
        "augmented"    : result["augmented"],
        "total_real"   : result["total_real"],
        "total_pool"   : result["total_pool"],
        "finetune_pid" : result["finetune_pid"],
        "diff_stats"   : diff_stats,
        "heatmap_preview": heatmap_preview,
        "learning_insights": learning_insights,
        "client_instruction": extracted_instruction[:500] if extracted_instruction else None,
        "instruction_source": client_brief_filename or ("Direct Text Note" if instruction_text else None)
    }


@app.get("/api/real-pair-log")
async def real_pair_log_api(limit: int = Query(20)):
    from isbd.self_learner import get_real_pair_log, get_real_pair_count
    return {
        "log"        : get_real_pair_log(limit),
        "total_real" : get_real_pair_count(),
    }


@app.get("/api/training/toggle")
async def toggle_training_api(action: str = Query(..., regex="^(start|stop|status)$")):
    """Start, stop or check status of the 24/7 autonomous continuous training service."""
    try:
        if action == "status":
            res = subprocess.run(["systemctl", "--user", "is-active", "isbd-train"], capture_output=True, text=True)
            active = res.stdout.strip() == "active"
            return {"ok": True, "active": active}
        
        elif action == "start":
            subprocess.run(["systemctl", "--user", "start", "isbd-train"], check=True)
            return {"ok": True, "active": True, "message": "২৪/৭ সেলফ-লার্নিং ট্রেনিং সফলভাবে চালু করা হয়েছে!"}
            
        elif action == "stop":
            subprocess.run(["systemctl", "--user", "stop", "isbd-train"], check=True)
            return {"ok": True, "active": False, "message": "২৪/৭ সেলফ-লার্নিং ট্রেনিং সাময়িকভাবে বন্ধ (পজ) করা হয়েছে!"}
            
    except Exception as e:
        raise HTTPException(500, f"সার্ভিস কমান্ড ব্যর্থ: {str(e)}")


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
async def detect_api(
    image: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.05, le=0.9),
    mode: str = Query("full_body", description="Mode: 'full_body' (Anatomy & Apparel) or 'general' (Standard YOLO)")
):
    """Detect and classify human body parts, apparel or general objects with Bengali descriptions."""
    raw = await image.read()
    if not raw:
        raise HTTPException(400, "ছবি পাওয়া যায়নি")
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        
        if mode == "full_body":
            from isbd.part_tracker import parse_human_body_and_apparel
            res = parse_human_body_and_apparel(img, confidence=conf)
            annotated_img = res["annotated_image"]
            summary = res["summary"]
            total_objs = res["total_parts"]
            detections = res["parts"]
            crops = res.get("crops", [])
        else:
            from isbd.detector import detect_objects_in_image
            annotated_img, detections, summary = detect_objects_in_image(img, conf_threshold=conf)
            total_objs = len(detections)
            crops = []

        buf = io.BytesIO()
        annotated_img.save(buf, format="JPEG", quality=85)
        buf.seek(0)
        img_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.read()).decode("utf-8")

        return {
            "ok": True,
            "mode": mode,
            "total_objects": total_objs,
            "summary": summary,
            "detections": detections,
            "annotated_image": img_b64,
            "crops": crops
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
HTML_PATH = ROOT / "isbd" / "templates" / "index.html"


@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_PATH.read_text(encoding="utf-8")
