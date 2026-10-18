"""
ISBD v1.00 — Advanced Self-Supervised & Multi-Scale Autonomous Learner
Master Architecture:
  1. Multi-Scale Patch Harvesting:
     - Extracts high-resolution 256×256 salient patches centered on designer modifications
     - Multi-level pyramid (256px patch + 64px global macro)
  2. Categorized Memory Buckets for 21 Commercial Studio Services:
     - Isolated experience pools per service (clipping, retouching, jewelry, etc.)
     - Class-balanced replay to prevent catastrophic forgetting
  3. Hard-Example Activity Mining:
     - Auto-detects highest-delta regions where professional human editing occurred
"""
import time
import os
import json
import random
import subprocess
import sys
import numpy as np
from pathlib import Path
from PIL import Image

ROOT          = Path(__file__).resolve().parent.parent
DATA          = ROOT / "data"
DATA.mkdir(exist_ok=True)
REF_ASSETS_DIR = DATA / "ref_assets"
REF_ASSETS_DIR.mkdir(exist_ok=True)
COMPOSITE_ASSETS_DIR = DATA / "composite_assets"
COMPOSITE_ASSETS_DIR.mkdir(exist_ok=True)
BUCKETS_DIR   = DATA / "buckets"
BUCKETS_DIR.mkdir(exist_ok=True)

NPZ           = DATA / "pairs.npz"
SELF_POOL     = DATA / "self_pool.npz"
REAL_POOL     = DATA / "real_pairs.npz"          # Global real pool (64x64)
PATCH_POOL    = DATA / "real_patches_256.npz"    # High-resolution 256x256 patch pool
STATE_FILE    = DATA / "self_learn_state.json"
REAL_LOG      = DATA / "real_pairs_log.json"

IMG_BASE      = 64
PATCH_SIZE    = 256

# Recognized 21 Commercial Studio Services
COMMERCIAL_SERVICES = [
    "clipping_path", "multi_clipping", "hair_masking", "neck_joint", "retouching",
    "face_beauty", "jewelry_shiner", "wrinkle_cleaner", "shadow_making", "cast_shadow_3d",
    "reflection", "ghost_mannequin", "color_fix", "recolor", "raster_to_vector",
    "inpaint", "super_res", "denoise", "exposure_balance", "hdr_fusion", "custom"
]


# ── STORAGE HELPERS ────────────────────────────────────────────────────────────

def _load_npz(path, img_size=IMG_BASE):
    p = Path(path)
    if p.exists():
        try:
            with np.load(p) as d:
                return d["X"], d["Y"]
        except Exception:
            pass
    return (np.empty((0, 3, img_size, img_size), dtype=np.float32),
            np.empty((0, 3, img_size, img_size), dtype=np.float32))


def _save_npz(path, X, Y, cap=4000):
    if len(X) > cap:
        X, Y = X[-cap:], Y[-cap:]
    np.savez(path, X=X.astype(np.float32), Y=Y.astype(np.float32))


def _img_to_tensor(pil_img, size=IMG_BASE):
    """Resize to size×size, normalise → (3,size,size) float32 array."""
    small = pil_img.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(small, dtype=np.float32).transpose(2, 0, 1) / 255.0


def _append_to_main_npz(x, y):
    """Append a single pair (3,64,64) to the shared pairs.npz used by continuous trainer."""
    mX, mY = _load_npz(NPZ, img_size=IMG_BASE)
    mX = np.concatenate([mX, x[None]], axis=0)
    mY = np.concatenate([mY, y[None]], axis=0)
    _save_npz(NPZ, mX, mY, cap=6000)


# ── SALIENT PATCH HARVESTER (HARD-EXAMPLE MINING) ───────────────────────────────

def extract_salient_patches(before_pil: Image.Image,
                             after_pil:  Image.Image,
                             max_patches: int = 4,
                             patch_size:  int = PATCH_SIZE) -> tuple[list, list]:
    """
    Identifies locations where the human artist made the most intense edits
    (highest pixel delta) and harvests aligned (patch_size × patch_size) crops.
    """
    b_rgb = before_pil.convert("RGB")
    a_rgb = after_pil.convert("RGB")

    # Match dimensions if slight discrepancy
    if b_rgb.size != a_rgb.size:
        a_rgb = a_rgb.resize(b_rgb.size, Image.Resampling.LANCZOS)

    w, h = b_rgb.size
    patches_b = []
    patches_a = []

    if w < patch_size or h < patch_size:
        # Image smaller than patch: pad/resize directly
        p_b = b_rgb.resize((patch_size, patch_size), Image.Resampling.LANCZOS)
        p_a = a_rgb.resize((patch_size, patch_size), Image.Resampling.LANCZOS)
        return [p_b], [p_a]

    # Convert to grayscale numpy arrays to calculate spatial difference heatmap
    arr_b = np.asarray(b_rgb.convert("L"), dtype=np.float32)
    arr_a = np.asarray(a_rgb.convert("L"), dtype=np.float32)
    diff = np.abs(arr_a - arr_b)

    # Boundary Hard-Negative Mining (Pillar 1) using Sobel edge energy
    try:
        import cv2
        sobel_x = cv2.Sobel(diff, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(diff, cv2.CV_32F, 0, 1, ksize=3)
        edge_energy = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    except Exception:
        gy, gx = np.gradient(diff)
        edge_energy = np.sqrt(gx ** 2 + gy ** 2)

    # Grid search across the image for candidate patch windows
    step = max(32, patch_size // 2)
    candidates = []

    for y in range(0, max(1, h - patch_size + 1), step):
        for x in range(0, max(1, w - patch_size + 1), step):
            win_diff = diff[y:y + patch_size, x:x + patch_size]
            win_edge = edge_energy[y:y + patch_size, x:x + patch_size]
            mean_delta = float(win_diff.mean())
            std_delta = float(win_diff.std())
            edge_delta = float(win_edge.mean())
            # Hardness Priority: combines overall delta + edge complexity (hair/wrinkles/jewelry)
            score = (mean_delta * 0.4 + std_delta * 0.6) + 2.5 * edge_delta
            candidates.append((score, x, y))

    # Always include bottom/right boundary windows if stepped over
    if (w - patch_size) > 0 and (h - patch_size) > 0:
        win_diff = diff[-patch_size:, -patch_size:]
        win_edge = edge_energy[-patch_size:, -patch_size:]
        score = (float(win_diff.mean()) * 0.4 + float(win_diff.std()) * 0.6) + 2.5 * float(win_edge.mean())
        candidates.append((score, w - patch_size, h - patch_size))

    # Sort descending by edit intensity
    candidates.sort(key=lambda c: c[0], reverse=True)

    # Select non-overlapping top patches
    chosen = []
    for score, cx, cy in candidates:
        overlap = False
        for _, ox, oy in chosen:
            if abs(cx - ox) < patch_size // 2 and abs(cy - oy) < patch_size // 2:
                overlap = True
                break
        if not overlap:
            chosen.append((score, cx, cy))
            if len(chosen) >= max_patches:
                break

    # If chosen is empty (e.g. identical or no change), take center crop
    if not chosen:
        cx = max(0, (w - patch_size) // 2)
        cy = max(0, (h - patch_size) // 2)
        chosen = [(0.0, cx, cy)]

    for _, cx, cy in chosen:
        crop_box = (cx, cy, cx + patch_size, cy + patch_size)
        patches_b.append(b_rgb.crop(crop_box))
        patches_a.append(a_rgb.crop(crop_box))

    return patches_b, patches_a


# ── BUCKETED EXPERIENCE MANAGEMENT ─────────────────────────────────────────────

def _normalize_service_labels(raw_label: str) -> list:
    """Normalize free-form tags or comma-separated list into standard service identifiers."""
    if not raw_label:
        return ["custom"]
    clean = raw_label.lower().replace("-", "_").strip()
    parts = [p.strip().replace(" ", "_") for p in clean.split(",") if p.strip()]
    found = []
    for p in parts:
        matched = False
        for svc in COMMERCIAL_SERVICES:
            if svc in p or p in svc:
                found.append(svc)
                matched = True
                break
        if not matched and p:
            found.append(p)
    return list(dict.fromkeys(found)) or ["custom"]


def _normalize_service_label(raw_label: str) -> str:
    labels = _normalize_service_labels(raw_label)
    return labels[0] if labels else "custom"


def _save_to_service_bucket(service_name: str, x_tensors: np.ndarray, y_tensors: np.ndarray):
    """Store samples in an isolated per-service bucket."""
    bucket_file = BUCKETS_DIR / f"{service_name}.npz"
    bX, bY = _load_npz(bucket_file, img_size=IMG_BASE)
    bX = np.concatenate([bX, x_tensors], axis=0)
    bY = np.concatenate([bY, y_tensors], axis=0)
    _save_npz(bucket_file, bX, bY, cap=300)  # Up to 300 curated pairs per service


def get_bucket_stats() -> dict:
    """Return dictionary of pair counts for every commercial service."""
    stats = {}
    for f in BUCKETS_DIR.glob("*.npz"):
        svc = f.stem
        try:
            with np.load(f) as d:
                stats[svc] = int(len(d["X"]))
        except Exception:
            stats[svc] = 0
    return stats


def _record_state(total_real, total_synthetic, note=""):
    state = {
        "last_harvest"            : time.time(),
        "total_real_pairs"        : total_real,
        "total_self_learned_pairs": total_synthetic,
        "bucket_counts"           : get_bucket_stats(),
        "note"                    : note,
        "status"                  : "Active — Multi-Scale Patch Engine (256px + 64px) & 21-Service Memory Buckets"
    }
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2))


def analyze_and_store_references(reference_pils: list, pair_id: str) -> list:
    """Store reference assets on disk and classify their design guidance role."""
    if not reference_pils:
        return []
    
    REF_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    ref_records = []
    
    role_translations = {
        "color_match": "কালার সোয়াচ / Color Match",
        "garment_part": "নেক / গার্মেন্টস পার্ট",
        "shadow_sample": "শ্যাডো স্যাম্পল",
        "mood_guide": "মুড / স্টাইল গাইড"
    }

    for idx, ref_img in enumerate(reference_pils):
        if not isinstance(ref_img, Image.Image):
            continue
        try:
            rgb_ref = ref_img.convert("RGB")
            filename = f"{pair_id}_ref_{idx+1}.jpg"
            save_path = REF_ASSETS_DIR / filename
            rgb_ref.save(save_path, format="JPEG", quality=90)

            w, h = rgb_ref.size
            arr = np.asarray(rgb_ref, dtype=np.float32)
            std_color = float(arr.std(axis=(0, 1)).mean())
            
            # Simple, robust role heuristic
            if min(w, h) < 250 or std_color < 18.0:
                role = "color_match"
            elif (w / max(1, h) > 1.8 or h / max(1, w) > 1.8) and std_color > 30.0:
                role = "garment_part"
            elif arr.mean() > 220 and std_color > 15.0:
                role = "shadow_sample"
            else:
                role = "mood_guide"

            ref_records.append({
                "index": idx + 1,
                "filename": filename,
                "path": str(save_path),
                "url": f"/data/ref_assets/{filename}",
                "role": role,
                "role_bn": role_translations.get(role, "রেফারেন্স গাইড"),
                "width": w,
                "height": h,
                "std_color": round(std_color, 2)
            })
        except Exception as e:
            print(f"[SelfLearner] Error processing reference image {idx+1}: {e}")

    return ref_records


def _log_real_pair(label, before_size, after_size, total, patches_count=0, references=None, mode="1_to_1", multi_info=None):
    """Keep a human-readable journal of every real pair added."""
    log = []
    if REAL_LOG.exists():
        try:
            log = json.loads(REAL_LOG.read_text())
        except Exception:
            pass
    entry = {
        "ts"            : time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode"          : mode,
        "label"         : label,
        "before_px"     : f"{before_size[0]}×{before_size[1]}",
        "after_px"      : f"{after_size[0]}×{after_size[1]}",
        "patches_256"   : patches_count,
        "total_pairs"   : total,
        "has_references": bool(references and len(references) > 0),
        "reference_count": len(references) if references else 0,
        "references"    : references or [],
        "multi_info"    : multi_info or {}
    }
    log.append(entry)
    if len(log) > 500:
        log = log[-500:]
    REAL_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2))


# ── MAIN LEARNING API ──────────────────────────────────────────────────────────

def learn_from_real_pair(before_pil: Image.Image,
                         after_pil:  Image.Image,
                         label:      str = "",
                         augment:    int = 8,
                         trigger_finetune: bool = True,
                         reference_pils: list | None = None) -> dict:
    """
    AUTONOMOUS MULTI-SCALE REAL PAIR LEARNING
    - Extracts global 64×64 thumbnail representation
    - Extracts up to 4 salient 256×256 high-resolution patches focused on modifications
    - Stores into dedicated 21-service bucket memory
    - Kicks off instant micro fine-tune with composite studio loss
    """
    detected_info = None
    if not label or label.strip().lower() == "auto":
        try:
            from isbd.vision_detector import detect_editing_services
            detected_info = detect_editing_services(before_pil, after_pil)
            service_keys = detected_info.get("detected_services", ["custom"])
            label = ", ".join(service_keys)
        except Exception:
            service_keys = ["custom"]
    else:
        service_keys = _normalize_service_labels(label)

    service_key = service_keys[0] if service_keys else "custom"

    # 1. Macro global representation
    x_base = _img_to_tensor(before_pil, size=IMG_BASE)
    y_base = _img_to_tensor(after_pil, size=IMG_BASE)

    pairs_x, pairs_y = [x_base], [y_base]

    # 2. Extract salient 256x256 high-detail patches
    p_b_list, p_a_list = extract_salient_patches(before_pil, after_pil, max_patches=4, patch_size=PATCH_SIZE)

    patch_tensors_x = []
    patch_tensors_y = []
    for pb, pa in zip(p_b_list, p_a_list):
        # 256px resolution tensor
        ptx = _img_to_tensor(pb, size=PATCH_SIZE)
        pty = _img_to_tensor(pa, size=PATCH_SIZE)
        patch_tensors_x.append(ptx)
        patch_tensors_y.append(pty)

        # Also add downscaled high-interest crop to the 64px trainer pool
        pairs_x.append(_img_to_tensor(pb, size=IMG_BASE))
        pairs_y.append(_img_to_tensor(pa, size=IMG_BASE))

    # Save 256px patches to PATCH_POOL
    if patch_tensors_x:
        px_arr = np.stack(patch_tensors_x, axis=0)
        py_arr = np.stack(patch_tensors_y, axis=0)
        hpX, hpY = _load_npz(PATCH_POOL, img_size=PATCH_SIZE)
        hpX = np.concatenate([hpX, px_arr], axis=0)
        hpY = np.concatenate([hpY, py_arr], axis=0)
        _save_npz(PATCH_POOL, hpX, hpY, cap=3000)

    # 3. Augmentations on global & patch crops (flips + rotations)
    src_b = before_pil.convert("RGB").resize((IMG_BASE, IMG_BASE), Image.Resampling.LANCZOS)
    src_a = after_pil.convert("RGB").resize((IMG_BASE, IMG_BASE), Image.Resampling.LANCZOS)

    aug_ops = [
        (Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_LEFT_RIGHT),
        (Image.Transpose.FLIP_TOP_BOTTOM, Image.Transpose.FLIP_TOP_BOTTOM),
        (Image.Transpose.ROTATE_90,       Image.Transpose.ROTATE_90),
        (Image.Transpose.ROTATE_180,      Image.Transpose.ROTATE_180),
        (Image.Transpose.ROTATE_270,      Image.Transpose.ROTATE_270),
        (Image.Transpose.TRANSPOSE,       Image.Transpose.TRANSPOSE),
        (Image.Transpose.TRANSVERSE,      Image.Transpose.TRANSVERSE),
    ]

    for op_b, op_a in aug_ops[:max(0, augment - len(pairs_x))]:
        try:
            ab = src_b.transpose(op_b)
            aa = src_a.transpose(op_a)
            pairs_x.append(np.asarray(ab, dtype=np.float32).transpose(2, 0, 1) / 255.0)
            pairs_y.append(np.asarray(aa, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        except Exception:
            pass

    aug_x = np.stack(pairs_x, axis=0)
    aug_y = np.stack(pairs_y, axis=0)

    # 4. Save to global real pool
    rX, rY = _load_npz(REAL_POOL, img_size=IMG_BASE)
    rX = np.concatenate([rX, aug_x], axis=0)
    rY = np.concatenate([rY, aug_y], axis=0)
    _save_npz(REAL_POOL, rX, rY, cap=8000)

    # 5. Save into service-specific memory bucket(s)
    for sk in service_keys:
        _save_to_service_bucket(sk, aug_x, aug_y)

    # 6. Append to shared pairs.npz for the 24/7 continuous trainer
    for xv, yv in zip(aug_x, aug_y):
        _append_to_main_npz(xv, yv)

    # 7. Update self pool and state
    sX, sY = _load_npz(SELF_POOL, img_size=IMG_BASE)
    sX = np.concatenate([sX, aug_x], axis=0)
    sY = np.concatenate([sY, aug_y], axis=0)
    _save_npz(SELF_POOL, sX, sY)

    total_real = int(len(rX))
    total_synth = int(len(sX))

    pair_id = f"pair_{int(time.time())}_{random.randint(1000, 9999)}"
    ref_meta = analyze_and_store_references(reference_pils, pair_id) if reference_pils else []

    _log_real_pair(label or service_key, before_pil.size, after_pil.size, total_real, patches_count=len(p_b_list), references=ref_meta)
    _record_state(total_real, total_synth,
                  note=f"Learned real pair '{service_key}': +{len(aug_x)} samples, {len(p_b_list)} 256px salient patches, {len(ref_meta)} refs")

    # 8. Micro fine-tune so model learns right now with composite studio loss
    ft_pid = None
    if trigger_finetune:
        try:
            ft_pid = _trigger_micro_finetune(steps=200, lr=5e-5)
        except Exception:
            pass

    return {
        "ok"           : True,
        "label"        : ", ".join(service_keys),
        "primary_label": service_key,
        "augmented"    : len(aug_x),
        "patches_256"  : len(p_b_list),
        "total_real"   : total_real,
        "total_pool"   : total_synth,
        "finetune_pid" : ft_pid,
        "buckets"      : get_bucket_stats(),
        "detected"     : detected_info,
        "has_references": bool(ref_meta),
        "reference_count": len(ref_meta),
        "references"   : ref_meta
    }


def _docker_safe_python() -> str:
    venv_py = ROOT / ".venv" / "bin" / "python"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def _trigger_micro_finetune(steps=200, lr=5e-5):
    cmd = [
        _docker_safe_python(),
        str(ROOT / "isbd" / "realfinetune.py"),
        "--steps", str(steps),
        "--lr", str(lr),
        "--batch", "4",
        "--wait-lock", "120",
    ]
    log_path = DATA / "micro_ft.log"
    with open(log_path, "w") as flog:
        p = subprocess.Popen(cmd, cwd=str(ROOT),
                             stdout=flog, stderr=subprocess.STDOUT)
    return p.pid


def get_real_pair_log(limit=25):
    if REAL_LOG.exists():
        try:
            log = json.loads(REAL_LOG.read_text())
            return log[-limit:]
        except Exception:
            pass
    return []


def get_real_pair_count():
    rX, _ = _load_npz(REAL_POOL, img_size=IMG_BASE)
    return int(len(rX))


def harvest_and_synthesize_pair(clean_pil_img: Image.Image):
    from isbd.data import degrade
    clean_small = clean_pil_img.convert("RGB").resize((IMG_BASE, IMG_BASE), Image.Resampling.LANCZOS)
    rng = random.Random(int(time.time() * 1000) % 1_000_000)
    degraded = degrade(clean_small, rng)
    x = np.asarray(degraded,    dtype=np.float32).transpose(2, 0, 1) / 255.0
    y = np.asarray(clean_small, dtype=np.float32).transpose(2, 0, 1) / 255.0

    sX, sY = _load_npz(SELF_POOL, img_size=IMG_BASE)
    sX = np.concatenate([sX, x[None]], axis=0)
    sY = np.concatenate([sY, y[None]], axis=0)
    _save_npz(SELF_POOL, sX, sY)
    _append_to_main_npz(x, y)

    total_real  = get_real_pair_count()
    total_synth = int(len(sX))
    _record_state(total_real, total_synth, note="Synthetic harvest from inference")
    return total_synth


def get_self_learn_stats():
    base_stats = {}
    if STATE_FILE.exists():
        try:
            base_stats = json.loads(STATE_FILE.read_text())
        except Exception:
            pass

    rX, _ = _load_npz(REAL_POOL, img_size=IMG_BASE)
    sX, _ = _load_npz(SELF_POOL, img_size=IMG_BASE)
    gX, _ = _load_npz(DATA / "golden_pool.npz", img_size=IMG_BASE)

    try:
        from isbd.curriculum import CurriculumScheduler
        curric_info = CurriculumScheduler().get_info()
    except Exception:
        curric_info = {"current_level": 1, "level_name": "solid_contrast", "moving_avg_loss": 0.035}

    ewc_active = (DATA / "ewc_anchor.pt").exists()

    result = {
        "last_harvest"            : base_stats.get("last_harvest", None),
        "total_real_pairs"        : int(len(rX)),
        "total_self_learned_pairs": int(len(sX)),
        "golden_pairs_count"      : int(len(gX)),
        "bucket_counts"           : get_bucket_stats(),
        "curriculum"              : curric_info,
        "ewc_anchor_active"       : ewc_active,
        "ewc_active"              : ewc_active,
        "status"                  : base_stats.get("status", "Active — Self-Learning v2.0 (Curriculum + PER + EWC)")
    }
    return result


def _augment_pair(src_b: Image.Image, src_a: Image.Image, num: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Generate rotation and flip augmented pairs as normalized (N, 3, 64, 64) numpy arrays."""
    b_small = src_b.convert("RGB").resize((IMG_BASE, IMG_BASE), Image.Resampling.LANCZOS)
    a_small = src_a.convert("RGB").resize((IMG_BASE, IMG_BASE), Image.Resampling.LANCZOS)

    px = [_img_to_tensor(b_small, size=IMG_BASE)]
    py = [_img_to_tensor(a_small, size=IMG_BASE)]

    aug_ops = [
        (Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_LEFT_RIGHT),
        (Image.Transpose.FLIP_TOP_BOTTOM, Image.Transpose.FLIP_TOP_BOTTOM),
        (Image.Transpose.ROTATE_90,       Image.Transpose.ROTATE_90),
        (Image.Transpose.ROTATE_180,      Image.Transpose.ROTATE_180),
        (Image.Transpose.ROTATE_270,      Image.Transpose.ROTATE_270),
        (Image.Transpose.TRANSPOSE,       Image.Transpose.TRANSPOSE),
        (Image.Transpose.TRANSVERSE,      Image.Transpose.TRANSVERSE),
    ]
    for op_b, op_a in aug_ops:
        if len(px) >= num:
            break
        try:
            ab = b_small.transpose(op_b)
            aa = a_small.transpose(op_a)
            px.append(np.asarray(ab, dtype=np.float32).transpose(2, 0, 1) / 255.0)
            py.append(np.asarray(aa, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        except Exception:
            pass

    return np.stack(px, axis=0), np.stack(py, axis=0)


def learn_from_multi_bundle(
    before_pils: list,
    after_pils: list,
    label: str = "",
    augment: int = 8,
    trigger_finetune: bool = True,
    reference_pils: list | None = None
) -> dict:
    """
    Unified Ingestion API for Studio Workflows:
    - 1-to-1: Single Before -> Single After
    - 1-to-N: 1 Raw Before -> Multiple Deliverables (Clipping, Shadow, Recolor, etc.)
    - N-to-1: Multiple Raw Befores -> 1 Master Composite (Ghost Mannequin / Neck Joint / Bracketing / Stacking)
    - N-to-M: Multi-pair Batch Set
    """
    nb = len(before_pils)
    na = len(after_pils)
    if nb == 0 or na == 0:
        return {"ok": False, "error": "Before ও After উভয় ছবি প্রয়োজন"}

    # 1. Single pair fallback
    if nb == 1 and na == 1:
        res = learn_from_real_pair(
            before_pils[0], after_pils[0],
            label=label,
            augment=augment,
            trigger_finetune=trigger_finetune,
            reference_pils=reference_pils
        )
        res["mode"] = "1_to_1"
        return res

    from isbd.vision_detector import detect_multi_pair
    det = detect_multi_pair(before_pils, after_pils)
    mode = det.get("mode", "1_to_1")
    bundle_id = f"bundle_{int(time.time())}_{random.randint(1000, 9999)}"
    ref_records = analyze_and_store_references(reference_pils, bundle_id) if reference_pils else []

    # 2. Case: 1-to-N Multi-Deliverables
    if nb == 1 and na > 1:
        raw_b = before_pils[0]
        outputs_learned = []
        all_new_rx = []
        all_new_ry = []
        all_patches_b = []
        all_patches_a = []
        
        rX, rY = _load_npz(REAL_POOL, img_size=IMG_BASE)
        pX, pY = _load_npz(PATCH_POOL, img_size=PATCH_SIZE)
        
        b_rgb = raw_b.convert("RGB")
        b_small = b_rgb.resize((IMG_BASE, IMG_BASE), Image.Resampling.LANCZOS)
        
        for idx, out_img in enumerate(after_pils):
            a_rgb = out_img.convert("RGB")
            a_small = a_rgb.resize((IMG_BASE, IMG_BASE), Image.Resampling.LANCZOS)
            
            detected_out_services = det.get("outputs", [])[idx].get("services", []) if idx < len(det.get("outputs", [])) else []
            out_label = label if label else (", ".join(detected_out_services) if detected_out_services else "custom")
            norm_services = _normalize_service_labels(out_label)
            
            aug_b, aug_a = _augment_pair(b_small, a_small, num=max(2, augment // max(1, na)))
            for svc in norm_services:
                _save_to_service_bucket(svc, aug_b, aug_a)
                
            all_new_rx.append(aug_b)
            all_new_ry.append(aug_a)
            
            pb, pa = extract_salient_patches(raw_b, out_img, max_patches=2, patch_size=PATCH_SIZE)
            all_patches_b.extend(pb)
            all_patches_a.extend(pa)
            
            outputs_learned.append({
                "index": idx + 1,
                "services": norm_services,
                "services_bn": det.get("outputs", [])[idx].get("services_bn", []) if idx < len(det.get("outputs", [])) else [],
                "patches_extracted": len(pb)
            })

        cat_rx = np.concatenate(all_new_rx, axis=0)
        cat_ry = np.concatenate(all_new_ry, axis=0)
        rX = np.concatenate([rX, cat_rx], axis=0)
        rY = np.concatenate([rY, cat_ry], axis=0)
        _save_npz(REAL_POOL, rX, rY, cap=3000)
        
        if all_patches_b:
            pt_b = np.stack([np.asarray(p, dtype=np.float32).transpose(2,0,1)/255.0 for p in all_patches_b], axis=0)
            pt_a = np.stack([np.asarray(p, dtype=np.float32).transpose(2,0,1)/255.0 for p in all_patches_a], axis=0)
            pX = np.concatenate([pX, pt_b], axis=0)
            pY = np.concatenate([pY, pt_a], axis=0)
            _save_npz(PATCH_POOL, pX, pY, cap=1500)
            
            for pb_img, pa_img in zip(all_patches_b, all_patches_a):
                pb_s = pb_img.resize((IMG_BASE, IMG_BASE), Image.Resampling.BILINEAR)
                pa_s = pa_img.resize((IMG_BASE, IMG_BASE), Image.Resampling.BILINEAR)
                bx = np.asarray(pb_s, dtype=np.float32).transpose(2,0,1)/255.0
                by = np.asarray(pa_s, dtype=np.float32).transpose(2,0,1)/255.0
                _append_to_main_npz(bx, by)

        for bx, by in zip(cat_rx, cat_ry):
            _append_to_main_npz(bx, by)
            
        total_real = int(len(rX))
        total_patches = int(len(pX))
        
        combined_label = label if label else ", ".join(det.get("detected_services", ["multi_deliverables"]))
        _log_real_pair(
            label=combined_label,
            before_size=raw_b.size,
            after_size=after_pils[0].size,
            total=total_real,
            patches_count=len(all_patches_b),
            references=ref_records,
            mode="1_to_n",
            multi_info={
                "deliverables_count": na,
                "outputs": outputs_learned,
                "summary_bn": det.get("summary_bn", "")
            }
        )
        
        ft_pid = None
        if trigger_finetune:
            ft_pid = _trigger_micro_finetune(steps=250, lr=5e-5)
            
        return {
            "ok": True,
            "mode": "1_to_n",
            "mode_title": det.get("mode_title", "মাল্টি-ডেলিভারেবল (1-to-N)"),
            "deliverables_count": na,
            "outputs": outputs_learned,
            "augmented_samples": int(len(cat_rx)),
            "patches_extracted": len(all_patches_b),
            "total_real_pairs": total_real,
            "total_patches_256": total_patches,
            "detected_services": det.get("detected_services", []),
            "summary_bn": det.get("summary_bn", ""),
            "micro_ft_triggered": bool(ft_pid),
            "micro_ft_pid": ft_pid
        }

    # 3. Case: N-to-1 Composite Assembly (Neck Joint, HDR Bracketing, Focus Stacking)
    if nb > 1 and na == 1:
        primary_b = before_pils[0]
        master_a = after_pils[0]
        comp_type = det.get("composite_type", "neck_joint")
        
        pair_prefix = f"pair_{int(time.time())}_{random.randint(1000, 9999)}"
        comp_records = []
        for c_idx, c_img in enumerate(before_pils[1:]):
            c_filename = f"{pair_prefix}_comp_{c_idx+1}.jpg"
            c_save_path = COMPOSITE_ASSETS_DIR / c_filename
            try:
                c_img.convert("RGB").save(c_save_path, quality=92)
                comp_records.append({
                    "index": c_idx + 2,
                    "filename": c_filename,
                    "url": f"/data/composite_assets/{c_filename}",
                    "role": "Inner Collar / Back Neck" if comp_type == "neck_joint" and c_idx == 0 else f"Component #{c_idx+2}",
                    "size": f"{c_img.width}×{c_img.height}"
                })
            except Exception as e:
                print(f"[SelfLearner] Warning saving composite asset: {e}")

        rX, rY = _load_npz(REAL_POOL, img_size=IMG_BASE)
        pX, pY = _load_npz(PATCH_POOL, img_size=PATCH_SIZE)
        
        pb_rgb = primary_b.convert("RGB")
        ma_rgb = master_a.convert("RGB")
        pb_s = pb_rgb.resize((IMG_BASE, IMG_BASE), Image.Resampling.LANCZOS)
        ma_s = ma_rgb.resize((IMG_BASE, IMG_BASE), Image.Resampling.LANCZOS)
        
        aug_pb, aug_ma = _augment_pair(pb_s, ma_s, num=augment)
        
        patches_b, patches_a = extract_salient_patches(primary_b, master_a, max_patches=3, patch_size=PATCH_SIZE)
        
        for c_img in before_pils[1:]:
            cp_b, cp_a = extract_salient_patches(c_img, master_a, max_patches=2, patch_size=PATCH_SIZE)
            patches_b.extend(cp_b)
            patches_a.extend(cp_a)

        norm_services = _normalize_service_labels(label or comp_type)
        if comp_type == "neck_joint":
            norm_services.extend(["neck_joint", "ghost_mannequin", "clipping_path"])
        elif comp_type == "hdr_bracket":
            norm_services.extend(["hdr_fusion", "exposure_balance"])
        norm_services = list(dict.fromkeys(norm_services))
        
        for svc in norm_services:
            _save_to_service_bucket(svc, aug_pb, aug_ma)
            
        rX = np.concatenate([rX, aug_pb], axis=0)
        rY = np.concatenate([rY, aug_ma], axis=0)
        _save_npz(REAL_POOL, rX, rY, cap=3000)
        
        if patches_b:
            pt_b = np.stack([np.asarray(p, dtype=np.float32).transpose(2,0,1)/255.0 for p in patches_b], axis=0)
            pt_a = np.stack([np.asarray(p, dtype=np.float32).transpose(2,0,1)/255.0 for p in patches_a], axis=0)
            pX = np.concatenate([pX, pt_b], axis=0)
            pY = np.concatenate([pY, pt_a], axis=0)
            _save_npz(PATCH_POOL, pX, pY, cap=1500)
            
            for pb_img, pa_img in zip(patches_b, patches_a):
                pb_crop_s = pb_img.resize((IMG_BASE, IMG_BASE), Image.Resampling.BILINEAR)
                pa_crop_s = pa_img.resize((IMG_BASE, IMG_BASE), Image.Resampling.BILINEAR)
                bx = np.asarray(pb_crop_s, dtype=np.float32).transpose(2,0,1)/255.0
                by = np.asarray(pa_crop_s, dtype=np.float32).transpose(2,0,1)/255.0
                _append_to_main_npz(bx, by)

        for bx, by in zip(aug_pb, aug_ma):
            _append_to_main_npz(bx, by)
            
        total_real = int(len(rX))
        total_patches = int(len(pX))
        
        _log_real_pair(
            label=", ".join(norm_services),
            before_size=primary_b.size,
            after_size=master_a.size,
            total=total_real,
            patches_count=len(patches_b),
            references=ref_records,
            mode="n_to_1",
            multi_info={
                "composite_type": comp_type,
                "components_count": len(before_pils),
                "component_assets": comp_records,
                "summary_bn": det.get("summary_bn", "")
            }
        )
        
        ft_pid = None
        if trigger_finetune:
            ft_pid = _trigger_micro_finetune(steps=250, lr=5e-5)
            
        return {
            "ok": True,
            "mode": "n_to_1",
            "mode_title": det.get("mode_title", "কম্পোজিট অ্যাসেম্বলি (N-to-1)"),
            "composite_type": comp_type,
            "components_count": len(before_pils),
            "component_assets": comp_records,
            "augmented_samples": int(len(aug_pb)),
            "junction_patches": len(patches_b),
            "total_real_pairs": total_real,
            "total_patches_256": total_patches,
            "detected_services": norm_services,
            "summary_bn": det.get("summary_bn", ""),
            "micro_ft_triggered": bool(ft_pid),
            "micro_ft_pid": ft_pid
        }

    # 4. Case: N-to-M Batch Set
    results = []
    pairs_count = min(nb, na)
    for i in range(pairs_count):
        sub_res = learn_from_real_pair(
            before_pils[i], after_pils[i],
            label=label,
            augment=max(2, augment // pairs_count),
            trigger_finetune=(trigger_finetune and i == pairs_count - 1),
            reference_pils=reference_pils if i == 0 else None
        )
        results.append(sub_res)
        
    return {
        "ok": True,
        "mode": "n_to_m",
        "batch_pairs_learned": pairs_count,
        "results": results
    }
