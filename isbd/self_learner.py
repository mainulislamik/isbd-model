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

    # Grid search across the image for candidate patch windows
    step = max(32, patch_size // 2)
    candidates = []

    for y in range(0, max(1, h - patch_size + 1), step):
        for x in range(0, max(1, w - patch_size + 1), step):
            score = float(diff[y:y + patch_size, x:x + patch_size].mean())
            candidates.append((score, x, y))

    # Always include bottom/right boundary windows if stepped over
    if (w - patch_size) > 0 and (h - patch_size) > 0:
        candidates.append((float(diff[-patch_size:, -patch_size:].mean()), w - patch_size, h - patch_size))

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

def _normalize_service_label(raw_label: str) -> str:
    """Normalize free-form tag or comma-separated list into standard service identifier."""
    if not raw_label:
        return "custom"
    clean = raw_label.lower().replace("-", "_").replace(" ", "_").strip()
    first = clean.split(",")[0].strip()
    for svc in COMMERCIAL_SERVICES:
        if svc in first or first in svc:
            return svc
    return first or "custom"


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


def _log_real_pair(label, before_size, after_size, total, patches_count=0):
    """Keep a human-readable journal of every real pair added."""
    log = []
    if REAL_LOG.exists():
        try:
            log = json.loads(REAL_LOG.read_text())
        except Exception:
            pass
    log.append({
        "ts"            : time.strftime("%Y-%m-%d %H:%M:%S"),
        "label"         : label,
        "before_px"     : f"{before_size[0]}×{before_size[1]}",
        "after_px"      : f"{after_size[0]}×{after_size[1]}",
        "patches_256"   : patches_count,
        "total_pairs"   : total
    })
    if len(log) > 500:
        log = log[-500:]
    REAL_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2))


# ── MAIN LEARNING API ──────────────────────────────────────────────────────────

def learn_from_real_pair(before_pil: Image.Image,
                         after_pil:  Image.Image,
                         label:      str = "",
                         augment:    int = 8,
                         trigger_finetune: bool = True) -> dict:
    """
    AUTONOMOUS MULTI-SCALE REAL PAIR LEARNING
    - Extracts global 64×64 thumbnail representation
    - Extracts up to 4 salient 256×256 high-resolution patches focused on modifications
    - Stores into dedicated 21-service bucket memory
    - Kicks off instant micro fine-tune with composite studio loss
    """
    service_key = _normalize_service_label(label)

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

    # 5. Save into service-specific memory bucket
    _save_to_service_bucket(service_key, aug_x, aug_y)

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

    _log_real_pair(label or service_key, before_pil.size, after_pil.size, total_real, patches_count=len(p_b_list))
    _record_state(total_real, total_synth,
                  note=f"Learned real pair '{service_key}': +{len(aug_x)} samples, {len(p_b_list)} 256px salient patches")

    # 8. Micro fine-tune so model learns right now with composite studio loss
    ft_pid = None
    if trigger_finetune:
        try:
            ft_pid = _trigger_micro_finetune(steps=200, lr=5e-5)
        except Exception:
            pass

    return {
        "ok"           : True,
        "label"        : service_key,
        "augmented"    : len(aug_x),
        "patches_256"  : len(p_b_list),
        "total_real"   : total_real,
        "total_pool"   : total_synth,
        "finetune_pid" : ft_pid,
        "buckets"      : get_bucket_stats()
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
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    rX, _ = _load_npz(REAL_POOL, img_size=IMG_BASE)
    sX, _ = _load_npz(SELF_POOL, img_size=IMG_BASE)
    return {
        "last_harvest"            : None,
        "total_real_pairs"        : int(len(rX)),
        "total_self_learned_pairs": int(len(sX)),
        "bucket_counts"           : get_bucket_stats(),
        "status"                  : "Ready"
    }
