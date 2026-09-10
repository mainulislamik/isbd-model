"""
ISBD v1.00 — Self-Supervised + Real Pair Learner
Two learning modes:
  1. Synthetic: harvest_and_synthesize_pair(clean_img) — auto-degrades & self-learns
  2. Real:      learn_from_real_pair(before_img, after_img) — learns from true human edits
                This is the most powerful mode — the AI sees exactly what a professional
                editor did and memorises the transformation.
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

ROOT        = Path(__file__).resolve().parent.parent
DATA        = ROOT / "data"
DATA.mkdir(exist_ok=True)
NPZ         = DATA / "pairs.npz"
SELF_POOL   = DATA / "self_pool.npz"
REAL_POOL   = DATA / "real_pairs.npz"     # NEW: stores genuine human-edited pairs
STATE_FILE  = DATA / "self_learn_state.json"
REAL_LOG    = DATA / "real_pairs_log.json"
IMG         = 64


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_npz(path):
    if Path(path).exists():
        try:
            with np.load(path) as d:
                return d["X"], d["Y"]
        except Exception:
            pass
    return (np.empty((0, 3, IMG, IMG), dtype=np.float32),
            np.empty((0, 3, IMG, IMG), dtype=np.float32))


def _save_npz(path, X, Y, cap=4000):
    if len(X) > cap:
        X, Y = X[-cap:], Y[-cap:]
    np.savez(path, X=X.astype(np.float32), Y=Y.astype(np.float32))


def _img_to_tensor(pil_img):
    """Resize to 64×64, normalise → (3,64,64) float32 array."""
    small = pil_img.convert("RGB").resize((IMG, IMG), Image.Resampling.LANCZOS)
    return np.asarray(small, dtype=np.float32).transpose(2, 0, 1) / 255.0


def _append_to_main_npz(x, y):
    """Append a single pair (3,64,64) to the shared pairs.npz used by continuous trainer."""
    mX, mY = _load_npz(NPZ)
    mX = np.concatenate([mX, x[None]], axis=0)
    mY = np.concatenate([mY, y[None]], axis=0)
    _save_npz(NPZ, mX, mY, cap=6000)


def _record_state(total_real, total_synthetic, note=""):
    state = {
        "last_harvest"            : time.time(),
        "total_real_pairs"        : total_real,
        "total_self_learned_pairs": total_synthetic,
        "note"                    : note,
        "status"                  : "Active — learning from REAL human edits + self-synthesis"
    }
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False))


def _log_real_pair(label, before_size, after_size, total):
    """Keep a human-readable journal of every real pair added."""
    log = []
    if REAL_LOG.exists():
        try:
            log = json.loads(REAL_LOG.read_text())
        except Exception:
            pass
    log.append({
        "ts"          : time.strftime("%Y-%m-%d %H:%M:%S"),
        "label"       : label,
        "before_px"   : f"{before_size[0]}×{before_size[1]}",
        "after_px"    : f"{after_size[0]}×{after_size[1]}",
        "total_pairs" : total
    })
    if len(log) > 500:
        log = log[-500:]
    REAL_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2))


# ── PUBLIC API ─────────────────────────────────────────────────────────────────

def learn_from_real_pair(before_pil: Image.Image,
                         after_pil:  Image.Image,
                         label:      str = "",
                         augment:    int = 8,
                         trigger_finetune: bool = True) -> dict:
    """
    REAL PAIR LEARNING — the AI directly observes a human professional edit.

    Parameters
    ----------
    before_pil       : original / unedited image
    after_pil        : professionally edited result
    label            : optional tag, e.g. "retouching", "color_correction"
    augment          : how many augmented copies to generate from one pair (default 8)
                       gives more training signal without needing more photos
    trigger_finetune : if True, kicks off a 200-step micro fine-tune immediately so
                       the AI absorbs the new knowledge right away

    Returns
    -------
    dict with pair count, augmentation count, and finetune status
    """
    x_base = _img_to_tensor(before_pil)   # noisy / raw  → input  to AI
    y_base = _img_to_tensor(after_pil)    # clean / edited → target for AI

    # ── Data augmentation: flip + rotate → N copies from 1 pair ──────────────
    pairs_x, pairs_y = [x_base], [y_base]

    src_b = before_pil.convert("RGB").resize((IMG, IMG), Image.Resampling.LANCZOS)
    src_a = after_pil.convert("RGB").resize((IMG, IMG), Image.Resampling.LANCZOS)

    aug_ops = [
        (Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_LEFT_RIGHT),
        (Image.Transpose.FLIP_TOP_BOTTOM, Image.Transpose.FLIP_TOP_BOTTOM),
        (Image.Transpose.ROTATE_90,       Image.Transpose.ROTATE_90),
        (Image.Transpose.ROTATE_180,      Image.Transpose.ROTATE_180),
        (Image.Transpose.ROTATE_270,      Image.Transpose.ROTATE_270),
        (Image.Transpose.TRANSPOSE,       Image.Transpose.TRANSPOSE),
        (Image.Transpose.TRANSVERSE,      Image.Transpose.TRANSVERSE),
    ]

    for i, (op_b, op_a) in enumerate(aug_ops[:max(0, augment - 1)]):
        try:
            ab = src_b.transpose(op_b)
            aa = src_a.transpose(op_a)
            pairs_x.append(np.asarray(ab, dtype=np.float32).transpose(2, 0, 1) / 255.0)
            pairs_y.append(np.asarray(aa, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        except Exception:
            pass

    # stack
    aug_x = np.stack(pairs_x, axis=0)   # (N, 3, 64, 64)
    aug_y = np.stack(pairs_y, axis=0)

    # ── Save to real_pairs.npz ────────────────────────────────────────────────
    rX, rY = _load_npz(REAL_POOL)
    rX = np.concatenate([rX, aug_x], axis=0)
    rY = np.concatenate([rY, aug_y], axis=0)
    _save_npz(REAL_POOL, rX, rY, cap=8000)   # real pairs get a bigger cap

    # ── Sync to shared pairs.npz (used by 24/7 continuous trainer) ────────────
    for xv, yv in zip(aug_x, aug_y):
        _append_to_main_npz(xv, yv)

    # ── Also add to self-pool so the pool stats stay consistent ──────────────
    sX, sY = _load_npz(SELF_POOL)
    sX = np.concatenate([sX, aug_x], axis=0)
    sY = np.concatenate([sY, aug_y], axis=0)
    _save_npz(SELF_POOL, sX, sY)

    total_real = int(len(rX))
    total_synth = int(len(sX))

    _log_real_pair(label or "untagged", before_pil.size, after_pil.size, total_real)
    _record_state(total_real, total_synth,
                  note=f"Last real pair: '{label or 'untagged'}' (+{len(aug_x)} augmented samples)")

    # ── Micro fine-tune so model learns RIGHT NOW ─────────────────────────────
    ft_pid = None
    if trigger_finetune:
        try:
            ft_pid = _trigger_micro_finetune(steps=200, lr=5e-5)
        except Exception:
            pass

    return {
        "ok"           : True,
        "label"        : label or "untagged",
        "augmented"    : len(aug_x),
        "total_real"   : total_real,
        "total_pool"   : total_synth,
        "finetune_pid" : ft_pid
    }


def _docker_safe_python() -> str:
    """Resolve python interpreter for background fine-tune subprocesses.
    Prefers the project venv (native install); inside Docker falls back to
    the container's own python (sys.executable). Mirrors panel.py helper."""
    venv_py = ROOT / ".venv" / "bin" / "python"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def _trigger_micro_finetune(steps=200, lr=5e-5):
    """
    Launch a quick background fine-tune so the AI immediately bakes in the new pair.
    Uses realfinetune.py with --wait-lock so it never clashes with the 24/7 trainer.
    """
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


def get_real_pair_log(limit=20):
    """Return the last N real-pair learning events for the UI."""
    if REAL_LOG.exists():
        try:
            log = json.loads(REAL_LOG.read_text())
            return log[-limit:]
        except Exception:
            pass
    return []


def get_real_pair_count():
    rX, _ = _load_npz(REAL_POOL)
    return int(len(rX))


# ── EXISTING synthetic harvester (unchanged) ──────────────────────────────────

def harvest_and_synthesize_pair(clean_pil_img: Image.Image):
    """
    Synthetic Pair Generator (original mode):
    Takes any clean image, applies realistic degradations, self-learns from it.
    """
    from isbd.data import degrade
    clean_small = clean_pil_img.convert("RGB").resize((IMG, IMG), Image.Resampling.LANCZOS)
    rng = random.Random(int(time.time() * 1000) % 1_000_000)
    degraded = degrade(clean_small, rng)
    x = np.asarray(degraded,    dtype=np.float32).transpose(2, 0, 1) / 255.0
    y = np.asarray(clean_small, dtype=np.float32).transpose(2, 0, 1) / 255.0

    sX, sY = _load_npz(SELF_POOL)
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
    rX, _ = _load_npz(REAL_POOL)
    sX, _ = _load_npz(SELF_POOL)
    return {
        "last_harvest"            : None,
        "total_real_pairs"        : int(len(rX)),
        "total_self_learned_pairs": int(len(sX)),
        "status"                  : "Ready"
    }
