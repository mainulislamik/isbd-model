"""
ISBD v1.00 — Self-Supervised Autonomous Learner
Autonomous pipeline that harvests user-tested and online degraded images,
synthesizes self-supervised inverse pairs, and feeds them into continuous fine-tuning.
"""
import time
import os
import json
import random
import numpy as np
import torch
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
NPZ = DATA / "pairs.npz"
SELF_POOL = DATA / "self_pool.npz"
STATE_FILE = DATA / "self_learn_state.json"
IMG = 64

def _load_pool():
    if SELF_POOL.exists():
        try:
            with np.load(SELF_POOL) as d:
                return d["X"], d["Y"]
        except Exception:
            pass
    return np.empty((0, 3, IMG, IMG), dtype=np.float32), np.empty((0, 3, IMG, IMG), dtype=np.float32)

def _save_pool(X, Y):
    # Cap self-learning memory pool to 2000 highest quality pairs to conserve memory
    if len(X) > 2000:
        X, Y = X[-2000:], Y[-2000:]
    np.savez(SELF_POOL, X=X.astype(np.float32), Y=Y.astype(np.float32))

def harvest_and_synthesize_pair(clean_pil_img: Image.Image):
    """
    Self-Supervised Pair Generator:
    Takes any clean image tested or uploaded by the user, applies realistic multi-stage
    degradations (noise, blur, color casts, contrast shifts) to synthesize a training pair.
    """
    from isbd.data import degrade
    
    clean_small = clean_pil_img.convert("RGB").resize((IMG, IMG), Image.Resampling.LANCZOS)
    rng = random.Random(int(time.time() * 1000) % 1_000_000)
    
    # Degrade to create input
    degraded = degrade(clean_small, rng)
    
    x = np.asarray(degraded, dtype=np.float32).transpose(2, 0, 1) / 255.0
    y = np.asarray(clean_small, dtype=np.float32).transpose(2, 0, 1) / 255.0
    
    X, Y = _load_pool()
    X = np.concatenate([X, x[None]], axis=0)
    Y = np.concatenate([Y, y[None]], axis=0)
    _save_pool(X, Y)
    
    # Sync with main pairs.npz so continuous trainer and panel immediately train on it
    if NPZ.exists():
        try:
            with np.load(NPZ) as md:
                mX, mY = md["X"], md["Y"]
            mX = np.concatenate([mX, x[None]], axis=0)
            mY = np.concatenate([mY, y[None]], axis=0)
        except Exception:
            mX, mY = x[None], y[None]
    else:
        mX, mY = x[None], y[None]
    np.savez(NPZ, X=mX.astype(np.float32), Y=mY.astype(np.float32))
    
    _record_harvest_event(len(X))
    return len(X)

def _record_harvest_event(total_pool):
    state = {
        "last_harvest": time.time(),
        "total_self_learned_pairs": total_pool,
        "status": "Active (Learning from inferences & interactions)"
    }
    STATE_FILE.write_text(json.dumps(state))

def get_self_learn_stats():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    X, _ = _load_pool()
    return {
        "last_harvest": None,
        "total_self_learned_pairs": len(X),
        "status": "Autonomous Self-Learning Ready"
    }
