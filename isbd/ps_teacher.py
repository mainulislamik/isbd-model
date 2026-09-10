"""
ISBD v1.00 — Photoshop 2026 Technique Teacher
Feeds the model REAL before/after pairs built from pro Photoshop techniques
(Sept 2026 skill update): LAB cast removal + color pop, Frequency Separation,
PHLEARN pro retouch order, Dodge & Burn, LAB L-channel sharpening.

Entry: python isbd/ps_teacher.py --source samples/selftest_input.png
Runs inside isbd-panel container (cv2 + torch present).
"""
import argparse
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from isbd.self_learner import learn_from_real_pair


def _pil(cv_img):
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


# ── Photoshop technique implementations (OpenCV equivalents) ──────────────────

def tech_lab_cast_removal(img):
    """LAB a/b cast fix + symmetric steepen = color pop (skill S18)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    for ch in (a, b):
        skew = float(ch.mean() - 127.5)
        if abs(skew) > 1.0:
            ch -= np.clip(skew * 0.6, -8, 8)
    lab[..., 1] = 127.5 + (lab[..., 1] - 127.5) * 1.18
    lab[..., 2] = 127.5 + (lab[..., 2] - 127.5) * 1.18
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def tech_frequency_separation(img):
    """True 8-bit FS recipe (skill S6): blemish smoothing, texture kept."""
    low = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    high = cv2.subtract(img, cv2.GaussianBlur(img, (9, 9), 2))
    return cv2.addWeighted(low, 0.85, high, 0.55, 0)


def tech_pro_retouch_order(img):
    """PHLEARN pro order (skill S20): exposure even -> FS -> D&B -> color -> sharpen."""
    # 1. Even exposure: CLAHE on L
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    L2 = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8)).apply(L)
    out = cv2.cvtColor(cv2.merge([L2, a, b]), cv2.COLOR_LAB2BGR)
    # 2. FS smoothing
    low = cv2.bilateralFilter(out, d=9, sigmaColor=80, sigmaSpace=80)
    high = cv2.subtract(out, cv2.GaussianBlur(out, (9, 9), 2))
    out = cv2.addWeighted(low, 0.86, high, 0.58, 0)
    # 3. D&B luminance sculpt
    lab2 = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    Lf = lab2[..., 0].astype(np.float32)
    Lf = Lf + 0.08 * (Lf - cv2.GaussianBlur(Lf, (0, 0), 12))
    lab2[..., 0] = np.clip(Lf, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    # 4. Vibrance-like boost (protect already-saturated pixels)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    s = hsv[..., 1]
    hsv[..., 1] = np.clip(s * np.where(s < 120, 1 + (120 - s) / 120 * 0.35, 1.06), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    # 5. Sharpen L only (fringe-free)
    lab3 = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    L3 = lab3[..., 0]
    lab3[..., 0] = cv2.addWeighted(L3, 1.35, cv2.GaussianBlur(L3, (0, 0), 1.2), -0.35, 0)
    return cv2.cvtColor(lab3, cv2.COLOR_LAB2BGR)


def tech_dodge_burn(img):
    """50%-gray Overlay equivalent (skill S6): local contrast sculpt."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[..., 0].astype(np.float32)
    detail = L - cv2.GaussianBlur(L, (0, 0), 8)
    lab[..., 0] = np.clip(L + 0.45 * detail, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def tech_lab_sharpen_l(img):
    """LAB L-channel-only sharpening (skill S18)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[..., 0]
    lab[..., 0] = cv2.addWeighted(L, 1.5, cv2.GaussianBlur(L, (0, 0), 1.5), -0.50, 0)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


TECHNIQUES = [
    ("lab_cast_removal",     tech_lab_cast_removal),
    ("frequency_separation", tech_frequency_separation),
    ("pro_retouch_order",    tech_pro_retouch_order),
    ("dodge_burn_sculpt",    tech_dodge_burn),
    ("lab_sharpen_l",        tech_lab_sharpen_l),
]


def make_before(img, rng):
    """Degrade source to RAW/unedited look: random cast + flat tone + soft focus."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 1] += rng.uniform(-10, 10)
    lab[..., 2] += rng.uniform(-10, 10)
    lab[..., 0] = 128 + (lab[..., 0] - 128) * 0.88
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    out = cv2.GaussianBlur(out, (5, 5), 1.0)
    return np.clip(out * rng.uniform(0.95, 1.05), 0, 255).astype(np.uint8)


def teach_from_source(source_path, rounds=3, augment=8):
    """Generate one teaching pair per technique per round and feed the model."""
    src = cv2.imread(str(source_path))
    if src is None:
        raise SystemExit(f"Cannot read source image: {source_path}")
    results = []
    for r in range(rounds):
        rng = random.Random(int(time.time() * 1000) % 999983 + r)
        before = make_before(src, rng)
        for name, fn in TECHNIQUES:
            after = fn(before)
            res = learn_from_real_pair(
                before_pil=_pil(before),
                after_pil=_pil(after),
                label=f"ps2026_{name}",
                augment=augment,
                trigger_finetune=True,
            )
            results.append({"technique": name, "round": r + 1,
                            "pairs": res["total_real"], "samples": res["augmented"]})
            print(f"[ps_teacher] r{r+1} {name}: +{res['augmented']} samples "
                  f"(total real pairs: {res['total_real']})", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser(description="ISBD Photoshop Technique Teacher")
    ap.add_argument("--source", default=str(ROOT / "samples" / "selftest_input.png"))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--augment", type=int, default=8)
    args = ap.parse_args()
    results = teach_from_source(Path(args.source), args.rounds, args.augment)
    print(f"\n[ps_teacher] DONE — {len(results)} teaching pairs ingested")
    for r in results:
        print(f"  - {r['technique']} (round {r['round']}): {r['samples']} samples")


if __name__ == "__main__":
    main()
