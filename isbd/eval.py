"""
ISBD v1.00 — Evaluation on unseen data
Usage: .venv/bin/python isbd/eval.py [--full]
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from isbd.model import TinyUNet, SmallUNet
from isbd.data import make_pair

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"


def evaluate(ckpt: str = "", n: int = 100, seed_start: int = 9000):
    ckpt = ckpt or str(CKPT / "best.pt")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model_type = state.get("model_type", "tiny")
    if model_type == "small":
        model = SmallUNet()
    else:
        model = TinyUNet()
    model.load_state_dict(state["model"])
    model.eval()

    l1i, l1m, msei, msem = [], [], [], []
    for seed in range(seed_start, seed_start + n):
        x, y = make_pair(seed)
        with torch.no_grad():
            p = model(torch.from_numpy(x)[None])[0].numpy()
        l1i.append(np.abs(x - y).mean())
        l1m.append(np.abs(p - y).mean())
        msei.append(np.mean((x - y) ** 2))
        msem.append(np.mean((p - y) ** 2))

    step = state["step"]
    l1i_m, l1m_m = float(np.mean(l1i)), float(np.mean(l1m))
    msei_m, msem_m = float(np.mean(msei)), float(np.mean(msem))
    psnr = 10 * np.log10(1 / msem_m)
    line = (
        f"step {step} | L1 id {l1i_m:.4f} -> {l1m_m:.4f} "
        f"({(1 - l1m_m / l1i_m) * 100:+.1f}%) | "
        f"MSE id {msei_m:.5f} -> {msem_m:.5f} ({(1 - msem_m / msei_m) * 100:+.1f}%) | "
        f"PSNR {psnr:.2f}dB"
    )
    print(line)
    return {"step": step, "psnr": float(psnr), "mse_improve_pct": (1 - msem_m / msei_m) * 100, "line": line}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="500 samples instead of 100")
    args = ap.parse_args()
    evaluate(n=500 if args.full else 100)
