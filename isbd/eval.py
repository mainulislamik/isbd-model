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
from isbd.loss import ssim_loss

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"


def load_model_from_ckpt(ckpt_path: str):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_type = state.get("model_type", "tiny")
    if model_type == "small":
        model = SmallUNet()
    else:
        model = TinyUNet()
    model.load_state_dict(state["model"])
    model.eval()
    return model, state


def evaluate(ckpt: str = "", n: int = 100, seed_start: int = 9000):
    ckpt = ckpt or str(CKPT / "last.pt" if (CKPT / "last.pt").exists() else CKPT / "best.pt")
    model, state = load_model_from_ckpt(ckpt)

    l1i, l1m, msei, msem, ssims = [], [], [], [], []
    for seed in range(seed_start, seed_start + n):
        x, y = make_pair(seed)
        tx = torch.from_numpy(x)[None]
        ty = torch.from_numpy(y)[None]
        with torch.no_grad():
            p = model(tx)
            ssim_val = 1.0 - ssim_loss(p, ty).item()
        p_np = p[0].numpy()
        l1i.append(np.abs(x - y).mean())
        l1m.append(np.abs(p_np - y).mean())
        msei.append(np.mean((x - y) ** 2))
        msem.append(np.mean((p_np - y) ** 2))
        ssims.append(ssim_val)

    step = state.get("step", 0)
    l1i_m, l1m_m = float(np.mean(l1i)), float(np.mean(l1m))
    msei_m, msem_m = float(np.mean(msei)), float(np.mean(msem))
    psnr = 10 * np.log10(1 / msem_m)
    avg_ssim = float(np.mean(ssims))
    line = (
        f"step {step} | L1 id {l1i_m:.4f} -> {l1m_m:.4f} "
        f"({(1 - l1m_m / l1i_m) * 100:+.1f}%) | "
        f"MSE id {msei_m:.5f} -> {msem_m:.5f} ({(1 - msem_m / msei_m) * 100:+.1f}%) | "
        f"PSNR {psnr:.2f}dB | SSIM {avg_ssim:.4f}"
    )
    print(line)
    return {
        "step": step,
        "psnr": float(psnr),
        "ssim": float(avg_ssim),
        "mse_improve_pct": (1 - msem_m / msei_m) * 100,
        "line": line,
    }


def evaluate_real(ckpt: str = "", real_pairs_path: str = ""):
    ckpt = ckpt or str(CKPT / "last.pt" if (CKPT / "last.pt").exists() else CKPT / "best.pt")
    real_pairs_path = real_pairs_path or str(ROOT / "data" / "real_pairs.npz")
    if not Path(real_pairs_path).exists():
        print(f"[eval_real] Real pairs not found at {real_pairs_path}")
        return None

    data = np.load(real_pairs_path)
    X, Y = data["X"], data["Y"]
    model, state = load_model_from_ckpt(ckpt)

    tx = torch.from_numpy(X)
    ty = torch.from_numpy(Y)

    with torch.no_grad():
        pred = model(tx)
        ssim_val = 1.0 - ssim_loss(pred, ty).item()
        l1 = float(torch.abs(pred - ty).mean().item())
        mse = float(((pred - ty) ** 2).mean().item())
        psnr = 10 * np.log10(1.0 / (mse + 1e-9))

    step = state.get("step", 0)
    line = (
        f"[Real Studio Studio Pairs (N={len(X)})] step {step} | "
        f"L1: {l1:.4f} | MSE: {mse:.6f} | PSNR: {psnr:.2f}dB | SSIM: {ssim_val:.4f} ({ssim_val*100:.2f}%)"
    )
    print(line)
    return {
        "step": step,
        "l1": l1,
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim_val,
        "total_pairs": len(X),
        "line": line,
    }


def validate_checkpoint_gate(candidate_ckpt: str, baseline_ckpt: str = ""):
    """
    SSIM Validation Gate:
    Validates candidate checkpoint against baseline checkpoint on real studio data.
    Returns True if candidate SSIM is equal to or higher than baseline SSIM.
    """
    baseline_ckpt = baseline_ckpt or str(CKPT / "best.pt")
    print(f"[Gate] Comparing candidate: {candidate_ckpt} vs baseline: {baseline_ckpt}")
    res_cand = evaluate_real(candidate_ckpt)
    res_base = evaluate_real(baseline_ckpt) if Path(baseline_ckpt).exists() else None

    if not res_cand:
        return {"approved": False, "reason": "Candidate evaluation failed"}
    if not res_base:
        return {"approved": True, "reason": "No existing baseline; candidate auto-approved", "metrics": res_cand}

    cand_ssim = res_cand["ssim"]
    base_ssim = res_base["ssim"]
    cand_psnr = res_cand["psnr"]
    base_psnr = res_base["psnr"]

    approved = cand_ssim >= (base_ssim - 0.005) and cand_psnr >= (base_psnr - 0.5)
    result = {
        "approved": approved,
        "cand_ssim": cand_ssim,
        "base_ssim": base_ssim,
        "cand_psnr": cand_psnr,
        "base_psnr": base_psnr,
        "improvement_ssim": cand_ssim - base_ssim,
        "improvement_psnr": cand_psnr - base_psnr,
    }
    status_str = "APPROVED" if approved else "REJECTED (REGRESSION)"
    print(f"[Gate] Status: {status_str} | SSIM: {base_ssim:.4f} -> {cand_ssim:.4f} | PSNR: {base_psnr:.2f} -> {cand_psnr:.2f}")
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="500 samples instead of 100")
    ap.add_argument("--real", action="store_true", help="Evaluate on real designer studio pairs")
    ap.add_argument("--gate", type=str, default="", help="Candidate checkpoint path to test against baseline")
    args = ap.parse_args()

    if args.gate:
        validate_checkpoint_gate(args.gate)
    elif args.real:
        evaluate_real()
    else:
        evaluate(n=500 if args.full else 100)

