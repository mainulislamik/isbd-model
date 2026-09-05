"""
ISBD v1.00 — Real-pair fine-tuner (panel backend)
Fine-tunes the live model on designer before/after pairs (data/pairs.npz).
Waits for the 24/7 trainer's train.lock so the two NEVER collide.
Uploaded images are never stored — panel.py keeps only 64px tensors.
"""
import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from isbd.model import TinyUNet
from isbd.data import torch_dataset

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"
NPZ = ROOT / "data" / "pairs.npz"
LOCK = CKPT / "train.lock"
HIST = CKPT / "history.json"


class RealPairs(Dataset):
    def __init__(self, X, Y):
        self.X, self.Y = X, Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x, y = self.X[i].copy(), self.Y[i].copy()
        # light augmentation: h-flip + rot90 (keeps few pairs from overfitting)
        if np.random.rand() < 0.5:
            x, y = x[:, :, ::-1].copy(), y[:, :, ::-1].copy()
        k = np.random.randint(4)
        x, y = np.rot90(x, k, (1, 2)).copy(), np.rot90(y, k, (1, 2)).copy()
        return torch.from_numpy(x), torch.from_numpy(y)


def holdout_metrics(model, X, Y):
    model.eval()
    with torch.no_grad():
        xt, yt = torch.from_numpy(X), torch.from_numpy(Y)
        pred = model(xt).clamp(0, 1)
        mse = ((pred - yt) ** 2).mean().item()
        idmse = ((xt - yt) ** 2).mean().item()
        m = {
            "l1": (pred - yt).abs().mean().item(),
            "idl1": (xt - yt).abs().mean().item(),
            "psnr": 10 * np.log10(1.0 / mse) if mse > 0 else 99.0,
            "idpsnr": 10 * np.log10(1.0 / idmse) if idmse > 0 else 99.0,
        }
    model.train()
    return m


def append_history(step, losses):
    try:
        h = json.loads(HIST.read_text())
    except Exception:
        h = {"steps": [], "losses": [], "total_steps": 0}
    base = step - len(losses)
    for i, l in enumerate(losses):
        h["steps"].append(base + i + 1)
        h["losses"].append(round(l, 5))
    h["total_steps"] = step
    if len(h["steps"]) > 4000:  # same bounded policy as train.py
        h["steps"] = h["steps"][:2000][::20] + h["steps"][2000:]
        h["losses"] = h["losses"][:2000][::20] + h["losses"][2000:]
    HIST.write_text(json.dumps(h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wait-lock", type=int, default=900)
    args = ap.parse_args()

    torch.set_num_threads(4)
    if not NPZ.exists():
        print("[ft] data/pairs.npz নেই — প্রথমে প্যানেলে পেয়ার আপলোড করুন")
        sys.exit(1)
    d = np.load(NPZ)
    X, Y = d["X"].astype(np.float32), d["Y"].astype(np.float32)
    n = len(X)
    print(f"[ft] {n} real pair(s) | {args.steps} steps | lr {args.lr}", flush=True)

    # holdout split (20% if we have >=5 pairs) for honest before/after metrics
    idx = np.arange(n)
    rng = np.random.default_rng(7)
    rng.shuffle(idx)
    n_hold = min(max(1, n // 5), n - 1) if n >= 5 else 0
    hold, tr = idx[:n_hold], idx[n_hold:]
    Xtr, Ytr = X[tr], Y[tr]
    Xh, Yh = X[hold], Y[hold]

    # wait for the 24/7 trainer's lock (it frees between rounds)
    lock = open(LOCK, "a+")
    t0 = time.time()
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except OSError:
            if time.time() - t0 > args.wait_lock:
                print("[ft] lock পাওয়া যায়নি — বাতিল; একটু পরে আবার চেষ্টা করুন", flush=True)
                sys.exit(2)
            print("[ft] 24/7 ট্রেনারের রাউন্ড চলছে — lock-এর অপেক্ষায়…", flush=True)
            time.sleep(10)

    try:
        model = TinyUNet()
        state = torch.load(CKPT / "last.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        start_step, best = state.get("step", 0), state.get("best", float("inf"))
        print(f"[ft] resume from step {start_step}", flush=True)

        if n_hold:
            m = holdout_metrics(model, Xh, Yh)
            print(f"[ft] BEFORE holdout: L1 {m['l1']:.4f} (id {m['idl1']:.4f}) | PSNR {m['psnr']:.2f}dB (id {m['idpsnr']:.2f}dB)", flush=True)

        # ~40% real / 60% synthetic mix (synthetic keeps old skills alive)
        reps = max(1, int(round(1200 / max(1, len(Xtr)))))
        real_ds = RealPairs(np.repeat(Xtr, reps, axis=0), np.repeat(Ytr, reps, axis=0))
        ds = ConcatDataset([real_ds, torch_dataset()(2000)])
        dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5000)  # match train.py
        model.train()
        step, done, losses = start_step, 0, []
        while done < args.steps:
            for x, y in dl:
                if done >= args.steps:
                    break
                pred = model(x)
                loss = nn.functional.l1_loss(pred, y) + 0.1 * nn.functional.mse_loss(pred, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                done += 1
                step += 1
                losses.append(loss.item())
                if done % 20 == 0:
                    print(f"ft step {step} | loss {sum(losses[-20:]) / 20:.4f} | {done}/{args.steps}", flush=True)

        avg = sum(losses[-100:]) / max(1, len(losses[-100:]))
        new_best = min(best, avg)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "step": step, "best": new_best}, CKPT / "last.pt")
        if avg < best:
            torch.save({"model": model.state_dict(), "step": step, "best": new_best}, CKPT / "best.pt")
            print(f"[ft] new best {new_best:.4f} -> best.pt", flush=True)
        append_history(step, losses)
        print(f"[ft] done: step {start_step} -> {step} | avg loss {avg:.4f}", flush=True)
        if n_hold:
            m = holdout_metrics(model, Xh, Yh)
            print(f"[ft] AFTER holdout: L1 {m['l1']:.4f} (id {m['idl1']:.4f}) | PSNR {m['psnr']:.2f}dB (id {m['idpsnr']:.2f}dB)", flush=True)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    main()
