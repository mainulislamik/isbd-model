"""
ISBD v1.00 — Training Loop (resumable, checkpoint-based)
Trains degraded->clean image restoration. Slow CPU training: keep steps small,
checkpoint often, resume automatically from checkpoints/last.pt
"""
import os
import sys
import time
import json
import fcntl
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from isbd.model import TinyUNet, param_count
from isbd.data import torch_dataset, IMG_SIZE

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"
CKPT.mkdir(exist_ok=True)

HISTORY = CKPT / "history.json"


def load_history():
    if HISTORY.exists():
        h = json.loads(HISTORY.read_text())
        # keep history bounded: downsample to every 20th step beyond the recent 2000
        if len(h.get("losses", [])) > 4000:
            old = h["losses"][:-2000]
            recent = h["losses"][-2000:]
            old_steps = h["steps"][:-2000]
            recent_steps = h["steps"][-2000:]
            h["losses"] = old[::20] + recent
            h["steps"] = old_steps[::20] + recent_steps
        return h
    return {"steps": [], "losses": [], "total_steps": 0}


def save_history(h):
    HISTORY.write_text(json.dumps(h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200, help="training steps this run")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--resume", action="store_true", default=True)
    args = ap.parse_args()

    torch.manual_seed(7)
    device = "cpu"

    # single-instance lock: never two trainings on the same checkpoint
    lockfile = CKPT / "train.lock"
    lockfile.parent.mkdir(exist_ok=True)
    lock = open(lockfile, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("[lock] another training is running — exiting")
        sys.exit(0)
    lock.write(str(os.getpid()))

    model = TinyUNet()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5000)

    start_step = 0
    best = float("inf")
    last_path = CKPT / "last.pt"
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start_step = state["step"]
        best = state.get("best", float("inf"))
        print(f"[resume] from step {start_step} (best loss {best:.4f})")

    print(f"ISBD v1.00 | params {param_count(model):,} | img {IMG_SIZE}px | batch {args.batch} | device {device}")

    ds = torch_dataset()(3000)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)

    h = load_history()
    model.train()
    step = start_step
    t0 = time.time()
    running = []

    for epoch in range(10000):
        for x, y in dl:
            if step >= start_step + args.steps:
                break
            pred = model(x)
            loss = nn.functional.l1_loss(pred, y) + 0.1 * nn.functional.mse_loss(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            step += 1
            running.append(loss.item())
            h["steps"].append(step)
            h["losses"].append(round(loss.item(), 5))
            h["total_steps"] = step

            if step % 25 == 0:
                dt = time.time() - t0
                print(f"step {step:6d} | loss {sum(running)/len(running):.4f} | {dt/25:.1f}s/step", flush=True)
                running = []
                t0 = time.time()

            if step % 100 == 0 or step >= start_step + args.steps:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "step": step, "best": best}, last_path)
                if loss.item() < best:
                    best = loss.item()
                    torch.save({"model": model.state_dict(), "step": step},
                               CKPT / "best.pt")
        if step >= start_step + args.steps:
            break

    # final save
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "step": step, "best": best}, last_path)
    save_history(h)
    print(f"[done] reached step {step} | last loss {h['losses'][-1]} | best {best:.4f}")


if __name__ == "__main__":
    main()
