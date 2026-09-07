"""
ISBD v1.00 — Training Loop v2 (resumable, checkpoint-based)
Upgrades from v1:
- Gradient accumulation (effective larger batch on CPU)
- CosineAnnealingWarmRestarts (better loss landscape exploration)
- Sample image saving every 500 steps (progress monitoring)
- Compatible with SmallUNet and TinyUNet
"""
import os
import sys
import time
import json
import fcntl
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from isbd.model import SmallUNet, TinyUNet, param_count
from isbd.data import torch_dataset, IMG_SIZE
from isbd.thermal_guard import auto_cool_if_needed, get_cpu_temp

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"
CKPT.mkdir(exist_ok=True)
SAMPLES = ROOT / "samples"
PAUSE_FLAG = ROOT / "data" / "trainer_paused.flag"


def _check_pause():
    """Block training loop while pause flag exists. Prints status every 30s."""
    if not PAUSE_FLAG.exists():
        return
    print("[trainer] ⏸️  Pause flag detected — training paused. Waiting for resume...", flush=True)
    while PAUSE_FLAG.exists():
        time.sleep(5)
    print("[trainer] ▶️  Resumed!", flush=True)
SAMPLES.mkdir(exist_ok=True)

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


def save_sample_grid(model, device, step):
    """Save a 3-panel grid: degraded → restored → clean for visual monitoring."""
    try:
        from PIL import Image
        from isbd.data import make_pair
        # Use a fixed seed for consistent comparison across steps
        x, y = make_pair(42)
        x_t = torch.from_numpy(x).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x_t)
        # Convert to images
        x_img = (x.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        pred_img = (pred[0].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        y_img = (y.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        # Make grid
        h, w = IMG_SIZE, IMG_SIZE
        grid = Image.new("RGB", (w * 3 + 20, h + 10))
        grid.paste(Image.fromarray(x_img), (0, 0))
        grid.paste(Image.fromarray(pred_img), (w + 10, 0))
        grid.paste(Image.fromarray(y_img), (w * 2 + 20, 0))
        grid.save(SAMPLES / f"step_{step:06d}.png")
        # Keep only last20 samples
        samples = sorted(SAMPLES.glob("step_*.png"))
        if len(samples) > 20:
            for s in samples[:-20]:
                s.unlink()
    except Exception as e:
        print(f"[monitor] sample save skipped: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200, help="training steps this run")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--model", choices=["tiny", "small"], default="small",
                    help="Model architecture: tiny (117K) or small (500K)")
    ap.add_argument("--accum-steps", type=int, default=2,
                    help="Gradient accumulation steps (effective batch = batch * accum_steps)")
    ap.add_argument("--warm-restart", type=int, default=3000,
                    help="Cosine annealing warm restart period (0=disabled)")
    args = ap.parse_args()

    torch.set_num_threads(1)
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

    # Model selection
    if args.model == "small":
        model = SmallUNet()
    else:
        model = TinyUNet()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Scheduler: warm restart or standard cosine
    if args.warm_restart > 0:
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=args.warm_restart, T_mult=2, eta_min=1e-6)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5000)

    # ── Resume from checkpoint (backwards-compatible with old TinyUNet checkpoints) ──
    start_step = 0
    best = float("inf")
    last_path = CKPT / "last.pt"
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=True)
        ckpt_arch = state.get("model_type", "tiny")  # old checkpoints have no model_type → "tiny"
        try:
            model.load_state_dict(state["model"])
            opt.load_state_dict(state["opt"])
            sched.load_state_dict(state["sched"])
            start_step = state["step"]
            best = state.get("best", float("inf"))
            print(f"[resume] from step {start_step} (best loss {best:.4f})")
        except RuntimeError:
            # Architecture mismatch (TinyUNet→SmallUNet) — cannot load weights,
            # but ALWAYS recover the step count from history.json so total is preserved
            print(f"[resume] Architecture mismatch ({ckpt_arch}→{args.model}) — weights fresh, step count preserved")
            hist = load_history()
            start_step = hist.get("total_steps", 0)
            best = hist.get("best", float("inf"))
            print(f"[resume] recovered total_steps={start_step} from history.json")

    effective_batch = args.batch * args.accum_steps
    print(f"ISBD v1.00 | {args.model} | params {param_count(model):,} | img {IMG_SIZE}px "
          f"| batch {args.batch}×{args.accum_steps}={effective_batch} | lr {args.lr} | device {device}",
          flush=True)

    ds = torch_dataset(augment_factor=12)(3000)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True, prefetch_factor=3, persistent_workers=True)

    h = load_history()
    model.train()
    step = start_step
    t0 = time.time()
    running = []
    accum_count = 0

    opt.zero_grad(set_to_none=True)

    for epoch in range(10000):
        for x, y in dl:
            if step >= start_step + args.steps:
                break
            pred = model(x)
            loss = nn.functional.l1_loss(pred, y) + 0.1 * nn.functional.mse_loss(pred, y)

            # Gradient accumulation: scale loss by accumulation steps
            loss_scaled = loss / args.accum_steps
            loss_scaled.backward()
            accum_count += 1

            if accum_count >= args.accum_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                accum_count = 0

            step += 1
            running.append(loss.item())
            h["steps"].append(step)
            h["losses"].append(round(loss.item(), 5))
            h["total_steps"] = step

            # ── Steady Low-Heat Pacing & Dynamic Guard ──
            cpu_t = get_cpu_temp()
            if cpu_t >= 82.0:
                time.sleep(0.10)  # cool-down pacing
            elif cpu_t >= 76.0:
                time.sleep(0.04)  # moderate pacing
            else:
                time.sleep(0.01)  # minimal breather

            if step % 25 == 0:
                dt = time.time() - t0
                lr_now = opt.param_groups[0]["lr"]
                print(f"step {step:6d} | loss {sum(running)/len(running):.4f} | {dt/25:.2f}s/step | lr {lr_now:.2e} | CPU {cpu_t:.0f}°C", flush=True)
                running = []
                t0 = time.time()

            # Smart Thermal Guard check every 10 steps
            if step % 10 == 0:
                auto_cool_if_needed(high_threshold=86.0, target_cool=74.0)
                _check_pause()  # Honor pause flag from panel UI

            # ── Sample monitoring: save grid every500 steps ──
            if step % 500 == 0:
                save_sample_grid(model, device, step)

            # ── Checkpoint save ──
            if step % 100 == 0 or step >= start_step + args.steps:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "step": step, "best": best,
                            "model_type": args.model}, last_path)
                if loss.item() < best:
                    best = loss.item()
                    torch.save({"model": model.state_dict(), "step": step,
                                "model_type": args.model}, CKPT / "best.pt")
        if step >= start_step + args.steps:
            break

    # final save
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "step": step, "best": best,
                "model_type": args.model}, last_path)
    save_history(h)
    save_sample_grid(model, device, step)
    print(f"[done] reached step {step} | last loss {h['losses'][-1]} | best {best:.4f}")


if __name__ == "__main__":
    main()
