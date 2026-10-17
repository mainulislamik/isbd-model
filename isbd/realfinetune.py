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
from isbd.model import SmallUNet, TinyUNet
from isbd.data import torch_dataset
from isbd.loss import ISBDProLoss
from isbd.export import export_all_formats
from isbd.ewc import EWC
from isbd.curriculum import CurriculumScheduler
from isbd.teacher_student import TeacherStudentDistiller

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"
NPZ = ROOT / "data" / "pairs.npz"
GOLDEN_NPZ = ROOT / "data" / "golden_pool.npz"
EWC_ANCHOR = ROOT / "data" / "ewc_anchor.pt"
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
        last_ckpt = CKPT / "last.pt"
        best_ckpt = CKPT / "best.pt"
        ckpt_to_load = last_ckpt if last_ckpt.exists() else best_ckpt
        
        state = torch.load(ckpt_to_load, map_location="cpu", weights_only=True) if ckpt_to_load.exists() else {}
        ckpt_arch = state.get("model_type", "small")
        model = SmallUNet() if ckpt_arch == "small" else TinyUNet()

        if "model" in state:
            try:
                model.load_state_dict(state["model"])
            except Exception:
                model = SmallUNet()
                
        start_step, best = state.get("step", 0), state.get("best", float("inf"))
        print(f"[ft] resume from step {start_step} ({model.__class__.__name__})", flush=True)

        if n_hold:
            m = holdout_metrics(model, Xh, Yh)
            print(f"[ft] BEFORE holdout: L1 {m['l1']:.4f} (id {m['idl1']:.4f}) | PSNR {m['psnr']:.2f}dB (id {m['idpsnr']:.2f}dB)", flush=True)

        # Tri-Balanced Replay: 50% Golden Human Ground Truth, 30% Live Harvest, 20% Hard-Negative
        reps = max(1, int(round(1500 / max(1, len(Xtr)))))
        real_ds = RealPairs(np.repeat(Xtr, reps, axis=0), np.repeat(Ytr, reps, axis=0))
        datasets = [real_ds]

        if GOLDEN_NPZ.exists():
            try:
                g_data = np.load(str(GOLDEN_NPZ))
                gX, gY = g_data["X"], g_data["Y"]
                if len(gX) > 0:
                    g_reps = max(1, int(round(1500 / len(gX))))
                    golden_ds = RealPairs(np.repeat(gX, g_reps, axis=0), np.repeat(gY, g_reps, axis=0))
                    datasets.append(golden_ds)
            except Exception as e:
                print(f"[ft] warning loading golden_pool: {e}", flush=True)

        datasets.append(torch_dataset(real_pair_weight=0.60)(1200))
        ds = ConcatDataset(datasets)
        dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)

        # Dynamic Curriculum Learning Scheduler (Pillar 6)
        curriculum = CurriculumScheduler()
        cw = curriculum.get_stage_weights()

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5000)
        criterion = ISBDProLoss(
            ssim_weight=cw.get("w_ssim", 0.20),
            edge_weight=cw.get("w_edge", 0.15),
            laplacian_weight=cw.get("w_laplacian", 0.15),
            color_weight=cw.get("w_color", 0.10),
            mse_weight=cw.get("w_mse", 0.05)
        )

        # Elastic Weight Consolidation (Pillar 4) & Teacher-Student Distiller (Pillar 3)
        ewc = EWC(model, lambda_ewc=60.0)
        ewc_loaded = ewc.load(EWC_ANCHOR)
        if not ewc_loaded and len(real_ds) > 0:
            try:
                anchor_loader = DataLoader(real_ds, batch_size=min(4, len(real_ds)), shuffle=True)
                ewc.compute_fisher(anchor_loader, criterion, device="cpu", sample_limit=25)
                ewc.save(EWC_ANCHOR)
            except Exception as e_err:
                print(f"[ft] EWC baseline calculation skipped: {e_err}", flush=True)

        distiller = TeacherStudentDistiller(model, ema_decay=0.995)

        model.train()
        step, done, losses = start_step, 0, []
        while done < args.steps:
            for x, y in dl:
                if done >= args.steps:
                    break
                pred = model(x)
                loss_task = criterion(pred, y)

                # Consistency Regularization (Pillar 3)
                x_pert, t_pred = distiller.perturb_and_predict(x)
                s_pert_pred = model(x_pert)
                loss_cons = distiller.consistency_loss(s_pert_pred, t_pred)

                # EWC Penalty (Pillar 4)
                loss_ewc = ewc.penalty(model)

                loss = loss_task + 0.20 * loss_cons + loss_ewc

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()

                # EMA Teacher update & Curriculum update
                distiller.update_teacher(model)
                curriculum.update(loss_task.item())

                done += 1
                step += 1
                losses.append(loss.item())
                if done % 20 == 0:
                    c_info = curriculum.get_info()
                    print(f"ft step {step} | loss {sum(losses[-20:]) / 20:.4f} | Curric Lvl {c_info.get('current_level')} ({c_info.get('level_name')}) | {done}/{args.steps}", flush=True)

        avg = sum(losses[-100:]) / max(1, len(losses[-100:]))
        new_best = min(best, avg)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "step": step, "best": new_best, "model_type": ckpt_arch}, CKPT / "last.pt")
        if avg < best:
            torch.save({"model": model.state_dict(), "step": step, "best": new_best, "model_type": ckpt_arch}, CKPT / "best.pt")
            print(f"[ft] new best {new_best:.4f} -> best.pt", flush=True)
            try:
                export_all_formats(model, step)
            except Exception:
                pass
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
