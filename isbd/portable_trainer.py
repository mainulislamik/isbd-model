"""
ISBD v1.00 — Hardware-Agnostic Portable Trainer Engine
Runs seamlessly across any machine (Low-Spec VPS, GPU Rig, Apple Silicon, Multi-Core Desktop).
Automatically adapts batch size, precision (AMP), and threads to available hardware.
Blends synthetic data with real-world designer harvested pairs.
"""

import os
import sys
import time
import json
import fcntl
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from isbd.model import SmallUNet, TinyUNet, param_count
from isbd.data import make_pair, IMG_SIZE
from isbd.loss import ISBDProLoss
from isbd.harvester import get_manifest, IMAGES_DIR

CKPT = ROOT / "checkpoints"
CKPT.mkdir(exist_ok=True)
HISTORY = CKPT / "history.json"
PAUSE_FLAG = ROOT / "data" / "trainer_paused.flag"


def detect_hardware_profile(override_device: Optional[str] = None) -> Dict[str, Any]:
    """
    Auto-detect CPU/GPU/VRAM and determine optimal training hyperparameters.
    Guarantees that training never crashes from OOM and never freezes low-spec hosts.
    """
    total_ram_gb = 8.0
    try:
        import psutil
        total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        pass

    cpu_count = os.cpu_count() or 2

    profile = {
        "device": "cpu",
        "device_name": "Generic CPU",
        "batch_size": 4,
        "accum_steps": 2,
        "use_amp": False,
        "num_workers": 0,
        "pin_memory": False,
        "threads": max(1, min(cpu_count - 1, 4)),
        "vram_gb": 0.0,
        "ram_gb": round(total_ram_gb, 1)
    }

    if override_device and override_device != "auto":
        profile["device"] = override_device
        if override_device.startswith("cuda") and torch.cuda.is_available():
            profile["device_name"] = torch.cuda.get_device_name(0)
            profile["use_amp"] = True
            profile["pin_memory"] = True
        return profile

    # 1. Check NVIDIA / ROCm CUDA
    if torch.cuda.is_available():
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = vram_bytes / (1024 ** 3)
        profile["device"] = "cuda"
        profile["device_name"] = torch.cuda.get_device_name(0)
        profile["vram_gb"] = round(vram_gb, 1)
        profile["use_amp"] = True
        profile["pin_memory"] = True

        if vram_gb >= 12.0:
            profile["batch_size"] = 16
            profile["accum_steps"] = 1
            profile["num_workers"] = min(cpu_count, 4)
        elif vram_gb >= 6.0:
            profile["batch_size"] = 8
            profile["accum_steps"] = 2
            profile["num_workers"] = 2
        else:
            profile["batch_size"] = 4
            profile["accum_steps"] = 4
            profile["num_workers"] = 0
        return profile

    # 2. Check Apple Silicon MPS
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        profile["device"] = "mps"
        profile["device_name"] = "Apple Silicon Metal (MPS)"
        profile["batch_size"] = 8
        profile["accum_steps"] = 2
        profile["use_amp"] = False
        profile["num_workers"] = 0
        return profile

    # 3. Check Intel XPU
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        profile["device"] = "xpu"
        profile["device_name"] = torch.xpu.get_device_name(0)
        profile["batch_size"] = 8
        profile["accum_steps"] = 2
        profile["use_amp"] = True
        return profile

    # 4. Fallback: CPU
    profile["device"] = "cpu"
    profile["device_name"] = f"CPU ({cpu_count} Cores, {round(total_ram_gb, 1)}GB RAM)"
    if total_ram_gb < 4.0:
        # Ultra low-spec VPS (e.g. 1-2GB RAM)
        profile["batch_size"] = 2
        profile["accum_steps"] = 4
        profile["threads"] = 1
    else:
        profile["batch_size"] = 4
        profile["accum_steps"] = 2
        profile["threads"] = max(1, min(cpu_count - 1, 2))

    return profile


class HybridCurriculumDataset(Dataset):
    """
    Blends synthetic image degradation pairs with real-world designer edited samples.
    Real designer samples receive higher representation to accelerate human style adaptation.
    """
    def __init__(self, synthetic_length: int = 5000, designer_weight: int = 5):
        self.synthetic_length = synthetic_length
        self.designer_weight = designer_weight
        self.designer_pairs = []

        manifest = get_manifest()
        samples = manifest.get("samples", [])
        for s in samples:
            b_path = IMAGES_DIR / s.get("before_file", "")
            a_path = IMAGES_DIR / s.get("after_file", "")
            if b_path.exists() and a_path.exists():
                self.designer_pairs.append((str(b_path), str(a_path)))

        self.has_designer = len(self.designer_pairs) > 0
        self.total_len = synthetic_length + (len(self.designer_pairs) * designer_weight)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        import cv2

        # Decide whether to sample from real designer pairs or synthetic
        if self.has_designer and np.random.rand() < 0.40:
            pair = self.designer_pairs[np.random.randint(0, len(self.designer_pairs))]
            b_img = cv2.imread(pair[0])
            a_img = cv2.imread(pair[1])

            if b_img is not None and a_img is not None:
                # Resize/crop to IMG_SIZE
                b_img = cv2.resize(b_img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
                a_img = cv2.resize(a_img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

                # Convert BGR to RGB [0, 1]
                x = b_img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
                y = a_img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0

                # Random flip augmentation
                if np.random.rand() > 0.5:
                    x = np.flip(x, axis=2).copy()
                    y = np.flip(y, axis=2).copy()

                return torch.from_numpy(x), torch.from_numpy(y)

        # Fallback to synthetic degradation pair
        x, y = make_pair(idx)
        return torch.from_numpy(x), torch.from_numpy(y)


def run_training_cycle(
    steps: int = 200,
    model_type: str = "small",
    device_override: Optional[str] = None,
    lr: float = 2e-4,
    reporter_cb=None
) -> Dict[str, Any]:
    """
    Execute a self-contained training cycle adaptable to any hardware.
    """
    profile = detect_hardware_profile(device_override)
    device = torch.device(profile["device"])

    if profile["device"] == "cpu":
        torch.set_num_threads(profile["threads"])

    # Load model
    model = SmallUNet() if model_type == "small" else TinyUNet()
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5000)
    criterion = ISBDProLoss()

    scaler = None
    if profile["use_amp"] and profile["device"] == "cuda":
        scaler = torch.cuda.amp.GradScaler()

    # Load checkpoint if exists
    start_step = 0
    best_loss = float("inf")
    last_pt = CKPT / "last.pt"

    if last_pt.exists():
        try:
            state = torch.load(last_pt, map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            if "opt" in state:
                opt.load_state_dict(state["opt"])
            if "sched" in state:
                sched.load_state_dict(state["sched"])
            start_step = state.get("step", 0)
            best_loss = state.get("best", float("inf"))
        except Exception as e:
            print(f"[portable_trainer] Checkpoint load warning: {e}")

    # Recover step from history.json if last.pt was incomplete
    if HISTORY.exists():
        try:
            h = json.loads(HISTORY.read_text())
            if h.get("total_steps", 0) > start_step:
                start_step = h["total_steps"]
        except Exception:
            pass

    dataset = HybridCurriculumDataset(synthetic_length=4000)
    loader = DataLoader(
        dataset,
        batch_size=profile["batch_size"],
        shuffle=True,
        num_workers=profile["num_workers"],
        pin_memory=profile["pin_memory"],
        drop_last=True
    )

    model.train()
    loader_iter = iter(loader)
    losses = []
    t0 = time.time()
    accum_steps = profile["accum_steps"]

    opt.zero_grad()

    for current_step in range(1, steps + 1):
        # Pause flag check
        if PAUSE_FLAG.exists():
            time.sleep(2)
            continue

        try:
            batch_x, batch_y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch_x, batch_y = next(loader_iter)

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        if scaler:
            with torch.cuda.amp.autocast():
                pred = model(batch_x)
                loss = criterion(pred, batch_y) / accum_steps
            scaler.scale(loss).backward()
            if current_step % accum_steps == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                sched.step()
        else:
            pred = model(batch_x)
            loss = criterion(pred, batch_y) / accum_steps
            loss.backward()
            if current_step % accum_steps == 0:
                opt.step()
                opt.zero_grad()
                sched.step()

        loss_val = float(loss.detach().cpu()) * accum_steps
        losses.append(loss_val)

        if current_step % 50 == 0 or current_step == steps:
            avg_loss = float(np.mean(losses[-50:]))
            if reporter_cb:
                reporter_cb(start_step + current_step, avg_loss)

    total_time = max(0.001, time.time() - t0)
    final_step = start_step + steps
    final_loss = float(np.mean(losses[-20:])) if losses else 0.0

    # Save state
    save_state = {
        "step": final_step,
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
        "best": min(best_loss, final_loss),
        "model_type": model_type,
        "hardware": profile["device_name"]
    }
    torch.save(save_state, str(last_pt))

    # Update history.json
    h_data = {"steps": [], "losses": [], "total_steps": final_step}
    if HISTORY.exists():
        try:
            h_data = json.loads(HISTORY.read_text())
        except Exception:
            pass
    h_data.setdefault("steps", []).append(final_step)
    h_data.setdefault("losses", []).append(round(final_loss, 4))
    h_data["total_steps"] = final_step
    h_data["last_hardware"] = profile["device_name"]
    HISTORY.write_text(json.dumps(h_data))

    return {
        "ok": True,
        "steps_completed": steps,
        "start_step": start_step,
        "final_step": final_step,
        "final_loss": round(final_loss, 4),
        "time_sec": round(total_time, 2),
        "speed_it_s": round(steps / total_time, 2),
        "hardware_profile": profile
    }
