"""
ISBD v1.00 — Live Show (for the 2-hour live report)
Builds ONE visual card: 'আমি এখন যা করছি' —
  current step, live loss, and a BEFORE -> AFTER of the model fixing a degraded photo.
No matplotlib needed — pure PIL. Output: samples/live_card.png (10KB-40KB).
"""
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from isbd.model import TinyUNet
from isbd.data import make_pair, IMG_SIZE

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"
OUT = ROOT / "samples" / "live_card.png"

# ---------- gather live state ----------
def live_state():
    last = {"step": 0, "loss": 0.0, "sps": 0.2}
    try:
        out = subprocess.run(
            ["journalctl", "--user", "-u", "isbd-train", "--no-pager", "-n", "30", "-o", "cat"],
            capture_output=True, text=True, timeout=15,
        ).stdout
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("step"):
                try:
                    p = line.replace("|", " ").split()
                    last = {
                        "step": int(p[1]),
                        "loss": float(p[3]),
                        "sps": float(p[4].replace("s/step", "").replace("s", "")),
                    }
                except Exception:
                    pass
    except Exception:
        pass
    return last


def load_best():
    m = TinyUNet()
    state = torch_load_best()
    m.load_state_dict(state["model"])
    m.eval()
    return m


def torch_load_best():
    import torch
    return torch.load(CKPT / "best.pt", map_location="cpu", weights_only=True)


# ---------- image helpers ----------
def grid_img(seed, size):
    """Render the synthetic degraded scene at a displayable size."""
    x, y = make_pair(seed)
    d = Image.fromarray((x.transpose(1, 2, 0) * 255).astype(np.uint8))
    return d.resize((size, size), Image.Resampling.LANCZOS)


def restored_img(model, seed, size):
    """Model's fix of that same degraded scene."""
    import torch
    x, _ = make_pair(seed)
    t = torch.from_numpy(x)[None]
    with torch.no_grad():
        out = model(t)[0].clamp(0, 1).numpy().transpose(1, 2, 0)
    return Image.fromarray((out * 255).astype(np.uint8)).resize((size, size), Image.Resampling.LANCZOS)


def load_font(sz):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if Path(p).exists():
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def hr_font(sz):
    p = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
    if Path(p).exists():
        return ImageFont.truetype(z := p, sz)
    return load_font(sz)


# ---------- build the card ----------
def build(seed=None):
    seed = seed if seed is not None else random.randrange(1 << 30)
    st = live_state()
    model = load_best()

    W = 1000
    pad = 28
    tile = 280
    label_h = 46
    title_h = 96
    H = title_h + tile + label_h + 2 * pad

    img = Image.new("RGB", (W, H), (16, 18, 24))
    d = ImageDraw.Draw(img)

    f_title = load_font(34)
    f_big   = load_font(30)
    f_lab   = load_font(24)
    f_small = load_font(18)

    # title bar
    d.text((pad, 18), f"ISBD v1.00 — LIVE  |  step {st['step']:,}  |  loss {st['loss']:.4f}",
           font=f_title, fill=(120, 220, 255))
    d.line([(pad, title_h - 8), (W - pad, title_h - 8)], fill=(45, 60, 80), width=2)

    # before / arrow / after
    before = grid_img(seed, tile)
    after = restored_img(model, seed, tile)
    y0 = title_h + pad
    img.paste(before, (pad, y0))
    img.paste(after, (W // 2 + 40, y0))

    # arrow
    ay = y0 + tile // 2
    for i in range(10):
        d.line([(W // 2 - 80 + i * 8, ay - i), (W // 2 - 80 + (i + 1) * 8, ay + i)], fill=(255, 200, 60), width=3)
    d.text((W // 2 - 58, ay + 14), "ISBD", font=f_small, fill=(255, 200, 60))

    # labels
    ly = y0 + tile + 10
    d.text((pad, ly), "INPUT: degraded (noise / blur / dark)", font=f_lab, fill=(255, 130, 130))
    d.text((W // 2 + 40, ly), "ISBD OUTPUT: fixed", font=f_lab, fill=(110, 255, 130))
    d.text((pad, ly + 30), "24/7 learning — this fix got made while you were away.",
           font=f_small, fill=(160, 160, 180))

    OUT.parent.mkdir(exist_ok=True)
    img.save(OUT, optimize=True)
    print(f"{OUT} step={st['step']} seed={seed}")
    return OUT


if __name__ == "__main__":
    build()
