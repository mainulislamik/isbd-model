"""
ISBD v1.00 — Data Generator
Builds paired (degraded, clean) training images using the image-editing skill's
professional operations as the curriculum:
  - Levels/Curves  -> brightness/contrast/gamma correction
  - Hue/Sat/Vibrance -> color cast + saturation shifts
  - Unsharp Mask    -> blur degradation (model learns to sharpen)
  - Reduce Noise    -> gaussian noise (model learns to denoise)
  - White balance   -> per-channel color casts (model learns WB fix)
Each sample: (input = degraded, target = clean). Model learns degraded -> clean.
All ops derived from Hermes skills: photoshop-image-editing + image-editing-tools.
"""
import os
import random
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFilter

IMG_SIZE = 64  # small on purpose — Surface Pro 3 CPU training


def _rand_img(rng: random.Random) -> Image.Image:
    """Synthetic 'photo-like' scene: gradients + shapes + noise texture."""
    w = h = IMG_SIZE
    # base gradient sky/field: (h,1,3) broadcast to (h,w,3)
    top = np.array([rng.uniform(40, 200), rng.uniform(80, 220), rng.uniform(120, 255)])
    bot = np.array([rng.uniform(20, 120), rng.uniform(60, 180), rng.uniform(30, 140)])
    yy = np.linspace(0, 1, h)[:, None, None]
    grad = bot[None, None, :] + (top - bot)[None, None, :] * yy  # (h,1,3)
    arr = np.broadcast_to(grad, (h, w, 3)).copy()
    # a few shapes (sun, hills, buildings)
    n_shapes = rng.randint(2, 5)
    for _ in range(n_shapes):
        cx, cy = rng.randint(8, w - 8), rng.randint(8, h - 8)
        r = rng.randint(5, 20)
        col = np.array([rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)])
        xx = np.arange(w)[None, :]
        ygrid = np.arange(h)[:, None]
        mask = (xx - cx) ** 2 + (ygrid - cy) ** 2 < r ** 2
        arr[mask] = col
    # texture noise so scenes aren't flat
    arr += rng.gauss(0, 1) * np.random.default_rng(rng.randrange(1 << 30)).normal(0, 8, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def degrade(img: Image.Image, rng: random.Random) -> Image.Image:
    """Apply ONE random professional-edit inverse (skill §4 pipeline)."""
    op = rng.choice(["exposure", "contrast", "color_cast", "blur", "noise", "saturation"])

    if op == "exposure":          # inverse of Levels correction
        f = rng.uniform(0.55, 0.85) if rng.random() < 0.5 else rng.uniform(1.2, 1.6)
        return ImageEnhance.Brightness(img).enhance(f)

    if op == "contrast":          # inverse of Curves S-curve
        f = rng.uniform(0.6, 0.85)
        return ImageEnhance.Contrast(img).enhance(f)

    if op == "color_cast":        # inverse of white-balance fix
        arr = np.asarray(img).astype(np.float32)
        casts = {
            "warm":  [1.18, 1.02, 0.86],   # too red/yellow
            "cool":  [0.86, 1.0, 1.2],     # too blue
            "green": [1.0, 1.2, 0.9],
        }
        k = np.array(casts[rng.choice(list(casts))], dtype=np.float32)
        arr *= k[None, None, :]
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    if op == "blur":              # inverse of Unsharp Mask
        r = rng.choice([ImageFilter.GaussianBlur(1.6), ImageFilter.GaussianBlur(2.4)])
        return img.filter(r)

    if op == "noise":             # inverse of Reduce Noise
        arr = np.asarray(img).astype(np.float32)
        sigma = rng.uniform(10, 26)
        g = np.random.default_rng(rng.randrange(1 << 30))
        arr += g.normal(0, sigma, arr.shape)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    if op == "saturation":        # inverse of Hue/Sat correction
        f = rng.uniform(0.3, 0.65)
        return ImageEnhance.Color(img).enhance(f)
    return img


def make_pair(seed: int):
    """Return (input_np_uint8, target_np_uint8) as HWC float arrays in [0,1]."""
    rng = random.Random(seed)
    clean = _rand_img(rng)
    dirty = degrade(clean, rng)
    x = np.asarray(dirty, dtype=np.float32) / 255.0
    y = np.asarray(clean, dtype=np.float32) / 255.0
    return x.transpose(2, 0, 1), y.transpose(2, 0, 1)  # CHW


def torch_dataset():
    import torch.utils.data as D
    base_seed = int(os.environ.get("ISBD_SEED", "7"))

    class _DS(D.Dataset):
        def __init__(self, n: int):
            self.n = n

        def __len__(self):
            return self.n

        def __getitem__(self, i):
            x, y = make_pair(base_seed * 1_000_003 + i)
            import torch
            return torch.from_numpy(x), torch.from_numpy(y)

    return _DS
