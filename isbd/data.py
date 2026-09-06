"""
ISBD v1.00 — Advanced Data Generator v2
- On-the-fly augmentation: each synthetic pair gets8-16x augmented variants
- Flip, rotate, color jitter, brightness, contrast, noise variations
- Total effective training samples:3000 base × ~12 augmentations = ~36,000
"""
import os
import random
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageDraw

IMG_SIZE = 64  # Optimized for CPU training


def _rand_img(rng: random.Random) -> Image.Image:
    """Creates synthetic realistic photo-like scenes with varied textures and skin-like / natural tones."""
    w = h = IMG_SIZE

    # Random base landscape / portrait gradient
    is_portrait = rng.random() < 0.4
    if is_portrait:
        # Skin-tone gamut (European, Asian, South Asian, African tones)
        skin_tones = [
            (rng.uniform(220, 255), rng.uniform(180, 220), rng.uniform(150, 190)),
            (rng.uniform(190, 230), rng.uniform(140, 180), rng.uniform(100, 140)),
            (rng.uniform(150, 190), rng.uniform(100, 140), rng.uniform(60, 100)),
            (rng.uniform(90, 130), rng.uniform(60, 90), rng.uniform(40, 70)),
        ]
        base_color = np.array(rng.choice(skin_tones), dtype=np.float32)
        arr = np.ones((h, w, 3), dtype=np.float32) * base_color[None, None, :]

        # Add facial contours / shadows
        cx, cy = rng.randint(20, 44), rng.randint(20, 44)
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        arr *= np.clip(1.1 - dist / 50.0, 0.7, 1.2)[:, :, None]
    else:
        # Sky / Nature / Indoor scene
        top = np.array([rng.uniform(40, 200), rng.uniform(80, 220), rng.uniform(120, 255)])
        bot = np.array([rng.uniform(20, 120), rng.uniform(60, 180), rng.uniform(30, 140)])
        yy = np.linspace(0, 1, h)[:, None, None]
        grad = bot[None, None, :] + (top - bot)[None, None, :] * yy
        arr = np.broadcast_to(grad, (h, w, 3)).copy()

    # Add geometric & organic scene elements
    n_shapes = rng.randint(2, 6)
    for _ in range(n_shapes):
        cx, cy = rng.randint(8, w - 8), rng.randint(8, h - 8)
        r = rng.randint(4, 18)
        col = np.array([rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)])
        xx, ygrid = np.arange(w)[None, :], np.arange(h)[:, None]
        mask = (xx - cx) ** 2 + (ygrid - cy) ** 2 < r ** 2
        arr[mask] = col

    # Fine natural texture
    arr += np.random.default_rng(rng.randrange(1 << 30)).normal(0, 6, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def degrade(img: Image.Image, rng: random.Random) -> Image.Image:
    """Advanced Multi-Stage degradation pipeline."""
    op = rng.choice([
        "selective_dermis_blemish", "studio_shadow_clip", "skin_tone_blotch",
        "color_temperature_skew", "texture_blur", "iso_dermal_noise", "exposure_curve"
    ])

    if op == "selective_dermis_blemish":
        arr = np.array(img).copy()
        n_spots = rng.randint(4, 12)
        for _ in range(n_spots):
            bx, by = rng.randint(4, IMG_SIZE - 5), rng.randint(4, IMG_SIZE - 5)
            rad = rng.randint(1, 3)
            tint = np.array([rng.uniform(170, 255), rng.uniform(30, 90), rng.uniform(30, 80)])
            for dy in range(-rad, rad + 1):
                for dx in range(-rad, rad + 1):
                    if 0 <= by + dy < IMG_SIZE and 0 <= bx + dx < IMG_SIZE:
                        if dx**2 + dy**2 <= rad**2:
                            arr[by + dy, bx + dx] = np.clip(arr[by + dy, bx + dx] * 0.35 + tint * 0.65, 0, 255)
        return Image.fromarray(arr)

    elif op == "studio_shadow_clip":
        arr = np.array(img).astype(np.float32)
        arr = np.where(arr < 128, arr * rng.uniform(0.55, 0.8), arr * rng.uniform(0.95, 1.15))
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    elif op == "skin_tone_blotch":
        arr = np.array(img).astype(np.float32)
        bx, by = rng.randint(10, IMG_SIZE - 10), rng.randint(10, IMG_SIZE - 10)
        xx, yy = np.meshgrid(np.arange(IMG_SIZE), np.arange(IMG_SIZE))
        dist = np.sqrt((xx - bx)**2 + (yy - by)**2)
        blotch = np.exp(-dist**2 / (2 * (rng.uniform(8, 16))**2))
        factor = rng.uniform(0.45, 1.55)
        arr = arr * (1.0 - blotch[:, :, None] * (1.0 - factor))
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    elif op == "color_temperature_skew":
        arr = np.array(img).astype(np.float32)
        if rng.random() < 0.5:
            arr[:, :, 0] = np.clip(arr[:, :, 0] * rng.uniform(1.15, 1.4), 0, 255)
            arr[:, :, 2] = np.clip(arr[:, :, 2] * rng.uniform(0.65, 0.85), 0, 255)
        else:
            arr[:, :, 2] = np.clip(arr[:, :, 2] * rng.uniform(1.15, 1.35), 0, 255)
            arr[:, :, 0] = np.clip(arr[:, :, 0] * rng.uniform(0.7, 0.88), 0, 255)
        return Image.fromarray(arr.astype(np.uint8))

    elif op == "texture_blur":
        return img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.8, 2.2)))

    elif op == "iso_dermal_noise":
        arr = np.array(img).astype(np.float32)
        noise = np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(10, 26), arr.shape)
        return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

    elif op == "exposure_curve":
        f = rng.uniform(0.5, 0.8) if rng.random() < 0.5 else rng.uniform(1.25, 1.65)
        return ImageEnhance.Brightness(img).enhance(f)

    return img


def augment_pair(x: np.ndarray, y: np.ndarray, rng: random.Random) -> tuple:
    """
    On-the-fly augmentation for a single (3,H,W) pair.
    Applies random flip, rotation, color jitter, brightness, noise.
    Returns augmented (x, y) pair.
    """
    # Convert back to PIL for spatial transforms
    x_img = Image.fromarray((x.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8))
    y_img = Image.fromarray((y.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8))

    # Spatial transforms (same for x and y to preserve correspondence)
    if rng.random() < 0.5:
        x_img = ImageOps.mirror(x_img)
        y_img = ImageOps.mirror(y_img)
    if rng.random() < 0.5:
        x_img = ImageOps.flip(x_img)
        y_img = ImageOps.flip(y_img)
    if rng.random() < 0.3:
        angle = rng.choice([90, 180, 270])
        x_img = x_img.rotate(angle)
        y_img = y_img.rotate(angle)

    x_aug = np.asarray(x_img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    y_aug = np.asarray(y_img, dtype=np.float32).transpose(2, 0, 1) / 255.0

    # Color jitter on INPUT only (degraded side gets more variation)
    if rng.random() < 0.4:
        brightness = rng.uniform(0.85, 1.15)
        x_aug = np.clip(x_aug * brightness, 0, 1)
    if rng.random() < 0.3:
        noise = np.random.default_rng(rng.randint(0, 2**31)).normal(0, rng.uniform(0.01, 0.03), x_aug.shape).astype(np.float32)
        x_aug = np.clip(x_aug + noise, 0, 1)

    return x_aug, y_aug


def make_pair(seed: int):
    """Return (x, y) as CHW float arrays in [0,1]."""
    rng = random.Random(seed)
    clean = _rand_img(rng)
    dirty = degrade(clean, rng)
    x = np.asarray(dirty, dtype=np.float32) / 255.0
    y = np.asarray(clean, dtype=np.float32) / 255.0
    return x.transpose(2, 0, 1), y.transpose(2, 0, 1)


def torch_dataset(augment_factor: int = 12):
    """
    PyTorch Dataset with on-the-fly augmentation.
    augment_factor: each base pair generates this many augmented variants.
    Total effective samples = base_n × augment_factor.
    """
    import torch.utils.data as D
    import torch
    base_seed = int(os.environ.get("ISBD_SEED", "7"))

    class _DS(D.Dataset):
        def __init__(self, n: int):
            self.n = n
            self.aug = augment_factor

        def __len__(self):
            return self.n * self.aug

        def __getitem__(self, idx):
            # idx → (base_index, aug_index)
            base_idx = idx // self.aug
            aug_idx = idx % self.aug

            # Each base pair + augmentation gets a deterministic seed
            seed = base_seed * 1_000_003 + base_idx
            x, y = make_pair(seed)

            if aug_idx > 0:  # first variant is always the original
                aug_rng = random.Random(seed * 7919 + aug_idx * 13)
                x, y = augment_pair(x, y, aug_rng)

            return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())

    return _DS
