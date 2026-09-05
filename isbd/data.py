"""
ISBD v1.00 — Advanced Data Generator & Pro Retouching Curriculum
Simulates realistic multi-layer professional Photoshop retouching tasks:
1. Skin Retouching & Blemish Removal (Frequency Separation simulation: low freq blotches + high freq blemishes)
2. Auto Color Grading & White Balance Casts (Warm, Cool, Green, Magenta shifts)
3. S-Curve Dynamic Contrast & Shadow Recovery (CLAHE / Tone curve inverse)
4. Highlight Burn & Exposure Fixing (Over/under exposure recovery)
5. Lens Blur & Defocus Softening (Unsharp Mask inverse)
6. Digital Sensor ISO Noise & Grain (Sensor noise removal)
7. Color Vibrance & Desaturation (Vibrance/Saturation inverse)
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
            (rng.uniform(220, 255), rng.uniform(180, 220), rng.uniform(150, 190)), # Fair
            (rng.uniform(190, 230), rng.uniform(140, 180), rng.uniform(100, 140)), # Medium
            (rng.uniform(150, 190), rng.uniform(100, 140), rng.uniform(60, 100)),  # Warm Brown
            (rng.uniform(90, 130), rng.uniform(60, 90), rng.uniform(40, 70)),     # Deep
        ]
        base_color = np.array(rng.choice(skin_tones), dtype=np.float32)
        arr = np.ones((h, w, 3), dtype=np.float32) * base_color[None, None, :]
        
        # Add facial facial contours / shadows
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
    """
    Applies multi-stage Photoshop Retouching inversions:
    - Blemish & Spot adding (Retouch / Spot Healing inverse)
    - Skin Blotching & Uneven tone (Frequency Separation inverse)
    - Color cast & Tint (White balance / Curves inverse)
    - Exposure & Shadow clipping (Levels / Curves inverse)
    - Lens blur (Unsharp mask inverse)
    - ISO Sensor noise (Denoise inverse)
    """
    op = rng.choice([
        "blemish_retouch", "skin_blotch", "exposure_curve",
        "color_balance", "lens_blur", "iso_noise", "vibrance"
    ])

    if op == "blemish_retouch":
        # Simulate blemishes, acne spots, and dust particles
        arr = np.array(img).copy()
        n_spots = rng.randint(3, 8)
        for _ in range(n_spots):
            bx, by = rng.randint(4, IMG_SIZE - 5), rng.randint(4, IMG_SIZE - 5)
            rad = rng.randint(1, 3)
            tint = np.array([rng.uniform(160, 255), rng.uniform(40, 100), rng.uniform(40, 90)]) # red blemish / dark mole
            for dy in range(-rad, rad + 1):
                for dx in range(-rad, rad + 1):
                    if 0 <= by + dy < IMG_SIZE and 0 <= bx + dx < IMG_SIZE:
                        if dx**2 + dy**2 <= rad**2:
                            arr[by + dy, bx + dx] = np.clip(arr[by + dy, bx + dx] * 0.4 + tint * 0.6, 0, 255)
        return Image.fromarray(arr)

    elif op == "skin_blotch":
        # Uneven lighting, dark circles or redness (Frequency Separation low-freq inverse)
        arr = np.array(img).astype(np.float32)
        bx, by = rng.randint(10, IMG_SIZE - 10), rng.randint(10, IMG_SIZE - 10)
        xx, yy = np.meshgrid(np.arange(IMG_SIZE), np.arange(IMG_SIZE))
        dist = np.sqrt((xx - bx)**2 + (yy - by)**2)
        blotch = np.exp(-dist**2 / (2 * (rng.uniform(8, 16))**2))
        factor = rng.uniform(0.5, 1.4)
        arr = arr * (1.0 - blotch[:, :, None] * (1.0 - factor))
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    elif op == "exposure_curve":
        f = rng.uniform(0.5, 0.8) if rng.random() < 0.5 else rng.uniform(1.25, 1.65)
        return ImageEnhance.Brightness(img).enhance(f)

    elif op == "color_balance":
        arr = np.asarray(img).astype(np.float32)
        casts = {
            "warm_yellow": [1.2, 1.05, 0.8],
            "cool_blue":   [0.8, 0.95, 1.25],
            "magenta":     [1.15, 0.85, 1.15],
            "green_tint":  [0.9, 1.2, 0.85],
        }
        k = np.array(casts[rng.choice(list(casts))], dtype=np.float32)
        arr *= k[None, None, :]
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    elif op == "lens_blur":
        r = rng.choice([ImageFilter.GaussianBlur(1.4), ImageFilter.GaussianBlur(2.2)])
        return img.filter(r)

    elif op == "iso_noise":
        arr = np.asarray(img).astype(np.float32)
        sigma = rng.uniform(12, 28)
        g = np.random.default_rng(rng.randrange(1 << 30))
        arr += g.normal(0, sigma, arr.shape)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    elif op == "vibrance":
        f = rng.uniform(0.25, 0.6)
        return ImageEnhance.Color(img).enhance(f)

    return img


def make_pair(seed: int):
    """Return (input_np_uint8, target_np_uint8) as CHW float arrays in [0,1]."""
    rng = random.Random(seed)
    clean = _rand_img(rng)
    dirty = degrade(clean, rng)
    x = np.asarray(dirty, dtype=np.float32) / 255.0
    y = np.asarray(clean, dtype=np.float32) / 255.0
    return x.transpose(2, 0, 1), y.transpose(2, 0, 1)


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
