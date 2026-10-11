"""
ISBD v1.00 — Frequency Separation & Guided Texture Injection Engine
Mathematical separation of high-frequency texture (skin pores, fabric grain, hair strands)
from low-frequency tones (lighting, shadows, skin tone blemishes).
Prevents plastic/cartoonish over-smoothing in commercial retouching.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any


def decompose_frequencies(
    img_bgr: np.ndarray,
    radius: int = 9,
    filter_type: str = "bilateral"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decomposes an input BGR image into:
    - low_freq (tones, color, smooth gradients)
    - high_freq (pores, texture, hair details, specular micro-highlights)
    Returns:
    (low_freq_bgr, high_freq_float)
    """
    img_float = img_bgr.astype(np.float32)

    if filter_type == "bilateral":
        # Bilateral filter preserves sharp geometric edges while smoothing tones
        low_freq = cv2.bilateralFilter(img_bgr, d=radius, sigmaColor=75, sigmaSpace=75).astype(np.float32)
    else:
        ksize = radius if radius % 2 != 0 else radius + 1
        low_freq = cv2.GaussianBlur(img_float, (ksize, ksize), 0)

    # High frequency is the residual difference
    high_freq = img_float - low_freq
    return np.clip(low_freq, 0, 255).astype(np.uint8), high_freq


def inject_frequency_texture(
    processed_tone_bgr: np.ndarray,
    original_high_freq: np.ndarray,
    texture_weight: float = 1.0,
    edge_boost: float = 1.15
) -> np.ndarray:
    """
    Recombines AI-processed low/mid tones with preserved high-frequency texture.
    Ensures 100% authentic skin pores and fabric weave.
    """
    proc_float = processed_tone_bgr.astype(np.float32)

    # If resolutions differ, resize high frequency to match
    h_proc, w_proc = proc_float.shape[:2]
    h_hi, w_hi = original_high_freq.shape[:2]
    if (h_proc, w_proc) != (h_hi, w_hi):
        high_freq = cv2.resize(original_high_freq, (w_proc, h_proc), interpolation=cv2.INTER_CUBIC)
    else:
        high_freq = original_high_freq

    # Modulate high frequency with edge boost
    reconstructed = proc_float + (high_freq * texture_weight * edge_boost)
    return np.clip(reconstructed, 0, 255).astype(np.uint8)


def commercial_frequency_retouch(
    img_bgr: np.ndarray,
    skin_smooth_strength: float = 0.55,
    texture_retention: float = 1.10,
    radius: int = 9
) -> Dict[str, Any]:
    """
    Applies high-end Commercial Frequency Separation retouch:
    1. Extracts true high-pass skin texture.
    2. Smoothes tonal blemishes on low-pass branch.
    3. Seamlessly injects high-pass texture back over smooth tones.
    """
    low_freq, high_freq = decompose_frequencies(img_bgr, radius=radius, filter_type="bilateral")

    # Smooth the low frequency tone layer while preserving overall morphology
    ksize = int(radius * 1.5)
    if ksize % 2 == 0:
        ksize += 1
    tone_smoothed = cv2.GaussianBlur(low_freq.astype(np.float32), (ksize, ksize), 0)

    # Blend original tones with smoothed tones based on strength
    blended_tone = (1.0 - skin_smooth_strength) * low_freq.astype(np.float32) + (skin_smooth_strength * tone_smoothed)

    # Recombine with high frequency texture
    result_bgr = inject_frequency_texture(
        blended_tone.astype(np.uint8),
        high_freq,
        texture_weight=texture_retention,
        edge_boost=1.05
    )

    # Measure texture preservation score (correlation between original and final high pass)
    _, res_high = decompose_frequencies(result_bgr, radius=radius)
    orig_norm = np.linalg.norm(high_freq) + 1e-6
    res_norm = np.linalg.norm(res_high) + 1e-6
    corr = float(np.sum(high_freq * res_high) / (orig_norm * res_norm))

    return {
        "ok": True,
        "processed_bgr": result_bgr,
        "low_freq_bgr": low_freq,
        "texture_retention_score": round(max(0.0, min(1.0, corr)), 4),
        "texture_weight": texture_retention,
        "skin_smooth_strength": skin_smooth_strength
    }
