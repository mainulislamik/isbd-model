"""
ISBD v1.00 — PyMatting Closed-Form Alpha Matting Engine
Extracts sub-pixel flyaway hair strands, fur, lace, and semi-transparent glass
using PyMatting's closed-form Laplacian matting and automated Trimap generation.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any


def generate_auto_trimap(
    coarse_mask_uint8: np.ndarray,
    erode_radius: int = 12,
    dilate_radius: int = 16
) -> np.ndarray:
    """
    Generates a 3-region Trimap from a coarse binary/grayscale mask:
    - 1.0 (255): Definite Foreground
    - 0.0 (0):   Definite Background
    - 0.5 (128): Unknown Boundary Band (Hair strands, glass, edges)
    """
    # Threshold coarse mask
    _, binary = cv2.threshold(coarse_mask_uint8, 127, 255, cv2.THRESH_BINARY)

    kernel_fg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_radius, erode_radius))
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_radius, dilate_radius))

    fg_definite = cv2.erode(binary, kernel_fg)
    bg_dilated = cv2.dilate(binary, kernel_bg)

    trimap = np.full(coarse_mask_uint8.shape, 0.5, dtype=np.float64)
    trimap[bg_dilated == 0] = 0.0
    trimap[fg_definite == 255] = 1.0

    return trimap


def extract_alpha_matte(
    img_bgr: np.ndarray,
    coarse_mask_uint8: np.ndarray,
    erode_radius: int = 12,
    dilate_radius: int = 16
) -> Dict[str, Any]:
    """
    Computes precise continuous alpha matte using PyMatting closed-form solver.
    """
    try:
        from pymatting import estimate_alpha_cf, estimate_foreground_ml

        # Normalize image to [0, 1] RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
        trimap = generate_auto_trimap(coarse_mask_uint8, erode_radius, dilate_radius)

        # Solve closed-form alpha matting
        alpha = estimate_alpha_cf(img_rgb, trimap)
        alpha = np.clip(alpha, 0.0, 1.0)

        # Estimate un-contaminated foreground colors (removes background color spill on hair edges)
        foreground = estimate_foreground_ml(img_rgb, alpha)
        fg_bgr = cv2.cvtColor((np.clip(foreground, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Assemble clean RGBA image
        alpha_u8 = (alpha * 255.0).astype(np.uint8)
        rgba = cv2.cvtColor(fg_bgr, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = alpha_u8

        return {
            "ok": True,
            "rgba": rgba,
            "alpha_mask": alpha_u8,
            "trimap_preview": (trimap * 255.0).astype(np.uint8),
            "method": "PyMatting Closed-Form Laplacian"
        }
    except Exception as e:
        # Graceful fallback: morphological guided alpha
        blurred_alpha = cv2.GaussianBlur(coarse_mask_uint8, (9, 9), 0)
        rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = blurred_alpha
        return {
            "ok": False,
            "rgba": rgba,
            "alpha_mask": blurred_alpha,
            "trimap_preview": coarse_mask_uint8,
            "error": str(e),
            "method": "Fallback Guided Alpha"
        }
