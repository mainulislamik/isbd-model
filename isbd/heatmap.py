"""
ISBD v1.00 — AI Pixel Heatmap & Difference Visualizer Module
Calculates precise per-pixel error/changes between Before & After images
and renders thermal colormaps (JET, TURBO, INFERNO, MAGMA) to instantly highlight modifications.
"""
from PIL import Image
import numpy as np
import cv2

COLORMAPS = {
    "jet": cv2.COLORMAP_JET,
    "turbo": cv2.COLORMAP_TURBO,
    "inferno": cv2.COLORMAP_INFERNO,
    "magma": cv2.COLORMAP_MAGMA,
    "hot": cv2.COLORMAP_HOT,
}

def generate_difference_heatmap(
    before_img: Image.Image,
    after_img: Image.Image,
    colormap_type: str = "turbo",
    blend_alpha: float = 0.55,
    boost_sensitivity: float = 1.5
):
    """
    Generates a high-precision heat map showing exact changes between Before and After images.
    Returns:
      - heatmap_img: Clean colored heatmap
      - overlay_img: Heatmap overlaid on the After image
      - stats: dict containing mean difference, max difference, and % of altered pixels
    """
    # Resize before to match after if sizes differ
    if before_img.size != after_img.size:
        before_img = before_img.resize(after_img.size, Image.Resampling.LANCZOS)

    b_arr = np.array(before_img.convert("RGB"))
    a_arr = np.array(after_img.convert("RGB"))

    # Compute per-pixel absolute difference in RGB
    diff = cv2.absdiff(b_arr, a_arr)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Compute statistical metrics
    mean_diff = float(np.mean(diff_gray))
    max_diff = float(np.max(diff_gray))
    altered_pixels_pct = float(round(float((np.count_nonzero(diff_gray > 3) / diff_gray.size) * 100), 2))

    # Boost difference for subtle changes
    scaled_diff = np.clip(diff_gray * boost_sensitivity, 0, 255).astype(np.uint8)

    # Apply Colormap
    cmap_flag = COLORMAPS.get(colormap_type.lower(), cv2.COLORMAP_TURBO)
    heatmap_colored = cv2.applyColorMap(scaled_diff, cmap_flag)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Create blended overlay on top of the modified/after image
    overlay_arr = cv2.addWeighted(a_arr, 1.0 - blend_alpha, heatmap_colored_rgb, blend_alpha, 0)

    stats = {
        "mean_diff": round(mean_diff, 2),
        "max_diff": round(max_diff, 2),
        "altered_pct": altered_pixels_pct,
        "interpretation": f"ছবির প্রায় {altered_pixels_pct}% পিক্সেলে পরিবর্তন শনাক্ত হয়েছে।"
    }

    return Image.fromarray(heatmap_colored_rgb), Image.fromarray(overlay_arr), stats
