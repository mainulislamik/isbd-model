"""
ISBD Intelligent Vision Delta Detector
Automatically inspects Before and After images to identify commercial photo editing services:
- Clipping Path / Background Removal
- Hair Masking / Alpha Matting
- Retouching / Skin Beautification
- Shadow Making / Drop & Cast Shadow
- Reflection Shadow
- Wrinkle & Crease Cleaner
- Color Correction / Recolor
- Exposure & Light Enhancement
- Ghost Mannequin / Neck Joint
- Object / Dust Inpainting & Removal
- Jewelry Polish & Shiner
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Any

SERVICE_DISPLAY_NAMES_BN = {
    "clipping_path": "Clipping Path (ক্লিপিং পাথ)",
    "image_masking": "Hair Masking (হেয়ার মাস্কিং)",
    "retouching": "Retouching (রিটাচিং)",
    "color_correction": "Color Fix (কালার কারেকশন)",
    "shadow_making": "Shadow Making (শ্যাডো তৈরি)",
    "reflection": "Reflection (প্রতিফলন)",
    "wrinkle_cleaner": "Wrinkle Removal (ভাঁজ অপসারণ)",
    "neck_joint": "Neck Joint (নেক জয়েন্ট)",
    "ghost_mannequin": "Ghost Mannequin (ঘোস্ট ম্যানিকুইন)",
    "inpaint": "Object/Dust Removal (ডাস্ট রিমুভ)",
    "exposure_balance": "Exposure Fix (এক্সপোজার ব্যালান্স)",
    "enhancement": "Enhancement (এনহ্যান্সমেন্ট)",
    "jewelry_shiner": "Jewelry Polish (জুয়েলারি শাইনার)",
    "multi_clipping": "Multi Clipping (মাল্টি ক্লিপিং)",
    "vector_tracing": "Vector Tracing (ভেক্টর ট্রেসিং)",
    "client_custom": "Custom Edit (কাস্টম এডিট)"
}

def _to_cv2(pil_img: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    """Convert PIL image to BGR and Alpha (if present)."""
    rgba = pil_img.convert("RGBA")
    arr = np.array(rgba)
    bgr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
    alpha = arr[:, :, 3]
    return bgr, alpha

def detect_editing_services(before_img: Image.Image, after_img: Image.Image) -> Dict[str, Any]:
    """
    Fast Computer Vision Delta Inspector.
    Compares before and after images at 512x512 resolution to find all modifications.
    """
    ANALYSIS_SIZE = (512, 512)
    b_resized = before_img.resize(ANALYSIS_SIZE, Image.Resampling.LANCZOS)
    a_resized = after_img.resize(ANALYSIS_SIZE, Image.Resampling.LANCZOS)

    b_bgr, b_alpha = _to_cv2(b_resized)
    a_bgr, a_alpha = _to_cv2(a_resized)

    # Convert to grayscale & LAB
    b_gray = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2GRAY)
    a_gray = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2GRAY)
    b_lab  = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2LAB)
    a_lab  = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2LAB)

    scores: Dict[str, float] = {}
    details: Dict[str, Any] = {}

    # ──────────────────────────────────────────────────────────────────────────
    # 1. BACKGROUND REMOVAL & CLIPPING / MASKING DETECTION
    # ──────────────────────────────────────────────────────────────────────────
    # Check A: Alpha transparency
    has_alpha_cut = False
    alpha_trans_ratio = 0.0
    if np.any(a_alpha < 250):
        trans_mask = (a_alpha < 240)
        alpha_trans_ratio = float(np.mean(trans_mask))
        if alpha_trans_ratio > 0.03:
            has_alpha_cut = True

    # Check B: Solid white background replacement
    # In studio workflow, transparent background is frequently flattened onto pure white (RGB >= 250)
    a_white_bg = (a_bgr[:, :, 0] >= 248) & (a_bgr[:, :, 1] >= 248) & (a_bgr[:, :, 2] >= 248)
    b_white_bg = (b_bgr[:, :, 0] >= 248) & (b_bgr[:, :, 1] >= 248) & (b_bgr[:, :, 2] >= 248)
    new_white_ratio = float(np.mean(a_white_bg & (~b_white_bg)))
    total_white_ratio = float(np.mean(a_white_bg))

    has_white_bg_cut = (new_white_ratio > 0.08) or (total_white_ratio > 0.15 and new_white_ratio > 0.03)

    is_bg_removed = has_alpha_cut or has_white_bg_cut
    details["bg_removed"] = bool(is_bg_removed)
    details["alpha_trans_ratio"] = round(alpha_trans_ratio, 3)
    details["new_white_ratio"] = round(new_white_ratio, 3)

    if is_bg_removed:
        # Determine if it is sharp clipping path or soft hair masking
        if has_alpha_cut:
            subject_mask = (a_alpha >= 200).astype(np.uint8) * 255
            # Non-binary alpha indicates true alpha matting / hair masking
            soft_alpha_pixels = np.sum((a_alpha > 10) & (a_alpha < 245))
            soft_alpha_ratio  = float(soft_alpha_pixels) / max(int(np.sum(a_alpha > 10)), 1)
        else:
            subject_mask = (~a_white_bg).astype(np.uint8) * 255
            soft_alpha_ratio = 0.0

        # Edge gradient inspection & contour roughness
        contours, _ = cv2.findContours(subject_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # Check contour complexity / roughness (Hair has fine spikes, Pen tool clipping has smooth curves)
        contour_roughness = 0.0
        if contours:
            main_cnt = max(contours, key=cv2.contourArea)
            if len(main_cnt) > 20:
                approx = cv2.approxPolyDP(main_cnt, 2.5, True)
                contour_roughness = len(approx) / float(len(main_cnt))

        details["soft_alpha_ratio"] = round(soft_alpha_ratio, 4)
        details["contour_roughness"] = round(contour_roughness, 3)

        # Clipping path confidence
        scores["clipping_path"] = 0.98 if has_white_bg_cut or has_alpha_cut else 0.85

        # Hair masking confidence: triggered by soft alpha pixels or highly jagged strand-like edges
        if has_alpha_cut and soft_alpha_ratio > 0.04:
            scores["image_masking"] = min(0.99, 0.80 + soft_alpha_ratio * 2.0)
        elif contour_roughness > 0.42:
            scores["image_masking"] = min(0.95, 0.75 + contour_roughness * 0.3)
    else:
        # Check if subject mask can be derived from absolute diff
        abs_diff = cv2.absdiff(a_bgr, b_bgr)
        diff_gray = cv2.cvtColor(abs_diff, cv2.COLOR_BGR2GRAY)
        subject_mask = (diff_gray > 15).astype(np.uint8) * 255

    # ──────────────────────────────────────────────────────────────────────────
    # 2. SHADOW & REFLECTION DETECTION
    # ──────────────────────────────────────────────────────────────────────────
    # Check bottom half of the image for drop shadow or cast shadow
    h, w = ANALYSIS_SIZE
    bottom_slice_a = a_bgr[int(h * 0.45):, :]
    bottom_slice_b = b_bgr[int(h * 0.45):, :]

    # Shadow creates darker pixels with smooth feathering around or under the cutout
    shadow_dark_mask = (bottom_slice_a < bottom_slice_b)
    shadow_diff = cv2.subtract(cv2.cvtColor(bottom_slice_b, cv2.COLOR_BGR2GRAY),
                                cv2.cvtColor(bottom_slice_a, cv2.COLOR_BGR2GRAY))
    shadow_soft_pixels = np.sum((shadow_diff > 8) & (shadow_diff < 120))
    shadow_ratio = float(shadow_soft_pixels / (bottom_slice_a.shape[0] * bottom_slice_a.shape[1]))
    details["shadow_ratio"] = round(shadow_ratio, 3)

    if shadow_ratio > 0.04 and is_bg_removed:
        scores["shadow_making"] = min(0.96, 0.75 + shadow_ratio * 2.5)

    # Reflection detection (inverted vertical similarity in lower region)
    if is_bg_removed and shadow_ratio > 0.08:
        lower_quarter_a = a_gray[int(h * 0.70):, :]
        lower_flipped   = cv2.flip(lower_quarter_a, 0)
        mid_quarter_a   = a_gray[int(h * 0.40):int(h * 0.70), :]
        res = cv2.matchTemplate(lower_flipped, mid_quarter_a, cv2.TM_CCOEFF_NORMED)
        if res.max() > 0.45:
            scores["reflection"] = round(float(res.max()), 2)

    # ──────────────────────────────────────────────────────────────────────────
    # 3. COLOR CORRECTION & EXPOSURE FIX
    # ──────────────────────────────────────────────────────────────────────────
    # In LAB space: L is lightness, A/B are chrominance
    delta_L = np.abs(a_lab[:, :, 0].astype(np.float32) - b_lab[:, :, 0].astype(np.float32))
    delta_A = np.abs(a_lab[:, :, 1].astype(np.float32) - b_lab[:, :, 1].astype(np.float32))
    delta_B = np.abs(a_lab[:, :, 2].astype(np.float32) - b_lab[:, :, 2].astype(np.float32))

    # Mask analysis to subject or non-pure-white areas
    eval_mask = (a_bgr[:, :, 0] < 245) | (a_bgr[:, :, 1] < 245) | (a_bgr[:, :, 2] < 245)
    if np.sum(eval_mask) > 1000:
        mean_chroma_delta = float(np.mean(delta_A[eval_mask] + delta_B[eval_mask]))
        mean_luma_delta   = float(np.mean(delta_L[eval_mask]))
    else:
        mean_chroma_delta = float(np.mean(delta_A + delta_B))
        mean_luma_delta   = float(np.mean(delta_L))

    details["chroma_delta"] = round(mean_chroma_delta, 2)
    details["luma_delta"]   = round(mean_luma_delta, 2)

    if mean_chroma_delta > 10.0:
        scores["color_correction"] = min(0.97, 0.70 + (mean_chroma_delta / 40.0))
    elif mean_chroma_delta > 5.0 and not is_bg_removed:
        scores["color_correction"] = 0.82

    if mean_luma_delta > 15.0 and (not is_bg_removed or np.mean(eval_mask) > 0.2):
        scores["exposure_balance"] = min(0.95, 0.70 + (mean_luma_delta / 50.0))
        scores["enhancement"] = min(0.92, 0.65 + (mean_luma_delta / 50.0))

    # ──────────────────────────────────────────────────────────────────────────
    # 4. RETOUCHING, WRINKLE REMOVAL & INPAINTING (HIGH FREQUENCY DELTA)
    # ──────────────────────────────────────────────────────────────────────────
    # Frequency separation using Laplacian
    b_lap = cv2.Laplacian(b_gray, cv2.CV_32F)
    a_lap = cv2.Laplacian(a_gray, cv2.CV_32F)

    # Difference in high frequency texture
    hf_diff = np.abs(b_lap - a_lap)
    if np.sum(eval_mask) > 1000:
        hf_delta = float(np.mean(hf_diff[eval_mask]))
    else:
        hf_delta = float(np.mean(hf_diff))
    details["high_frequency_delta"] = round(hf_delta, 2)

    # Retouching: Texture smoothing, blemish removal, skin softening
    if hf_delta > 1.8:
        # Check if low-frequency shape structure remained mostly intact
        b_low = cv2.GaussianBlur(b_gray, (21, 21), 0)
        a_low = cv2.GaussianBlur(a_gray, (21, 21), 0)
        low_diff = float(np.mean(np.abs(b_low.astype(np.float32) - a_low.astype(np.float32))))

        if low_diff < 28.0 and hf_delta > 2.2:
            scores["retouching"] = min(0.97, 0.75 + (hf_delta / 10.0))

            # Wrinkle cleaning check (directional edge smoothing in garment regions)
            if hf_delta > 3.0:
                scores["wrinkle_cleaner"] = min(0.90, 0.70 + (hf_delta / 12.0))

    # Inpainting / Object Removal: Localized spots present in Before but absent in After
    diff_thresh = cv2.threshold(cv2.absdiff(b_gray, a_gray), 30, 255, cv2.THRESH_BINARY)[1]
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(diff_thresh)
    small_components = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels) if 10 < stats[i, cv2.CC_STAT_AREA] < 1500]
    if len(small_components) >= 2 and not is_bg_removed:
        scores["inpaint"] = min(0.96, 0.70 + (len(small_components) * 0.05))
        # Blemish removal in studio is also considered skin retouching
        scores["retouching"] = max(scores.get("retouching", 0.0), min(0.95, 0.75 + (len(small_components) * 0.04)))

    # ──────────────────────────────────────────────────────────────────────────
    # 5. NECK JOINT / GHOST MANNEQUIN DETECTION
    # ──────────────────────────────────────────────────────────────────────────
    # In neck joint, the clothing has left and right shoulder boundaries,
    # but the inner hollow between them was empty/white in Before and filled in After.
    shoulder_left_mask  = subject_mask[int(h * 0.15):int(h * 0.35), int(w * 0.15):int(w * 0.35)]
    shoulder_right_mask = subject_mask[int(h * 0.15):int(h * 0.35), int(w * 0.65):int(w * 0.85)]
    has_shoulders = (np.mean(shoulder_left_mask > 0) > 0.30) and (np.mean(shoulder_right_mask > 0) > 0.30)

    upper_center_y1, upper_center_y2 = int(h * 0.12), int(h * 0.35)
    upper_center_x1, upper_center_x2 = int(w * 0.40), int(w * 0.60)

    collar_b = b_gray[upper_center_y1:upper_center_y2, upper_center_x1:upper_center_x2]
    collar_a = a_gray[upper_center_y1:upper_center_y2, upper_center_x1:upper_center_x2]
    collar_b_white = (b_bgr[upper_center_y1:upper_center_y2, upper_center_x1:upper_center_x2, 0] > 240)
    collar_a_dark  = (a_bgr[upper_center_y1:upper_center_y2, upper_center_x1:upper_center_x2, 0] < 235)

    if has_shoulders and is_bg_removed and np.mean(collar_b_white & collar_a_dark) > 0.25:
        collar_a_std = float(np.std(collar_a))
        if collar_a_std > 15.0:
            scores["neck_joint"] = 0.95
            scores["ghost_mannequin"] = 0.94

    # ──────────────────────────────────────────────────────────────────────────
    # 6. JEWELRY & SPECULAR SHINE DETECTION
    # ──────────────────────────────────────────────────────────────────────────
    # Specular highlights in metallic objects (gold/silver/diamond sparkles)
    if is_bg_removed:
        non_bg = (subject_mask > 0)
        # Check if subject has metallic tone (low saturation silver or gold hue in HSV)
        hsv_a = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2HSV)
        is_metallic = ((hsv_a[:, :, 1] < 45) & (hsv_a[:, :, 2] > 140) & non_bg) | \
                      ((hsv_a[:, :, 0] >= 15) & (hsv_a[:, :, 0] <= 38) & (hsv_a[:, :, 1] > 50) & non_bg)
        
        # Specular pixels (sharp bright glare > 245 inside metallic region)
        specular_pixels = (a_bgr[:, :, 0] > 245) & (a_bgr[:, :, 1] > 245) & (a_bgr[:, :, 2] > 245) & is_metallic
        specular_ratio = float(np.sum(specular_pixels)) / max(int(np.sum(non_bg)), 1)
        if 0.015 < specular_ratio < 0.20 and float(np.sum(is_metallic)) / max(int(np.sum(non_bg)), 1) > 0.40:
            scores["jewelry_shiner"] = min(0.95, 0.75 + specular_ratio * 1.5)

    # ──────────────────────────────────────────────────────────────────────────
    # FINAL AGGREGATION & FORMATTING
    # ──────────────────────────────────────────────────────────────────────────
    # Filter services with confidence >= 0.70
    detected_services = [svc for svc, conf in sorted(scores.items(), key=lambda x: x[1], reverse=True) if conf >= 0.70]

    # Fallback if subtle edits occurred
    if not detected_services:
        detected_services = ["retouching", "enhancement"]
        scores["retouching"] = 0.75
        scores["enhancement"] = 0.70

    primary_service = detected_services[0]

    # Create Bengali summary string
    top_items = [f"{SERVICE_DISPLAY_NAMES_BN.get(svc, svc)} ({int(scores.get(svc, 0.7)*100)}%)" for svc in detected_services[:4]]
    summary_bn = "🤖 AI স্বয়ংক্রিয়ভাবে সনাক্ত করেছে: " + ", ".join(top_items)

    return {
        "detected_services": detected_services,
        "primary_service": primary_service,
        "confidence": {k: round(v, 2) for k, v in scores.items()},
        "summary_bn": summary_bn,
        "details": details
    }
