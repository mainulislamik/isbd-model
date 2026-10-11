"""
ISBD v1.00 — Google MediaPipe 468-Point Face Mesh Retouching Engine
Provides anatomical, non-destructive commercial portrait retouching:
- Teeth Whitening (LAB space luminance boost + desaturation of yellow/brown cast)
- Eye Iris Pop & Sclera Clarity
- Lip Sheen & Gloss Enhancement
- Anatomical Facial Masking (Skin-only smoothing without blurring eyebrows, eyes, nostrils or lips)
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, Optional, List, Tuple


_DETECTOR = None

def get_face_landmarker():
    global _DETECTOR
    if _DETECTOR is not None:
        return _DETECTOR
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        model_path = "/app/data/models/face_landmarker.task"
        if not os.path.exists(model_path):
            model_path = "/home/imon/isbd_model/data/models/face_landmarker.task"

        if os.path.exists(model_path):
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                num_faces=1
            )
            _DETECTOR = vision.FaceLandmarker.create_from_options(options)
            return _DETECTOR
    except Exception as e:
        print(f"[MediaPipe] Init error: {e}")
    return None


# Standard Canonical MediaPipe 468 Landmark Indices
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
             377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


def _get_polygon(landmarks, indices: List[int], w: int, h: int) -> np.ndarray:
    pts = []
    for idx in indices:
        if idx < len(landmarks):
            lm = landmarks[idx]
            pts.append([int(lm.x * w), int(lm.y * h)])
    return np.array(pts, dtype=np.int32)


def apply_teeth_whitening(img_bgr: np.ndarray, inner_lips_poly: np.ndarray, strength: float = 0.65) -> np.ndarray:
    """
    Whitens teeth inside the mouth opening:
    - Increases LAB luminance L
    - Reduces yellow/brown saturation (a and b channels closer to neutral 128)
    """
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(inner_lips_poly) >= 3:
        cv2.fillPoly(mask, [inner_lips_poly], 255)
        # Soft feather
        mask = cv2.GaussianBlur(mask, (15, 15), 0)

    if mask.max() == 0:
        return img_bgr

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)

    # Detect lighter/yellowish pixels in mouth opening (likely teeth, not dark throat or red tongue)
    # Yellowish in LAB: b > 128
    teeth_candidate = (l > 90) & (b >= 126)
    factor = (mask.astype(np.float32) / 255.0) * teeth_candidate.astype(np.float32) * strength

    # Boost lightness
    l_whitened = l + (factor * 35.0)
    # Desaturate yellow cast
    b_whitened = b - (factor * (b - 128.0) * 0.70)
    a_whitened = a - (factor * (a - 128.0) * 0.30)

    l_whitened = np.clip(l_whitened, 0, 255)
    a_whitened = np.clip(a_whitened, 0, 255)
    b_whitened = np.clip(b_whitened, 0, 255)

    lab_res = cv2.merge((l_whitened, a_whitened, b_whitened)).astype(np.uint8)
    return cv2.cvtColor(lab_res, cv2.COLOR_LAB2BGR)


def apply_eye_enhancement(img_bgr: np.ndarray, left_poly: np.ndarray, right_poly: np.ndarray) -> np.ndarray:
    """
    Iris Pop, Sclera brightening, and specular reflection boost for lifelike eyes.
    """
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(left_poly) >= 3:
        cv2.fillPoly(mask, [left_poly], 255)
    if len(right_poly) >= 3:
        cv2.fillPoly(mask, [right_poly], 255)

    if mask.max() == 0:
        return img_bgr

    mask_soft = (cv2.GaussianBlur(mask, (9, 9), 0).astype(np.float32) / 255.0)[:, :, np.newaxis]

    # Sharpen and contrast eyes
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l_clahe = clahe.apply(l)
    sharpened_bgr = cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)

    # Sub-pixel iris highlight boost
    enhanced_bgr = cv2.addWeighted(img_bgr, 0.45, sharpened_bgr, 0.55, 0)
    out = (img_bgr.astype(np.float32) * (1.0 - mask_soft)) + (enhanced_bgr.astype(np.float32) * mask_soft)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_lip_sheen(img_bgr: np.ndarray, outer_poly: np.ndarray, inner_poly: np.ndarray) -> np.ndarray:
    """
    Applies natural lip color richness and subtle glossy highlight.
    """
    h, w = img_bgr.shape[:2]
    mask_outer = np.zeros((h, w), dtype=np.uint8)
    mask_inner = np.zeros((h, w), dtype=np.uint8)

    if len(outer_poly) >= 3:
        cv2.fillPoly(mask_outer, [outer_poly], 255)
    if len(inner_poly) >= 3:
        cv2.fillPoly(mask_inner, [inner_poly], 255)

    lip_mask = cv2.subtract(mask_outer, mask_inner)
    lip_mask_soft = (cv2.GaussianBlur(lip_mask, (11, 11), 0).astype(np.float32) / 255.0)[:, :, np.newaxis]

    if lip_mask.max() == 0:
        return img_bgr

    # Enhance vibrancy in HSV space
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    s_ch = np.clip(s_ch * 1.15, 0, 255)
    v_ch = np.clip(v_ch * 1.05, 0, 255)
    rich_lips_bgr = cv2.cvtColor(cv2.merge((h_ch, s_ch, v_ch)).astype(np.uint8), cv2.COLOR_HSV2BGR)

    out = (img_bgr.astype(np.float32) * (1.0 - lip_mask_soft)) + (rich_lips_bgr.astype(np.float32) * lip_mask_soft)
    return np.clip(out, 0, 255).astype(np.uint8)


def mediapipe_pro_retouch(
    img_bgr: np.ndarray,
    teeth_whiten: bool = True,
    eye_pop: bool = True,
    lip_gloss: bool = True,
    dermis_retouch: bool = True
) -> Dict[str, Any]:
    """
    Runs full MediaPipe 468-point 3D landmark mesh portrait retouch.
    """
    h, w = img_bgr.shape[:2]
    detector = get_face_landmarker()

    landmarks = None
    if detector is not None:
        try:
            import mediapipe as mp
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = detector.detect(mp_img)
            if res.face_landmarks and len(res.face_landmarks) > 0:
                landmarks = res.face_landmarks[0]
        except Exception as e:
            print(f"[MediaPipe] Detect error: {e}")

    result = img_bgr.copy()
    features_applied = []

    if landmarks is not None:
        outer_lips = _get_polygon(landmarks, LIPS_OUTER, w, h)
        inner_lips = _get_polygon(landmarks, LIPS_INNER, w, h)
        left_eye = _get_polygon(landmarks, LEFT_EYE, w, h)
        right_eye = _get_polygon(landmarks, RIGHT_EYE, w, h)

        if teeth_whiten and len(inner_lips) >= 3:
            result = apply_teeth_whitening(result, inner_lips, strength=0.70)
            features_applied.append("Teeth Whitening")

        if eye_pop and (len(left_eye) >= 3 or len(right_eye) >= 3):
            result = apply_eye_enhancement(result, left_eye, right_eye)
            features_applied.append("Eye Iris Pop")

        if lip_gloss and len(outer_lips) >= 3:
            result = apply_lip_sheen(result, outer_lips, inner_lips)
            features_applied.append("Lip Gloss Sheen")

        if dermis_retouch:
            from isbd.frequency_engine import commercial_frequency_retouch
            freq_res = commercial_frequency_retouch(result, skin_smooth_strength=0.50, texture_retention=1.10)
            result = freq_res["processed_bgr"]
            features_applied.append("Dermis Frequency Separation")
    else:
        # Fallback to pure frequency engine if no human face detected
        from isbd.frequency_engine import commercial_frequency_retouch
        freq_res = commercial_frequency_retouch(result, skin_smooth_strength=0.50, texture_retention=1.10)
        result = freq_res["processed_bgr"]
        features_applied.append("Universal Frequency Texture Retouch")

    desc = f"MediaPipe 468pt Mesh: {', '.join(features_applied)} সম্পন্ন" if landmarks is not None else "Face Mesh (সরাসরি টেক্সচার ও ফ্রিকোয়েন্সি রিটাচ সম্পন্ন)"

    return {
        "ok": True,
        "processed_bgr": result,
        "face_detected": landmarks is not None,
        "features_applied": features_applied,
        "landmarks_count": 468 if landmarks is not None else 0,
        "description": desc
    }
