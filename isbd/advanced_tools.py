"""
ISBD v1.00 — Advanced Commercial Photo Editing Suite & AI Training Generators
Contains 6 Advanced High-End Studio Photo Manipulation Engines:
1. Dress Wrinkle & Crease Cleaner (Garment structure-texture smoothing)
2. 3D Cast Shadow & Perspective Grounding (Directional soft perspective shadow)
3. Jewelry & Metal Shiner (Specular highlight extraction & scratch healing)
4. Apparel Color Swatch Recolor (Luminance & drape-preserving hue transfer)
5. Studio Denoise & High-ISO Healer (Fast Non-Local / Bilateral Grain cleaner)
6. Spot & Thread Dust Inpainting (Fast local blemish & dust remover)
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageOps
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent


def dress_wrinkle_cleaner(pil_img: Image.Image) -> tuple[Image.Image, str]:
    """
    Garment wrinkle & fold smoothing.
    Decomposes image into low-frequency lighting/shape and high-frequency fabric texture,
    attenuating medium-frequency harsh wrinkle shadows.
    """
    cv_img = np.array(pil_img.convert("RGB"))
    
    # 1. Edge-preserving surface blur (smooths harsh wrinkle transitions)
    smoothed = cv2.edgePreservingFilter(cv_img, flags=1, sigma_s=50, sigma_r=0.35)
    
    # 2. Extract subtle fabric micro-texture via unsharp difference
    gray_orig = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
    blurred_orig = cv2.GaussianBlur(cv_img, (7, 7), 0)
    high_freq = cv2.subtract(cv_img, blurred_orig) + 128
    
    # 3. Blend smoothed drape with preserved textile texture
    wrinkle_free = cv2.addWeighted(smoothed, 0.88, high_freq, 0.12, 0)
    
    return Image.fromarray(wrinkle_free), "Dress Wrinkle Cleaner (পোশাকের ভাঁজ দূরীকরণ ও ফ্যাব্রিক মসৃণ সম্পন্ন)"


def cast_shadow_3d(pil_img: Image.Image, angle: float = 45.0, opacity: float = 0.55, blur_radius: int = 12) -> tuple[Image.Image, str]:
    """
    Directional 3D Perspective Cast Shadow.
    Projects the object's alpha silhouette onto a 3D ground plane with realistic falloff.
    """
    w, h = pil_img.size
    cv_img = np.array(pil_img.convert("RGB"))
    
    # Extract foreground alpha / silhouette
    gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    
    # Construct 3D perspective shadow canvas
    shadow_h = int(h * 0.4)
    canvas_h = h + shadow_h
    canvas = Image.new("RGB", (w, canvas_h), (255, 255, 255))
    
    # Create shadow projection mask using affine shear
    shadow_mask = Image.fromarray(thresh).resize((w, int(h * 0.35)))
    shadow_mask = shadow_mask.transform(
        (w, shadow_h),
        Image.Transform.AFFINE,
        (1, -0.4, 0, 0, 1, 0),
        Image.Resampling.BILINEAR
    )
    
    # Apply Gaussian blur for soft shadow edge
    shadow_mask = shadow_mask.filter(ImageFilter.GaussianBlur(blur_radius))
    
    # Apply opacity fade
    shadow_layer = Image.new("RGBA", (w, shadow_h), (30, 30, 35, int(255 * opacity)))
    
    canvas.paste(shadow_layer, (0, h - int(shadow_h * 0.3)), shadow_mask)
    canvas.paste(pil_img, (0, 0))
    
    return canvas.resize((w, h)), "3D Cast Shadow (বাস্তবসম্মত ডিরেকশনাল ৩D শ্যাডো তৈরি সম্পন্ন)"


def jewelry_metal_shiner(pil_img: Image.Image) -> tuple[Image.Image, str]:
    """
    Jewelry & Metal Surface Polisher.
    Enhances specular reflections, diamond brilliance, and metallic luster.
    """
    cv_img = np.array(pil_img.convert("RGB"))
    
    # 1. Specular highlight mask (bright reflective peaks)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
    _, high_peaks = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
    peaks_blur = cv2.GaussianBlur(high_peaks, (9, 9), 0)
    
    # 2. Local contrast enhancement (CLAHE for metallic micro-detail)
    lab = cv2.cvtColor(cv_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    contrast_rgb = cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2RGB)
    
    # 3. Add specular glow onto metal highlights
    glow = cv2.cvtColor(peaks_blur, cv2.COLOR_GRAY2RGB)
    shined = cv2.addWeighted(contrast_rgb, 0.9, glow, 0.15, 0)
    
    return Image.fromarray(shined), "Jewelry & Metal Shiner (গহনা ও মেটালের প্রিমিয়াম গ্লস ও শাইন সম্পন্ন)"


def apparel_recolor(pil_img: Image.Image, target_hue: int = 110) -> tuple[Image.Image, str]:
    """
    Garment & Apparel Color Swatch Recolor.
    Transfers hue while preserving shadows, cloth highlights, and texture folds.
    """
    cv_img = np.array(pil_img.convert("RGB"))
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    
    # Mask out extreme whites and dark shadows so only colored cloth changes
    cloth_mask = cv2.inRange(s, 30, 255)
    
    # Shift hue towards target
    new_h = np.full_like(h, target_hue, dtype=np.uint8)
    h_recolored = np.where(cloth_mask > 0, new_h, h)
    s_boosted = np.where(cloth_mask > 0, np.clip(s.astype(np.int32) + 20, 0, 255).astype(np.uint8), s)
    
    merged_hsv = cv2.merge((h_recolored, s_boosted, v))
    recolored_rgb = cv2.cvtColor(merged_hsv, cv2.COLOR_HSV2RGB)
    
    return Image.fromarray(recolored_rgb), "Apparel Recolor (পোশাকের কালার ভ্যারিয়েন্ট সোয়াচ সম্পন্ন)"


def studio_denoise_healer(pil_img: Image.Image) -> tuple[Image.Image, str]:
    """
    High-ISO Sensor Noise & Compression Healer.
    Fast bilateral chrominance + luminance denoise.
    """
    cv_img = np.array(pil_img.convert("RGB"))
    denoised = cv2.fastNlMeansDenoisingColored(cv_img, None, 8, 8, 7, 21)
    return Image.fromarray(denoised), "Studio Denoise (হাই-আইএসও নয়েজ ও গ্রেইন রিমুভাল সম্পন্ন)"


def spot_dust_remover(pil_img: Image.Image) -> tuple[Image.Image, str]:
    """
    Spot, Dust & Thread Defect Remover.
    Locates isolated micro-spots/dust and performs Navier-Stokes inpainting.
    """
    cv_img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
    
    # Detect micro dark specks and bright dust particles
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    blemish_mask = cv2.add(blackhat, tophat)
    _, mask = cv2.threshold(blemish_mask, 18, 255, cv2.THRESH_BINARY)
    
    # Inpaint detected micro-spots
    cleaned = cv2.inpaint(cv_img, mask, 3, cv2.INPAINT_TELEA)
    return Image.fromarray(cleaned), "Spot & Dust Remover (পোশাকের খুঁত ও ধূলিকণা রিমুভাল সম্পন্ন)"
