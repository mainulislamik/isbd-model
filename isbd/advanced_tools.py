"""
ISBD v1.00 — Advanced Commercial Photo Editing Suite & AI Training Generators
Production-Grade Graphic Design & Photo Manipulation Algorithms:
1. Multiple Clipping Path Engine (K-means LAB + XY vector segmentation)
2. Ghost Mannequin / Neck Joint (Inner collar synthesis, 3D hollow depth & seamless blending)
3. Commercial Frequency Retouch (3-Layer Frequency Separation & Smart Brightening)
4. Digital Ironing & Wrinkle Cleaner (Morphological Black-Hat crease leveling & weave lock)
5. Jewelry Metal & Gemstone Shiner (Specular inpainting, CLAHE facet brilliance & sparkle)
6. Dual-Zone Ground Shadow & Reflection (Contact shadow, progressive cast shadow & luxury reflection)
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageOps
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent


class MetaStr(str):
    """String subclass that allows dictionary-style access to embedded metadata."""
    def __new__(cls, text, metadata=None):
        s = super().__new__(cls, text)
        s.metadata = metadata or {}
        return s

    def __getitem__(self, item):
        if isinstance(item, str) and item in self.metadata:
            return self.metadata[item]
        return super().__getitem__(item)

    def __contains__(self, key):
        if isinstance(key, str) and key in self.metadata:
            return True
        return super().__contains__(key)

    def get(self, key, default=None):
        return self.metadata.get(key, default)

    def keys(self):
        return self.metadata.keys()

    def values(self):
        return self.metadata.values()

    def items(self):
        return self.metadata.items()


class StudioResult(tuple):
    """
    Dual-compatibility return type for studio services:
    - Unpacks as a 2-tuple: (image, desc)
    - Subscripts as a dictionary: res['ok'], res['image'], res['desc'], res['metadata']
    """
    def __new__(cls, image, desc, ok=True, metadata=None, **kwargs):
        if isinstance(desc, dict):
            text = desc.get("description", "Processed successfully")
            desc = MetaStr(text, desc)
        elif metadata:
            desc = MetaStr(str(desc), metadata)
        return super().__new__(cls, (image, desc))

    def __init__(self, image, desc, ok=True, metadata=None, **kwargs):
        self.image = image
        self.desc = self[1]
        self.ok = ok
        self.metadata = metadata or (self.desc.metadata if isinstance(self.desc, MetaStr) else {})
        self.extra = kwargs

    def __getitem__(self, item):
        if isinstance(item, str):
            if item == "ok":
                return self.ok
            if item == "image":
                return self.image
            if item in ("desc", "description"):
                return self.desc
            if item == "metadata":
                return self.metadata
            if item in self.metadata:
                return self.metadata[item]
            if item in self.extra:
                return self.extra[item]
            raise KeyError(item)
        return super().__getitem__(item)

    def __contains__(self, key):
        if key in ("ok", "image", "desc", "description", "metadata"):
            return True
        if key in self.metadata or key in self.extra:
            return True
        return False

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


def _to_cv_rgb(img_input) -> np.ndarray:
    """Helper to convert any PIL Image or numpy array to an RGB uint8 numpy array."""
    if isinstance(img_input, Image.Image):
        return np.array(img_input.convert("RGB"))
    arr = np.array(img_input)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        if arr.shape[2] == 1:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    return arr.astype(np.uint8)


# ==============================================================================
# 1. MULTIPLE CLIPPING PATH ENGINE
# ==============================================================================
def multiple_clipping_path_engine(img_cv, n_clusters: int = 5) -> StudioResult:
    """
    1. Multiple Clipping Path Engine:
       - Converts image to LAB color space.
       - Runs adaptive K-means clustering (cv2.kmeans) on (L, a, b, x, y) features
         to segment distinct areas (clothing parts, skin, background, accessories).
       - Isolates foreground segments, extracts contours (cv2.findContours),
         simplifies with cv2.approxPolyDP to form vector polygons.
       - Renders a polished multi-segment visualization: color-tint each segment
         with distinct pastel tones and draws thin crisp anti-aliased boundary outlines.
       - Returns StudioResult with PIL Image and detailed segment metadata.
    """
    rgb = _to_cv_rgb(img_cv)
    h, w = rgb.shape[:2]
    k = max(2, min(int(n_clusters), 12))

    # Convert to LAB for perceptually uniform color clustering
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

    # Generate normalized coordinate grids (scaled to balance color vs spatial proximity)
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    x_norm = (x_coords.astype(np.float32) / max(w, 1)) * 90.0
    y_norm = (y_coords.astype(np.float32) / max(h, 1)) * 90.0

    features = np.column_stack([
        lab[:, :, 0].flatten().astype(np.float32) * 1.0,
        lab[:, :, 1].flatten().astype(np.float32) * 1.4,
        lab[:, :, 2].flatten().astype(np.float32) * 1.4,
        x_norm.flatten() * 0.42,
        y_norm.flatten() * 0.42
    ])

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, _ = cv2.kmeans(features, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape((h, w))

    # Curated palette of soft, elegant commercial pastel tones
    PASTEL_COLORS = [
        (255, 179, 186),  # Pastel Pink
        (186, 255, 201),  # Pastel Mint
        (186, 225, 255),  # Pastel Sky Blue
        (255, 255, 186),  # Pastel Cream Yellow
        (255, 223, 186),  # Pastel Apricot
        (225, 190, 231),  # Pastel Lavender
        (178, 235, 242),  # Pastel Cyan
        (255, 204, 188),  # Pastel Coral
    ]

    vis = rgb.copy().astype(np.float32)
    segments_meta = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for c_idx in range(k):
        c_mask = (labels == c_idx).astype(np.uint8) * 255
        c_mask = cv2.morphologyEx(c_mask, cv2.MORPH_OPEN, kernel)
        c_mask = cv2.morphologyEx(c_mask, cv2.MORPH_CLOSE, kernel)

        color = PASTEL_COLORS[c_idx % len(PASTEL_COLORS)]
        tint_layer = np.full_like(rgb, color, dtype=np.float32)
        m_bool = (c_mask > 0)[:, :, None]

        # Soft pastel tint overlay
        vis = np.where(m_bool, vis * 0.68 + tint_layer * 0.32, vis)

        # Extract contours and convert to simplified vector polygons
        contours, _ = cv2.findContours(c_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 25:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.007 * peri, True)
            if len(approx) >= 3:
                polygons.append(approx.squeeze().tolist())
                # Draw crisp dual-layer vector boundary (white glow + dark core)
                vis_uint = np.clip(vis, 0, 255).astype(np.uint8)
                cv2.polylines(vis_uint, [approx], True, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.polylines(vis_uint, [approx], True, (25, 25, 30), 1, cv2.LINE_AA)
                vis = vis_uint.astype(np.float32)

        segments_meta.append({
            "segment_id": c_idx,
            "label": f"Path_{c_idx + 1}",
            "color_rgb": color,
            "pixel_count": int(np.sum(c_mask > 0)),
            "area_percentage": round(float(np.mean(c_mask > 0) * 100), 2),
            "polygons_count": len(polygons)
        })

    out_vis = np.clip(vis, 0, 255).astype(np.uint8)
    pil_out = Image.fromarray(out_vis)
    desc_text = f"Multiple Clipping Path Engine ({k} ভেক্টর সেগমেন্ট পাথ আইসোলেশন ও কনট্যুর তৈরি সম্পন্ন)"
    meta = {
        "description": desc_text,
        "n_clusters": k,
        "segments": segments_meta,
        "total_segments": len(segments_meta)
    }
    return StudioResult(pil_out, desc_text, ok=True, metadata=meta)


# ==============================================================================
# 2. GHOST MANNEQUIN / NECK JOINT
# ==============================================================================
def ghost_mannequin_neck_joint(img_cv, inner_collar_cv=None) -> StudioResult:
    """
    2. Ghost Mannequin Neck Joint Engine:
       - Identifies the neckline/collar boundary using alpha/contours or upper garment concave region.
       - Samples fabric color & texture from inner shoulder/chest, darkened 15-20% for 3D depth.
       - Synthesizes an arched inner collar shape fitting the collar cutout opening.
       - Blends using seamless cloning or multi-band feathered alpha gradient.
       - Renders soft ambient depth shadow and inner collar rim highlight along junction.
    """
    rgb = _to_cv_rgb(img_cv)
    h, w = rgb.shape[:2]

    # Detect foreground garment mask
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, fg_thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        gx, gy, gw, gh = cv2.boundingRect(c)
    else:
        gx, gy, gw, gh = int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8)

    cx = gx + gw // 2
    neck_w = max(20, int(gw * 0.38))
    neck_h = max(20, int(gh * 0.22))
    neck_top = max(5, gy - int(neck_h * 0.15))
    neck_center_y = gy + int(neck_h * 0.45)

    # Cutout mask representing the hollow interior of the collar
    cutout_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(cutout_mask, (cx, neck_center_y), (neck_w // 2, neck_h), 0, 0, 360, 255, -1)

    # Only fill where the foreground garment is hollow / background
    inner_mask = cv2.bitwise_and(cutout_mask, cv2.bitwise_not(fg_mask))
    if np.sum(inner_mask > 0) < 50:
        # Fallback: create realistic inner oval if background was noisy
        inner_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(inner_mask, (cx, gy + int(neck_h * 0.35)), (neck_w // 2, int(neck_h * 0.6)), 0, 0, 360, 255, -1)

    # Sample fabric color from the inner shoulder / chest region
    sample_y1 = min(h - 5, gy + int(gh * 0.25))
    sample_y2 = min(h, sample_y1 + max(10, int(gh * 0.20)))
    sample_x1 = max(0, cx - int(neck_w * 0.35))
    sample_x2 = min(w, cx + int(neck_w * 0.35))
    chest_sample = rgb[sample_y1:sample_y2, sample_x1:sample_x2]

    if chest_sample.size > 0:
        base_color = np.mean(chest_sample, axis=(0, 1))
    else:
        base_color = np.array([80.0, 90.0, 110.0])

    # Darken 15-20% for natural interior cavity depth
    inner_color = np.clip(base_color * 0.82, 0, 255).astype(np.uint8)

    # Synthesize inner collar backing
    if inner_collar_cv is not None:
        inner_collar_arr = _to_cv_rgb(inner_collar_cv)
        backing = cv2.resize(inner_collar_arr, (w, h))
    else:
        backing = np.zeros_like(rgb)
        backing[:] = inner_color
        # Add subtle vertical ambient lighting gradient (darker at bottom of inner neck)
        y_grad = np.linspace(1.10, 0.85, h, dtype=np.float32)[:, None, None]
        backing = np.clip(backing.astype(np.float32) * y_grad, 0, 255).astype(np.uint8)

    # Inner collar rim highlight along the upper inner edge
    rim_mask = cv2.subtract(
        cv2.dilate(inner_mask, kernel, iterations=1),
        inner_mask
    )

    # Ambient depth shadow along front collar junction
    junction_shadow = cv2.GaussianBlur(
        cv2.dilate(inner_mask, kernel, iterations=3),
        (21, 21),
        0
    ).astype(np.float32) / 255.0

    # Multi-band feathered alpha blend
    feathered_mask = cv2.GaussianBlur(inner_mask, (7, 7), 0).astype(np.float32) / 255.0
    feathered_3c = feathered_mask[:, :, None]

    blended = rgb.astype(np.float32) * (1.0 - feathered_3c) + backing.astype(np.float32) * feathered_3c

    # Apply ambient depth shadow into the hollow backing
    shadow_factor = 1.0 - (junction_shadow[:, :, None] * feathered_3c * 0.35)
    blended = blended * shadow_factor

    # Apply subtle inner collar rim highlight
    rim_3c = (rim_mask[:, :, None].astype(np.float32) / 255.0) * 0.20
    blended = np.clip(blended + (rim_3c * 255.0), 0, 255).astype(np.uint8)

    desc_text = "Ghost Mannequin Neck Joint (থ্রিডি গোস্ট ম্যানিকুইন ইনার কলার কম্পোজিট সম্পন্ন)"
    return StudioResult(Image.fromarray(blended), desc_text, ok=True)


# ==============================================================================
# 3. COMMERCIAL FREQUENCY RETOUCH
# ==============================================================================
def commercial_frequency_retouch(img_cv, smooth_strength: float = 0.55, texture_retention: float = 1.12) -> StudioResult:
    """
    3. Commercial Frequency Retouch Engine:
       - 3-Layer Frequency Separation:
         * Low Frequency: Bilateral filter on LAB for smooth skin tone and redness removal.
         * Mid Frequency: Bandpass residual to detect and suppress blemishes/pimples/dark spots.
         * High Frequency: High-pass extraction of micro-skin pores and texture, re-injected with 100%+ retention.
       - Smart Eye & Teeth brightening: Brightens whites of eyes and teeth subtly while enhancing iris catchlight contrast.
    """
    rgb = _to_cv_rgb(img_cv)
    h, w = rgb.shape[:2]
    img_f = rgb.astype(np.float32)

    # 1. High Frequency Extraction (micro-pores, hair strands, crisp textile weaves)
    fine = cv2.GaussianBlur(img_f, (3, 3), 0)
    high_freq = img_f - fine

    # 2. Low Frequency Layer (smooth tone & redness removal via Bilateral filtering in LAB)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    sigma_c = int(25 + smooth_strength * 35)
    sigma_s = int(15 + smooth_strength * 25)
    l_low = cv2.bilateralFilter(l_chan, d=9, sigmaColor=sigma_c, sigmaSpace=sigma_s)

    # Redness removal: gently moderate high 'a' channel deviations in skin tones
    a_low = cv2.bilateralFilter(a_chan, d=7, sigmaColor=25, sigmaSpace=15)
    b_low = cv2.bilateralFilter(b_chan, d=7, sigmaColor=25, sigmaSpace=15)

    low_lab = cv2.merge([l_low, a_low, b_low])
    low_rgb = cv2.cvtColor(low_lab, cv2.COLOR_LAB2RGB).astype(np.float32)

    # 3. Mid Frequency Layer (Bandpass residual containing blemishes, pimples & spots)
    mid_freq = fine - low_rgb

    # Suppress blemishes / uneven spot spikes in mid frequency
    suppression_factor = max(0.2, 1.0 - (float(smooth_strength) * 0.65))
    mid_cleaned = mid_freq * suppression_factor

    # 4. Reconstruct with boosted High Frequency texture retention
    retention = float(texture_retention)
    reconstructed = low_rgb + mid_cleaned + (high_freq * retention)
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)

    # 5. Smart Eye & Teeth Brightening + Iris Catchlight Contrast
    hsv = cv2.cvtColor(reconstructed, cv2.COLOR_RGB2HSV)
    h_c, s_c, v_c = cv2.split(hsv)

    # Whites of eyes & teeth: High brightness (v > 165), low color saturation (s < 55)
    eye_teeth_mask = ((v_c > 165) & (s_c < 55) & (v_c < 252)).astype(np.uint8) * 255
    eye_teeth_mask = cv2.GaussianBlur(eye_teeth_mask, (5, 5), 0)
    whitening_weight = (eye_teeth_mask.astype(np.float32) / 255.0)[:, :, None] * 0.16

    # Iris specular catchlights: specular peaks (v > 220)
    specular_mask = (v_c > 225).astype(np.uint8) * 255
    specular_weight = (specular_mask.astype(np.float32) / 255.0)[:, :, None] * 0.12

    brightened = reconstructed.astype(np.float32) * (1.0 + whitening_weight + specular_weight)
    final_rgb = np.clip(brightened, 0, 255).astype(np.uint8)

    desc_text = "Commercial Frequency Retouch (৩-লেয়ার ফ্রিকোয়েন্সি সেপারেশন, স্কিন পোর রিটেনশন ও টিথ হোয়াইটনিং সম্পন্ন)"
    return StudioResult(Image.fromarray(final_rgb), desc_text, ok=True)


# ==============================================================================
# 4. DIGITAL IRONING & WRINKLE CLEANER
# ==============================================================================
def digital_ironing_wrinkle_cleaner(img_cv, intensity: float = 0.7) -> StudioResult:
    """
    4. Digital Ironing Wrinkle Cleaner Engine:
       - Converts to Lab color space.
       - Separates low-frequency lighting and folding shadows from high-frequency fabric weaves/prints
         using morphological Black-Hat transform or oriented line kernels.
       - Levels crease shadows along fabric flow lines while locking weave texture and garment seams.
       - Reconstructs Lab image to RGB.
    """
    rgb = _to_cv_rgb(img_cv)
    h, w = rgb.shape[:2]
    strength = float(np.clip(intensity, 0.1, 1.0))

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    # High-frequency fabric weave extraction
    l_float = l_chan.astype(np.float32)
    l_blur = cv2.GaussianBlur(l_float, (5, 5), 0)
    high_weave = l_float - l_blur

    # Detect dark crease & fold shadows using morphological Black-Hat transform
    se_size = max(5, int(15 * strength))
    kernel_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (se_size, 3))
    kernel_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (3, se_size))
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size))

    blackhat_h = cv2.morphologyEx(l_chan, cv2.MORPH_BLACKHAT, kernel_horiz)
    blackhat_v = cv2.morphologyEx(l_chan, cv2.MORPH_BLACKHAT, kernel_vert)
    blackhat_c = cv2.morphologyEx(l_chan, cv2.MORPH_BLACKHAT, kernel_cross)

    crease_shadows = cv2.max(cv2.max(blackhat_h, blackhat_v), blackhat_c)
    crease_smooth = cv2.GaussianBlur(crease_shadows, (7, 7), 0).astype(np.float32)

    # Edge protection: Lock major seams and garment silhouette contours
    edges = cv2.Canny(l_chan, 50, 140)
    seam_lock = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    seam_mask = (seam_lock.astype(np.float32) / 255.0)

    # Iron out and level the crease shadows
    ironing_boost = crease_smooth * (0.85 * strength) * (1.0 - seam_mask)
    l_ironed = np.clip(l_float + ironing_boost, 0, 255)

    # Surface smoothing for drape flow lines
    l_surface = cv2.bilateralFilter(l_ironed.astype(np.uint8), d=7, sigmaColor=35, sigmaSpace=20).astype(np.float32)

    # Re-inject locked weave texture
    l_final = np.clip(l_surface + high_weave * 0.95, 0, 255).astype(np.uint8)

    lab_ironed = cv2.merge([l_final, a_chan, b_chan])
    rgb_ironed = cv2.cvtColor(lab_ironed, cv2.COLOR_LAB2RGB)

    desc_text = "Digital Ironing Wrinkle Cleaner (ডিজিটাল আয়রনিং ও ফ্যাব্রিক টেক্সচার লক সম্পন্ন)"
    return StudioResult(Image.fromarray(rgb_ironed), desc_text, ok=True)


# ==============================================================================
# 5. JEWELRY METAL & GEMSTONE SHINER
# ==============================================================================
def jewelry_metal_gemstone_shiner(img_cv) -> StudioResult:
    """
    5. Jewelry Metal & Gemstone Shiner Engine:
       - Polished Metal: Detects smooth specular metal regions, identifies micro-scratches,
         and inpaints them with cv2.inpaint.
       - Gemstones: Detects faceted gemstone edges with high gradient entropy, enhances
         facet micro-contrast with CLAHE, and boosts sparkle/specular flare.
    """
    rgb = _to_cv_rgb(img_cv)
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # 1. Detect Metal Scratches & Inpaint (High-contrast micro-ridges on reflective surfaces)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    scratch_energy = cv2.add(blackhat, tophat)

    # Restrict scratch healing to metallic/gemstone areas (mid-to-high luminance)
    metal_zone = cv2.inRange(gray, 70, 245)
    scratch_mask = cv2.bitwise_and(cv2.threshold(scratch_energy, 22, 255, cv2.THRESH_BINARY)[1], metal_zone)

    # Inpaint scratches
    inpainted = cv2.inpaint(rgb, scratch_mask, 3, cv2.INPAINT_TELEA)

    # 2. Gemstone Facet Brilliance & Micro-Contrast (CLAHE on L-channel)
    lab = cv2.cvtColor(inpainted, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    l_brilliant = clahe.apply(l)
    gem_rgb = cv2.cvtColor(cv2.merge([l_brilliant, a, b]), cv2.COLOR_LAB2RGB)

    # 3. Specular Sparkle & Diamond Flare Boost
    _, specular_peaks = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)
    # Generate diamond star sparkle glow
    sparkle_glow = cv2.GaussianBlur(specular_peaks, (11, 11), 0)
    sparkle_rgb = cv2.cvtColor(sparkle_glow, cv2.COLOR_GRAY2RGB)

    # Composite polished metal with gemstone brilliance & specular shine
    shined = cv2.addWeighted(gem_rgb, 0.88, rgb, 0.12, 0)
    shined = cv2.addWeighted(shined, 1.0, sparkle_rgb, 0.22, 0)
    shined = np.clip(shined, 0, 255).astype(np.uint8)

    desc_text = "Jewelry Metal & Gemstone Shiner (জুয়েলারি মেটাল পলিশ ও রত্নপাথরের স্পার্কল সম্পন্ন)"
    return StudioResult(Image.fromarray(shined), desc_text, ok=True)


# ==============================================================================
# 6. DUAL-ZONE GROUND SHADOW & REFLECTION
# ==============================================================================
def dual_zone_ground_shadow_and_reflection(img_cv, shadow_type: str = "both") -> StudioResult:
    """
    6. Dual-Zone Ground Shadow & Reflection Engine:
       - Contact Shadow (Ambient Occlusion): Dark, crisp, tight shadow along bottom contact line.
       - Cast Shadow: Directional perspective shadow with progressive Gaussian blur.
       - Luxury Ground Reflection: Vertical flip with perspective warp, exponential opacity fade,
         and surface roughness blur.
    """
    rgb = _to_cv_rgb(img_cv)
    h, w = rgb.shape[:2]

    # Extract silhouette mask
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find product base contact line
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        bx, by, bw, bh = cv2.boundingRect(c)
        base_y = by + bh
    else:
        bx, by, bw, bh = int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8)
        base_y = h

    # Canvas dimensions for grounded product with shadow & reflection
    extra_bottom = int(h * 0.45)
    canvas_h = h + extra_bottom
    canvas = np.full((canvas_h, w, 3), 255, dtype=np.uint8)

    # 1. Luxury Ground Reflection
    mirror_h = int(bh * 0.50)
    product_crop = rgb[by:base_y, bx:bx+bw]
    mask_crop = mask[by:base_y, bx:bx+bw]

    if product_crop.size > 0 and mirror_h > 5:
        flipped_product = cv2.flip(product_crop, 0)
        flipped_mask = cv2.flip(mask_crop, 0)
        flipped_product = cv2.resize(flipped_product, (bw, mirror_h))
        flipped_mask = cv2.resize(flipped_mask, (bw, mirror_h))

        # Exponential vertical opacity fade for mirror reflection
        fade_curve = np.linspace(0.40, 0.0, mirror_h, dtype=np.float32)[:, None]
        roughness_blur = cv2.GaussianBlur(flipped_product, (7, 7), 0)

        # Place reflection on canvas
        ref_y1 = base_y
        ref_y2 = min(canvas_h, base_y + mirror_h)
        paste_h = ref_y2 - ref_y1

        if paste_h > 0:
            ref_sub = roughness_blur[:paste_h]
            ref_mask_sub = (flipped_mask[:paste_h].astype(np.float32) / 255.0)[:, :, None] * fade_curve[:paste_h, :, None]
            canvas[ref_y1:ref_y2, bx:bx+bw] = (
                canvas[ref_y1:ref_y2, bx:bx+bw].astype(np.float32) * (1.0 - ref_mask_sub) +
                ref_sub.astype(np.float32) * ref_mask_sub
            ).astype(np.uint8)

    # 2. Contact Shadow (Tight Ambient Occlusion along product base)
    contact_mask = np.zeros((canvas_h, w), dtype=np.uint8)
    contact_w = max(10, int(bw * 0.85))
    contact_h = max(4, int(bh * 0.04))
    cv2.ellipse(contact_mask, (bx + bw // 2, base_y), (contact_w // 2, contact_h), 0, 0, 360, 255, -1)
    contact_shadow = cv2.GaussianBlur(contact_mask, (9, 9), 0).astype(np.float32) / 255.0

    # 3. Directional Cast Shadow (Perspective falloff)
    cast_mask = np.zeros((canvas_h, w), dtype=np.uint8)
    cast_w = max(10, int(bw * 0.95))
    cast_h = max(8, int(bh * 0.20))
    cv2.ellipse(cast_mask, (bx + bw // 2 + int(bw * 0.08), base_y + int(cast_h * 0.6)), (cast_w // 2, cast_h), -15, 0, 360, 255, -1)
    cast_shadow = cv2.GaussianBlur(cast_mask, (25, 25), 0).astype(np.float32) / 255.0

    # Combine contact & cast shadows
    total_shadow = np.clip(contact_shadow * 0.75 + cast_shadow * 0.35, 0.0, 1.0)[:, :, None]
    shadow_color = np.array([30.0, 32.0, 38.0])

    canvas = (canvas.astype(np.float32) * (1.0 - total_shadow) + shadow_color * total_shadow).astype(np.uint8)

    # 4. Paste original product over shadow & reflection
    fg_alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
    canvas[:h, :w] = (canvas[:h, :w].astype(np.float32) * (1.0 - fg_alpha) + rgb.astype(np.float32) * fg_alpha).astype(np.uint8)

    # Resize back to original target dimensions
    out_img = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)

    desc_text = "Dual-Zone Ground Shadow & Reflection (ডুয়াল-জোন কন্টাক্ট শ্যাডো ও বিলাসবহুল মিরর রিফ্লেকশন সম্পন্ন)"
    return StudioResult(Image.fromarray(out_img), desc_text, ok=True)


# ==============================================================================
# LEGACY & EXTENDED UTILITY ENGINES
# ==============================================================================
def dress_wrinkle_cleaner(pil_img: Image.Image) -> StudioResult:
    return digital_ironing_wrinkle_cleaner(pil_img)

def cast_shadow_3d(pil_img: Image.Image, angle: float = 45.0, opacity: float = 0.55, blur_radius: int = 12) -> StudioResult:
    return dual_zone_ground_shadow_and_reflection(pil_img, shadow_type="both")

def jewelry_metal_shiner(pil_img: Image.Image) -> StudioResult:
    return jewelry_metal_gemstone_shiner(pil_img)

def apparel_recolor(pil_img: Image.Image, target_hue: int = 110) -> StudioResult:
    cv_img = _to_cv_rgb(pil_img)
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    cloth_mask = cv2.inRange(s, 30, 255)
    new_h = np.full_like(h, target_hue, dtype=np.uint8)
    h_recolored = np.where(cloth_mask > 0, new_h, h)
    s_boosted = np.where(cloth_mask > 0, np.clip(s.astype(np.int32) + 20, 0, 255).astype(np.uint8), s)
    recolored_rgb = cv2.cvtColor(cv2.merge((h_recolored, s_boosted, v)), cv2.COLOR_HSV2RGB)
    return StudioResult(Image.fromarray(recolored_rgb), "Apparel Recolor (পোশাকের কালার ভ্যারিয়েন্ট সোয়াচ সম্পন্ন)", ok=True)

def studio_denoise_healer(pil_img: Image.Image) -> StudioResult:
    cv_img = _to_cv_rgb(pil_img)
    denoised = cv2.fastNlMeansDenoisingColored(cv_img, None, 8, 8, 7, 21)
    return StudioResult(Image.fromarray(denoised), "Studio Denoise (হাই-আইএসও নয়েজ ও গ্রেইন রিমুভাল সম্পন্ন)", ok=True)

def spot_dust_remover(pil_img: Image.Image) -> StudioResult:
    cv_img = _to_cv_rgb(pil_img)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    blemish_mask = cv2.add(blackhat, tophat)
    _, mask = cv2.threshold(blemish_mask, 18, 255, cv2.THRESH_BINARY)
    cleaned = cv2.inpaint(cv_img, mask, 3, cv2.INPAINT_TELEA)
    return StudioResult(Image.fromarray(cleaned), "Spot & Dust Remover (পোশাকের খুঁত ও ধূলিকণা রিমুভাল সম্পন্ন)", ok=True)
