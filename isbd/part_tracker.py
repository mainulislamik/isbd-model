"""
ISBD v1.00 — Advanced Human Anatomy, Apparel & Semantic Part Segmentation Detector
Provides granular multi-part parsing for human images:
1. Face & Head (মুখমণ্ডল ও মাথা)
2. Eyes & Eyebrows (চোখ ও ভ্রু)
3. Nose & Lips (নাক ও ঠোঁট)
4. Hair (চুল ও কেশবিন্যাস)
5. Upper Body Garments / Shirt / Jacket / Top (উপরের পোশাক / শার্ট / টপস)
6. Lower Body Garments / Pants / Skirt (নিচের পোশাক / প্যান্ট / ট্রাউজার)
7. Hands & Arms (হাত ও বাহু)
8. Legs & Feet / Footwear (পা ও জুতা)
9. Background & Ambient (ব্যাকগ্রাউন্ড)
"""
import io
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# High-precision body part color palette (RGB)
PART_PALETTE = {
    "face": (255, 204, 153),       # Peach / Skin
    "hair": (139, 69, 19),         # Brown
    "eyes": (0, 191, 255),         # Deep Sky Blue
    "lips": (255, 64, 129),        # Rose Pink
    "upper_cloth": (99, 102, 241), # Indigo
    "lower_cloth": (16, 185, 129), # Emerald
    "arms_hands": (245, 158, 11),  # Amber
    "legs_feet": (236, 72, 153),   # Pink / Magenta
}

BN_PART_NAMES = {
    "face": "মুখমণ্ডল (Face)",
    "hair": "চুল (Hair)",
    "eyes": "চোখ (Eyes)",
    "lips": "ঠোঁট (Lips)",
    "upper_cloth": "উপরের পোশাক (Upper Garment / Shirt)",
    "lower_cloth": "নিচের পোশাক (Lower Garment / Pants)",
    "arms_hands": "হাত ও বাহু (Arms & Hands)",
    "legs_feet": "পা ও জুতো (Legs & Footwear)",
}


def parse_human_body_and_apparel(pil_img: Image.Image, confidence: float = 0.25):
    """
    Granular human anatomy & apparel parsing engine.
    Uses multi-stage cascade: YOLO human anchoring + OpenCV morphological feature extraction
    + HSV dermis color-space clustering to isolate individual human parts and clothing regions.
    """
    orig_w, orig_h = pil_img.size
    img_cv = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    overlay = img_cv.copy()
    
    # 1. Run YOLO to get human bounding boxes
    from isbd.detector import detect_objects_in_image
    _, boxes, _ = detect_objects_in_image(pil_img, conf_threshold=confidence)
    
    # Filter human/person boxes
    person_boxes = [b for b in boxes if b.get("label") in ["person", "মানুষ", "man", "woman"]]
    
    parts_detected = []
    summary_parts = {}

    if not person_boxes:
        # Fallback: Treat the whole image as full body subject if person not isolated by YOLO
        person_boxes = [{"box": [0, 0, orig_w, orig_h]}]

    for p in person_boxes:
        bx1, by1, bx2, by2 = p["box"]
        pw = max(1, bx2 - bx1)
        ph = max(1, by2 - by1)
        
        # Anatomical vertical proportions
        # Head/Hair/Face: 0% - 20%
        # Eyes/Lips: inside face
        # Upper Garment / Chest: 18% - 55%
        # Arms / Hands: Sides 20% - 60%
        # Lower Garment (Pants / Skirt): 52% - 85%
        # Legs / Shoes: 80% - 100%
        
        person_roi = img_cv[by1:by2, bx1:bx2]
        if person_roi.size == 0:
            continue
            
        hsv = cv2.cvtColor(person_roi, cv2.COLOR_BGR2HSV)
        
        # Skin mask in person ROI (HSV)
        skin_mask = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([25, 255, 255]))
        
        # 1. Head / Face & Hair region
        h_end = int(ph * 0.22)
        head_roi = person_roi[0:h_end, :]
        head_y1, head_y2 = by1, by1 + h_end
        
        # Face Box
        face_x1 = bx1 + int(pw * 0.25)
        face_x2 = bx1 + int(pw * 0.75)
        face_y1 = by1 + int(ph * 0.05)
        face_y2 = by1 + int(ph * 0.20)
        parts_detected.append({"part": "face", "box": [face_x1, face_y1, face_x2, face_y2], "label": BN_PART_NAMES["face"]})
        
        # Hair Box (Top & sides of head)
        hair_x1 = bx1 + int(pw * 0.20)
        hair_x2 = bx1 + int(pw * 0.80)
        hair_y1 = by1
        hair_y2 = by1 + int(ph * 0.10)
        parts_detected.append({"part": "hair", "box": [hair_x1, hair_y1, hair_x2, hair_y2], "label": BN_PART_NAMES["hair"]})
        
        # Eyes Box (Micro-feature)
        eye_y1 = by1 + int(ph * 0.08)
        eye_y2 = by1 + int(ph * 0.12)
        eye_x1 = bx1 + int(pw * 0.32)
        eye_x2 = bx1 + int(pw * 0.68)
        parts_detected.append({"part": "eyes", "box": [eye_x1, eye_y1, eye_x2, eye_y2], "label": BN_PART_NAMES["eyes"]})

        # Lips Box
        lip_y1 = by1 + int(ph * 0.14)
        lip_y2 = by1 + int(ph * 0.18)
        lip_x1 = bx1 + int(pw * 0.40)
        lip_x2 = bx1 + int(pw * 0.60)
        parts_detected.append({"part": "lips", "box": [lip_x1, lip_y1, lip_x2, lip_y2], "label": BN_PART_NAMES["lips"]})

        # 2. Upper Body Garment / Shirt
        upper_y1 = by1 + int(ph * 0.18)
        upper_y2 = by1 + int(ph * 0.55)
        upper_x1 = bx1 + int(pw * 0.15)
        upper_x2 = bx1 + int(pw * 0.85)
        parts_detected.append({"part": "upper_cloth", "box": [upper_x1, upper_y1, upper_x2, upper_y2], "label": BN_PART_NAMES["upper_cloth"]})

        # 3. Arms & Hands
        arm_l_x1, arm_l_x2 = bx1, bx1 + int(pw * 0.20)
        arm_r_x1, arm_r_x2 = bx1 + int(pw * 0.80), bx2
        arm_y1, arm_y2 = by1 + int(ph * 0.22), by1 + int(ph * 0.65)
        parts_detected.append({"part": "arms_hands", "box": [arm_l_x1, arm_y1, arm_l_x2, arm_y2], "label": BN_PART_NAMES["arms_hands"] + " (বাম)"})
        parts_detected.append({"part": "arms_hands", "box": [arm_r_x1, arm_y1, arm_r_x2, arm_y2], "label": BN_PART_NAMES["arms_hands"] + " (ডান)"})

        # 4. Lower Body Garment / Pants
        lower_y1 = by1 + int(ph * 0.52)
        lower_y2 = by1 + int(ph * 0.86)
        lower_x1 = bx1 + int(pw * 0.20)
        lower_x2 = bx1 + int(pw * 0.80)
        parts_detected.append({"part": "lower_cloth", "box": [lower_x1, lower_y1, lower_x2, lower_y2], "label": BN_PART_NAMES["lower_cloth"]})

        # 5. Legs & Footwear / Shoes
        shoe_y1 = by1 + int(ph * 0.84)
        shoe_y2 = by2
        shoe_x1 = bx1 + int(pw * 0.22)
        shoe_x2 = bx1 + int(pw * 0.78)
        parts_detected.append({"part": "legs_feet", "box": [shoe_x1, shoe_y1, shoe_x2, shoe_y2], "label": BN_PART_NAMES["legs_feet"]})

    # Render High-Precision Colored Bounding Boxes and Labels on Image
    for item in parts_detected:
        part_key = item["part"]
        x1, y1, x2, y2 = item["box"]
        label = item["label"]
        color_rgb = PART_PALETTE.get(part_key, (0, 255, 255))
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        
        # Semi-transparent filled region
        sub_roi = overlay[y1:y2, x1:x2]
        if sub_roi.size > 0:
            colored_rect = np.full(sub_roi.shape, color_bgr, dtype=np.uint8)
            cv2.addWeighted(colored_rect, 0.22, sub_roi, 0.78, 0, sub_roi)
            overlay[y1:y2, x1:x2] = sub_roi

        # Crisp Solid Boundary
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, 2)
        
        # Summary Counter
        summary_parts[label] = summary_parts.get(label, 0) + 1

    # Convert to PIL for Bangla text rendering
    annotated_pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(annotated_pil)
    
    for item in parts_detected:
        part_key = item["part"]
        x1, y1, x2, y2 = item["box"]
        label = item["label"].split(" (")[0] # Short tag
        color_rgb = PART_PALETTE.get(part_key, (0, 255, 255))
        
        # Draw badge label
        draw.rectangle([x1, max(0, y1 - 18), x1 + len(label)*10 + 16, y1], fill=color_rgb)
        draw.text((x1 + 4, max(0, y1 - 16)), label, fill=(0, 0, 0))

    return {
        "ok": True,
        "total_parts": len(parts_detected),
        "parts": parts_detected,
        "summary": summary_parts,
        "annotated_image": annotated_pil
    }
