"""
ISBD v1.00 — Multi-Engine AI Vision & Computer Vision Toolkit
Engines:
1. YOLOv8 (Deep Learning Object Tracking & Recognition with Bengali translation)
2. OpenCV & Scikit-Image (Edge Detection, Face & Eye detection, Color Palette Analyzer, Adaptive Denoising)
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import skimage.color
import skimage.filters
import skimage.exposure

ROOT = Path(__file__).resolve().parent.parent

# 80 COCO Classes Translated to Bengali
COCO_BN = {
    "person": "মানুষ (Person)", "bicycle": "সাইকেল (Bicycle)", "car": "গাড়ি (Car)",
    "motorcycle": "মোটরসাইকেল (Motorcycle)", "airplane": "উড়োজাহাজ (Airplane)", "bus": "বাস (Bus)",
    "train": "ট্রেন (Train)", "truck": "ট্রাক (Truck)", "boat": "নৌকা (Boat)",
    "traffic light": "ট্রাফিক লাইট (Traffic Light)", "fire hydrant": "ফায়ার হাইড্রেন্ট",
    "stop sign": "স্টপ সাইন (Stop Sign)", "parking meter": "পার্কিং মিটার", "bench": "বেঞ্চ (Bench)",
    "bird": "পাখি (Bird)", "cat": "বিড়াল (Cat)", "dog": "কুকুর (Dog)", "horse": "ঘোড়া (Horse)",
    "sheep": "ভেড়া (Sheep)", "cow": "গরু (Cow)", "elephant": "হাতি (Elephant)", "bear": "ভালুক (Bear)",
    "zebra": "জেব্রা (Zebra)", "giraffe": "জিরাফ (Giraffe)", "backpack": "ব্যাগ / ব্যাকপ্যাক",
    "umbrella": "ছাতা (Umbrella)", "handbag": "হ্যান্ডব্যাগ (Handbag)", "tie": "টাই (Tie)",
    "suitcase": "সুটকেস (Suitcase)", "frisbee": "ফ্রিসবি (Frisbee)", "skis": "স্কি (Skis)",
    "snowboard": "স্নোবোর্ড", "sports ball": "বল (Sports Ball)", "kite": "ঘুড়ি (Kite)",
    "baseball bat": "বেসবল ব্যাট", "baseball glove": "বেসবল গ্লাভস", "skateboard": "স্কেটবোর্ড",
    "surfboard": "সার্ফবোর্ড", "tennis racket": "টেনিস র্যাকেট", "bottle": "বোতল (Bottle)",
    "wine glass": "গ্লাস (Glass)", "cup": "কাপ / মগ (Cup)", "fork": "কাঁটাচামচ (Fork)",
    "knife": "ছুরি (Knife)", "spoon": "চামচ (Spoon)", "bowl": "বাটি (Bowl)",
    "banana": "কলা (Banana)", "apple": "আপেল (Apple)", "sandwich": "স্যান্ডউইচ (Sandwich)",
    "orange": "কমলা (Orange)", "broccoli": "ব্রোকলি (Broccoli)", "carrot": "গাজর (Carrot)",
    "hot dog": "হট ডগ", "pizza": "পিজ্জা (Pizza)", "donut": "ডোনাট (Donut)", "cake": "কেক (Cake)",
    "chair": "চেয়ার (Chair)", "couch": "সোফা (Couch)", "potted plant": "গাছের টব (Plant)",
    "bed": "বিছানা (Bed)", "dining table": "ডাইনিং টেবিল (Table)", "toilet": "টয়লেট (Toilet)",
    "tv": "টিভি / মনিটর (TV)", "laptop": "ল্যাপটপ (Laptop)", "mouse": "মাউস (Mouse)",
    "remote": "রিমোট (Remote)", "keyboard": "কীবোর্ড (Keyboard)", "cell phone": "মোবাইল ফোন (Phone)",
    "microwave": "মাইক্রোওয়েভ", "oven": "ওভেন (Oven)", "toaster": "টোস্টার", "sink": "সিঙ্ক (Sink)",
    "refrigerator": "ফ্রিজ (Fridge)", "book": "বই (Book)", "clock": "ঘড়ি (Clock)", "vase": "ফুলদানি (Vase)",
    "scissors": "কাঁচি (Scissors)", "teddy bear": "টেরি বিয়ার (Teddy Bear)", "hair drier": "হেয়ার ড্রায়ার",
    "toothbrush": "টুথব্রাশ (Toothbrush)"
}

_yolo_model = None

def get_detector():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        import torch
        torch.set_num_threads(4)
        model_path = ROOT / "yolov8n.pt"
        _yolo_model = YOLO(str(model_path))
    return _yolo_model

def detect_objects_in_image(img: Image.Image, conf_threshold: float = 0.25):
    """
    Detect objects in PIL Image using YOLOv8.
    """
    model = get_detector()
    res = model(img, conf=conf_threshold, verbose=False)[0]
    
    boxes = res.boxes
    detections = []
    summary_counts = {}

    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    w, h = img.size

    palette = [
        (99, 102, 241),   # indigo
        (6, 182, 212),    # cyan
        (16, 185, 129),   # green
        (245, 158, 11),   # amber
        (239, 68, 68),    # red
        (236, 72, 153),   # pink
        (168, 85, 247),   # purple
    ]

    for i, b in enumerate(boxes):
        cls_id = int(b.cls[0])
        raw_name = model.names[cls_id]
        name_bn = COCO_BN.get(raw_name, raw_name)
        conf = float(b.conf[0])
        box = [float(x) for x in b.xyxy[0].tolist()]
        x1, y1, x2, y2 = box

        color = palette[cls_id % len(palette)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        label = f"{raw_name} {conf*100:.0f}%"
        pad = 4
        text_bbox = draw.textbbox((x1, y1), label)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
        
        badge_y1 = max(0, y1 - th - pad * 2)
        badge_y2 = badge_y1 + th + pad * 2
        badge_x2 = min(w, x1 + tw + pad * 2)
        
        draw.rectangle([x1, badge_y1, badge_x2, badge_y2], fill=color)
        draw.text((x1 + pad, badge_y1 + pad), label, fill=(255, 255, 255))

        detections.append({
            "name": raw_name,
            "name_bn": name_bn,
            "conf": round(conf, 3),
            "box": [round(v, 1) for v in box]
        })
        summary_counts[name_bn] = summary_counts.get(name_bn, 0) + 1

    return annotated, detections, summary_counts


def apply_cv_filter(img: Image.Image, filter_type: str):
    """
    Applies professional Computer Vision algorithms (OpenCV & Scikit-Image):
    - 'canny': Canny Edge & Boundary Detection
    - 'clahe': Adaptive Histogram Contrast Equalization (CLAHE)
    - 'sketch': Pencil Sketch Conversion
    - 'denoise': Fast Non-Local Means Image Denoising
    - 'palette': Dominant Color Palette Extraction
    """
    cv_img = np.array(img.convert("RGB"))
    
    if filter_type == "canny":
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(edges_rgb), "Canny Edge Detection (বর্ডার ও লাইন ট্রেসিং)"

    elif filter_type == "clahe":
        # Scikit-image / OpenCV CLAHE
        lab = cv2.cvtColor(cv_img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)
        return Image.fromarray(enhanced), "CLAHE Adaptive Contrast (উচ্চমানের কনট্রাস্ট ও শ্যাডো বুস্ট)"

    elif filter_type == "sketch":
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        sketch_rgb = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(sketch_rgb), "AI Pencil Sketch (আর্ট স্কেচ ফিল্টার)"

    elif filter_type == "denoise":
        denoised = cv2.fastNlMeansDenoisingColored(cv_img, None, 10, 10, 7, 21)
        return Image.fromarray(denoised), "Fast NLM Denoising (অ্যাডভান্সড নয়েজ রিমুভাল)"

    return img, "Original"


def analyze_image_colors(img: Image.Image, num_colors: int = 5):
    """Extracts dominant color palette using K-Means clustering (OpenCV)."""
    cv_img = np.array(img.resize((150, 150)).convert("RGB"))
    pixels = cv_img.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    palette = []
    counts = np.bincount(labels.flatten())
    total_pts = len(labels)

    for i in np.argsort(-counts):
        c = centers[i].astype(int)
        hex_code = f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"
        pct = float(round(float((counts[i] / total_pts) * 100), 1))
        palette.append({"hex": hex_code, "rgb": [int(c[0]), int(c[1]), int(c[2])], "percent": pct})

    return palette
