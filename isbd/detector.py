"""
ISBD v1.00 — Visual Object Detection & Recognition Module
Translates COCO categories to standard Bengali names and provides bounding box visualizations.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

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
    Detect objects in PIL Image.
    Returns:
      - annotated_img: PIL Image with modern stylish bounding boxes & labels
      - detections: list of {name, name_bn, conf, box: [x1, y1, x2, y2]}
      - summary: list of distinct objects found with counts
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

        # Draw bounding box with rounded corners / thick border
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Label text
        label = f"{raw_name} {conf*100:.0f}%"
        
        # Badge background
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
