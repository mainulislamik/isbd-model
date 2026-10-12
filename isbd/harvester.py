"""
ISBD v1.00 — Designer Live Action Harvester & Dataset Manager
Collects, validates, and manages real-world human editing pairs (Before -> After)
captured silently from Adobe Photoshop designers.
"""

import os
import json
import uuid
import time
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "harvested_samples"
IMAGES_DIR = DATA_DIR / "images"
MANIFEST_PATH = DATA_DIR / "manifest.json"


def ensure_dirs():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    if not MANIFEST_PATH.exists():
        MANIFEST_PATH.write_text(json.dumps({
            "version": "1.0.0",
            "total_samples": 0,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "samples": []
        }, indent=2, ensure_ascii=False))


def get_manifest() -> Dict[str, Any]:
    ensure_dirs()
    try:
        return json.loads(MANIFEST_PATH.read_text())
    except Exception:
        return {"version": "1.0.0", "total_samples": 0, "samples": []}


def save_manifest(m: Dict[str, Any]):
    ensure_dirs()
    MANIFEST_PATH.write_text(json.dumps(m, indent=2, ensure_ascii=False))


def save_designer_sample(
    before_bytes: bytes,
    after_bytes: bytes,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Validate and save a pair of Before (original) and After (designer edited) images.
    Returns details of saved sample or error.
    """
    ensure_dirs()

    if not before_bytes or not after_bytes:
        return {"ok": False, "error": "Before বা After ইমেজ বাইটস পাওয়া যায়নি"}

    # Decode images to verify integrity
    np_before = np.frombuffer(before_bytes, np.uint8)
    np_after = np.frombuffer(after_bytes, np.uint8)

    img_before = cv2.imdecode(np_before, cv2.IMREAD_COLOR)
    img_after = cv2.imdecode(np_after, cv2.IMREAD_COLOR)

    if img_before is None or img_after is None:
        return {"ok": False, "error": "ইমেজ ফরম্যাট অকার্যকর বা ডিকোড করা যায়নি"}

    h_b, w_b = img_before.shape[:2]
    h_a, w_a = img_after.shape[:2]

    # Ensure reasonable size
    if min(h_b, w_b) < 64 or min(h_a, w_a) < 64:
        return {"ok": False, "error": "ছবির রেজোলিউশন অত্যন্ত কম (কমপক্ষে ৬৪ পিক্সেল আবশ্যক)"}

    # Verify that designer actually edited the image (not identical)
    # Check MSE difference
    if (h_b, w_b) == (h_a, w_a):
        diff = np.mean(np.abs(img_before.astype(np.float32) - img_after.astype(np.float32)))
        if diff < 0.5:
            return {"ok": False, "error": "ছবিতে কোনো পরিবর্তন লক্ষ্য করা যায়নি (Before ও After অভিন্ন)"}
    else:
        # Resize after to match before for consistent pairing if designer cropped/resized
        img_after = cv2.resize(img_after, (w_b, h_b), interpolation=cv2.INTER_AREA)

    sample_id = f"sample_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    before_filename = f"{sample_id}_before.png"
    after_filename = f"{sample_id}_after.png"

    before_path = IMAGES_DIR / before_filename
    after_path = IMAGES_DIR / after_filename

    # Save lossless PNGs
    cv2.imwrite(str(before_path), img_before)
    cv2.imwrite(str(after_path), img_after)

    meta = metadata or {}
    sample_entry = {
        "id": sample_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "width": w_b,
        "height": h_b,
        "designer_id": meta.get("designer_id", "designer_pc"),
        "doc_name": meta.get("doc_name", "untitled.psd"),
        "tools_used": meta.get("tools_used", ["retouch", "curves"]),
        "duration_sec": meta.get("duration_sec", 0),
        "before_file": before_filename,
        "after_file": after_filename,
        "file_size_kb": round((os.path.getsize(before_path) + os.path.getsize(after_path)) / 1024, 1)
    }

    manifest = get_manifest()
    manifest.setdefault("samples", []).append(sample_entry)
    manifest["total_samples"] = len(manifest["samples"])
    manifest["last_updated"] = sample_entry["timestamp"]
    save_manifest(manifest)

    return {
        "ok": True,
        "sample_id": sample_id,
        "total_samples": manifest["total_samples"],
        "details": sample_entry
    }


def get_harvester_stats() -> Dict[str, Any]:
    """Return summary statistics of collected designer training data."""
    manifest = get_manifest()
    samples = manifest.get("samples", [])
    total_size_mb = 0.0
    for s in samples:
        total_size_mb += s.get("file_size_kb", 0.0) / 1024.0

    today_str = time.strftime("%Y-%m-%d")
    today_count = sum(1 for s in samples if s.get("timestamp", "").startswith(today_str))

    return {
        "ok": True,
        "total_samples": len(samples),
        "today_samples": today_count,
        "total_size_mb": round(total_size_mb, 2),
        "last_updated": manifest.get("last_updated", "—"),
        "recent_samples": samples[-10:][::-1]
    }


def export_harvested_dataset_zip(zip_path: Path) -> Path:
    """Create a compressed zip file of all harvested designer samples for remote trainers."""
    ensure_dirs()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        if MANIFEST_PATH.exists():
            zf.write(MANIFEST_PATH, "manifest.json")
        for f in IMAGES_DIR.glob("*.png"):
            zf.write(f, f"images/{f.name}")
    return zip_path
