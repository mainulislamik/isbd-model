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
DATA = ROOT / "data"
GOLDEN_POOL = DATA / "golden_pool.npz"

_DEDUP_CACHE = []  # Stores (dhash_str, timestamp)
_DEDUP_FILTERED_COUNT = 0

CANONICAL_DOMAINS = {
    "retouch": "skin_retouching",
    "skin_retouching": "skin_retouching",
    "matting": "hair_matting",
    "hair_matting": "hair_matting",
    "isolate": "hair_matting",
    "ecommerce": "ecommerce_product",
    "ecommerce_product": "ecommerce_product",
    "lighting": "lighting_hdr",
    "lighting_hdr": "lighting_hdr",
    "color_grade": "lighting_hdr",
    "inpaint": "object_inpainting",
    "object_inpainting": "object_inpainting",
}


def _compute_dhash(img_bgr: np.ndarray) -> str:
    """Compute 64-bit difference perceptual hash (dHash) on downsampled grayscale image."""
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
        diff = resized[:, 1:] > resized[:, :-1]
        bits = 0
        for val in diff.flatten():
            bits = (bits << 1) | int(val)
        return f"{bits:016x}"
    except Exception:
        return "0" * 16


def _hamming_distance(h1: str, h2: str) -> int:
    try:
        return bin(int(h1, 16) ^ int(h2, 16)).count("1")
    except Exception:
        return 64


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
    proposal_bytes: Optional[bytes] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Validate and save a pair of Before (original) and After (designer edited) images.
    If proposal_bytes (AI proposal) is provided, calculates human correction delta and
    triggers immediate Human Residual Adapter training.
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
    # Check difference (allow localized blemish edits as well as full transforms)
    if (h_b, w_b) == (h_a, w_a):
        abs_diff = np.abs(img_before.astype(np.float32) - img_after.astype(np.float32))
        max_diff = float(np.max(abs_diff))
        sum_diff = float(np.sum(abs_diff))
        if max_diff < 4.0 or sum_diff < 30.0:
            return {"ok": False, "error": "ছবিতে কোনো দৃশ্যমান পরিবর্তন লক্ষ্য করা যায়নি (Before ও After অভিন্ন)"}
    else:
        # Resize after to match before for consistent pairing if designer cropped/resized
        img_after = cv2.resize(img_after, (w_b, h_b), interpolation=cv2.INTER_AREA)

    # Perceptual Deduplication check (Pillar 5)
    global _DEDUP_CACHE, _DEDUP_FILTERED_COUNT
    now_ts = time.time()
    cur_hash = _compute_dhash(img_after)
    # Evict cache older than 15 minutes (900s)
    _DEDUP_CACHE = [(h, ts) for h, ts in _DEDUP_CACHE if (now_ts - ts) < 900]
    for h, ts in _DEDUP_CACHE:
        if _hamming_distance(cur_hash, h) <= 2:
            _DEDUP_FILTERED_COUNT += 1
            return {
                "ok": True,
                "duplicate": True,
                "note": "Perceptual duplicate filtered to conserve GPU memory and prevent overfitting",
                "sample_id": f"dedup_{int(now_ts)}",
                "total_samples": get_manifest().get("total_samples", 0),
                "details": {"filtered": True, "reason": "dhash_match"}
            }
    _DEDUP_CACHE.append((cur_hash, now_ts))

    sample_id = f"sample_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    before_filename = f"{sample_id}_before.png"
    after_filename = f"{sample_id}_after.png"

    before_path = IMAGES_DIR / before_filename
    after_path = IMAGES_DIR / after_filename

    # Save lossless PNGs
    cv2.imwrite(str(before_path), img_before)
    cv2.imwrite(str(after_path), img_after)

    meta = metadata or {}
    raw_domain = meta.get("domain") or meta.get("task_domain", "skin_retouching")
    task_domain = CANONICAL_DOMAINS.get(str(raw_domain).lower(), "skin_retouching")
    designer_id = meta.get("designer_id") or meta.get("editor_id", "designer_pc")
    tools_used = meta.get("tools_used", ["retouch", "curves"])
    stroke_count = int(meta.get("stroke_count", 1))
    undos_filtered = int(meta.get("undos_filtered", 0))
    roi_cropped = bool(meta.get("roi_cropped", False))
    layer_info = meta.get("layer_info", {})
    is_correction = bool(proposal_bytes is not None)

    proposal_filename = None
    adapter_train_info = None

    if proposal_bytes:
        np_prop = np.frombuffer(proposal_bytes, np.uint8)
        img_prop = cv2.imdecode(np_prop, cv2.IMREAD_COLOR)
        if img_prop is not None:
            if img_prop.shape[:2] != (h_b, w_b):
                img_prop = cv2.resize(img_prop, (w_b, h_b), interpolation=cv2.INTER_AREA)
            proposal_filename = f"{sample_id}_proposal.png"
            proposal_path = IMAGES_DIR / proposal_filename
            cv2.imwrite(str(proposal_path), img_prop)

            # Trigger immediate fast adapter learning from human correction
            try:
                from PIL import Image
                from isbd.opensource_ai import train_adapter_on_human_pair
                p_orig = Image.fromarray(cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB))
                p_human = Image.fromarray(cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB))
                p_prop = Image.fromarray(cv2.cvtColor(img_prop, cv2.COLOR_BGR2RGB))
                adapter_train_info = train_adapter_on_human_pair(
                    p_orig, p_human, p_prop, domain=task_domain, steps=15
                )
            except Exception as e:
                adapter_train_info = {"ok": False, "error": str(e)}
    else:
        # Live human edit (without prior AI proposal): train domain adapter directly
        try:
            from PIL import Image
            from isbd.opensource_ai import train_adapter_on_human_pair
            p_orig = Image.fromarray(cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB))
            p_human = Image.fromarray(cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB))
            adapter_train_info = train_adapter_on_human_pair(
                p_orig, p_human, p_orig, domain=task_domain, steps=10
            )
        except Exception as e:
            adapter_train_info = {"ok": False, "error": str(e)}

    # Auto-append to self_pool.npz for continuous 24/7 UNet backbone learning
    try:
        from PIL import Image
        from isbd.self_learner import _load_npz, _save_npz, _img_to_tensor, SELF_POOL, IMG_BASE
        p_b = Image.fromarray(cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB))
        p_a = Image.fromarray(cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB))
        x_t = _img_to_tensor(p_b, size=IMG_BASE)
        y_t = _img_to_tensor(p_a, size=IMG_BASE)
        cur_X, cur_Y = _load_npz(SELF_POOL, img_size=IMG_BASE)
        cur_X = np.concatenate([cur_X, x_t[None]], axis=0)
        cur_Y = np.concatenate([cur_Y, y_t[None]], axis=0)
        _save_npz(SELF_POOL, cur_X, cur_Y, cap=5000)

        # Golden Ground-Truth validation (Pillar 2: Zero Undos & High Dwell)
        is_golden = bool(meta.get("is_golden", False)) or (float(meta.get("dwell_sec", 0)) >= 30.0 and undos_filtered == 0)
        dwell_sec = float(meta.get("dwell_sec", 0.0))
        correction_type = str(meta.get("correction_type", "direct_refine"))

        if is_golden:
            cur_gX, cur_gY = _load_npz(GOLDEN_POOL, img_size=IMG_BASE)
            cur_gX = np.concatenate([cur_gX, x_t[None]], axis=0)
            cur_gY = np.concatenate([cur_gY, y_t[None]], axis=0)
            _save_npz(GOLDEN_POOL, cur_gX, cur_gY, cap=3000)
    except Exception as pool_err:
        print(f"[Harvester] pool append error: {pool_err}")

    is_golden = bool(meta.get("is_golden", False)) or (float(meta.get("dwell_sec", 0)) >= 30.0 and undos_filtered == 0)
    dwell_sec = float(meta.get("dwell_sec", 0.0))
    correction_type = str(meta.get("correction_type", "direct_refine"))

    # Dynamic validation score based on designer certitude (Undo penalty)
    if is_golden:
        validation_score = 1.0
    elif undos_filtered > 2:
        validation_score = 0.50  # Hard negative / refined after multiple mistakes
    else:
        validation_score = 0.75

    sample_entry = {
        "id": sample_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "width": w_b,
        "height": h_b,
        "designer_id": designer_id,
        "doc_name": meta.get("doc_name", "untitled.psd"),
        "tools_used": tools_used,
        "task_domain": task_domain,
        "stroke_count": stroke_count,
        "undos_filtered": undos_filtered,
        "roi_cropped": roi_cropped,
        "layer_info": layer_info,
        "is_correction": is_correction,
        "is_golden": is_golden,
        "dwell_sec": dwell_sec,
        "correction_type": correction_type,
        "validation_score": 1.0 if is_golden else 0.75,
        "duration_sec": meta.get("duration_sec", 0),
        "before_file": before_filename,
        "after_file": after_filename,
        "proposal_file": proposal_filename,
        "adapter_train_info": adapter_train_info,
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
    """Return summary statistics of collected designer training data across all editors."""
    manifest = get_manifest()
    samples = manifest.get("samples", [])
    total_size_mb = 0.0
    by_task: Dict[str, int] = {
        "skin_retouching": 0,
        "hair_matting": 0,
        "ecommerce_product": 0,
        "lighting_hdr": 0,
        "object_inpainting": 0,
    }
    by_editor: Dict[str, Dict[str, int]] = {}
    corrections_count = 0
    total_strokes = 0
    total_undos_filtered = 0
    roi_crops = 0

    for s in samples:
        total_size_mb += s.get("file_size_kb", 0.0) / 1024.0
        raw_t = s.get("task_domain", "skin_retouching")
        t = CANONICAL_DOMAINS.get(str(raw_t).lower(), "skin_retouching")
        by_task[t] = by_task.get(t, 0) + 1
        
        ed = s.get("designer_id", "designer_pc")
        if ed not in by_editor:
            by_editor[ed] = {"samples": 0, "strokes": 0}
        by_editor[ed]["samples"] += 1
        by_editor[ed]["strokes"] += s.get("stroke_count", 1)

        total_strokes += s.get("stroke_count", 1)
        total_undos_filtered += s.get("undos_filtered", 0)
        if s.get("roi_cropped"):
            roi_crops += 1

        if s.get("is_correction"):
            corrections_count += 1

    today_str = time.strftime("%Y-%m-%d")
    today_count = sum(1 for s in samples if s.get("timestamp", "").startswith(today_str))
    golden_samples = sum(1 for s in samples if s.get("is_golden"))

    return {
        "ok": True,
        "total_samples": len(samples),
        "today_samples": today_count,
        "corrections_count": corrections_count,
        "golden_samples": golden_samples,
        "dedup_filtered_count": _DEDUP_FILTERED_COUNT,
        "by_task": by_task,
        "by_editor": by_editor,
        "total_strokes": total_strokes,
        "total_undos_filtered": total_undos_filtered,
        "roi_crops": roi_crops,
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
