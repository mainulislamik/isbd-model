"""
ISBD v1.00 — Inference
Restores a degraded image with the trained checkpoint. Saves before/after.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from isbd.model import TinyUNet
from isbd.data import IMG_SIZE, make_pair

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"


def restore(img_path: str, out_path: str = "", ckpt: str = ""):
    ckpt = ckpt or str(CKPT / "best.pt")
    model = TinyUNet()
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()

    img = Image.open(img_path).convert("RGB")
    orig_size = img.size
    small = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    x = torch.from_numpy(np.asarray(small, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]

    with torch.no_grad():
        out = model(x)[0].clamp(0, 1).numpy().transpose(1, 2, 0)
    restored = Image.fromarray((out * 255).astype(np.uint8)).resize(orig_size, Image.Resampling.LANCZOS)

    # side-by-side comparison
    w, h = orig_size
    cmp = Image.new("RGB", (w * 2 + 8, h), (20, 20, 20))
    cmp.paste(img, (0, 0))
    cmp.paste(restored, (w + 8, 0))

    out_path = out_path or str(ROOT / "samples" / "compare.png")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cmp.save(out_path)
    restored_only = Image.new("RGB", orig_size)
    restored_only = restored
    restored.save(str(Path(out_path).with_name(Path(out_path).stem + "_restored.png")))
    print(f"saved: {out_path} (+ restored)")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) > 1:
        restore(sys.argv[1])
    else:
        # self-test on synthetic degraded sample
        import random
        x, y = make_pair(999)
        dirty = Image.fromarray((x.transpose(1, 2, 0) * 255).astype(np.uint8))
        p = ROOT / "samples" / "selftest_input.png"
        p.parent.mkdir(exist_ok=True)
        dirty.resize((256, 256), Image.Resampling.NEAREST).save(p)
        restore(str(p))
