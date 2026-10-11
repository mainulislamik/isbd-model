"""
ISBD v1.00 — Adobe Photoshop Multi-Layer Non-Destructive PSD Exporter
Generates genuine .PSD documents containing distinct layers:
- Layer 0: Original Photo (Locked background)
- Layer 1: Enhanced / Retouched Dermis Tone
- Layer 2: Cutout Subject with Sub-pixel Alpha Transparency
- Layer 3: Drop Shadow / Ambient Occlusion
"""

import io
import cv2
import numpy as np
from typing import Optional, Dict, Any


def create_commercial_psd(
    original_bgr: np.ndarray,
    cutout_rgba: Optional[np.ndarray] = None,
    retouched_bgr: Optional[np.ndarray] = None,
    shadow_rgba: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Assembles a professional multi-layer Photoshop (.psd) file.
    """
    try:
        import pytoshop
        from pytoshop import layers, enums

        h, w = original_bgr.shape[:2]
        psd = pytoshop.core.PsdFile(num_channels=3, height=h, width=w)

        layer_records = []

        # 1. Background Layer (Original Photo)
        orig_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        bg_layer = layers.LayerRecord(
            channels={
                0: layers.ChannelImageData(image=orig_rgb[:, :, 0]),
                1: layers.ChannelImageData(image=orig_rgb[:, :, 1]),
                2: layers.ChannelImageData(image=orig_rgb[:, :, 2]),
            },
            top=0, left=0, bottom=h, right=w,
            blend_mode=enums.BlendMode.normal,
            name="Original Photo (Base)"
        )
        layer_records.append(bg_layer)

        # 2. Retouched Tone & Texture Layer (if provided)
        if retouched_bgr is not None:
            ret_rgb = cv2.cvtColor(retouched_bgr, cv2.COLOR_BGR2RGB)
            ret_layer = layers.LayerRecord(
                channels={
                    0: layers.ChannelImageData(image=ret_rgb[:, :, 0]),
                    1: layers.ChannelImageData(image=ret_rgb[:, :, 1]),
                    2: layers.ChannelImageData(image=ret_rgb[:, :, 2]),
                },
                top=0, left=0, bottom=h, right=w,
                blend_mode=enums.BlendMode.normal,
                name="Retouched Dermis & Tone"
            )
            layer_records.append(ret_layer)

        # 3. Drop Shadow Layer (if provided)
        if shadow_rgba is not None:
            sh_rgb = cv2.cvtColor(shadow_rgba[:, :, :3], cv2.COLOR_BGR2RGB)
            sh_alpha = shadow_rgba[:, :, 3]
            sh_layer = layers.LayerRecord(
                channels={
                    -1: layers.ChannelImageData(image=sh_alpha),
                    0: layers.ChannelImageData(image=sh_rgb[:, :, 0]),
                    1: layers.ChannelImageData(image=sh_rgb[:, :, 1]),
                    2: layers.ChannelImageData(image=sh_rgb[:, :, 2]),
                },
                top=0, left=0, bottom=h, right=w,
                blend_mode=enums.BlendMode.multiply,
                name="Cast Drop Shadow"
            )
            layer_records.append(sh_layer)

        # 4. Cutout Subject with Alpha Transparency (if provided)
        if cutout_rgba is not None:
            cut_rgb = cv2.cvtColor(cutout_rgba[:, :, :3], cv2.COLOR_BGR2RGB)
            cut_alpha = cutout_rgba[:, :, 3]
            cut_layer = layers.LayerRecord(
                channels={
                    -1: layers.ChannelImageData(image=cut_alpha),
                    0: layers.ChannelImageData(image=cut_rgb[:, :, 0]),
                    1: layers.ChannelImageData(image=cut_rgb[:, :, 1]),
                    2: layers.ChannelImageData(image=cut_rgb[:, :, 2]),
                },
                top=0, left=0, bottom=h, right=w,
                blend_mode=enums.BlendMode.normal,
                name="Cutout Subject (Clipping)"
            )
            layer_records.append(cut_layer)

        psd.layer_and_mask_info.layer_info.layer_records.extend(layer_records)

        buf = io.BytesIO()
        psd.write(buf)
        psd_bytes = buf.getvalue()

        return {
            "ok": True,
            "psd_bytes": psd_bytes,
            "byte_size": len(psd_bytes),
            "layers_count": len(layer_records),
            "dimensions": f"{w}x{h}"
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "psd_bytes": b"",
            "layers_count": 0
        }
