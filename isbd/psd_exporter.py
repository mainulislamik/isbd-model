"""
ISBD v1.00 — Adobe Photoshop Multi-Layer Non-Destructive PSD Exporter
Generates genuine industry-grade .PSD documents containing distinct layers:
- Layer 0: Original Photo (Base locked)
- Layer 1: Frequency Separation - Low Frequency (Colors & Smooth Tones)
- Layer 2: Frequency Separation - High Frequency (Micro-Texture & Pores, Linear Light)
- Layer 3: Retouched Dermis & Tone (Enhanced studio finish)
- Layer 4: Cast Drop Shadow (Multiply blend mode)
- Layer 5: Cutout Subject with Sub-pixel Alpha Transparency
- Layer 6: High-Pass Unsharp Detail (Overlay blend mode)
"""

import io
import cv2
import numpy as np
from typing import Optional, Dict, Any


def create_commercial_psd(
    original_bgr: np.ndarray,
    cutout_rgba: Optional[np.ndarray] = None,
    retouched_bgr: Optional[np.ndarray] = None,
    shadow_rgba: Optional[np.ndarray] = None,
    include_frequency_separation: bool = True,
    include_high_pass: bool = True
) -> Dict[str, Any]:
    """
    Assembles a professional multi-layer Photoshop (.psd) file.
    Supports Hollywood/Commercial Agency Frequency Separation & Layered Non-Destructive editing.
    """
    try:
        import pytoshop
        from pytoshop import layers, enums

        h, w = original_bgr.shape[:2]
        psd = pytoshop.core.PsdFile(num_channels=3, height=h, width=w)

        layer_records = []

        # 1. Background Layer (Original Photo - Base)
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

        # 2 & 3. Frequency Separation Layers (High-End Retouching Standard)
        if include_frequency_separation:
            # Low Frequency: Gaussian blur to capture lighting, tones, skin gradients
            ksize = max(5, int((min(h, w) // 100) * 2 + 1))
            low_rgb = cv2.GaussianBlur(orig_rgb, (ksize, ksize), 0)

            # High Frequency: Micro-texture, fine pores, wrinkles with 128 neutral gray offset
            orig_f = orig_rgb.astype(np.float32)
            low_f = low_rgb.astype(np.float32)
            high_f = np.clip((orig_f - low_f) + 128.0, 0, 255).astype(np.uint8)

            low_layer = layers.LayerRecord(
                channels={
                    0: layers.ChannelImageData(image=low_rgb[:, :, 0]),
                    1: layers.ChannelImageData(image=low_rgb[:, :, 1]),
                    2: layers.ChannelImageData(image=low_rgb[:, :, 2]),
                },
                top=0, left=0, bottom=h, right=w,
                blend_mode=enums.BlendMode.normal,
                name="FS: Low Frequency (Tones & Colors)"
            )
            layer_records.append(low_layer)

            high_layer = layers.LayerRecord(
                channels={
                    0: layers.ChannelImageData(image=high_f[:, :, 0]),
                    1: layers.ChannelImageData(image=high_f[:, :, 1]),
                    2: layers.ChannelImageData(image=high_f[:, :, 2]),
                },
                top=0, left=0, bottom=h, right=w,
                blend_mode=enums.BlendMode.linear_light,
                name="FS: High Frequency (Skin Pores & Texture)"
            )
            layer_records.append(high_layer)

        # 4. Retouched Tone & Dermis Layer (if provided)
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

        # 5. Drop Shadow / Ambient Occlusion (if provided, Multiply blend)
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

        # 6. High-Pass Unsharp Detail Layer (Overlay blend mode)
        if include_high_pass:
            # High-pass filter: Image - Blur + 128
            blur_hp = cv2.GaussianBlur(orig_rgb, (3, 3), 0).astype(np.float32)
            hp_f = np.clip((orig_rgb.astype(np.float32) - blur_hp) * 1.5 + 128.0, 0, 255).astype(np.uint8)
            hp_layer = layers.LayerRecord(
                channels={
                    0: layers.ChannelImageData(image=hp_f[:, :, 0]),
                    1: layers.ChannelImageData(image=hp_f[:, :, 1]),
                    2: layers.ChannelImageData(image=hp_f[:, :, 2]),
                },
                top=0, left=0, bottom=h, right=w,
                blend_mode=enums.BlendMode.overlay,
                name="High-Pass Micro Sharpening"
            )
            layer_records.append(hp_layer)

        # 7. Cutout Subject with Sub-pixel Alpha Transparency
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
                name="Cutout Subject (Clipping Mask)"
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
            "layer_names": [l.name for l in layer_records],
            "dimensions": f"{w}x{h}"
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "psd_bytes": b"",
            "layers_count": 0,
            "layer_names": []
        }
