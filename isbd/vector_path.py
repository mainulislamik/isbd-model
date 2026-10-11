"""
ISBD v1.00 — Bézier Vector Clipping Path Engine (SVG / Pen Tool Exporter)
Extracts vector contours from binary/alpha masks and converts them into smooth
cubic Bézier curve control points formatted as standard SVG Work Paths.
"""

import cv2
import numpy as np
from typing import Dict, Any, List


def mask_to_bezier_svg(
    mask_u8: np.ndarray,
    epsilon_ratio: float = 0.003,
    stroke_color: str = "#2563eb",
    stroke_width: int = 2
) -> Dict[str, Any]:
    """
    Traces high-precision boundary contours and generates smooth cubic Bézier SVG path.
    """
    h, w = mask_u8.shape[:2]
    # Threshold mask
    _, binary = cv2.threshold(mask_u8, 127, 255, cv2.THRESH_BINARY)

    # Find external contours with hierarchy
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

    if not contours:
        return {
            "ok": False,
            "svg": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}"></svg>',
            "points_count": 0,
            "paths_count": 0
        }

    svg_paths = []
    total_anchor_points = 0

    for c in contours:
        # Filter tiny noise specs
        if cv2.contourArea(c) < 50:
            continue

        # Douglas-Peucker polygon approximation for clean pen-tool vertices
        epsilon = epsilon_ratio * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        pts = approx.reshape(-1, 2)
        n = len(pts)

        if n < 3:
            continue

        total_anchor_points += n

        # Build smooth cubic Bézier SVG path (C c1_x,c1_y c2_x,c2_y p_x,p_y)
        d_cmd = [f"M {pts[0][0]},{pts[0][1]}"]

        for i in range(n):
            p0 = pts[(i - 1) % n]
            p1 = pts[i]
            p2 = pts[(i + 1) % n]
            p3 = pts[(i + 2) % n]

            # Catmull-Rom to Cubic Bézier conversion
            # c1 = p1 + (p2 - p0) / 6
            # c2 = p2 - (p3 - p1) / 6
            c1_x = p1[0] + (p2[0] - p0[0]) / 6.0
            c1_y = p1[1] + (p2[1] - p0[1]) / 6.0
            c2_x = p2[0] - (p3[0] - p1[0]) / 6.0
            c2_y = p2[1] - (p3[1] - p1[1]) / 6.0

            d_cmd.append(f"C {c1_x:.2f},{c1_y:.2f} {c2_x:.2f},{c2_y:.2f} {p2[0]},{p2[1]}")

        d_cmd.append("Z")
        svg_paths.append(" ".join(d_cmd))

    svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">
  <defs>
    <style>
      .clipping-path {{ fill: none; stroke: {stroke_color}; stroke-width: {stroke_width}; stroke-linecap: round; stroke-linejoin: round; }}
    </style>
  </defs>
"""
    for p in svg_paths:
        svg_content += f'  <path class="clipping-path" d="{p}" />\n'
    svg_content += "</svg>"

    return {
        "ok": True,
        "svg": svg_content,
        "points_count": total_anchor_points,
        "paths_count": len(svg_paths),
        "width": w,
        "height": h
    }
