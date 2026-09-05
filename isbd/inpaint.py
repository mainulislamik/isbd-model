"""
ISBD v1.00 — Inpainting & Object Removal Module
Removes unwanted objects/watermarks/blemishes using AI & Navier-Stokes/Telea inpainting.
"""
from PIL import Image
import numpy as np
import cv2

def inpaint_image(img: Image.Image, mask: Image.Image, radius: int = 5, method: str = "telea") -> Image.Image:
    """
    Inpaint/Remove objects using mask.
    - method: 'telea' (Fast marching) or 'ns' (Navier-Stokes fluid dynamics)
    """
    cv_img = np.array(img.convert("RGB"))
    
    # Ensure mask is single channel grayscale with binary 0/255
    mask_gray = np.array(mask.convert("L").resize((img.width, img.height), Image.Resampling.NEAREST))
    _, mask_bin = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    
    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    inpainted = cv2.inpaint(cv_img, mask_bin, inpaintRadius=max(1, radius), flags=flag)
    
    return Image.fromarray(inpainted)
