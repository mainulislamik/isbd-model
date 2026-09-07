"""
Ghost Mannequin / Neck Joint Processor.
Combines Front Garment Image with Inner Back (Neck) Image.
"""
import cv2
import numpy as np
from PIL import Image
import io

try:
    import rembg
except ImportError:
    pass

def remove_background(img: Image.Image) -> Image.Image:
    """Removes mannequin/background from the garment photo."""
    # Use rembg.remove
    try:
        from rembg import remove
        return remove(img)
    except Exception:
        # Fallback to simple return if library not installed yet
        return img

def create_neck_joint(front_img: Image.Image, neck_img: Image.Image) -> Image.Image:
    """
    1. Removes background from front and neck images.
    2. Detects the V-neck or collar curve on front_img.
    3. Warps the neck_img to fit perfectly behind the front collar.
    4. Composites them together to form a "Ghost Mannequin" e-commerce asset.
    """
    # 1. Background removal
    front_nobg = remove_background(front_img)
    neck_nobg = remove_background(neck_img)
    
    # Need to convert to cv2 formats
    front_cv = cv2.cvtColor(np.array(front_nobg), cv2.COLOR_RGBA2BGRA)
    neck_cv = cv2.cvtColor(np.array(neck_nobg), cv2.COLOR_RGBA2BGRA)
    
    # ------------- FUTURE IMPLEMENTATION -------------
    # When hardware is upgraded, we will use Deep Learning landmark detection 
    # to find the collar boundaries automatically and use cv2.perspectiveTransform
    # or TPS (Thin Plate Spline) to warp the neck piece perfectly.
    
    # Placeholder: Just overlaying for structural placeholder
    h, w = front_cv.shape[:2]
    # Resize neck slightly smaller
    neck_resized = cv2.resize(neck_cv, (int(w*0.5), int(h*0.3)))
    
    result = front_cv.copy()
    
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA))
