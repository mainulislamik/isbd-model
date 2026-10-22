"""
ISBD v1.00 — Professional Commercial Studio Sector Engine
Full implementation of the 11 Commercial Graphic Design & Photo Editing Services:
1. Clipping Path (Background cutout & precise alpha extraction)
2. Multiple Clipping Path (Multi-part color isolation & segmentation)
3. Image Masking (Hair/fur feather masking & edge refinement)
4. Neck Joint / Ghost Mannequin (Apparel compositing & seamless stitching)
5. Image Retouching (Blemish healing, frequency separation skin smoothing)
6. Shadow Making (Natural drop shadow, contact shadow creation)
7. Reflection (Mirror ground reflection & glossy surface effect)
8. Color Correction (White balance, exposure curve, color grading)
9. Image Enhancement (HDR detail pop, unsharp sharpness, CLAHE boost)
10. Image Manipulation (Creative compositing, tone blend & stylization)
11. Raster To Vector (Edge tracing, contours, SVG vectorization)
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageOps
import numpy as np
import cv2
import skimage.filters
import skimage.restoration

ROOT = Path(__file__).resolve().parent.parent

SERVICES_CONFIG = [
    {"id": "clipping_path", "title": "Clipping Path", "title_bn": "ক্লিপিং পাথ (ব্যাকগ্রাউন্ড কাটআউট)", "icon": "✂️", "desc": "অবজেক্ট নিখুঁতভাবে ব্যাকগ্রাউন্ড থেকে আলাদা করে স্বচ্ছ করা"},
    {"id": "background_removal", "title": "Magic Background Removal", "title_bn": "ম্যাজিক ব্যাকগ্রাউন্ড রিমুভার (U²-Net)", "icon": "🪄", "desc": "এক ক্লিকে আল্ট্রা-শার্প ট্রান্সপারেন্ট ব্যাকগ্রাউন্ড রিমুভাল"},
    {"id": "dress_wrinkle_cleaner", "title": "Dress Wrinkle Cleaner", "title_bn": "পোশাকের ভাঁজ ও রিংকেল ক্লিনার", "icon": "👔", "desc": "কাপড়ের কুঁচকানো দাগ ও ভাঁজ দূর করে আসল টেক্সচার মসৃণ করা"},
    {"id": "cast_shadow_3d", "title": "3D Cast Shadow", "title_bn": "৩D কাস্ট শ্যাডো ও পারস্পেক্টিভ", "icon": "📐", "desc": "আলোর কোণ অনুযায়ী বাস্তবসম্মত ৩D ড্রপ ও কাস্ট শ্যাডো তৈরি"},
    {"id": "jewelry_metal_shiner", "title": "Jewelry & Metal Shiner", "title_bn": "জুয়েলারি ও মেটাল শাইনার", "icon": "💍", "desc": "গহনা, হীরা ও মেটালের ধুলোবালি মুছে স্পেকুলার গ্লস ও শাইন আনা"},
    {"id": "apparel_recolor", "title": "Apparel Recolor", "title_bn": "ড্রেস ও কালার সোয়াচ রিকলার", "icon": "🎨", "desc": "কাপড়ের ভাঁজ ও আলো অক্ষুণ্ণ রেখে যেকোনো নতুন কালারে রূপান্তর"},
    {"id": "studio_denoise", "title": "Studio Denoise", "title_bn": "হাই-আইএসও স্টুডিও ডিনয়েজ", "icon": "🧼", "desc": "কম আলো ও কম্প্রেশনের গ্রেইন/নয়েজ দূর করে ক্রিস্টাল ক্লিন লুক"},
    {"id": "spot_dust_remover", "title": "Spot & Dust Inpainting", "title_bn": "স্পট ও ডাস্ট ইনপেইন্টিং", "icon": "🖌️", "desc": "পোশাকের সুতো, দাগ ও সেন্সর ডাস্ট নিখুঁতভাবে মুছে ফেলা"},
    {"id": "super_resolution_4k", "title": "4K Super-Resolution", "title_bn": "৪K সুপার রেজোলিউশন ও আপস্কেলার", "icon": "🔍", "desc": "ছবিকে ২x-৪x ক্রিস্টাল ক্লিয়ার হাই-রেজোলিউশন ও শার্প টেক্সচারে রূপান্তর"},
    {"id": "face_beauty_retouch", "title": "Face & Skin Beauty Pro", "title_bn": "এআই ফেস বিউটি ও স্কিন রিটাচিং", "icon": "💎", "desc": "পোর্ট্রেট ও ফ্যাশন মডেলদের স্কিন টোন, চোখ ও চেহারার দাগ নিখুঁত রিটাচ"},
    {"id": "smart_part_selector", "title": "Smart Part Selector", "title_bn": "১-ক্লিক স্মার্ট পার্ট ও কলার সিলেক্টর", "icon": "🎯", "desc": "পোশাকের কলার, বোতাম বা আলাদা অংশ এক ক্লিকে সিলেক্ট ও কাটআউট"},
    {"id": "multi_clipping_path", "title": "Multiple Clipping Path", "title_bn": "মাল্টিপল ক্লিপিং পাথ (কালার সেগমেন্টেশন)", "icon": "🎨", "desc": "ছবির বিভিন্ন অংশকে আলাদা আলাদা কালার ও লেয়ারে বিভক্ত করা"},
    {"id": "image_masking", "title": "Image Masking", "title_bn": "ইমেজ মাস্কিং (চুল/পশম রিফাইনমেন্ট)", "icon": "🎭", "desc": "জটিল চুল বা পশমের সূক্ষ্ম বর্ডার অক্ষুণ্ণ রেখে ব্যাকগ্রাউন্ড রিমুভাল"},
    {"id": "neck_joint", "title": "Neck Joint / Ghost Mannequin", "title_bn": "নেক জয়েন্ট / গোস্ট ম্যানিকুইন", "icon": "👔", "desc": "পোশাকের ভেতর ও বাইরের কলার নিখুঁতভাবে জয়েন্ট করে থ্রিডি লুক"},
    {"id": "image_retouching", "title": "Image Retouching", "title_bn": "ইমেজ রিটাচিং (স্কিন ও গ্ল্যামার ফিনিশ)", "icon": "✨", "desc": "ব্রণ, দাগ ও রিঙ্কেল মুছে ন্যাচারাল স্কিন টেক্সচার ফ্রিকোয়েন্সি সেপারেশন"},
    {"id": "shadow_making", "title": "Shadow Making", "title_bn": "শ্যাডো মেকিং (ড্রপ ও কন্টাক্ট শ্যাডো)", "icon": "👥", "desc": "প্রোডাক্টে রিয়েলিস্টিক ড্রপ শ্যাডো ও কন্টাক্ট শ্যাডো যুক্ত করা"},
    {"id": "reflection", "title": "Reflection", "title_bn": "রিফ্লেকশন শ্যাডো (মিরর গ্লস ইফেক্ট)", "icon": "🪞", "desc": "গ্লাস বা মার্বেল ফ্লোরে বিলাসবহুল মিরর রিফ্লেকশন তৈরি"},
    {"id": "color_correction", "title": "Color Correction", "title_bn": "কালার কারেকশন (হোয়াইট ব্যালেন্স ও টোন)", "icon": "🌈", "desc": "অপ্রাকৃতিক কালার কাস্ট মুছে পারফেক্ট অটো হোয়াইট ব্যালেন্স ও এস-কার্ভ"},
    {"id": "image_enhancement", "title": "Image Enhancement", "title_bn": "ইমেজ এনহ্যান্সমেন্ট (এইচডিআর ও শার্পনেস)", "icon": "🔮", "desc": "এইচডিআর ডিটেইল বুস্ট, আনশার্প মাস্কিং ও শ্যাডো রিকভারি"},
    {"id": "image_manipulation", "title": "Image Manipulation", "title_bn": "ইমেজ ম্যানিপুলেশন (ক্রিয়েটিভ ব্লেন্ডিং)", "icon": "🌌", "desc": "একাধিক এলিমেন্ট ও লাইটিংয়ের ক্রিয়েটিভ ড্রামাটিক কম্পোজিটিং"},
    {"id": "raster_to_vector", "title": "Raster To Vector", "title_bn": "রাস্টার টু ভেক্টর (এজ ও লাইন ট্রেসিং)", "icon": "📐", "desc": "পিক্সেল ছবিকে পরিষ্কার ভেক্টর কনট্যুর ও হাই-রেজ লাইন ট্রেসিংয়ে রূপান্তর"}
]


def execute_studio_service(img: Image.Image, service_id: str, mask: Image.Image = None) -> (Image.Image, str):
    """
    Executes one of the 11 Professional Studio Sector services on the given image.
    """
    cv_img = np.array(img.convert("RGB"))
    w, h = img.size

    from isbd.advanced_tools import (
        multiple_clipping_path_engine,
        ghost_mannequin_neck_joint,
        commercial_frequency_retouch,
        digital_ironing_wrinkle_cleaner,
        jewelry_metal_gemstone_shiner,
        dual_zone_ground_shadow_and_reflection,
        apparel_recolor,
        studio_denoise_healer,
        spot_dust_remover,
        color_correction_studio,
        smart_part_selector,
        raster_to_vector_engine,
        shadow_making,
        reflection_studio,
        StudioResult,
    )

    if service_id in ("clipping_path", "multi_clipping_path"):
        return multiple_clipping_path_engine(img)

    elif service_id in ("ghost_mannequin", "neck_joint"):
        return ghost_mannequin_neck_joint(img)

    elif service_id in ("image_retouching", "skin_retouch"):
        return commercial_frequency_retouch(img)

    elif service_id in ("dress_wrinkle_cleaner", "wrinkle_cleaner"):
        return digital_ironing_wrinkle_cleaner(img)

    elif service_id in ("jewelry_metal_shiner", "jewelry_retouch"):
        return jewelry_metal_gemstone_shiner(img)

    elif service_id in ("cast_shadow_3d", "natural_shadow"):
        return dual_zone_ground_shadow_and_reflection(img, shadow_type="both")

    elif service_id == "shadow_making":
        return shadow_making(img)

    elif service_id == "reflection":
        return reflection_studio(img)

    elif service_id == "background_removal":
        try:
            from isbd.opensource_ai import birefnet_hair_matting
            out_img, desc = birefnet_hair_matting(img)
            return StudioResult(out_img, f"Background Removal (BiRefNet/IS-Net ম্যাজিক রিমুভ সম্পন্ন)", ok=True)
        except Exception as e:
            try:
                from isbd.opensource_ai import get_rembg_session, clean_outer_artifacts
                import rembg
                s = get_rembg_session()
                if s:
                    out_img = rembg.remove(img, session=s, post_process_mask=True)
                    out_img = clean_outer_artifacts(out_img)
                    return StudioResult(out_img, "Background Removal (U2Net/IS-Net AI ম্যাজিক রিমুভ সম্পন্ন)", ok=True)
            except Exception:
                pass
            return StudioResult(img, f"Background Removal (ত্রুটি: {str(e)})", ok=False)

    elif service_id == "apparel_recolor":
        return apparel_recolor(img)

    elif service_id == "studio_denoise":
        return studio_denoise_healer(img)

    elif service_id == "spot_dust_remover":
        return spot_dust_remover(img)

    elif service_id == "super_resolution_4k":
        from isbd.opensource_ai import super_resolution_4k
        out_img, desc = super_resolution_4k(img, scale=2)
        return StudioResult(out_img, desc, ok=True)

    elif service_id == "face_beauty_retouch":
        try:
            from isbd.opensource_ai import face_beauty_retouch
            out_img, desc = face_beauty_retouch(img)
            return StudioResult(out_img, desc, ok=True)
        except Exception:
            return commercial_frequency_retouch(img)

    elif service_id == "smart_part_selector":
        try:
            from isbd.opensource_ai import smart_part_selector as ai_smart_part
            out_img, desc = ai_smart_part(img)
            return StudioResult(out_img, desc, ok=True)
        except Exception:
            return smart_part_selector(img)

    elif service_id == "image_masking":
        # Commercial hair/fur feather masking with Guided Filter alpha refinement
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        # Multi-level threshold for soft hair edge detection
        _, thresh_low = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        _, thresh_high = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
        coarse_fg = cv2.bitwise_and(thresh_low, thresh_high)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        coarse_fg = cv2.morphologyEx(coarse_fg, cv2.MORPH_CLOSE, kernel, iterations=2)
        # Edge-aware alpha refinement via guided box filter
        alpha_f = coarse_fg.astype(np.float32) / 255.0
        guide = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        r = 4
        eps = 0.01
        mean_alpha = cv2.boxFilter(alpha_f, -1, (2 * r + 1, 2 * r + 1))
        mean_guide = cv2.boxFilter(guide, -1, (2 * r + 1, 2 * r + 1))
        mean_ag = cv2.boxFilter(alpha_f * guide, -1, (2 * r + 1, 2 * r + 1))
        cov_ag = mean_ag - mean_alpha * mean_guide
        mean_gg = cv2.boxFilter(guide * guide, -1, (2 * r + 1, 2 * r + 1))
        var_g = mean_gg - mean_guide * mean_guide
        a_k = cov_ag / (var_g + eps)
        b_k = mean_alpha - a_k * mean_guide
        mean_ak = cv2.boxFilter(a_k, -1, (2 * r + 1, 2 * r + 1))
        mean_bk = cv2.boxFilter(b_k, -1, (2 * r + 1, 2 * r + 1))
        refined_alpha = np.clip(mean_ak * guide + mean_bk, 0.0, 1.0)
        refined_mask = (refined_alpha * 255).astype(np.uint8)
        rgba = cv2.cvtColor(cv_img, cv2.COLOR_RGB2RGBA)
        rgba[:, :, 3] = refined_mask
        return StudioResult(Image.fromarray(rgba), "Image Masking (গাইডেড ফিল্টার ফাইন হেয়ার ফেদার মাস্কিং সম্পন্ন)", ok=True)

    elif service_id == "color_correction":
        return color_correction_studio(img)

    elif service_id == "image_enhancement":
        # HDR tone map + Unsharp masking
        hdr = cv2.detailEnhance(cv_img, sigma_s=12, sigma_r=0.18)
        arr_f = hdr.astype(np.float32) / 255.0
        sharpened = skimage.filters.unsharp_mask(arr_f, radius=1.8, amount=1.8)
        enhanced = np.ascontiguousarray(np.clip(sharpened * 255, 0, 255).astype(np.uint8))
        return StudioResult(Image.fromarray(enhanced), "Image Enhancement (এইচডিআর টেক্সচার ও হাই-পাস শার্পেনিং)", ok=True)

    elif service_id == "image_manipulation":
        # Creative dynamic lighting & cinematic tone blend
        stylized = cv2.stylization(cv_img, sigma_s=40, sigma_r=0.08)
        lab = cv2.cvtColor(stylized, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        manipulated = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)
        return StudioResult(Image.fromarray(manipulated), "Image Manipulation (ক্রিয়েটিভ লাইটিং ও ড্রামাটিক টোন)", ok=True)

    elif service_id == "raster_to_vector":
        return raster_to_vector_engine(img)

    return StudioResult(img, "Original", ok=True)
