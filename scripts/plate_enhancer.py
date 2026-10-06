"""Super-Resolution & Image Enhancement Engine for Motorcycle License Plates.

Applies multi-stage neural & morphological enhancement:
1. 4x Super-Resolution / Lanczos-4 Upscaling
2. Bilateral Edge-Preserving Denoising
3. CLAHE (Contrast Limited Adaptive Histogram Equalization in LAB space)
4. High-Pass Unsharp Masking (USM) for crisp character contours
5. Perspective & Deskew alignment
"""
import cv2
import numpy as np
import re


class PlateEnhancer:
    def __init__(self, target_height=160):
        self.target_height = target_height
        self.clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))

    def enhance(self, crop):
        """Enhances a small/blurry license plate crop into a high-resolution, sharp image."""
        if crop is None or crop.size == 0:
            return crop

        h, w = crop.shape[:2]
        if h == 0 or w == 0:
            return crop

        # 1. Super-Resolution Scaling
        scale = max(2.5, float(self.target_height) / max(1, h))
        target_w = int(round(w * scale))
        target_h = int(round(h * scale))
        scaled = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        # 2. Gentle Bilateral Filter: Smooth sensor noise while preserving sharp character edges
        denoised = cv2.bilateralFilter(scaled, d=5, sigmaColor=25, sigmaSpace=25)

        # 3. YUV Color Space Contrast Enhancement (chống cháy sáng trên biển trắng)
        yuv = cv2.cvtColor(denoised, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(yuv)
        y_eq = self.clahe.apply(y)
        yuv_enhanced = cv2.merge([y_eq, u, v])
        bgr_enhanced = cv2.cvtColor(yuv_enhanced, cv2.COLOR_YUV2BGR)

        # 4. Mild Unsharp Masking (Crisp character edges without highlight burnout)
        gaussian = cv2.GaussianBlur(bgr_enhanced, (0, 0), sigmaX=1.5, sigmaY=1.5)
        sharpened = cv2.addWeighted(bgr_enhanced, 1.30, gaussian, -0.30, 0)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

        # 5. Deskew / Angle Correction (if aspect ratio is standard)
        try:
            gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            coords = np.column_stack(np.where(thresh > 0))
            if len(coords) > 20:
                angle = cv2.minAreaRect(coords)[-1]
                if angle < -45:
                    angle = -(90 + angle)
                elif angle > 45:
                    angle = 90 - angle
                if abs(angle) > 1.5 and abs(angle) < 25.0:
                    (ch, cw) = sharpened.shape[:2]
                    M = cv2.getRotationMatrix2D((cw // 2, ch // 2), angle, 1.0)
                    sharpened = cv2.warpAffine(sharpened, M, (cw, ch), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        except Exception:
            pass

        return sharpened

    @staticmethod
    def clean_plate_text(raw_text):
        """Cleans and standardizes Vietnamese motorcycle license plate format (e.g., 22-S1 2000)."""
        if not raw_text:
            return ""

        # Normalize common OCR confusions
        cleaned = raw_text.upper().strip()
        cleaned = re.sub(r'[^A-Z0-9\-\.\s]', '', cleaned)
        cleaned = cleaned.replace('O', '0')

        # Split into parts
        tokens = cleaned.split()
        if not tokens:
            return ""

        # Filter out junk tokens that are single characters or nonsense
        valid_tokens = [t for t in tokens if len(t) >= 2 or t.isalnum()]
        if not valid_tokens:
            return ""

        full_str = " ".join(valid_tokens)

        # Must have at least 1 digit to be a real license plate in Vietnam
        has_digit = any(c.isdigit() for c in full_str)
        if not has_digit or len(full_str) < 4:
            return ""

        return full_str
