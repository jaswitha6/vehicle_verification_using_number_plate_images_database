import cv2
import numpy as np
import re
import logging
import torch
from collections import Counter
from PIL import Image

try:
    import easyocr
except ImportError:
    easyocr = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

logger = logging.getLogger('VehicleVerification')

if pytesseract is not None:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize EasyOCR once globally
try:
    if easyocr is not None:
        reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        logger.info(f"[OCR] EasyOCR initialized | GPU: {torch.cuda.is_available()}")
    else:
        reader = None
        logger.warning("[OCR] EasyOCR not installed; skipping EasyOCR backend")
except Exception as e:
    reader = None
    logger.warning(f"[OCR] EasyOCR init failed: {e}")

HOLOGRAM_NOISE = ['INDIA', 'IND', 'DIA', 'INIA', 'NDIA', 'VION', 'ION',
                  'YIUN', 'YIUNI', 'INDI', 'VIOND', 'NI', 'DI', 'IA']


def correct_plate(text):
    """Correct common OCR misreads for Indian plates."""
    if not text:
        return ""
    text = re.sub(r'[^A-Z0-9]', '', text.upper())

    # Try to extract valid Indian plate pattern
    patterns = [
        r'[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}',
        r'[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}',
        r'[A-Z]{2}\d{2}\d{4}',
    ]
    for pat in patterns:
        match = re.search(pat, text)
        if match:
            return match.group()

    # Common OCR fixes: fix digits in last 4 chars
    char_fixes = {'O': '0', 'I': '1', 'l': '1', 'S': '5', 'B': '8', 'G': '6', 'Z': '2', 'Q': '0'}
    if len(text) >= 4:
        prefix = text[:-4]
        suffix = ''
        for c in text[-4:]:
            suffix += char_fixes.get(c, c) if c.isalpha() else c
        text = prefix + suffix

    return text


def get_preprocessing_variants(img_bgr):
    """Returns preprocessing variants for Indian number plates."""
    variants = []
    h, w = img_bgr.shape[:2]

    scale = max(1, 800 // max(h, w, 1))
    if scale > 1:
        img_bgr = cv2.resize(img_bgr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    variants.append(clahe.apply(gray))

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)
    variants.append(cv2.bitwise_not(otsu))

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, blur_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(blur_otsu)

    median = cv2.medianBlur(gray, 3)
    variants.append(clahe.apply(median))

    return variants


def is_hologram_noise(text):
    clean = re.sub(r'[^A-Z0-9]', '', text.upper()).replace('IND', '')
    for noise in HOLOGRAM_NOISE:
        if clean == noise:
            return True
    if len(clean) <= 2:
        return True
    if re.match(r'^(IN|ND|DI|IA|INDI|INDIA|VION|ION)+$', clean):
        return True
    return False


def clean_text(text):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    for noise in ['INDIA', 'INDI', 'IND', 'VION', 'YIUN', 'YIUNI',
                  'NDIA', 'INIA', 'DIA', 'ION', 'NI', 'DI', 'IA']:
        text = text.replace(noise, '')
    text = re.sub(r'(.)\1{2,}', '', text)
    return text


def sort_regions_by_reading_order(results):
    if not results:
        return results
    heights = [max(pt[1] for pt in bbox) - min(pt[1] for pt in bbox) for (bbox, _, _) in results]
    avg_height = sum(heights) / len(heights) if heights else 20
    row_threshold = avg_height * 0.7

    def get_cy(bbox): return sum(pt[1] for pt in bbox) / 4
    def get_cx(bbox): return sum(pt[0] for pt in bbox) / 4

    items = [(r, get_cy(r[0]), get_cx(r[0])) for r in results]
    items.sort(key=lambda x: x[1])

    rows, current_row = [], [items[0]]
    for item in items[1:]:
        if abs(item[1] - current_row[-1][1]) <= row_threshold:
            current_row.append(item)
        else:
            rows.append(current_row)
            current_row = [item]
    rows.append(current_row)

    sorted_results = []
    for row in rows:
        row.sort(key=lambda x: x[2])
        for item in row:
            sorted_results.append(item[0])
    return sorted_results


def filter_noise_regions(results):
    filtered = []
    for (bbox, text, conf) in results:
        clean = re.sub(r'[^A-Z0-9]', '', text.upper())
        if not clean:
            continue
        if is_hologram_noise(text):
            continue
        if re.match(r'^\d{6,}$', clean):
            continue
        filtered.append((bbox, text, conf))
    return filtered


def run_easyocr(variant):
    if reader is None:
        return '', 0.0
    try:
        results = reader.readtext(
            variant,
            detail=1,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            paragraph=False,
            width_ths=0.9,
            contrast_ths=0.1
        )
        if not results:
            return '', 0.0

        results = filter_noise_regions(results)
        if not results:
            return '', 0.0

        results = sort_regions_by_reading_order(results)
        candidates = []

        combined = clean_text(''.join([r[1].upper() for r in results]))
        avg_conf = sum(r[2] for r in results) / len(results)
        if len(combined) >= 5:
            candidates.append((combined, avg_conf))

        high_conf = [r for r in results if r[2] >= 0.45]
        if high_conf:
            hc_text = clean_text(''.join([r[1].upper() for r in high_conf]))
            hc_conf = sum(r[2] for r in high_conf) / len(high_conf)
            if len(hc_text) >= 5:
                candidates.append((hc_text, hc_conf))

        for (bbox, text, conf) in results:
            t = clean_text(text)
            if len(t) >= 5:
                candidates.append((t, conf))

        if not candidates:
            return '', 0.0

        best_text, best_score = '', -1
        for text, conf in candidates:
            length_score = 1.0 - abs(len(text) - 10) * 0.1
            score = conf * 0.5 + length_score * 0.5
            if score > best_score:
                best_score = score
                best_text = text

        return best_text, best_score

    except Exception as e:
        logger.warning(f"[OCR] EasyOCR failed: {e}")
        return '', 0.0


def run_tesseract_variant(variant):
    if pytesseract is None:
        return '', 0.0
    try:
        pil_img = Image.fromarray(variant)
        config = '--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text = pytesseract.image_to_string(pil_img, config=config).strip()
        text = clean_text(text)
        return text, 0.5
    except Exception:
        return '', 0.0


def score_reading(text, conf):
    length_score = 1.0 - abs(len(text) - 10) * 0.1
    return conf * 0.6 + length_score * 0.4


def extract_plate_text(image_path):
    logger.info(f"[OCR] Processing: {image_path}")
    try:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return "", 0.0

        variants = get_preprocessing_variants(img_bgr)
        all_readings = []

        for variant in variants:
            text, conf = run_easyocr(variant)
            if text:
                cleaned = correct_plate(text)
                if cleaned and len(cleaned) >= 5:
                    logger.info(f"[OCR] EasyOCR: '{cleaned}' conf={conf:.2f}")
                    all_readings.append((cleaned, conf))

            text, conf = run_tesseract_variant(variant)
            if text:
                cleaned = correct_plate(text)
                if cleaned and len(cleaned) >= 5:
                    logger.info(f"[OCR] Tesseract: '{cleaned}' conf={conf:.2f}")
                    all_readings.append((cleaned, conf))

        if not all_readings:
            logger.warning("[OCR] No text extracted.")
            return "", 0.0

        valid_readings = [r for r in all_readings if 7 <= len(r[0]) <= 12]
        if valid_readings:
            valid_readings.sort(key=lambda x: x[1], reverse=True)
            best = valid_readings[0]
            logger.info(f"[OCR] Best: '{best[0]}' conf={best[1]:.2f}")
            return best[0], min(best[1], 1.0)

        all_readings.sort(key=lambda x: score_reading(x[0], x[1]), reverse=True)
        top_texts = [r[0] for r in all_readings[:6]]
        freq = Counter(top_texts)
        best = freq.most_common(1)[0][0]
        logger.info(f"[OCR] Frequency best: '{best}'")
        return best, 0.5

    except Exception as e:
        logger.error(f"[OCR] Fatal error: {e}")
        return "", 0.0


def run_ocr_pipeline(image_path, *args, **kwargs):
    return extract_plate_text(image_path)
