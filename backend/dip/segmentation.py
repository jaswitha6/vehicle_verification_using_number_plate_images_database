"""
DIP Module: Image Segmentation
Covers: U3-T3 Image Segmentation techniques, U3-T4 Single/Multi-level segmentation,
        U5-T1 Object recognition, U5-T4 Character recognition
"""

import cv2
import numpy as np


def detect_plate_region(img: np.ndarray) -> tuple:
    """
    U3-T3: Segmentation using contour-based approach.
    Detects the number plate bounding box from the full vehicle image.
    
    Returns: (plate_crop, bbox) where bbox = (x, y, w, h)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    
    # Edge detection (Canny) for plate boundary
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate to connect edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort by area (largest = most likely plate region)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    plate_crop = None
    best_bbox = None
    
    for contour in contours[:10]:
        area = cv2.contourArea(contour)
        if area < 500:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Indian plates are roughly 3:1 to 5:1 aspect ratio
        if 2.0 <= aspect_ratio <= 7.0 and 30 < h < 200:
            plate_crop = img[y:y+h, x:x+w] if len(img.shape) == 3 else gray[y:y+h, x:x+w]
            best_bbox = (x, y, w, h)
            break
    
    # Fallback: if no plate detected, return center crop
    if plate_crop is None:
        h, w = img.shape[:2]
        # Assume plate is in bottom half center
        y1, y2 = int(h * 0.4), int(h * 0.85)
        x1, x2 = int(w * 0.1), int(w * 0.9)
        plate_crop = img[y1:y2, x1:x2]
        best_bbox = (x1, y1, x2 - x1, y2 - y1)
    
    return plate_crop, best_bbox


def multi_level_segment(gray: np.ndarray) -> dict:
    """
    U3-T4: Multi-level segmentation.
    Level 1: Segment plate from vehicle
    Level 2: Segment characters from plate
    """
    # Level 1: Plate detection
    _, edges = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Level 2: Character segmentation within plate
    # Connected components analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        edges, connectivity=8
    )
    
    characters = []
    h, w = gray.shape
    
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        aspect = cw / ch if ch > 0 else 0
        rel_h = ch / h if h > 0 else 0
        
        # Filter for character-sized blobs
        if (0.3 <= aspect <= 1.2 and 
            0.3 <= rel_h <= 0.9 and
            50 < area < 5000):
            char_img = gray[y:y+ch, x:x+cw]
            characters.append({
                "bbox": (x, y, cw, ch),
                "image": char_img,
                "centroid": centroids[i]
            })
    
    # Sort characters left to right
    characters = sorted(characters, key=lambda c: c["bbox"][0])
    
    return {
        "level1": edges,
        "num_characters": len(characters),
        "characters": characters
    }


def preprocess_plate_for_ocr(plate_img: np.ndarray) -> np.ndarray:
    """
    U5-T4: Specific preprocessing for OCR/character recognition.
    Optimizes the plate region for text extraction.
    """
    # Ensure it's grayscale
    if len(plate_img.shape) == 3:
        plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    else:
        plate_gray = plate_img.copy()
    
    # Resize to standard height for OCR consistency
    target_h = 80
    ratio = target_h / plate_gray.shape[0]
    target_w = max(int(plate_gray.shape[1] * ratio), 200)
    plate_resized = cv2.resize(plate_gray, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    
    # CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    plate_enhanced = clahe.apply(plate_resized)
    
    # Sharpen
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    plate_sharp = cv2.filter2D(plate_enhanced, -1, kernel)
    
    # Denoise
    plate_denoised = cv2.fastNlMeansDenoising(plate_sharp, h=10)
    
    return plate_denoised


def draw_detection(img: np.ndarray, bbox: tuple, label: str = "", color=(0, 255, 0)) -> np.ndarray:
    """Draw bounding box and label on image."""
    vis = img.copy()
    x, y, w, h = bbox
    cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
    if label:
        cv2.putText(vis, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return vis
