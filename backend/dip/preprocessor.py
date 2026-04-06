"""
DIP Module: Spatial Domain Enhancement
Covers: U1-T2 Gray-level transformations, U1-T3 Histogram, U1-T4 Spatial filtering,
        U2-T1 Smoothing and sharpening
"""

import cv2
import numpy as np
from config import DIP_CONFIG


def load_image(image_path: str) -> np.ndarray:
    """Load image and convert to RGB numpy array."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")
    return img


def resize_image(img: np.ndarray) -> np.ndarray:
    """
    U1-T1: Interpolation and 2D signals.
    Resize using bicubic interpolation to standardize input.
    """
    h, w = img.shape[:2]
    target_w = DIP_CONFIG["resize_width"]
    target_h = DIP_CONFIG["resize_height"]

    # Maintain aspect ratio
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return resized


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    U1-T2: Gray-level transformation.
    Convert BGR to grayscale using luminosity weights.
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_clahe(gray: np.ndarray) -> np.ndarray:
    """
    U1-T3: Histogram equalization using CLAHE (Contrast Limited Adaptive HE).
    Enhances local contrast without over-amplifying noise.
    """
    clahe = cv2.createCLAHE(
        clipLimit=DIP_CONFIG["clahe_clip"],
        tileGridSize=DIP_CONFIG["clahe_grid"]
    )
    return clahe.apply(gray)


def global_histogram_equalization(gray: np.ndarray) -> np.ndarray:
    """
    U1-T3: Global histogram equalization for comparison.
    Stretches histogram across full dynamic range.
    """
    return cv2.equalizeHist(gray)


def gaussian_smoothing(img: np.ndarray) -> np.ndarray:
    """
    U2-T1: Spatial filtering - Gaussian smoothing.
    Removes high-frequency noise while preserving edges.
    """
    kernel = DIP_CONFIG["gaussian_kernel"]
    sigma = DIP_CONFIG["gaussian_sigma"]
    return cv2.GaussianBlur(img, kernel, sigma)


def median_filter(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    U1-T4: Spatial filtering - Median filter.
    Best for salt-and-pepper noise (common in real-world captures).
    """
    return cv2.medianBlur(img, ksize)


def sharpen_image(img: np.ndarray) -> np.ndarray:
    """
    U2-T1: Spatial filtering - Unsharp masking for sharpening.
    Enhances edges to improve character readability.
    """
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    alpha = DIP_CONFIG["sharpen_amount"]
    sharpened = cv2.addWeighted(img, 1 + alpha, blurred, -alpha, 0)
    return sharpened


def laplacian_sharpening(gray: np.ndarray) -> np.ndarray:
    """
    U2-T1: Laplacian-based edge sharpening.
    Second-order derivative for high-frequency enhancement.
    """
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    sharpened = cv2.subtract(gray, laplacian)
    return sharpened


def gamma_correction(img: np.ndarray, gamma: float = 1.5) -> np.ndarray:
    """
    U1-T2: Power-law (gamma) transformation.
    Corrects for varying lighting conditions in outdoor captures.
    """
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in np.arange(0, 256)
    ]).astype("uint8")
    return cv2.LUT(img, table)


def binarize(gray: np.ndarray) -> np.ndarray:
    """
    Adaptive thresholding for segmenting characters from plate background.
    Uses local mean to handle uneven illumination.
    """
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )


def morphological_cleanup(binary: np.ndarray) -> np.ndarray:
    """
    Morphological operations to clean up binary image.
    Closes small gaps in characters.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    return cleaned


def full_dip_preprocess(image_path: str) -> dict:
    """
    Complete DIP preprocessing pipeline.
    Returns dict with all intermediate and final images.
    """
    # Load
    img = load_image(image_path)
    
    # Step 1: Resize (interpolation)
    img_resized = resize_image(img)
    
    # Step 2: Gamma correction for lighting
    img_gamma = gamma_correction(img_resized, gamma=1.2)
    
    # Step 3: Grayscale (gray-level transformation)
    gray = to_grayscale(img_gamma)
    
    # Step 4: Median filter (noise removal)
    gray_denoised = median_filter(gray, ksize=3)
    
    # Step 5: CLAHE histogram equalization
    gray_enhanced = apply_clahe(gray_denoised)
    
    # Step 6: Sharpening
    gray_sharp = sharpen_image(gray_enhanced)
    
    # Step 7: Binarize
    binary = binarize(gray_sharp)
    
    # Step 8: Morphological cleanup
    binary_clean = morphological_cleanup(binary)

    return {
        "original": img,
        "resized": img_resized,
        "gamma_corrected": img_gamma,
        "grayscale": gray,
        "denoised": gray_denoised,
        "clahe": gray_enhanced,
        "sharpened": gray_sharp,
        "binary": binary,
        "binary_clean": binary_clean,
        "final": gray_sharp,  # Best for OCR
    }
