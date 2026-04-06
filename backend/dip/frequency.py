"""
DIP Module: Frequency Domain Filtering
Covers: U2-T2 DFT, DCT, U2-T4 Frequency domain filtering - smoothing/sharpening,
        Filter banks and wavelets
"""

import cv2
import numpy as np
import pywt
from config import DIP_CONFIG


def apply_dft_lowpass(gray: np.ndarray, cutoff: int = None) -> np.ndarray:
    """
    U2-T2 & U2-T4: DFT-based low-pass filter.
    Removes high-frequency noise (periodic noise from camera sensor).
    
    Steps:
    1. Compute 2D DFT
    2. Shift zero-frequency to center
    3. Apply circular mask (low-pass)
    4. Inverse DFT
    """
    if cutoff is None:
        cutoff = DIP_CONFIG["dft_cutoff"]

    gray_float = np.float32(gray)
    dft = cv2.dft(gray_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft)

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    # Create circular low-pass mask
    mask = np.zeros((rows, cols, 2), np.uint8)
    cv2.circle(mask, (ccol, crow), cutoff, (1, 1), -1)

    # Apply mask
    filtered = dft_shifted * mask

    # Inverse DFT
    ishift = np.fft.ifftshift(filtered)
    img_back = cv2.idft(ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    
    # Normalize to 0-255
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(img_back)


def apply_dft_highpass(gray: np.ndarray, cutoff: int = 30) -> np.ndarray:
    """
    U2-T4: DFT-based high-pass filter (sharpening via frequency domain).
    Enhances edges and fine character details.
    """
    gray_float = np.float32(gray)
    dft = cv2.dft(gray_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft)

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    # High-pass: invert the low-pass mask
    mask = np.ones((rows, cols, 2), np.uint8)
    cv2.circle(mask, (ccol, crow), cutoff, (0, 0), -1)

    filtered = dft_shifted * mask
    ishift = np.fft.ifftshift(filtered)
    img_back = cv2.idft(ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(img_back)


def get_dft_spectrum(gray: np.ndarray) -> np.ndarray:
    """
    Compute and return the DFT magnitude spectrum (for visualization).
    """
    gray_float = np.float32(gray)
    dft = cv2.dft(gray_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft)
    magnitude = 20 * np.log(cv2.magnitude(dft_shifted[:, :, 0], dft_shifted[:, :, 1]) + 1)
    cv2.normalize(magnitude, magnitude, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(magnitude)


def apply_dct_enhance(gray: np.ndarray) -> np.ndarray:
    """
    U2-T2: DCT-based enhancement.
    Similar to JPEG compression concept — modify DCT coefficients,
    then reconstruct to sharpen the image.
    """
    gray_float = np.float32(gray) / 255.0
    dct = cv2.dct(gray_float)

    # Boost mid-frequency coefficients (character frequencies)
    rows, cols = dct.shape
    mid_r, mid_c = rows // 4, cols // 4
    dct[mid_r:mid_r * 2, mid_c:mid_c * 2] *= 1.5

    # Inverse DCT
    idct = cv2.idct(dct)
    idct = np.clip(idct * 255, 0, 255)
    return np.uint8(idct)


def apply_wavelet_denoise(gray: np.ndarray, wavelet: str = None, level: int = None) -> np.ndarray:
    """
    U2-T4: Wavelet-based denoising using filter banks.
    Uses multi-level decomposition (haar wavelet) to separate
    signal from noise, then reconstruct.
    
    This is the "Filter banks and wavelets" concept from the syllabus.
    """
    if wavelet is None:
        wavelet = DIP_CONFIG["wavelet"]
    if level is None:
        level = DIP_CONFIG["wavelet_level"]

    gray_float = np.float64(gray)
    
    # Multi-level wavelet decomposition
    coeffs = pywt.wavedec2(gray_float, wavelet=wavelet, level=level)
    
    # Soft thresholding on detail coefficients (noise removal)
    threshold = 20.0
    coeffs_thresh = list(coeffs)
    for i in range(1, len(coeffs_thresh)):
        coeffs_thresh[i] = tuple(
            pywt.threshold(c, threshold, mode='soft') for c in coeffs_thresh[i]
        )
    
    # Reconstruct
    denoised = pywt.waverec2(coeffs_thresh, wavelet=wavelet)
    denoised = np.clip(denoised, 0, 255)
    
    # Match original shape
    h, w = gray.shape
    denoised = denoised[:h, :w]
    return np.uint8(denoised)


def frequency_domain_pipeline(gray: np.ndarray) -> np.ndarray:
    """
    Combined frequency domain pipeline for number plate enhancement.
    1. Wavelet denoise (remove noise)
    2. DFT low-pass (remove remaining HF noise)
    3. DCT enhance (boost character frequencies)
    """
    step1 = apply_wavelet_denoise(gray)
    step2 = apply_dft_lowpass(step1, cutoff=60)
    step3 = apply_dct_enhance(step2)
    return step3
