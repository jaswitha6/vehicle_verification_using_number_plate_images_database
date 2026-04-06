import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Database
DB_PATH = os.path.join(BASE_DIR, "registered_plates.db")

# Upload folder
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "heic", "heif", "bmp", "webp"}

# DIP Settings
DIP_CONFIG = {
    "resize_width": 640,
    "resize_height": 480,
    "gaussian_kernel": (5, 5),
    "gaussian_sigma": 1.0,
    "clahe_clip": 2.0,
    "clahe_grid": (8, 8),
    "sharpen_amount": 1.5,
    "dft_cutoff": 30,
    "wavelet": "haar",
    "wavelet_level": 2,
}

# NLP Settings
NLP_CONFIG = {
    "indian_plate_pattern": r"[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}",
    "fuzzy_threshold": 80,
    "bert_model": "bert-base-uncased",
    "max_length": 64,
}

# Flask
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload
