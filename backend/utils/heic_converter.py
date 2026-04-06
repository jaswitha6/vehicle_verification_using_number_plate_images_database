"""
Utility: HEIC to JPG Converter
Handles iPhone HEIC images from the dataset.
"""

import os
import shutil
from pathlib import Path


def convert_heic_to_jpg(input_path: str, output_path: str = None) -> str:
    """
    Convert HEIC/HEIF image to JPG.
    Returns path to converted JPG file.
    """
    input_path = str(input_path)
    
    if output_path is None:
        output_path = str(Path(input_path).with_suffix('.jpg'))
    
    ext = Path(input_path).suffix.lower()
    
    # If already JPG/PNG, just return as-is
    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
        return input_path
    
    if ext not in ['.heic', '.heif']:
        return input_path
    
    try:
        from pillow_heif import register_heif_opener
        from PIL import Image
        
        register_heif_opener()
        img = Image.open(input_path)
        img = img.convert('RGB')
        img.save(output_path, 'JPEG', quality=95)
        return output_path
    
    except ImportError:
        print("[HEIC] pillow-heif not installed. Try: pip install pillow-heif")
        return input_path
    except Exception as e:
        print(f"[HEIC] Conversion failed: {e}")
        return input_path


def convert_upload(file_path: str) -> str:
    """
    Convert uploaded file to a format OpenCV can handle.
    Returns path to the usable image.
    """
    ext = Path(file_path).suffix.lower()
    
    if ext in ['.heic', '.heif']:
        jpg_path = file_path.replace(ext, '.jpg')
        return convert_heic_to_jpg(file_path, jpg_path)
    
    return file_path


def batch_convert_dataset(dataset_dir: str, output_dir: str = None):
    """
    Batch convert all HEIC images in a directory.
    Useful for pre-processing the 174-image dataset.
    """
    dataset_dir = Path(dataset_dir)
    
    if output_dir is None:
        output_dir = dataset_dir / "converted"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    heic_files = list(dataset_dir.glob("*.heic")) + list(dataset_dir.glob("*.HEIC"))
    jpg_files = list(dataset_dir.glob("*.jpg")) + list(dataset_dir.glob("*.JPG"))
    
    converted = 0
    
    for f in heic_files:
        out = output_dir / f.with_suffix('.jpg').name
        result = convert_heic_to_jpg(str(f), str(out))
        if result != str(f):
            converted += 1
    
    # Copy existing JPGs
    for f in jpg_files:
        shutil.copy(str(f), str(output_dir / f.name))
    
    print(f"[HEIC] Converted {converted} HEIC files. Copied {len(jpg_files)} JPGs to {output_dir}")
    return str(output_dir)
