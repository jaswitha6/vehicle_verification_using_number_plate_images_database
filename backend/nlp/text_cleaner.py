"""
NLP Module: Text Preprocessing & Cleaning
Covers: U1-T4 Data acquisition, Text extraction and cleanup, Preprocessing
        U1-T3 NLP Pipeline
"""

import re
import nltk
from nltk.tokenize import word_tokenize
from config import NLP_CONFIG

# Download NLTK data silently
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)


# Indian number plate format patterns
PLATE_PATTERNS = [
    # Standard: KA01AB1234
    r'\b[A-Z]{2}\s*[0-9]{1,2}\s*[A-Z]{1,3}\s*[0-9]{4}\b',
    # New BH series: 22BH1234AA
    r'\b[0-9]{2}BH[0-9]{4}[A-Z]{2}\b',
    # Temporary: TR-02-234
    r'\bTR[-\s]?[0-9]{2}[-\s]?[0-9]{3,4}\b',
    # Old format: KA-01-1234
    r'\b[A-Z]{2}[-\s]?[0-9]{2}[-\s]?[0-9]{4}\b',
]

# Common OCR confusion pairs (OCR often confuses these)
OCR_CORRECTIONS = {
    '0': 'O', 'O': '0',  # Context-dependent
    '1': 'I', 'I': '1',
    '8': 'B', 'B': '8',
    '5': 'S', 'S': '5',
    '6': 'G', 'G': '6',
    'Q': '0', 'D': '0',
}

# State codes for Indian plates
INDIAN_STATE_CODES = {
    'AP', 'AR', 'AS', 'BR', 'CG', 'GA', 'GJ', 'HR', 'HP', 'JH',
    'KA', 'KL', 'MP', 'MH', 'MN', 'ML', 'MZ', 'NL', 'OD', 'PB',
    'RJ', 'SK', 'TN', 'TS', 'TR', 'UP', 'UK', 'WB', 'AN', 'CH',
    'DD', 'DL', 'JK', 'LA', 'LD', 'PY'
}


def basic_cleanup(raw_text: str) -> str:
    """
    U1-T4: Basic text cleanup.
    Remove unwanted characters, normalize spaces.
    """
    if not raw_text:
        return ""
    
    # Uppercase
    text = raw_text.upper()
    
    # Remove common unwanted chars but keep alphanumeric and spaces
    text = re.sub(r'[^A-Z0-9\s\-]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove isolated single characters that aren't meaningful
    tokens = text.split()
    tokens = [t for t in tokens if len(t) > 1 or t.isdigit()]
    
    return ' '.join(tokens)


def extract_plate_candidates(text: str) -> list:
    """
    U1-T4 + U4-T1: Extract plate number candidates from raw OCR text.
    Uses regex pattern matching as a heuristic NLP approach.
    """
    text_clean = text.upper().replace(' ', '')
    candidates = []
    
    for pattern in PLATE_PATTERNS:
        # Try on spaced version
        matches = re.findall(pattern, text.upper())
        candidates.extend(matches)
        
        # Try on no-space version  
        matches2 = re.findall(pattern, text_clean)
        candidates.extend(matches2)
    
    # Remove spaces within candidates and deduplicate
    candidates = list(set([c.replace(' ', '').replace('-', '') for c in candidates]))
    return candidates


def correct_ocr_errors(text: str) -> list:
    """
    U1-T4: OCR error correction using common substitution rules.
    Generates multiple correction candidates.
    
    Indian plates follow: [STATE][DIST][SERIES][NUMBER]
    Position-based correction:
    - Positions 0-1: Letters only (state code)
    - Positions 2-3: Digits only (district)
    - Positions 4-6: Letters only (series)  
    - Positions 7-10: Digits only (number)
    """
    text = text.upper().replace(' ', '').replace('-', '')
    
    if len(text) < 6:
        return [text]
    
    corrected = list(text)
    
    # Apply position-based corrections for standard format (10 chars)
    if len(corrected) >= 10:
        # Positions 0,1: must be letters (state code)
        for i in [0, 1]:
            if corrected[i].isdigit():
                digit_to_letter = {'0': 'O', '1': 'I', '8': 'B', '5': 'S'}
                corrected[i] = digit_to_letter.get(corrected[i], corrected[i])
        
        # Positions 2,3: must be digits (district number)
        for i in [2, 3]:
            if corrected[i].isalpha():
                letter_to_digit = {'O': '0', 'I': '1', 'B': '8', 'S': '5', 'G': '6', 'Z': '2'}
                corrected[i] = letter_to_digit.get(corrected[i], corrected[i])
        
        # Last 4 positions: must be digits
        for i in range(len(corrected) - 4, len(corrected)):
            if i >= 0 and corrected[i].isalpha():
                letter_to_digit = {'O': '0', 'I': '1', 'B': '8', 'S': '5', 'G': '6', 'Z': '2'}
                corrected[i] = letter_to_digit.get(corrected[i], corrected[i])
    
    primary = ''.join(corrected)
    
    # Also return original
    return list(set([primary, text]))


def validate_indian_plate(plate: str) -> dict:
    """
    U1-T3 + U2-T1: Validate and classify Indian plate format.
    Returns validation result with plate type.
    """
    plate = plate.upper().replace(' ', '').replace('-', '')
    
    result = {
        "plate": plate,
        "is_valid": False,
        "plate_type": "unknown",
        "state_code": None,
        "district": None,
        "series": None,
        "number": None,
        "confidence": 0.0
    }
    
    # Standard 10-char format: KA01AB1234
    match = re.match(r'^([A-Z]{2})([0-9]{2})([A-Z]{1,3})([0-9]{4})$', plate)
    if match:
        state, dist, series, number = match.groups()
        result["is_valid"] = True
        result["plate_type"] = "standard"
        result["state_code"] = state
        result["district"] = dist
        result["series"] = series
        result["number"] = number
        result["confidence"] = 1.0 if state in INDIAN_STATE_CODES else 0.7
        return result
    
    # BH series: 22BH1234AA
    match = re.match(r'^([0-9]{2})(BH)([0-9]{4})([A-Z]{2})$', plate)
    if match:
        year, bh, number, series = match.groups()
        result["is_valid"] = True
        result["plate_type"] = "BH_series"
        result["series"] = series
        result["number"] = number
        result["confidence"] = 0.95
        return result
    
    # Partial match — incomplete plate
    if len(plate) >= 6:
        result["plate_type"] = "partial"
        result["confidence"] = 0.3
    
    return result


def full_nlp_preprocess(raw_ocr_text: str) -> dict:
    """
    Complete NLP preprocessing pipeline.
    U1-T3: Full NLP pipeline from raw text to structured output.
    """
    # Step 1: Basic cleanup
    cleaned = basic_cleanup(raw_ocr_text)
    
    # Step 2: Extract plate candidates
    candidates = extract_plate_candidates(cleaned)
    
    # Step 3: OCR correction on each candidate
    corrected_candidates = []
    for c in candidates:
        corrected_candidates.extend(correct_ocr_errors(c))
    
    # If no candidates found, try correction on full text
    if not corrected_candidates:
        full_no_space = cleaned.replace(' ', '')
        corrected_candidates = correct_ocr_errors(full_no_space)
    
    # Step 4: Validate each candidate
    validations = []
    for cand in set(corrected_candidates):
        if len(cand) >= 6:
            v = validate_indian_plate(cand)
            validations.append(v)
    
    # Sort by confidence
    validations = sorted(validations, key=lambda x: x["confidence"], reverse=True)
    
    best = validations[0] if validations else {
        "plate": cleaned.replace(' ', ''),
        "is_valid": False,
        "plate_type": "unknown",
        "confidence": 0.0
    }
    
    return {
        "raw": raw_ocr_text,
        "cleaned": cleaned,
        "candidates": list(set(corrected_candidates)),
        "best_candidate": best,
        "all_validations": validations
    }
