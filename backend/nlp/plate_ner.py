"""
NLP Module: Named Entity Recognition for Number Plates
Covers: U4-T1 IE Tasks, Key phrase Extraction
        U4-T2 Named Entity Recognition, Building an NER System
        U4-T3 Named Entity Disambiguation and Linking
"""

import re
import spacy
from typing import Optional

# Try to load spaCy model
_nlp_model = None


def get_spacy_model():
    global _nlp_model
    if _nlp_model is None:
        try:
            _nlp_model = spacy.load("en_core_web_sm")
            print("[NER] spaCy model loaded.")
        except OSError:
            print("[NER] spaCy model not found. Run: python -m spacy download en_core_web_sm")
            _nlp_model = None
    return _nlp_model


# Custom entity ruler patterns for Indian number plates
PLATE_ENTITY_PATTERNS = [
    # Standard: KA01AB1234
    {"label": "NUMBER_PLATE", "pattern": [
        {"TEXT": {"REGEX": "[A-Z]{2}"}},
        {"TEXT": {"REGEX": "[0-9]{1,2}"}},
        {"TEXT": {"REGEX": "[A-Z]{1,3}"}},
        {"TEXT": {"REGEX": "[0-9]{4}"}},
    ]},
    # Condensed: KA01AB1234 as single token
    {"label": "NUMBER_PLATE", "pattern": [
        {"TEXT": {"REGEX": "[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4}"}}
    ]},
    # BH Series
    {"label": "NUMBER_PLATE", "pattern": [
        {"TEXT": {"REGEX": "[0-9]{2}BH[0-9]{4}[A-Z]{2}"}}
    ]},
]


def setup_ner_pipeline():
    """
    U4-T2: Build a custom NER system using spaCy EntityRuler.
    Adds NUMBER_PLATE as a custom entity type.
    """
    nlp = get_spacy_model()
    if nlp is None:
        return None
    
    # Add entity ruler if not already present
    if "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        ruler.add_patterns(PLATE_ENTITY_PATTERNS)
    
    return nlp


def extract_plate_with_ner(text: str) -> list:
    """
    U4-T1 & U4-T2: Use NER to extract plate number entities from text.
    Returns list of detected plate entities with their positions.
    """
    nlp = setup_ner_pipeline()
    
    results = []
    
    if nlp:
        doc = nlp(text.upper())
        for ent in doc.ents:
            if ent.label_ in ["NUMBER_PLATE", "ORG", "GPE"]:
                # Validate if it looks like a plate
                text_clean = ent.text.replace(' ', '').replace('-', '')
                if is_plate_like(text_clean):
                    results.append({
                        "text": text_clean,
                        "label": "NUMBER_PLATE",
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "source": "spacy_ner"
                    })
    
    # Fallback: regex-based NER
    regex_results = regex_ner(text)
    for r in regex_results:
        if r["text"] not in [x["text"] for x in results]:
            results.append(r)
    
    return results


def regex_ner(text: str) -> list:
    """
    U4-T1: Heuristic/regex-based NER for plate extraction.
    U1-T2: Heuristics-Based NLP approach.
    Fallback when spaCy is unavailable or misses the entity.
    """
    text_upper = text.upper()
    patterns = [
        r'\b[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}\b',
        r'\b[0-9]{2}BH[0-9]{4}[A-Z]{2}\b',
        r'\b[A-Z]{2}[0-9]{4}\b',
    ]
    
    results = []
    for pattern in patterns:
        for match in re.finditer(pattern, text_upper):
            results.append({
                "text": match.group().replace(' ', '').replace('-', ''),
                "label": "NUMBER_PLATE",
                "start": match.start(),
                "end": match.end(),
                "source": "regex_ner"
            })
    
    return results


def is_plate_like(text: str) -> bool:
    """Check if a string looks like an Indian number plate."""
    text = text.upper().replace(' ', '').replace('-', '')
    if len(text) < 6 or len(text) > 12:
        return False
    
    # Must have both letters and numbers
    has_alpha = any(c.isalpha() for c in text)
    has_digit = any(c.isdigit() for c in text)
    return has_alpha and has_digit


def disambiguate_plate(candidates: list, context: str = "") -> Optional[str]:
    """
    U4-T3: Named Entity Disambiguation and Linking.
    When multiple plate candidates exist, pick the most likely one
    using contextual clues and confidence scoring.
    """
    if not candidates:
        return None
    
    if len(candidates) == 1:
        return candidates[0]
    
    scored = []
    for cand in candidates:
        score = 0
        text = cand.upper().replace(' ', '').replace('-', '')
        
        # Length score (standard 10-char plates score highest)
        if len(text) == 10:
            score += 10
        elif len(text) == 9:
            score += 7
        elif 7 <= len(text) <= 11:
            score += 4
        
        # State code check
        state_codes = {'KA', 'MH', 'DL', 'TN', 'AP', 'TS', 'GJ', 'RJ', 'UP', 'KL'}
        if text[:2] in state_codes:
            score += 5
        
        # Structure check: letters then digits pattern
        if re.match(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4}$', text):
            score += 8
        
        scored.append((text, score))
    
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0]


def run_ner_pipeline(raw_ocr_text: str, preprocessed_candidates: list) -> dict:
    """
    Full NER pipeline combining spaCy NER + regex NER + disambiguation.
    U4-T1: Complete IE pipeline.
    """
    # NER on raw OCR text
    ner_entities = extract_plate_with_ner(raw_ocr_text)
    
    # Collect all candidates
    all_candidates = [e["text"] for e in ner_entities]
    all_candidates.extend(preprocessed_candidates)
    all_candidates = list(set([c for c in all_candidates if c and len(c) >= 6]))
    
    # Disambiguate
    best_plate = disambiguate_plate(all_candidates, context=raw_ocr_text)
    
    return {
        "ner_entities": ner_entities,
        "all_candidates": all_candidates,
        "best_plate": best_plate,
        "num_candidates": len(all_candidates)
    }
