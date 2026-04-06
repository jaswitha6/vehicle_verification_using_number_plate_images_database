"""
Vehicle Verification System - Flask Backend
Integrates DIP + NLP pipeline for number plate recognition.
"""

import os
import sys
import uuid
import base64
import cv2
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS, MAX_CONTENT_LENGTH
import database as db
from utils.heic_converter import convert_upload
from utils.logger import logger

from dip.preprocessor import full_dip_preprocess
from dip.frequency import frequency_domain_pipeline
from dip.segmentation import detect_plate_region, preprocess_plate_for_ocr
from dip.ocr_engine import run_ocr_pipeline

from nlp.text_cleaner import full_nlp_preprocess
from nlp.plate_ner import run_ner_pipeline
from nlp.classifier import classify_plate
from nlp.bert_verifier import verify_plate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), 'frontend')

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

db.init_db()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(img: np.ndarray, ext: str = '.jpg') -> str:
    _, buffer = cv2.imencode(ext, img)
    return base64.b64encode(buffer).decode('utf-8')


def parse_ocr_result(ocr_result):
    if isinstance(ocr_result, tuple):
        text       = ocr_result[0] if len(ocr_result) > 0 else ""
        confidence = ocr_result[1] if len(ocr_result) > 1 else 0.0
        variant    = "easyocr"
    elif isinstance(ocr_result, dict):
        text       = ocr_result.get("text", "")
        confidence = ocr_result.get("confidence", 0.0)
        variant    = ocr_result.get("variant", "unknown")
    else:
        text       = str(ocr_result) if ocr_result else ""
        confidence = 0.0
        variant    = "unknown"
    return text, confidence, variant


def step(name, detail, status="ok"):
    return {"name": name, "detail": str(detail), "status": status}


@app.route('/')
def serve_frontend():
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "message": "Vehicle Verification API running"})


@app.route('/api/verify', methods=['POST'])
def verify_vehicle():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": f"File type not supported. Use: {ALLOWED_EXTENSIONS}"}), 400

    filename  = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    raw_path  = os.path.join(UPLOAD_FOLDER, filename)
    file.save(raw_path)
    image_path = raw_path

    dip_steps = []
    nlp_steps = []

    try:
        image_path = convert_upload(raw_path)
        logger.info(f"Processing: {image_path}")

        # ── DIP pipeline ──────────────────────────────────────────
        dip_result = full_dip_preprocess(image_path)
        preprocessed_gray = dip_result["final"]
        original_img      = dip_result["resized"]
        dip_steps.append(step("Image Resize & Load",   f"{original_img.shape[1]}×{original_img.shape[0]}px"))
        dip_steps.append(step("Grayscale Conversion",  "RGB → Grayscale"))
        dip_steps.append(step("CLAHE Enhancement",     "Contrast-limited adaptive histogram equalisation"))

        freq_enhanced = frequency_domain_pipeline(preprocessed_gray)
        dip_steps.append(step("Frequency Domain Filter", "FFT high-pass sharpening applied"))

        plate_crop, bbox = detect_plate_region(original_img)
        if plate_crop is not None and plate_crop.size > 0:
            dip_steps.append(step("Plate Region Detection", f"Bounding box: {bbox}"))
            plate_for_ocr = preprocess_plate_for_ocr(plate_crop)
            dip_steps.append(step("Plate Preprocessing",    "Upscaled + binarised for OCR"))
        else:
            dip_steps.append(step("Plate Region Detection", "No region found — using full image", "warn"))
            plate_for_ocr = preprocessed_gray

        ocr_result = run_ocr_pipeline(image_path, plate_for_ocr)
        raw_ocr_text, ocr_confidence, ocr_variant = parse_ocr_result(ocr_result)
        dip_steps.append(step("OCR Extraction",
                               f"Engine: {ocr_variant} | Raw: '{raw_ocr_text}' | Conf: {round(ocr_confidence*100,1)}%",
                               "ok" if raw_ocr_text else "warn"))

        logger.info(f"OCR extracted: '{raw_ocr_text}' (conf: {ocr_confidence:.2f})")

        # ── NLP pipeline ──────────────────────────────────────────
        nlp_preprocess = full_nlp_preprocess(raw_ocr_text)
        candidates     = nlp_preprocess["candidates"]
        best_candidate = nlp_preprocess["best_candidate"]
        nlp_steps.append(step("Text Cleaning",       f"Input: '{raw_ocr_text}'"))
        cand_labels = [c.get('plate','') if isinstance(c, dict) else str(c) for c in candidates[:4]]
        nlp_steps.append(step("Candidate Generation",
                               f"{len(candidates)} candidate(s): {', '.join(cand_labels)}"))

        if isinstance(best_candidate, dict):
            best_cand_plate = best_candidate.get("plate", raw_ocr_text)
            best_cand_type  = best_candidate.get("plate_type", "unknown")
            best_cand_valid = best_candidate.get("is_valid", False)
        else:
            best_cand_plate = str(best_candidate) if best_candidate else raw_ocr_text
            best_cand_type  = "unknown"
            best_cand_valid = False
        ner_result     = run_ner_pipeline(raw_ocr_text, candidates)
        extracted_plate = ner_result["best_plate"] or best_cand_plate
        nlp_steps.append(step("NER Plate Extraction",
                               f"Entities: {[e['text'] for e in ner_result['ner_entities']]}"))
        nlp_steps.append(step("Entity Disambiguation", f"Best plate: '{extracted_plate}'"))

        # Common OCR misread corrections for Indian plates
        extracted_plate = extracted_plate.replace('TH', 'TN') if extracted_plate.startswith('TH') else extracted_plate

        classification = classify_plate(extracted_plate) if extracted_plate else {"vehicle_type": "Unknown"}
        nlp_steps.append(step("Plate Classification",
                               f"Type: {classification.get('vehicle_type','Unknown')} | State: {classification.get('state','?')}"))

        exact_match  = db.check_plate(extracted_plate)
        all_plates   = [p["plate_number"] for p in db.get_all_plates()]
        verification = verify_plate(extracted_plate, all_plates)

        if exact_match:
            decision        = "ALLOW"
            vehicle_info    = exact_match
            final_plate     = extracted_plate
            final_confidence = min(1.0, ocr_confidence + 0.2)
            match_method    = "exact_db"
            nlp_steps.append(step("Database Lookup",   f"Exact match found: '{final_plate}'"))
            nlp_steps.append(step("BERT Verification", "Exact match — skipped fuzzy search"))
        elif verification["decision"] == "ALLOW" and verification["confidence"] >= 0.92:
            decision        = "ALLOW"
            final_plate     = verification["matched_plate"]
            vehicle_info    = db.check_plate(final_plate) or {}
            final_confidence = verification["confidence"]
            match_method    = "fuzzy_bert"
            nlp_steps.append(step("Database Lookup",   f"No exact match for '{extracted_plate}'", "warn"))
            nlp_steps.append(step("BERT Verification", f"Fuzzy match → '{final_plate}' ({round(verification['confidence']*100,1)}% conf)"))
        else:
            decision        = "DENY"
            vehicle_info    = {}
            final_plate     = extracted_plate
            final_confidence = verification.get("confidence", 0.0)
            match_method    = "no_match"
            nlp_steps.append(step("Database Lookup",   f"No match for '{extracted_plate}'", "warn"))
            nlp_steps.append(step("BERT Verification", f"Below threshold ({round(verification.get('confidence',0)*100,1)}%)", "error"))

        db.log_access(plate_number=final_plate, status=decision,
                      confidence=final_confidence, image_path=image_path)

        # ── Stage images ──────────────────────────────────────────
        stage_images = []
        try:
            stage_images.append({"label": "Original",          "data": image_to_base64(original_img)})
            stage_images.append({"label": "Grayscale",         "data": image_to_base64(cv2.cvtColor(dip_result["grayscale"], cv2.COLOR_GRAY2BGR))})
            stage_images.append({"label": "CLAHE Enhanced",    "data": image_to_base64(cv2.cvtColor(dip_result["clahe"],     cv2.COLOR_GRAY2BGR))})
            stage_images.append({"label": "Frequency Filtered","data": image_to_base64(cv2.cvtColor(freq_enhanced,           cv2.COLOR_GRAY2BGR))})
            if plate_crop is not None and plate_crop.size > 0:
                pc = plate_crop if len(plate_crop.shape) == 3 else cv2.cvtColor(plate_crop, cv2.COLOR_GRAY2BGR)
                stage_images.append({"label": "Plate Crop", "data": image_to_base64(pc)})
        except Exception as e:
            logger.warning(f"Stage image prep failed: {e}")

        return jsonify({
            "success":        True,
            "decision":       decision,
            "plate_number":   final_plate,
            "extracted_plate": final_plate,
            "ocr_raw":        raw_ocr_text,
            "ocr_confidence": round(ocr_confidence * 100, 1),
            "match_method":   match_method,
            "confidence":     round(final_confidence * 100, 1),
            "final_confidence": round(final_confidence * 100, 1),
            "vehicle_info":   vehicle_info,
            "classification": classification,
            "dip_steps":      dip_steps,
            "nlp_steps":      nlp_steps,
            "stage_images":   stage_images,
        })

    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        return jsonify({
            "success":   False,
            "error":     str(e),
            "dip_steps": dip_steps,
            "nlp_steps": nlp_steps,
        }), 500

    finally:
        if image_path != raw_path and os.path.exists(raw_path):
            try:
                os.remove(raw_path)
            except Exception:
                pass


@app.route('/api/plates', methods=['GET'])
def get_plates():
    plates = db.get_all_plates()
    return jsonify({"plates": plates, "count": len(plates)})


@app.route('/api/plates', methods=['POST'])
def add_plate():
    data     = request.json
    required = ['plate_number', 'owner_name', 'vehicle_type', 'vehicle_model', 'color']
    if not all(k in data for k in required):
        return jsonify({"error": f"Required fields: {required}"}), 400
    success = db.add_plate(
        data['plate_number'], data['owner_name'],
        data['vehicle_type'], data['vehicle_model'], data['color']
    )
    if success:
        return jsonify({"success": True, "message": f"Plate {data['plate_number']} registered."})
    else:
        return jsonify({"error": "Plate already registered or DB error"}), 409


@app.route('/api/logs', methods=['GET'])
def get_logs():
    limit = int(request.args.get('limit', 20))
    logs  = db.get_logs(limit)
    return jsonify({"logs": logs, "count": len(logs)})


@app.route('/api/manual-verify', methods=['POST'])
def manual_verify():
    data  = request.json
    plate = data.get('plate_number', '').upper().strip()
    if not plate:
        return jsonify({"error": "plate_number required"}), 400

    nlp_steps = []

    nlp_result     = full_nlp_preprocess(plate)
    classification = classify_plate(plate)
    nlp_steps.append(step("Text Cleaning",       f"Input: '{plate}'"))
    nlp_steps.append(step("Plate Classification",
                           f"Type: {classification.get('vehicle_type','Unknown')} | State: {classification.get('state','?')}"))

    exact      = db.check_plate(plate)
    all_plates = [p["plate_number"] for p in db.get_all_plates()]
    verification = verify_plate(plate, all_plates)

    if exact:
        decision     = "ALLOW"
        vehicle_info = exact
        confidence   = 100.0
        nlp_steps.append(step("Database Lookup",   f"Exact match found: '{plate}'"))
        nlp_steps.append(step("BERT Verification", "Exact match — skipped"))
    elif verification["decision"] == "ALLOW":
        decision     = "ALLOW"
        vehicle_info = db.check_plate(verification["matched_plate"]) or {}
        confidence   = round(verification["confidence"] * 100, 1)
        nlp_steps.append(step("Database Lookup",   f"No exact match", "warn"))
        nlp_steps.append(step("BERT Verification", f"Fuzzy match → '{verification['matched_plate']}' ({confidence}%)"))
    else:
        decision     = "DENY"
        vehicle_info = {}
        confidence   = round(verification.get("confidence", 0) * 100, 1)
        nlp_steps.append(step("Database Lookup",   "No match found", "warn"))
        nlp_steps.append(step("BERT Verification", f"Below threshold ({confidence}%)", "error"))

    db.log_access(plate, decision, confidence / 100)

    return jsonify({
        "success":         True,
        "decision":        decision,
        "plate_number":    plate,
        "extracted_plate": plate,
        "final_confidence": confidence,
        "confidence":      confidence,
        "vehicle_info":    vehicle_info,
        "classification":  classification,
        "dip_steps":       [],
        "nlp_steps":       nlp_steps,
        "stage_images":    [],
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
