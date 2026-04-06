# VeriPlate — Vehicle Verification Using Number Plate Database

> **Integrated DIP + NLP System** | Python · Flask · PaddleOCR · BERT · OpenCV

A full-stack vehicle access control system combining **Digital Image Processing** and **Natural Language Processing** to detect, extract, and verify Indian number plates in real-time.

---

## Project Overview

| Aspect | Detail |
|---|---|
| **Dataset** | 174 HEIC/JPG images from iPhone 12 Pro (Christ University, Kengeri Campus) |
| **Resolution** | 4032×3024 px (iPhone default) |
| **OCR Engine** | PaddleOCR (offline, no API key) |
| **Backend** | Python 3.11 + Flask |
| **Frontend** | HTML + CSS + Vanilla JS |
| **Database** | SQLite |

---

## DIP Concepts Used

| Concept | Module | Syllabus Ref |
|---|---|---|
| Bicubic interpolation | `preprocessor.py` | U1-T1 |
| Gray-level transformation | `preprocessor.py` | U1-T2 |
| Histogram equalization (CLAHE) | `preprocessor.py` | U1-T3 |
| Spatial filtering (Gaussian, Median) | `preprocessor.py` | U1-T4 / U2-T1 |
| Sharpening (Unsharp mask, Laplacian) | `preprocessor.py` | U2-T1 |
| DFT Low/High pass filtering | `frequency.py` | U2-T2 / U2-T4 |
| DCT enhancement | `frequency.py` | U2-T2 |
| Wavelet denoising (Haar, filter banks) | `frequency.py` | U2-T4 |
| Contour-based segmentation | `segmentation.py` | U3-T3 |
| Multi-level segmentation | `segmentation.py` | U3-T4 |
| Character recognition (OCR) | `ocr_engine.py` | U5-T4 |

## NLP Concepts Used

| Concept | Module | Syllabus Ref |
|---|---|---|
| Text cleanup + NLP pipeline | `text_cleaner.py` | U1-T4 / U1-T3 |
| Heuristics-based NLP | `text_cleaner.py` | U1-T2 |
| Regex + key phrase extraction | `plate_ner.py` | U4-T1 |
| Named Entity Recognition (spaCy) | `plate_ner.py` | U4-T2 |
| Entity disambiguation & linking | `plate_ner.py` | U4-T3 |
| Bag of Words / N-grams | `classifier.py` | U2-T1 |
| TF-IDF vectorization | `classifier.py` | U2-T2 |
| Text classification pipeline | `classifier.py` | U3-T1 |
| Character embeddings (cosine sim) | `bert_verifier.py` | U3-T2 / U2-T4 |
| BERT-style fuzzy verification | `bert_verifier.py` | U5-T3 / U5-T4 |

---

## Quick Start

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
cd backend

# macOS / Linux
bash setup.sh

# Windows
setup.bat
```

### Run

```bash
cd backend
source venv/bin/activate    # Windows: venv\Scripts\activate
python app.py
```

Then open: **http://127.0.0.1:5000**

---

## Folder Structure

```
vehicle-verification/
├── backend/
│   ├── app.py                  # Flask server
│   ├── config.py               # Settings
│   ├── database.py             # SQLite handler
│   ├── requirements.txt
│   ├── setup.sh / setup.bat
│   ├── dip/                    # DIP Module
│   │   ├── preprocessor.py     # Gray, CLAHE, histogram, filtering
│   │   ├── frequency.py        # DFT, DCT, Wavelets
│   │   ├── segmentation.py     # Plate region detection
│   │   └── ocr_engine.py       # PaddleOCR integration
│   ├── nlp/                    # NLP Module
│   │   ├── text_cleaner.py     # Preprocessing pipeline
│   │   ├── plate_ner.py        # spaCy NER + regex NER
│   │   ├── classifier.py       # TF-IDF vehicle classifier
│   │   └── bert_verifier.py    # BERT fuzzy verification
│   └── utils/
│       ├── heic_converter.py   # iPhone HEIC → JPG
│       └── logger.py
├── frontend/
│   ├── index.html              # UI
│   ├── style.css               # Dark cyberpunk theme
│   └── app.js                  # JS logic
├── notebooks/
│   ├── 01_dip_pipeline.ipynb   # DIP experiments
│   └── 02_nlp_pipeline.ipynb   # NLP experiments
└── dataset/                    # Your 174 images here
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/verify` | Upload image → full pipeline → decision |
| `POST` | `/api/manual-verify` | Manually enter plate → NLP + DB check |
| `GET` | `/api/plates` | List all registered vehicles |
| `POST` | `/api/plates` | Add new vehicle to registry |
| `GET` | `/api/logs` | Recent access logs |
| `GET` | `/api/health` | Health check |

---

## Pipeline Flow

```
Image Upload (HEIC/JPG)
        │
        ▼
  [DIP PHASE]
  ┌─────────────────────────────────────┐
  │ 1. HEIC → JPG conversion           │
  │ 2. Resize (bicubic interpolation)  │
  │ 3. Gamma correction                 │
  │ 4. Grayscale transformation         │
  │ 5. Median filter (noise removal)    │
  │ 6. CLAHE histogram equalization     │
  │ 7. Unsharp mask sharpening          │
  │ 8. DFT low-pass + DCT enhance       │
  │ 9. Wavelet denoising (Haar)         │
  │ 10. Contour segmentation → crop     │
  │ 11. PaddleOCR                       │
  └─────────────────────────────────────┘
        │ raw_text
        ▼
  [NLP PHASE]
  ┌─────────────────────────────────────┐
  │ 12. Text cleanup + regex extract    │
  │ 13. OCR error correction            │
  │ 14. spaCy NER (NUMBER_PLATE entity) │
  │ 15. Entity disambiguation           │
  │ 16. TF-IDF classification           │
  │ 17. Format validation               │
  │ 18. BERT/fuzzy DB verification      │
  │ 19. ALLOW / DENY decision           │
  └─────────────────────────────────────┘
        │
        ▼
    Response → Frontend UI
```

---

## Resume Bullet Points

```
• Built a full-stack vehicle access control system integrating Digital Image Processing 
  (DFT, CLAHE, wavelet denoising, segmentation) and NLP (NER, TF-IDF classification, 
  BERT-based fuzzy verification) using Python, Flask, and PaddleOCR on a 174-image 
  real-world dataset of Indian number plates.

• Designed a 19-step dual pipeline (DIP + NLP) achieving plate text extraction and 
  ALLOW/DENY decisions with OCR-error-resilient fuzzy matching, deployed as a 
  REST API with a responsive frontend interface.
```

---

## Tech Stack

- **Python 3.11** — Core language
- **OpenCV** — Image processing
- **PaddleOCR** — Offline OCR (no API key)
- **PyWavelets** — Wavelet transforms
- **spaCy** — NER pipeline
- **HuggingFace Transformers** — BERT
- **scikit-learn** — TF-IDF + classification
- **NLTK** — Text preprocessing
- **Flask** — REST API
- **SQLite** — Vehicle registry
- **pillow-heif** — iPhone HEIC support
