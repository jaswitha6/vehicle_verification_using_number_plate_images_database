#!/bin/bash
# ============================================================
# VeriPlate — Automated Setup Script
# Run: bash setup.sh
# ============================================================

echo ""
echo "  ██╗   ██╗███████╗██████╗ ██╗██████╗ ██╗      █████╗ ████████╗███████╗"
echo "  ██║   ██║██╔════╝██╔══██╗██║██╔══██╗██║     ██╔══██╗╚══██╔══╝██╔════╝"
echo "  ██║   ██║█████╗  ██████╔╝██║██████╔╝██║     ███████║   ██║   █████╗  "
echo "  ╚██╗ ██╔╝██╔══╝  ██╔══██╗██║██╔═══╝ ██║     ██╔══██║   ██║   ██╔══╝  "
echo "   ╚████╔╝ ███████╗██║  ██║██║██║     ███████╗██║  ██║   ██║   ███████╗"
echo "    ╚═══╝  ╚══════╝╚═╝  ╚═╝╚═╝╚═╝     ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝"
echo ""
echo "  Vehicle Verification Using Number Plate Database"
echo "  Integrating DIP + NLP | Python + Flask"
echo ""

set -e

# Step 1: Python check
echo "[1/6] Checking Python..."
python3 --version || { echo "ERROR: Python3 not found"; exit 1; }

# Step 2: Virtual environment
echo "[2/6] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Step 3: Pip upgrade
echo "[3/6] Upgrading pip..."
pip install --upgrade pip -q

# Step 4: Install requirements
echo "[4/6] Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt

# Step 5: Download spaCy model
echo "[5/6] Downloading spaCy English model..."
python3 -m spacy download en_core_web_sm

# Step 6: Init database
echo "[6/6] Initializing database..."
python3 database.py

echo ""
echo "  ✅ Setup complete!"
echo ""
echo "  To start the server:"
echo "    source venv/bin/activate"
echo "    python app.py"
echo ""
echo "  Then open: http://127.0.0.1:5000"
echo ""
