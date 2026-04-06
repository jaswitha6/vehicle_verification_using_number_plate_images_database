@echo off
echo.
echo  VeriPlate - Vehicle Verification System
echo  Setting up environment...
echo.

REM Step 1: Create venv
python -m venv venv
call venv\Scripts\activate

REM Step 2: Upgrade pip
pip install --upgrade pip

REM Step 3: Install requirements
pip install -r requirements.txt

REM Step 4: spaCy model
python -m spacy download en_core_web_sm

REM Step 5: Init DB
python database.py

echo.
echo  Setup complete!
echo  Run: python app.py
echo  Then open: http://127.0.0.1:5000
echo.
pause
