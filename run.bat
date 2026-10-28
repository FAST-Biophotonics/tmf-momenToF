@echo off
cd /d "%~dp0"
python -c "import PySide6, pyqtgraph, scipy, matplotlib" 2>nul
if errorlevel 1 (
  echo Installing packages - first run only...
  python -m pip install -r requirements.txt
)
if exist tmf_momenTOF.py (
  python tmf_momenTOF.py
) else (
  python app.py
)
if errorlevel 1 pause
