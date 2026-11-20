@echo off
title NephroScan AI - Flask Deep Learning Backend
color 0A
cd /d "%~dp0"
echo ===============================================================================
echo     NEPHROSCAN AI - DEEP LEARNING BACKEND (FLASK + YOLOv11 + SAM)
echo     Department of Computer Science - University of Baltistan, Skardu (UOBS)
echo ===============================================================================
echo.
echo Checking Python environment...
if not exist ".\venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .\venv\Scripts\python.exe
    pause
    exit /b 1
)

echo Starting Flask API Server on http://127.0.0.1:5000 ...
echo Loading MobileNetV3 Gatekeeper, YOLOv11, and Meta SAM ViT-B...
echo.
".\venv\Scripts\python.exe" app.py
pause
