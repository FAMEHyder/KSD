@echo off
title NephroScan AI - System Launcher
color 0B
cd /d "%~dp0"

echo ===============================================================================
echo     NEPHROSCAN AI - FULL STACK LAUNCHER
echo     YOLOv11m + Meta SAM ViT-B + MobileNetV3 Gatekeeper
echo     Department of Computer Science - University of Baltistan, Skardu (UOBS)
echo ===============================================================================
echo.
echo [1/2] Launching Flask Deep Learning Backend on http://127.0.0.1:5000 ...
start "NephroScan AI - Flask Backend (Port 5000)" cmd /k "cd /d ""%~dp0Backend"" && start_backend.bat"

echo.
echo Waiting 3 seconds for Backend to initialize...
timeout /t 3 /nobreak >nul

echo.
echo [2/2] Launching React Vite Frontend Workstation on http://localhost:3000 ...
start "NephroScan AI - React Frontend (Port 3000)" cmd /k "cd /d ""%~dp0Frontend"" && npm run dev"

echo.
echo ===============================================================================
echo Both Backend (Port 5000) and Frontend (Port 3000) are starting up!
echo Your default browser will open automatically once Vite starts.
echo ===============================================================================
echo.
pause
