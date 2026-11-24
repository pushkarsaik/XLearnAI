@echo off
REM X-Learn AI - run the Grad-CAM checks. Results in gradcam_check\grid.png
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run setup.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" verify_gradcam.py --n 10
echo.
echo Opening the picture...
start "" "gradcam_check\grid.png"
pause
