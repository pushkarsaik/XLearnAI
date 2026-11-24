@echo off
REM ============================================================
REM  X-Learn AI - one-time setup. Double-click this file.
REM  Creates a private Python environment (.venv) in this folder
REM  and installs everything the app needs into it.
REM ============================================================
cd /d "%~dp0"

echo.
echo [1/4] Checking Python...
python --version
if errorlevel 1 (
    echo Python was not found. Install Python 3.12 or 3.13 from python.org and run this again.
    pause
    exit /b 1
)

if exist ".venv\Scripts\python.exe" (
    echo [2/4] Environment already exists, reusing it.
) else (
    echo [2/4] Creating environment in .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo Could not create the environment.
        pause
        exit /b 1
    )
)

echo [3/4] Upgrading pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip

echo [4/4] Installing TensorFlow, Streamlit and the rest (about 500 MB, takes a few minutes)...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Installation failed. Take a screenshot of the error above and send it.
    pause
    exit /b 1
)

echo.
echo Checking that everything loads...
".venv\Scripts\python.exe" -c "import tensorflow as tf, streamlit, PIL, numpy; print('TensorFlow', tf.__version__, '- OK')"
if errorlevel 1 (
    echo The check failed. Take a screenshot and send it.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Setup finished. Double-click run_app.bat to start the app.
echo ============================================================
pause
