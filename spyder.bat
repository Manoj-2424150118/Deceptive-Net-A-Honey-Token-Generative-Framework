@echo off
title Deceptive-Net Main Executor
color 0A

echo ============================================================
echo              DECEPTIVE-NET EXECUTION SYSTEM
echo ============================================================
echo.

:: ------------------------------------------------------------
:: CHECK PYTHON
:: ------------------------------------------------------------
echo [1/7] Checking Python...

python --version >nul 2>&1

IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Python not installed.
    pause
    exit /b
)

python --version
echo.

:: ------------------------------------------------------------
:: CREATE VENV IF NOT EXISTS
:: ------------------------------------------------------------
echo [2/7] Checking virtual environment...

IF NOT EXIST venv (
    echo Creating virtual environment...
    python -m venv venv --copies
)

echo Virtual environment ready.
echo.

:: ------------------------------------------------------------
:: ACTIVATE VENV
:: ------------------------------------------------------------
echo [3/7] Activating virtual environment...

call venv\Scripts\activate.bat

IF %ERRORLEVEL% NEQ 0 (
    echo Failed activating venv.
    pause
    exit /b
)

echo Environment activated.
echo.

:: ------------------------------------------------------------
:: CHECK PIP
:: ------------------------------------------------------------
echo [4/7] Repairing pip...

python -m ensurepip --upgrade >nul 2>&1
python -m pip install --upgrade pip wheel
python -m pip install setuptools==81.0.0

echo Pip ready.
echo.

:: ------------------------------------------------------------
:: INSTALL LIBRARIES
:: ------------------------------------------------------------
echo [5/7] Checking required libraries...

python -m pip install ^
numpy ^
pandas ^
matplotlib ^
scipy ^
scikit-learn ^
torch ^
torchvision ^
torchaudio

echo.
echo Libraries ready.
echo.

:: ------------------------------------------------------------
:: RUN MAIN.PY
:: ------------------------------------------------------------
echo [6/7] Running main.py ...
echo ============================================================
echo.

python main.py

echo.
echo ============================================================
echo main.py execution completed.
echo ============================================================
echo.

:: ------------------------------------------------------------
:: KEEP WINDOW OPEN
:: ------------------------------------------------------------
echo [7/7] Process finished.
pause