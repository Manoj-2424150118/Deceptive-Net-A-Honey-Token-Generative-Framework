@echo off
setlocal EnableDelayedExpansion

title Deceptive-Net Docker Deployment
color 0A
mode con: cols=120 lines=40

:: =========================================================
:: CONFIGURATION
:: =========================================================
set PROJECT_NAME=deceptive-net
set IMAGE_NAME=deceptive-net-image
set CONTAINER_NAME=deceptive-net-container
set PORT=7860
set PYTHON_VERSION=3.10

:: =========================================================
:: HEADER
:: =========================================================
cls
echo ================================================================
echo               DECEPTIVE-NET AUTO DEPLOYMENT
echo ================================================================
echo.

:: =========================================================
:: CHECK PYTHON
:: =========================================================
echo [1/9] Checking Python installation...
python --version >nul 2>&1

if errorlevel 1 (
    echo.
    echo Python not found.
    echo Installing Python...
    winget install -e --id Python.Python.3.10

    echo.
    echo Please restart terminal after Python installation.
    pause
    exit
)

echo Python detected.
echo.

:: =========================================================
:: UPDATE PIP
:: =========================================================
echo [2/9] Updating pip...
python -m pip install --upgrade pip
echo.

:: =========================================================
:: CREATE REQUIREMENTS.TXT
:: =========================================================
echo [3/9] Creating requirements.txt...

(
echo numpy
echo pandas
echo matplotlib
echo scipy
echo scikit-learn
echo torch
echo torchvision
echo torchaudio
echo flask
echo fastapi
echo uvicorn
echo gradio
) > requirements.txt

echo requirements.txt created.
echo.

:: =========================================================
:: INSTALL DEPENDENCIES
:: =========================================================
echo [4/9] Installing dependencies...
pip install -r requirements.txt
echo.

:: =========================================================
:: CHECK DOCKER
:: =========================================================
echo [5/9] Checking Docker Desktop...

docker --version >nul 2>&1

if errorlevel 1 (
    echo.
    echo Docker Desktop not found.
    echo Installing Docker Desktop...

    winget install -e --id Docker.DockerDesktop

    echo.
    echo ====================================================
    echo Docker installed successfully.
    echo Restart your system if Docker does not start.
    echo ====================================================
    echo.

    pause
)

:: =========================================================
:: START DOCKER
:: =========================================================
echo [6/9] Starting Docker Desktop...

start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"

echo Waiting for Docker to initialize...
timeout /t 20 >nul

docker info >nul 2>&1

if errorlevel 1 (
    echo.
    echo Docker is still not running.
    echo Open Docker Desktop manually and retry.
    pause
    exit
)

echo Docker is running.
echo.

:: =========================================================
:: CREATE DOCKERFILE
:: =========================================================
echo [7/9] Creating Dockerfile...

(
echo FROM python:3.10
echo.
echo WORKDIR /app
echo.
echo COPY . /app
echo.
echo RUN pip install --upgrade pip
echo RUN pip install -r requirements.txt
echo.
echo EXPOSE 7860
echo.
echo CMD ["python", "main.py", "--mode", "full"]
) > Dockerfile

echo Dockerfile created.
echo.

:: =========================================================
:: BUILD IMAGE
:: =========================================================
echo [8/9] Building Docker image...

docker build -t %IMAGE_NAME% .

if errorlevel 1 (
    echo.
    echo ====================================================
    echo Docker image build failed.
    echo ====================================================
    pause
    exit
)

echo Docker image built successfully.
echo.

:: =========================================================
:: REMOVE OLD CONTAINER
:: =========================================================
docker rm -f %CONTAINER_NAME% >nul 2>&1

:: =========================================================
:: RUN CONTAINER
:: =========================================================
echo [9/9] Launching container...

docker run -dit ^
--name %CONTAINER_NAME% ^
-p %PORT%:%PORT% ^
%IMAGE_NAME%

if errorlevel 1 (
    echo.
    echo ====================================================
    echo Failed to start Docker container.
    echo ====================================================
    pause
    exit
)

echo.
echo ================================================================
echo                 DEPLOYMENT SUCCESSFUL
echo ================================================================
echo.

echo Container Name:
echo %CONTAINER_NAME%
echo.

echo Docker Image:
echo %IMAGE_NAME%
echo.

echo Local Access:
echo http://localhost:%PORT%
echo.

echo Source File:
echo main.py
echo.

echo ================================================================
echo.

:: =========================================================
:: OPTIONAL BROWSER OPEN
:: =========================================================
start http://localhost:%PORT%

:: =========================================================
:: SHOW LIVE LOGS
:: =========================================================
echo Showing live container logs...
echo Press CTRL + C to stop viewing logs.
echo.

docker logs -f %CONTAINER_NAME%

pause