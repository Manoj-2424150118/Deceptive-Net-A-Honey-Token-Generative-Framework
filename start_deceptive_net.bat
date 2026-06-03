@echo off
setlocal enabledelayedexpansion
title Deceptive-Net Boot Sequence

color 0d

echo.
echo    ================================================================
echo    Deceptive-Net ^| Tactical Financial Fraud Detection Platform
echo    ================================================================
echo.

echo [+] Initializing secure boot sequence...
ping 127.0.0.1 -n 2 > nul

echo [+] Verifying Docker daemon status...
docker info >nul 2>&1
if %errorlevel% neq 0 (
    color 0c
    echo [!] CRITICAL: Docker daemon not found or not running.
    echo [!] Attempting to start Docker Desktop...
    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    echo [!] Waiting for Docker to start ^(this may take up to 30 seconds^)...
    goto check_docker
) else (
    echo [+] Docker daemon is running.
    goto docker_ready
)

:check_docker
ping 127.0.0.1 -n 6 > nul
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [.] Still waiting for Docker...
    goto check_docker
)
color 0d
echo [+] Docker daemon connected successfully.

:docker_ready
echo.
echo [+] Compiling containers (this may take a few minutes)...
docker-compose build
if %errorlevel% neq 0 (
    color 0c
    echo [!] CRITICAL: Build failed. Check logs above.
    pause
    exit /b %errorlevel%
)

echo [+] Launching Deceptive-Net nodes...
docker-compose up -d

echo.
echo [+] Establishing secure channel to Backend API (Port 8000)...
:check_backend
ping 127.0.0.1 -n 3 > nul
curl -s http://localhost:8000/docs >nul
if %errorlevel% neq 0 (
    echo [.] Waiting for backend handshake...
    goto check_backend
)
echo [+] Backend API online.

echo [+] Establishing secure channel to Frontend UI (Port 8080)...
:check_frontend
ping 127.0.0.1 -n 3 > nul
curl -s http://localhost:8080/ >nul
if %errorlevel% neq 0 (
    echo [.] Waiting for frontend handshake...
    goto check_frontend
)
echo [+] Frontend UI online.

color 0a
echo.
echo    ================================================================
echo    [SYSTEM SECURE] Deceptive-Net is fully operational.
echo    ================================================================
echo.
echo    Dashboard: http://localhost:8080
echo    API Docs:  http://localhost:8000/docs
echo.
echo    Credentials:
echo      Admin:   admin / admin123
echo      Analyst: analyst / analyst123
echo.
echo [+] Opening secure dashboard in default browser...
start http://localhost:8080

echo.
echo Press any key to stop services and exit...
pause >nul

echo [+] Shutting down nodes...
docker-compose down
echo [+] Disconnected.
exit /b 0
