@echo off
setlocal enabledelayedexpansion
title Deceptive-Net: Automated Honey Token Test

echo =================================================================
echo   Deceptive-Net: Automated Honey Token Exfiltration Test
echo =================================================================
echo.

:: STEP 1: Authenticate as Analyst
echo [1/5] Authenticating as Rogue Analyst...
for /f "delims=" %%I in ('powershell -NoProfile -Command "(Invoke-RestMethod -Uri 'http://localhost:8000/auth/token' -Method Post -Body 'username=analyst&password=analyst123' -ContentType 'application/x-www-form-urlencoded').access_token" 2^>nul') do set ANALYST_TOKEN=%%I

if "!ANALYST_TOKEN!"=="" (
    echo [!] ERROR: Failed to get analyst token. Make sure the backend is running on port 8000!
    pause
    exit /b 1
)
echo [+] Analyst token acquired.

echo.
:: STEP 2: Download the Data
echo [2/5] Simulating Data Exfiltration (Downloading stolen_data.csv)...
powershell -NoProfile -Command "Invoke-RestMethod -Uri 'http://localhost:8000/api/transactions/export' -Headers @{Authorization='Bearer !ANALYST_TOKEN!'} -OutFile 'stolen_data.csv'"
echo [+] Data successfully exported to stolen_data.csv.

echo.
:: STEP 3: Inspect the Stolen Data
echo [3/5] Inspecting stolen_data.csv for the Honey Token...
set HONEY_TOKEN_ROW=
for /f "delims=" %%I in ('findstr "TOR_EXIT_NODE_HONEYPOT" stolen_data.csv') do set HONEY_TOKEN_ROW=%%I

if "!HONEY_TOKEN_ROW!"=="" (
    echo [!] ERROR: No Honey Token found in the exported data!
    pause
    exit /b 1
)

echo [+] HONEY TOKEN INJECTED SUCCESSFULLY. Found in CSV:
echo     !HONEY_TOKEN_ROW!

:: Extract the Watermark ID from the row (HTXN-XXXXXXXX)
for /f "tokens=1 delims=," %%A in ("!HONEY_TOKEN_ROW!") do set TXN_ID=%%A
set WATERMARK=!TXN_ID:HTXN-=!
echo [+] Extracted Watermark ID: !WATERMARK!

echo.
:: STEP 4: Authenticate as Admin
echo [4/5] Authenticating as System Admin to investigate...
for /f "delims=" %%I in ('powershell -NoProfile -Command "(Invoke-RestMethod -Uri 'http://localhost:8000/auth/token' -Method Post -Body 'username=admin&password=admin123' -ContentType 'application/x-www-form-urlencoded').access_token" 2^>nul') do set ADMIN_TOKEN=%%I
echo [+] Admin token acquired.

echo.
:: STEP 5: Verify the Audit Logs
echo [5/5] Checking Admin Audit Alerts for the matching Watermark...
powershell -NoProfile -Command "$logs = Invoke-RestMethod -Uri 'http://localhost:8000/api/audit/alerts' -Headers @{Authorization='Bearer !ADMIN_TOKEN!'}; $trap = $logs | Where-Object { $_.detail -match '!WATERMARK!' }; if ($trap) { Write-Host '================================================================' -ForegroundColor Cyan; Write-Host '[+] THE TRAP WORKED! The internal system caught the exfiltration.' -ForegroundColor Green; Write-Host '================================================================' -ForegroundColor Cyan; Write-Host 'Target Identified: ' -NoNewline; Write-Host $trap.actor -ForegroundColor Yellow; Write-Host 'Action Tracked   : ' -NoNewline; Write-Host $trap.action -ForegroundColor Red; Write-Host 'IP Address       : ' -NoNewline; Write-Host $trap.ip_address -ForegroundColor Yellow; Write-Host 'Evidence Match   : ' -NoNewline; Write-Host $trap.detail -ForegroundColor Yellow; } else { Write-Host '[-] Watermark not found in logs.' -ForegroundColor Red }"

echo.
echo =================================================================
echo   Test Complete. Cleaning up test files...
echo =================================================================
del stolen_data.csv
pause
