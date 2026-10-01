@echo off
title Handwritten Digit Recognition Web App
cd /d "%~dp0"
echo ====================================================
echo  Starting Handwritten Digit Recognition Web App...
echo ====================================================
echo.
echo  Please wait while the AI models are loading...
echo  This may take 10-15 seconds on first launch.
echo.

REM Start Flask server in the background
start /B python app.py

REM Wait 6 seconds for Flask + models to fully load
timeout /t 6 /nobreak > nul

REM Now open the browser
echo  Opening browser at http://localhost:5000 ...
start http://localhost:5000

echo.
echo ====================================================
echo  App is running. Close this window to STOP the app.
echo ====================================================
echo.

REM Keep the window open (keeps Flask running)
python app.py
