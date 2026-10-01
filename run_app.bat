@echo off
title Handwritten Digit Recognition Web App
cd /d "%~dp0"
echo ====================================================
echo Starting Handwritten Digit Recognition Web App...
echo ====================================================
start http://localhost:5000
python app.py
pause
