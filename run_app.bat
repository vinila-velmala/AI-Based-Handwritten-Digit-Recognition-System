@echo off
title AI Handwritten Digit Recognition Web App
cd /d "%~dp0"
echo ====================================================
echo  Starting AI Handwritten Digit Recognition Web App
echo ====================================================
echo.
echo  Please wait... Loading TensorFlow and AI models.
echo  Your browser will open automatically once ready!
echo.
echo  NOTE: Do NOT close this window while using the app.
echo ====================================================
echo.
python app.py
pause
