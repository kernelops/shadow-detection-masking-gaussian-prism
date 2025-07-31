@echo off
echo Installing Shadow Detection and Removal Project...
echo.

echo Checking Python installation...
python --version
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.7 or higher from https://python.org
    pause
    exit /b 1
)

echo.
echo Installing dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo Error: Failed to install dependencies
    echo Please check your internet connection and try again
    pause
    exit /b 1
)

echo.
echo Installation completed successfully!
echo.
echo To run the project:
echo 1. Place your input image as 'shadow.jpg' in this directory
echo 2. Run: python main.py
echo.
pause 