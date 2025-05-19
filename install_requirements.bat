@echo off
echo Installing required packages for Voice Cloning Application...
python -m pip install --upgrade pip
python -m pip install torch numpy sounddevice soundfile librosa matplotlib
if %ERRORLEVEL% NEQ 0 (
  echo Error installing packages. Please check your Python installation.
  echo Make sure Python is added to your PATH.
) else (
  echo All packages installed successfully.
)
echo.
echo Press any key to exit...
pause >nul 