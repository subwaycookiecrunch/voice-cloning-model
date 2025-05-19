@echo off
echo Starting Voice Cloning Application...
python "%~dp0voice_cloning_app.py"
if %ERRORLEVEL% NEQ 0 (
  echo Error starting application. Please make sure Python and required packages are installed.
  echo Press any key to exit...
  pause >nul
) 