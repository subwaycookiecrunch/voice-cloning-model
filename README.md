# Voice Cloning Application

A desktop application for voice cloning and real-time voice conversion using Retrieval-based Voice Conversion (RVC) technology.

## Features

- **Voice Data Collection**: Record and manage audio samples for training
- **Model Training**: Train custom voice conversion models
- **Voice Conversion**: Convert audio files or recordings using trained models
- **Real-time Conversion**: Use your voice models in real-time with Discord and other applications

## Quick Start

1. Double-click `run_voice_cloner.bat` to start the application
2. For real-time Discord voice conversion:
   - Install a virtual audio cable like [VB-Cable](https://vb-audio.com/Cable/)
   - Go to the "Real-time/Discord" tab
   - Select your microphone as input device
   - Select your virtual cable as output device
   - Load a trained model or create one
   - Click "Start Voice Conversion"
   - Set your Discord input to the virtual cable

## Requirements

- Python 3.7 or higher
- Required Python packages:
  - torch
  - numpy
  - sounddevice
  - soundfile
  - tkinter
  - librosa
  - matplotlib

## Installation (For Developers)

If the launcher doesn't work, you can set up the environment manually:

```bash
# Install required packages
pip install torch numpy sounddevice soundfile librosa matplotlib

# Launch the application
python voice_cloning_app.py
```

## Troubleshooting

If you encounter issues:

1. Make sure you have Python installed and added to PATH
2. Check that all required packages are installed
3. For audio issues, ensure your audio devices are properly configured
4. For real-time conversion, a virtual audio cable is required 