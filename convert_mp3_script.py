"""
Helper script to prepare an MP3 file for voice cloning
This will convert your MP3 to the proper format and place it in the dataset/target folder
"""

import os
import sys
import librosa
import soundfile as sf
import argparse

def process_mp3(mp3_path, output_dir):
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the MP3 file
        print(f"Loading {mp3_path}...")
        audio_data, sample_rate = librosa.load(mp3_path, sr=None)
        
        # Get the filename without extension
        filename = os.path.splitext(os.path.basename(mp3_path))[0]
        
        # Save as WAV file in output directory
        output_path = os.path.join(output_dir, f"{filename}.wav")
        print(f"Converting to WAV and saving to {output_path}...")
        sf.write(output_path, audio_data, sample_rate)
        
        print("Conversion complete! Your file is ready for voice cloning.")
        print("\nNext steps:")
        print("1. Run the voice cloning application with: python voice_cloning_app.py")
        print("2. In the Data Collection tab, set the Target Voice path to:", output_dir)
        print("3. Click 'Import Target Voice Data'")
        print("4. Continue with recording your source voice or using existing recordings")
        
        return True
    except Exception as e:
        print(f"Error processing MP3 file: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MP3 to WAV for voice cloning")
    parser.add_argument("mp3_file", help="Path to the MP3 file you want to use for voice cloning")
    args = parser.parse_args()
    
    mp3_path = args.mp3_file
    output_dir = os.path.join(os.getcwd(), "dataset", "target")
    
    if not os.path.exists(mp3_path):
        print(f"Error: The file {mp3_path} does not exist.")
        sys.exit(1)
    
    success = process_mp3(mp3_path, output_dir)
    if not success:
        print("Conversion failed. Please check the error message above.")
        sys.exit(1) 