"""
Voice Cloning Application using RVC (Retrieval-based Voice Conversion)
This script provides a local GUI application for training and using voice conversion models.
"""

import os
import sys
import time
import torch
import numpy as np
import sounddevice as sd
import soundfile as sf
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import librosa
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import queue
import warnings
warnings.filterwarnings("ignore")

# Import the real-time voice converter
try:
    from realtime_voice_converter import RealtimeConverterGUI
except ImportError:
    print("Realtime voice converter module not found. Real-time conversion will be disabled.")
    RealtimeConverterGUI = None

# Placeholder for RVC specific imports
# In a real implementation, you would need to install and import the RVC library
# These are placeholders to demonstrate the structure
class RVCModel:
    def __init__(self):
        self.model = None
        self.config = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
    
    def extract_features(self, audio_path, output_dir):
        """Extract features from audio files for training"""
        print(f"Extracting features from {audio_path} to {output_dir}")
        # In a real implementation, this would call the feature extraction 
        # functions from the RVC library
        return True
    
    def train(self, dataset_path, model_name, epochs=100):
        """Train the RVC model with the provided dataset"""
        print(f"Training model {model_name} with data from {dataset_path}")
        print(f"This would normally take several hours depending on your GPU")
        
        # Simulating training process
        for i in range(epochs):
            # In real implementation, this would be the actual training loop
            time.sleep(0.1)  # Just for demonstration
            yield i / epochs  # Return progress
            
        print("Training complete!")
        # In a real implementation, this would save the model
        self.model = f"Trained model: {model_name}"
        return True
    
    def convert_voice(self, audio_data, sr, target_pitch=0):
        """Convert voice using the trained model"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        print(f"Converting voice with target pitch shift: {target_pitch}")
        # In a real implementation, this would use the RVC model for inference
        
        # Simulate processing delay
        time.sleep(0.5)
        
        # Return the processed audio (this is just the input for simulation)
        # In a real system, this would be the converted audio
        return audio_data, sr
    
    def load_model(self, model_path):
        """Load a pretrained model"""
        print(f"Loading model from {model_path}")
        # In a real implementation, this would load the model weights
        self.model = f"Loaded model: {model_path}"
        return True

class VoiceCloningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Cloning Application")
        self.root.geometry("800x600")
        
        self.model = RVCModel()
        self.audio_queue = queue.Queue()
        self.recording = False
        self.playing = False
        
        self.setup_ui()
    
    def setup_ui(self):
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Data Collection
        self.tab1 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1, text="Data Collection")
        
        # Tab 2: Training
        self.tab2 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab2, text="Training")
        
        # Tab 3: Voice Conversion
        self.tab3 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab3, text="Voice Conversion")
        
        # Tab 4: Real-time Conversion (for Discord and other apps)
        self.tab4 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab4, text="Real-time/Discord")
        
        # Setup each tab's content
        self.setup_data_collection_tab()
        self.setup_training_tab()
        self.setup_conversion_tab()
        self.setup_realtime_tab()
    
    def setup_data_collection_tab(self):
        # Frame for target voice (your girlfriend's voice)
        target_frame = ttk.LabelFrame(self.tab1, text="Target Voice (Voice to Clone)")
        target_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(target_frame, text="Select folder containing target voice recordings:").pack(anchor=tk.W, padx=5, pady=5)
        
        target_path_frame = ttk.Frame(target_frame)
        target_path_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.target_path_var = tk.StringVar()
        ttk.Entry(target_path_frame, textvariable=self.target_path_var, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(target_path_frame, text="Browse", command=self.browse_target_folder).pack(side=tk.RIGHT, padx=5)
        
        ttk.Button(target_frame, text="Import Target Voice Data", command=self.import_target_data).pack(anchor=tk.W, padx=5, pady=5)
        
        # Frame for source voice (your voice)
        source_frame = ttk.LabelFrame(self.tab1, text="Source Voice (Your Voice)")
        source_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(source_frame, text="Record your voice or select existing recordings:").pack(anchor=tk.W, padx=5, pady=5)
        
        source_path_frame = ttk.Frame(source_frame)
        source_path_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.source_path_var = tk.StringVar()
        ttk.Entry(source_path_frame, textvariable=self.source_path_var, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(source_path_frame, text="Browse", command=self.browse_source_folder).pack(side=tk.RIGHT, padx=5)
        
        record_frame = ttk.Frame(source_frame)
        record_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.record_button = ttk.Button(record_frame, text="Start Recording", command=self.toggle_recording)
        self.record_button.pack(side=tk.LEFT, padx=5)
        
        self.recording_label = ttk.Label(record_frame, text="Not recording")
        self.recording_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(source_frame, text="Import Source Voice Data", command=self.import_source_data).pack(anchor=tk.W, padx=5, pady=5)
    
    def setup_training_tab(self):
        # Frame for training configuration
        config_frame = ttk.LabelFrame(self.tab2, text="Training Configuration")
        config_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(config_frame, text="Model Name:").pack(anchor=tk.W, padx=5, pady=5)
        self.model_name_var = tk.StringVar(value="my_voice_model")
        ttk.Entry(config_frame, textvariable=self.model_name_var, width=50).pack(anchor=tk.W, padx=5, pady=5, fill=tk.X)
        
        ttk.Label(config_frame, text="Number of Training Epochs:").pack(anchor=tk.W, padx=5, pady=5)
        self.epochs_var = tk.IntVar(value=100)
        ttk.Entry(config_frame, textvariable=self.epochs_var, width=10).pack(anchor=tk.W, padx=5, pady=5)
        
        # Frame for training actions
        train_frame = ttk.Frame(self.tab2)
        train_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.train_button = ttk.Button(train_frame, text="Start Training", command=self.start_training)
        self.train_button.pack(anchor=tk.W, padx=5, pady=5)
        
        ttk.Label(train_frame, text="Training Progress:").pack(anchor=tk.W, padx=5, pady=5)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(train_frame, variable=self.progress_var, maximum=1.0)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        self.train_status_label = ttk.Label(train_frame, text="Not started")
        self.train_status_label.pack(anchor=tk.W, padx=5, pady=5)
    
    def setup_conversion_tab(self):
        # Frame for model selection
        model_frame = ttk.LabelFrame(self.tab3, text="Model Selection")
        model_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(model_frame, text="Load Trained Model", command=self.load_model).pack(anchor=tk.W, padx=5, pady=5)
        self.model_status_label = ttk.Label(model_frame, text="No model loaded")
        self.model_status_label.pack(anchor=tk.W, padx=5, pady=5)
        
        # Frame for conversion settings
        settings_frame = ttk.LabelFrame(self.tab3, text="Conversion Settings")
        settings_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(settings_frame, text="Pitch Shift (semitones):").pack(anchor=tk.W, padx=5, pady=5)
        self.pitch_var = tk.IntVar(value=0)
        pitch_scale = ttk.Scale(settings_frame, from_=-12, to=12, variable=self.pitch_var, orient=tk.HORIZONTAL)
        pitch_scale.pack(fill=tk.X, padx=5, pady=5)
        pitch_value = ttk.Label(settings_frame, textvariable=self.pitch_var)
        pitch_value.pack(anchor=tk.W, padx=5, pady=5)
        
        # Frame for voice conversion
        conversion_frame = ttk.LabelFrame(self.tab3, text="Voice Conversion")
        conversion_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create a figure for the audio waveform
        self.fig, self.ax = plt.subplots(figsize=(6, 2), dpi=100)
        self.ax.set_title("Audio Waveform")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True)
        
        # Canvas for displaying the waveform
        self.canvas = FigureCanvasTkAgg(self.fig, master=conversion_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control buttons
        control_frame = ttk.Frame(conversion_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.record_convert_button = ttk.Button(control_frame, text="Record for Conversion", command=self.toggle_record_convert)
        self.record_convert_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Convert from File", command=self.convert_from_file).pack(side=tk.LEFT, padx=5)
        
        self.play_button = ttk.Button(control_frame, text="Play Converted", command=self.play_converted, state=tk.DISABLED)
        self.play_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Save Converted", command=self.save_converted, state=tk.DISABLED).pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.conversion_status_label = ttk.Label(conversion_frame, text="Ready")
        self.conversion_status_label.pack(anchor=tk.W, padx=5, pady=5)
        
        # Initialize properties for voice conversion
        self.source_audio = None
        self.converted_audio = None
        self.sr = 44100  # Sample rate
    
    def setup_realtime_tab(self):
        """Setup the real-time conversion tab for Discord integration"""
        if RealtimeConverterGUI is None:
            # Show a message if the module is not available
            ttk.Label(self.tab4, text="Real-time voice conversion module not found.").pack(padx=10, pady=10)
            ttk.Label(self.tab4, text="Please make sure the realtime_voice_converter.py file is in the same directory.").pack(padx=10, pady=10)
            return
        
        # Initialize the real-time converter GUI with our model if it's loaded
        self.realtime_gui = RealtimeConverterGUI(self.tab4, self.model if hasattr(self.model, 'model') and self.model.model is not None else None)
        
        # Add a note about Discord setup
        discord_frame = ttk.LabelFrame(self.tab4, text="Discord Setup Instructions")
        discord_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        instructions = """
1. Install a virtual audio cable like VB-Cable (https://vb-audio.com/Cable/)
2. Select your virtual audio output device in the dropdown above
3. In Discord settings, set the Input Device to your virtual audio cable
4. Start voice conversion before joining a voice channel
5. Speak normally - your voice will be converted in real-time!
        """
        
        ttk.Label(discord_frame, text=instructions, justify=tk.LEFT).pack(padx=10, pady=10, anchor=tk.W)
        
        # Add a button to refresh the model when it's loaded from another tab
        ttk.Button(self.tab4, text="Refresh Voice Model", command=self.refresh_realtime_model).pack(padx=10, pady=10)
    
    def refresh_realtime_model(self):
        """Update the real-time converter with the currently loaded model"""
        if hasattr(self, 'realtime_gui') and self.model.model is not None:
            self.realtime_gui.converter.set_model(self.model)
            self.realtime_gui.model_label.config(text=f"Model loaded: {self.model.model}")
            messagebox.showinfo("Success", "Voice model updated for real-time conversion!")
        else:
            messagebox.showerror("Error", "No model loaded. Please load a model in the Voice Conversion tab first.")
    
    def browse_target_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.target_path_var.set(folder)
    
    def browse_source_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.source_path_var.set(folder)
    
    def import_target_data(self):
        target_path = self.target_path_var.get()
        if not target_path or not os.path.isdir(target_path):
            messagebox.showerror("Error", "Please select a valid target voice folder")
            return
        
        # Create a dataset directory if it doesn't exist
        dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
        target_dir = os.path.join(dataset_dir, "target")
        os.makedirs(target_dir, exist_ok=True)
        
        # In a real implementation, this would process and prepare the target voice data
        messagebox.showinfo("Info", f"Target voice data would be processed and stored in {target_dir}")
    
    def import_source_data(self):
        source_path = self.source_path_var.get()
        if not source_path or not os.path.isdir(source_path):
            messagebox.showerror("Error", "Please select a valid source voice folder")
            return
        
        # Create a dataset directory if it doesn't exist
        dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
        source_dir = os.path.join(dataset_dir, "source")
        os.makedirs(source_dir, exist_ok=True)
        
        # In a real implementation, this would process and prepare the source voice data
        messagebox.showinfo("Info", f"Source voice data would be processed and stored in {source_dir}")
    
    def toggle_recording(self):
        if not self.recording:
            self.recording = True
            self.record_button.config(text="Stop Recording")
            self.recording_label.config(text="Recording...")
            
            # Start recording in a separate thread
            threading.Thread(target=self.record_audio, daemon=True).start()
        else:
            self.recording = False
            self.record_button.config(text="Start Recording")
            self.recording_label.config(text="Stopped")
    
    def record_audio(self):
        # Create a directory for recordings if it doesn't exist
        recordings_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
        os.makedirs(recordings_dir, exist_ok=True)
        
        # Set parameters for recording
        sample_rate = 44100
        channels = 1
        
        # Start recording
        with sd.InputStream(samplerate=sample_rate, channels=channels, callback=self.audio_callback):
            while self.recording:
                time.sleep(0.1)
        
        # Save the recorded audio
        if not self.audio_queue.empty():
            audio_data = []
            while not self.audio_queue.empty():
                audio_data.append(self.audio_queue.get())
            
            audio_data = np.concatenate(audio_data)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(recordings_dir, f"recording_{timestamp}.wav")
            
            sf.write(filename, audio_data, sample_rate)
            
            self.root.after(0, lambda: self.recording_label.config(text=f"Saved as {filename}"))
    
    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Status: {status}")
        self.audio_queue.put(indata.copy())
    
    def start_training(self):
        model_name = self.model_name_var.get()
        epochs = self.epochs_var.get()
        
        if not model_name:
            messagebox.showerror("Error", "Please enter a model name")
            return
        
        # Disable the train button during training
        self.train_button.config(state=tk.DISABLED)
        self.train_status_label.config(text="Preparing for training...")
        
        # Create a directory for models if it doesn't exist
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        os.makedirs(models_dir, exist_ok=True)
        
        # Start training in a separate thread
        threading.Thread(target=self.train_model, args=(model_name, epochs), daemon=True).start()
    
    def train_model(self, model_name, epochs):
        try:
            # In a real implementation, this would use the actual dataset path
            dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
            
            # Update status
            self.root.after(0, lambda: self.train_status_label.config(text="Training in progress..."))
            
            # Training progress
            for progress in self.model.train(dataset_path, model_name, epochs):
                self.root.after(0, lambda p=progress: self.progress_var.set(p))
            
            # Update status when done
            self.root.after(0, lambda: self.train_status_label.config(text="Training complete!"))
            self.root.after(0, lambda: messagebox.showinfo("Training Complete", "Model training has been completed successfully!"))
        except Exception as e:
            self.root.after(0, lambda: self.train_status_label.config(text=f"Error: {str(e)}"))
            self.root.after(0, lambda: messagebox.showerror("Error", f"Training failed: {str(e)}"))
        finally:
            # Re-enable the train button
            self.root.after(0, lambda: self.train_button.config(state=tk.NORMAL))
    
    def load_model(self):
        model_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Model Files", "*.pth")]
        )
        
        if not model_path:
            return
        
        try:
            self.model.load_model(model_path)
            self.model_status_label.config(text=f"Model loaded: {os.path.basename(model_path)}")
            messagebox.showinfo("Success", "Model loaded successfully!")
            
            # Update the real-time converter if it exists
            if hasattr(self, 'realtime_gui'):
                self.refresh_realtime_model()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def toggle_record_convert(self):
        if not hasattr(self, 'recording_for_conversion') or not self.recording_for_conversion:
            self.recording_for_conversion = True
            self.record_convert_button.config(text="Stop Recording")
            self.conversion_status_label.config(text="Recording for conversion...")
            
            # Reset audio data
            self.source_audio = None
            self.converted_audio = None
            self.audio_queue = queue.Queue()
            
            # Start recording in a separate thread
            threading.Thread(target=self.record_for_conversion, daemon=True).start()
        else:
            self.recording_for_conversion = False
            self.record_convert_button.config(text="Record for Conversion")
            self.conversion_status_label.config(text="Processing...")
            
            # Process the recorded audio after stopping
            threading.Thread(target=self.process_recorded_audio, daemon=True).start()
    
    def record_for_conversion(self):
        # Set parameters for recording
        sample_rate = 44100
        channels = 1
        
        # Start recording
        with sd.InputStream(samplerate=sample_rate, channels=channels, callback=self.audio_callback):
            while hasattr(self, 'recording_for_conversion') and self.recording_for_conversion:
                time.sleep(0.1)
    
    def process_recorded_audio(self):
        if self.audio_queue.empty():
            self.root.after(0, lambda: self.conversion_status_label.config(text="No audio recorded"))
            return
        
        # Collect audio data
        audio_data = []
        while not self.audio_queue.empty():
            audio_data.append(self.audio_queue.get())
        
        self.source_audio = np.concatenate(audio_data).flatten()
        self.sr = 44100
        
        # Plot the waveform
        self.plot_waveform(self.source_audio, self.sr, "Source Audio")
        
        # Convert the voice if model is loaded
        if self.model.model is not None:
            try:
                pitch_shift = self.pitch_var.get()
                self.converted_audio, self.sr = self.model.convert_voice(self.source_audio, self.sr, pitch_shift)
                
                # Update status and enable play button
                self.root.after(0, lambda: self.conversion_status_label.config(text="Conversion complete"))
                self.root.after(0, lambda: self.play_button.config(state=tk.NORMAL))
                
                # Plot the converted waveform
                self.plot_waveform(self.converted_audio, self.sr, "Converted Audio")
            except Exception as e:
                self.root.after(0, lambda: self.conversion_status_label.config(text=f"Error: {str(e)}"))
        else:
            self.root.after(0, lambda: self.conversion_status_label.config(text="Please load a model first"))
    
    def convert_from_file(self):
        if self.model.model is None:
            messagebox.showerror("Error", "Please load a model first")
            return
        
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio Files", "*.wav *.mp3 *.ogg")]
        )
        
        if not file_path:
            return
        
        try:
            # Load the audio file
            self.source_audio, self.sr = librosa.load(file_path, sr=None)
            
            # Plot the waveform
            self.plot_waveform(self.source_audio, self.sr, "Source Audio")
            
            # Update status
            self.conversion_status_label.config(text="Converting...")
            
            # Convert the voice in a separate thread
            threading.Thread(target=self.convert_file_audio, args=(file_path,), daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load audio: {str(e)}")
    
    def convert_file_audio(self, file_path):
        try:
            pitch_shift = self.pitch_var.get()
            self.converted_audio, self.sr = self.model.convert_voice(self.source_audio, self.sr, pitch_shift)
            
            # Update status and enable play button
            self.root.after(0, lambda: self.conversion_status_label.config(text="Conversion complete"))
            self.root.after(0, lambda: self.play_button.config(state=tk.NORMAL))
            
            # Plot the converted waveform
            self.plot_waveform(self.converted_audio, self.sr, "Converted Audio")
        except Exception as e:
            self.root.after(0, lambda: self.conversion_status_label.config(text=f"Error: {str(e)}"))
    
    def plot_waveform(self, audio, sr, title):
        duration = len(audio) / sr
        time_axis = np.linspace(0, duration, len(audio))
        
        self.ax.clear()
        self.ax.plot(time_axis, audio)
        self.ax.set_title(title)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.ax.set_ylim([-1, 1])
        self.ax.grid(True)
        
        self.canvas.draw()
    
    def play_converted(self):
        if self.converted_audio is None:
            messagebox.showerror("Error", "No converted audio to play")
            return
        
        if self.playing:
            sd.stop()
            self.playing = False
            self.play_button.config(text="Play Converted")
            return
        
        try:
            self.playing = True
            self.play_button.config(text="Stop Playback")
            
            # Play the audio in a separate thread
            threading.Thread(target=self.play_audio, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to play audio: {str(e)}")
            self.playing = False
            self.play_button.config(text="Play Converted")
    
    def play_audio(self):
        try:
            sd.play(self.converted_audio, self.sr)
            sd.wait()
        finally:
            self.root.after(0, lambda: self.play_button.config(text="Play Converted"))
            self.playing = False
    
    def save_converted(self):
        if self.converted_audio is None:
            messagebox.showerror("Error", "No converted audio to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Converted Audio",
            defaultextension=".wav",
            filetypes=[("WAV Files", "*.wav")]
        )
        
        if not file_path:
            return
        
        try:
            sf.write(file_path, self.converted_audio, self.sr)
            messagebox.showinfo("Success", f"Audio saved to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save audio: {str(e)}")

if __name__ == "__main__":
    # Create the main window
    root = tk.Tk()
    app = VoiceCloningApp(root)
    
    # Start the application
    root.mainloop()
