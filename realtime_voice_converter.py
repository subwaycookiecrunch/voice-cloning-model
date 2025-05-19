"""
Real-time Voice Converter for Discord and other applications
This module captures audio from a microphone, converts it using an RVC model,
and outputs it to a virtual audio device that can be selected in Discord.
"""

import os
import sys
import time
import queue
import threading
import numpy as np
import torch
import sounddevice as sd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

class RealtimeVoiceConverter:
    def __init__(self, model=None):
        self.model = model
        self.input_device = None
        self.output_device = None
        self.is_converting = False
        self.pitch_shift = 0
        self.buffer_size = 1024
        self.sample_rate = 44100
        self.audio_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.block_size = 2048  # Number of frames per block
        
    def set_model(self, model):
        """Set the voice conversion model"""
        self.model = model
        
    def list_audio_devices(self):
        """List available audio devices"""
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        output_devices = [d for d in devices if d['max_output_channels'] > 0]
        return input_devices, output_devices
    
    def set_devices(self, input_device_id, output_device_id):
        """Set input and output audio devices"""
        self.input_device = input_device_id
        self.output_device = output_device_id
        
    def set_pitch_shift(self, pitch_shift):
        """Set pitch shift amount in semitones"""
        self.pitch_shift = pitch_shift
        
    def input_callback(self, indata, frames, time_info, status):
        """Callback function for audio input stream"""
        if status:
            print(f"Input status: {status}")
        self.audio_queue.put(indata.copy())
            
    def output_callback(self, outdata, frames, time_info, status):
        """Callback function for audio output stream"""
        if status:
            print(f"Output status: {status}")
        
        if not self.output_queue.empty():
            outdata[:] = self.output_queue.get()
        else:
            outdata.fill(0)  # Output silence if no processed data is available
    
    def process_audio(self):
        """Process audio data from input to output"""
        while self.is_converting:
            if not self.audio_queue.empty() and self.model is not None:
                # Get input audio data
                audio_data = self.audio_queue.get()
                
                try:
                    # Convert voice using the model (in a real implementation)
                    # For now, we'll just simulate by applying a basic effect
                    if hasattr(self.model, 'convert_voice'):
                        # Use the actual model for conversion
                        converted_audio, _ = self.model.convert_voice(audio_data.flatten(), self.sample_rate, self.pitch_shift)
                        # Reshape to match output format
                        converted_audio = converted_audio.reshape((-1, 1))
                    else:
                        # Simple simulation (pitch shift simulation)
                        # In a real app, this would be the actual voice conversion
                        converted_audio = audio_data
                    
                    # Put processed audio in output queue
                    self.output_queue.put(converted_audio)
                    
                except Exception as e:
                    print(f"Error processing audio: {e}")
            else:
                # Sleep a bit to avoid hammering the CPU
                time.sleep(0.001)
    
    def start_conversion(self):
        """Start real-time voice conversion"""
        if self.model is None:
            raise ValueError("No model loaded for conversion")
        
        if self.input_device is None or self.output_device is None:
            raise ValueError("Input and output devices must be set")
        
        # Set conversion flag
        self.is_converting = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_audio, daemon=True)
        self.processing_thread.start()
        
        # Start input and output streams
        self.input_stream = sd.InputStream(
            device=self.input_device,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            callback=self.input_callback
        )
        
        self.output_stream = sd.OutputStream(
            device=self.output_device,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            callback=self.output_callback
        )
        
        self.input_stream.start()
        self.output_stream.start()
        
        print(f"Started real-time voice conversion with pitch shift {self.pitch_shift}")
        return True
    
    def stop_conversion(self):
        """Stop real-time voice conversion"""
        self.is_converting = False
        
        # Stop streams if they exist
        if hasattr(self, 'input_stream'):
            self.input_stream.stop()
            self.input_stream.close()
        
        if hasattr(self, 'output_stream'):
            self.output_stream.stop()
            self.output_stream.close()
        
        # Clear queues
        while not self.audio_queue.empty():
            self.audio_queue.get()
        
        while not self.output_queue.empty():
            self.output_queue.get()
        
        print("Stopped real-time voice conversion")
        return True

class RealtimeConverterGUI:
    def __init__(self, root, model=None):
        self.root = root
        self.converter = RealtimeVoiceConverter(model)
        
        # Main frame
        self.frame = ttk.LabelFrame(root, text="Real-time Voice Conversion")
        self.frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Device selection
        self.setup_device_selection()
        
        # Pitch shift control
        self.setup_pitch_control()
        
        # Model selection
        if model is None:
            self.setup_model_selection()
        else:
            self.model_label = ttk.Label(self.frame, text="Model loaded from main application")
            self.model_label.pack(anchor=tk.W, padx=5, pady=5)
        
        # Control buttons
        self.setup_control_buttons()
        
        # Status label
        self.status_label = ttk.Label(self.frame, text="Ready")
        self.status_label.pack(anchor=tk.W, padx=5, pady=5)
    
    def setup_device_selection(self):
        # Get available devices
        input_devices, output_devices = self.converter.list_audio_devices()
        
        # Input device selection
        input_frame = ttk.LabelFrame(self.frame, text="Input Device (Your Microphone)")
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.input_device_var = tk.StringVar()
        input_combo = ttk.Combobox(input_frame, textvariable=self.input_device_var, state="readonly")
        input_combo['values'] = [f"{i}: {d['name']}" for i, d in enumerate(input_devices)]
        if input_devices:
            input_combo.current(0)
            self.converter.set_devices(0, None)
        input_combo.pack(fill=tk.X, padx=5, pady=5)
        input_combo.bind('<<ComboboxSelected>>', self.on_input_device_changed)
        
        # Output device selection
        output_frame = ttk.LabelFrame(self.frame, text="Output Device (Virtual Mic for Discord)")
        output_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.output_device_var = tk.StringVar()
        output_combo = ttk.Combobox(output_frame, textvariable=self.output_device_var, state="readonly")
        output_combo['values'] = [f"{i}: {d['name']}" for i, d in enumerate(output_devices)]
        if output_devices:
            output_combo.current(0)
            self.converter.set_devices(None, 0)
        output_combo.pack(fill=tk.X, padx=5, pady=5)
        output_combo.bind('<<ComboboxSelected>>', self.on_output_device_changed)
        
        # Note about virtual audio device
        note_label = ttk.Label(output_frame, text="Note: You may need to install a virtual audio device like VB-Cable")
        note_label.pack(anchor=tk.W, padx=5, pady=5)
    
    def setup_pitch_control(self):
        pitch_frame = ttk.LabelFrame(self.frame, text="Voice Settings")
        pitch_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(pitch_frame, text="Pitch Shift (semitones):").pack(anchor=tk.W, padx=5, pady=5)
        
        self.pitch_var = tk.IntVar(value=0)
        pitch_scale = ttk.Scale(pitch_frame, from_=-12, to=12, variable=self.pitch_var, orient=tk.HORIZONTAL)
        pitch_scale.pack(fill=tk.X, padx=5, pady=5)
        
        pitch_value = ttk.Label(pitch_frame, textvariable=self.pitch_var)
        pitch_value.pack(anchor=tk.W, padx=5, pady=5)
        
        # Update converter when pitch changes
        self.pitch_var.trace_add("write", self.on_pitch_changed)
    
    def setup_model_selection(self):
        model_frame = ttk.LabelFrame(self.frame, text="Voice Model")
        model_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(model_frame, text="Load Voice Model", command=self.load_model).pack(anchor=tk.W, padx=5, pady=5)
        
        self.model_label = ttk.Label(model_frame, text="No model loaded")
        self.model_label.pack(anchor=tk.W, padx=5, pady=5)
    
    def setup_control_buttons(self):
        control_frame = ttk.Frame(self.frame)
        control_frame.pack(fill=tk.X, padx=5, pady=10)
        
        self.start_button = ttk.Button(control_frame, text="Start Voice Conversion", command=self.start_conversion)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop_conversion, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
    
    def on_input_device_changed(self, event):
        selected = self.input_device_var.get()
        device_id = int(selected.split(':')[0])
        self.converter.set_devices(device_id, self.converter.output_device)
        print(f"Input device set to: {selected}")
    
    def on_output_device_changed(self, event):
        selected = self.output_device_var.get()
        device_id = int(selected.split(':')[0])
        self.converter.set_devices(self.converter.input_device, device_id)
        print(f"Output device set to: {selected}")
    
    def on_pitch_changed(self, *args):
        pitch = self.pitch_var.get()
        self.converter.set_pitch_shift(pitch)
        if hasattr(self, 'converter') and self.converter.is_converting:
            print(f"Pitch shift changed to {pitch} semitones")
    
    def load_model(self):
        model_path = filedialog.askopenfilename(
            title="Select Voice Model File",
            filetypes=[("Model Files", "*.pth")]
        )
        
        if not model_path:
            return
        
        try:
            # In a real implementation, this would load the model
            # For demonstration, we'll create a dummy model object
            class DummyModel:
                def convert_voice(self, audio, sr, pitch):
                    return audio, sr
            
            self.converter.set_model(DummyModel())
            self.model_label.config(text=f"Model loaded: {os.path.basename(model_path)}")
            messagebox.showinfo("Success", "Model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def start_conversion(self):
        if self.converter.model is None:
            messagebox.showerror("Error", "Please load a voice model first")
            return
        
        try:
            success = self.converter.start_conversion()
            if success:
                self.start_button.config(state=tk.DISABLED)
                self.stop_button.config(state=tk.NORMAL)
                self.status_label.config(text="Converting voice in real-time")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start conversion: {str(e)}")
    
    def stop_conversion(self):
        try:
            success = self.converter.stop_conversion()
            if success:
                self.start_button.config(state=tk.NORMAL)
                self.stop_button.config(state=tk.DISABLED)
                self.status_label.config(text="Stopped")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop conversion: {str(e)}")

if __name__ == "__main__":
    # If run directly, create a standalone window
    root = tk.Tk()
    root.title("Real-time Voice Converter")
    root.geometry("600x500")
    app = RealtimeConverterGUI(root)
    root.mainloop() 