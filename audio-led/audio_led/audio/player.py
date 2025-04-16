#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import logging
import threading
import numpy as np

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False
    logging.warning("Sounddevice not available")

try:
    import pyaudio
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False
    logging.warning("PyAudio not available")

# Initialize the logger
logger = logging.getLogger(__name__)

class AudioPlayer:
    """
    Audio player for file inputs.
    Plays audio from the input handler to the system's default audio output.
    """
    
    def __init__(self, config):
        """
        Initialize the audio player.
        
        Args:
            config (dict): Environment configuration
        """
        self.config = config
        self.input_handler = None
        self.running = False
        self.thread = None
        self.stream = None
        self.pyaudio_instance = None
        
        # Default audio settings
        self.sample_rate = 44100
        self.channels = 1
        self.chunk_size = 1024
        self.buffer = []
        self.buffer_lock = threading.Lock()
        
    def connect_to_input(self, input_handler):
        """
        Connect to an audio input handler.
        
        Args:
            input_handler: The AudioInputHandler instance
        """
        self.input_handler = input_handler
        
        # Get audio settings from input handler if available
        if hasattr(input_handler, 'sample_rate'):
            self.sample_rate = input_handler.sample_rate
        if hasattr(input_handler, 'channels'):
            self.channels = input_handler.channels
        if hasattr(input_handler, 'chunk_size'):
            self.chunk_size = input_handler.chunk_size
            
        logger.info(f"Audio player connected to input handler (rate={self.sample_rate}, channels={self.channels}, chunk={self.chunk_size})")
        
    def _initialize_sounddevice(self):
        """Initialize playback using sounddevice"""
        if not HAS_SOUNDDEVICE:
            return False
            
        try:
            # Get default output device info
            device_info = sd.query_devices(kind='output')
            logger.info(f"Using output device: {device_info['name']}")
            
            # Start the stream
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.chunk_size,
                callback=self._sounddevice_callback
            )
            self.stream.start()
            return True
        except Exception as e:
            logger.error(f"Error initializing sounddevice: {e}")
            return False
            
    def _sounddevice_callback(self, outdata, frames, time, status):
        """Callback for sounddevice output stream"""
        if status:
            logger.warning(f"Sounddevice status: {status}")
            
        # Get data from buffer
        with self.buffer_lock:
            if len(self.buffer) > 0:
                # Get the first chunk from the buffer
                data = self.buffer.pop(0)
                
                # Reshape if needed
                if data.shape[0] != frames:
                    # Pad or trim
                    if data.shape[0] < frames:
                        # Pad with zeros
                        padding = np.zeros((frames - data.shape[0], self.channels), dtype=np.float32)
                        data = np.vstack((data, padding))
                    else:
                        # Trim
                        data = data[:frames]
                        
                # Copy to output buffer
                outdata[:] = data
            else:
                # No data available, output silence
                outdata.fill(0)
                
    def _initialize_pyaudio(self):
        """Initialize playback using PyAudio"""
        if not HAS_PYAUDIO:
            return False
            
        try:
            # Create PyAudio instance
            self.pyaudio_instance = pyaudio.PyAudio()
            
            # Find default output device
            default_device = self.pyaudio_instance.get_default_output_device_info()
            logger.info(f"Using output device: {default_device['name']}")
            
            # Open stream
            self.stream = self.pyaudio_instance.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._pyaudio_callback
            )
            self.stream.start_stream()
            return True
        except Exception as e:
            logger.error(f"Error initializing PyAudio: {e}")
            if self.pyaudio_instance:
                self.pyaudio_instance.terminate()
                self.pyaudio_instance = None
            return False
            
    def _pyaudio_callback(self, in_data, frame_count, time_info, status):
        """Callback for PyAudio output stream"""
        if status:
            logger.warning(f"PyAudio status: {status}")
            
        # Get data from buffer
        with self.buffer_lock:
            if len(self.buffer) > 0:
                # Get the first chunk from the buffer
                data = self.buffer.pop(0)
                
                # Reshape if needed
                if data.shape[0] != frame_count:
                    # Pad or trim
                    if data.shape[0] < frame_count:
                        # Pad with zeros
                        padding = np.zeros((frame_count - data.shape[0], self.channels), dtype=np.float32)
                        data = np.vstack((data, padding))
                    else:
                        # Trim
                        data = data[:frame_count]
                        
                # Convert to bytes
                return (data.astype(np.float32).tobytes(), pyaudio.paContinue)
            else:
                # No data available, output silence
                return (np.zeros((frame_count, self.channels), dtype=np.float32).tobytes(), pyaudio.paContinue)
                
    def _playback_thread(self):
        """Thread function for audio playback"""
        logger.info("Audio playback thread started")
        
        # Initialize audio output
        if not self._initialize_sounddevice() and not self._initialize_pyaudio():
            logger.error("Failed to initialize audio output")
            return
            
        # Main loop
        while self.running and self.input_handler:
            # Get audio data from input handler
            audio_data = self.input_handler.get_audio_data()
            
            if audio_data is not None:
                # Convert int16 to float32 if needed
                if audio_data.dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32768.0
                    
                # Reshape to (frames, channels) if needed
                if len(audio_data.shape) == 1:
                    audio_data = audio_data.reshape(-1, 1)
                    
                # Add to buffer
                with self.buffer_lock:
                    self.buffer.append(audio_data)
                    
                    # Limit buffer size
                    while len(self.buffer) > 5:
                        self.buffer.pop(0)
            else:
                # No data available
                time.sleep(0.01)
                
        logger.info("Audio playback thread stopped")
        
    def start(self):
        """Start audio playback"""
        if self.running:
            return
            
        if not self.input_handler:
            logger.error("No input handler connected")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._playback_thread)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        """Stop audio playback"""
        self.running = False
        
        # Stop stream
        if self.stream:
            try:
                if hasattr(self.stream, 'stop_stream'):
                    self.stream.stop_stream()
                if hasattr(self.stream, 'close'):
                    self.stream.close()
            except Exception as e:
                logger.error(f"Error stopping stream: {e}")
                
        # Terminate PyAudio
        if self.pyaudio_instance:
            try:
                self.pyaudio_instance.terminate()
            except Exception as e:
                logger.error(f"Error terminating PyAudio: {e}")
                
        # Wait for thread to finish
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
            
        self.stream = None
        self.pyaudio_instance = None
        self.thread = None 