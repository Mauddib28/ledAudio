#!/usr/bin/env python3
# -*- coding: utf-8 -*-

####
# Created on:        2025-04-15
# Created by:        AI Assistant
# Last modified on:  2025-04-15
####

#--------------------------------------
#   MODULE DESCRIPTION
#--------------------------------------

# This module provides audio processing functionality for the Audio LED Visualization System.
# It handles:
# - Audio input from various sources (microphone, files, network)
# - Real-time audio analysis (FFT, spectrum, beat detection)
# - Audio data transformation and normalization
# - Feature extraction (frequency bands, volume, beat, tempo)
#
# The processed audio data is used by visualization modules to create
# synchronized light effects.

#--------------------------------------
#       IMPORTS
#--------------------------------------

import os
import time
import queue
import logging
import threading
import numpy as np
from enum import Enum
from abc import ABC, abstractmethod

# Configure module logger
logger = logging.getLogger(__name__)

# Optional imports based on environment
try:
    import pyaudio
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False
    logger.warning("PyAudio not available")

try:
    import sounddevice as sd
    import soundfile as sf
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False
    logger.warning("Sounddevice not available")

try:
    from scipy.signal import butter, lfilter, find_peaks
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("SciPy not available, advanced processing features will be limited")

#--------------------------------------
#       CONSTANTS
#--------------------------------------

# Audio input methods
class AudioInputMethod(Enum):
    MICROPHONE = "microphone"
    FILE = "file"
    I2S_MICROPHONE = "i2s_microphone"
    NETWORK = "network"
    DUMMY = "dummy"

# Audio processing modes
class ProcessingMode(Enum):
    RAW = "raw"
    SPECTRUM = "spectrum"
    FREQUENCY_BANDS = "frequency_bands"
    BEAT_DETECTION = "beat_detection"

# Default frequency bands (in Hz)
DEFAULT_FREQUENCY_BANDS = [
    (20, 60),      # Sub-bass
    (60, 250),     # Bass
    (250, 500),    # Low midrange
    (500, 2000),   # Midrange
    (2000, 4000),  # Upper midrange
    (4000, 8000),  # Presence
    (8000, 20000)  # Brilliance
]

# Audio chunk processing constants
MAX_FREQ = 20000  # Maximum frequency to analyze (Hz)
MIN_FREQ = 20     # Minimum frequency to analyze (Hz)

#--------------------------------------
#   AUDIO INPUT BASE CLASS
#--------------------------------------

class AudioInput(ABC):
    """
    Abstract base class for audio input devices.
    
    This class defines the interface that all audio input devices must implement.
    """
    
    def __init__(self, config):
        """
        Initialize the audio input device.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.sample_rate = config.get("sample_rate", 44100)
        self.channels = config.get("channels", 1)
        self.chunk_size = config.get("chunk_size", 1024)
        self.overlap = config.get("overlap", 0.5)
        self.is_running = False
        self.audio_queue = queue.Queue(maxsize=100)
    
    @abstractmethod
    def open(self):
        """
        Open the audio input device.
        
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def close(self):
        """
        Close the audio input device.
        
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def read(self):
        """
        Read audio data from the device.
        
        Returns:
            numpy.ndarray: Audio data as a numpy array
        """
        pass
    
    def start(self):
        """
        Start the audio input device.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.is_running:
            logger.warning("Audio input already running")
            return True
        
        success = self.open()
        if success:
            self.is_running = True
            # Start the reading thread
            self.read_thread = threading.Thread(target=self._read_thread, daemon=True)
            self.read_thread.start()
            
            logger.info("Audio input started")
            return True
        else:
            logger.error("Failed to start audio input")
            return False
    
    def stop(self):
        """
        Stop the audio input device.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_running:
            logger.warning("Audio input already stopped")
            return True
        
        self.is_running = False
        # Wait for the reading thread to finish
        if hasattr(self, 'read_thread') and self.read_thread.is_alive():
            self.read_thread.join(timeout=1.0)
        
        success = self.close()
        if success:
            logger.info("Audio input stopped")
            return True
        else:
            logger.error("Failed to stop audio input")
            return False
    
    def get_audio_data(self, block=True, timeout=None):
        """
        Get audio data from the queue.
        
        Args:
            block (bool, optional): Whether to block until data is available
            timeout (float, optional): Timeout in seconds
            
        Returns:
            numpy.ndarray: Audio data as a numpy array
        """
        try:
            return self.audio_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None
    
    def _read_thread(self):
        """
        Thread function for reading audio data.
        """
        while self.is_running:
            try:
                # Read audio data
                data = self.read()
                if data is not None and len(data) > 0:
                    # Put data in the queue (non-blocking)
                    try:
                        self.audio_queue.put(data, block=False)
                    except queue.Full:
                        # Queue is full, remove the oldest item
                        try:
                            self.audio_queue.get_nowait()
                            self.audio_queue.put(data, block=False)
                        except queue.Empty:
                            pass
            except Exception as e:
                logger.error(f"Error reading audio data: {e}")
                time.sleep(0.1)

#--------------------------------------
#   MICROPHONE INPUT
#--------------------------------------

class MicrophoneInput(AudioInput):
    """
    Microphone input using PyAudio or SoundDevice.
    """
    
    def __init__(self, config):
        """
        Initialize the microphone input.
        
        Args:
            config (dict): Configuration dictionary
        """
        super().__init__(config)
        self.device_name = config.get("input_device", "default")
        self.device_index = None
        self.stream = None
        
        # Choose the audio library to use
        if HAS_PYAUDIO:
            self.use_pyaudio = True
        elif HAS_SOUNDDEVICE:
            self.use_pyaudio = False
        else:
            raise ImportError("No audio library available (PyAudio or SoundDevice required)")
    
    def open(self):
        """
        Open the microphone input.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.use_pyaudio:
                return self._open_pyaudio()
            else:
                return self._open_sounddevice()
        except Exception as e:
            logger.error(f"Error opening microphone: {e}")
            return False
    
    def _open_pyaudio(self):
        """
        Open the microphone using PyAudio.
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Create PyAudio instance
        self.pa = pyaudio.PyAudio()
        
        # Find the device index if a specific device name is specified
        if self.device_name != "default":
            for i in range(self.pa.get_device_count()):
                device_info = self.pa.get_device_info_by_index(i)
                if self.device_name.lower() in device_info['name'].lower() and device_info['maxInputChannels'] > 0:
                    self.device_index = i
                    break
            
            if self.device_index is None:
                logger.warning(f"Device '{self.device_name}' not found, using default")
        
        # Open the audio stream
        self.stream = self.pa.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._pyaudio_callback
        )
        
        return True
    
    def _open_sounddevice(self):
        """
        Open the microphone using SoundDevice.
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Find the device index if a specific device name is specified
        if self.device_name != "default":
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if self.device_name.lower() in device['name'].lower() and device['max_input_channels'] > 0:
                    self.device_index = i
                    break
            
            if self.device_index is None:
                logger.warning(f"Device '{self.device_name}' not found, using default")
        
        # Open the audio stream
        self.stream = sd.InputStream(
            device=self.device_index,
            channels=self.channels,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            dtype='float32'
        )
        
        self.stream.start()
        
        return True
    
    def _pyaudio_callback(self, in_data, frame_count, time_info, status):
        """
        PyAudio callback function.
        
        This is called by PyAudio when new audio data is available.
        
        Args:
            in_data (bytes): Raw audio data
            frame_count (int): Number of frames
            time_info (dict): Time information
            status (int): Status flags
            
        Returns:
            tuple: (None, pyaudio.paContinue)
        """
        # Convert the raw data to a numpy array
        data = np.frombuffer(in_data, dtype=np.float32)
        
        # Add to the queue if not full
        if not self.audio_queue.full():
            self.audio_queue.put_nowait(data)
        
        return (None, pyaudio.paContinue)
    
    def read(self):
        """
        Read audio data from the microphone.
        
        Returns:
            numpy.ndarray: Audio data as a numpy array
        """
        if not self.is_running:
            return None
        
        if self.use_pyaudio:
            # For PyAudio, data is read in the callback
            return None
        else:
            # For SoundDevice, read the data directly
            try:
                data, overflowed = self.stream.read(self.chunk_size)
                if overflowed:
                    logger.warning("Audio input buffer overflowed")
                
                return data.flatten()
            except Exception as e:
                logger.error(f"Error reading from microphone: {e}")
                return None
    
    def close(self):
        """
        Close the microphone input.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.use_pyaudio:
                if self.stream is not None:
                    self.stream.stop_stream()
                    self.stream.close()
                    self.stream = None
                
                if hasattr(self, 'pa'):
                    self.pa.terminate()
            else:
                if self.stream is not None:
                    self.stream.stop()
                    self.stream.close()
                    self.stream = None
            
            return True
        except Exception as e:
            logger.error(f"Error closing microphone: {e}")
            return False

#--------------------------------------
#   FILE INPUT
#--------------------------------------

class FileInput(AudioInput):
    """
    Audio input from a file.
    """
    
    def __init__(self, config):
        """
        Initialize the file input.
        
        Args:
            config (dict): Configuration dictionary
        """
        super().__init__(config)
        self.file_path = config.get("file_path")
        self.loop = config.get("loop", True)
        self.file_data = None
        self.position = 0
        
        if not self.file_path:
            raise ValueError("File path not specified")
        
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Audio file not found: {self.file_path}")
    
    def open(self):
        """
        Open the audio file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if HAS_SOUNDDEVICE:
                # Use soundfile to read the audio file
                self.file_data, file_sr = sf.read(self.file_path, dtype='float32')
                
                # Convert to mono if necessary
                if len(self.file_data.shape) > 1 and self.file_data.shape[1] > 1:
                    if self.channels == 1:
                        self.file_data = np.mean(self.file_data, axis=1)
                
                # Resample if necessary
                if file_sr != self.sample_rate:
                    if HAS_SCIPY:
                        from scipy import signal
                        num_samples = int(len(self.file_data) * self.sample_rate / file_sr)
                        self.file_data = signal.resample(self.file_data, num_samples)
                    else:
                        logger.warning("SciPy not available, cannot resample audio")
                
                self.position = 0
                return True
            else:
                logger.error("SoundFile required for file input")
                return False
        except Exception as e:
            logger.error(f"Error opening audio file: {e}")
            return False
    
    def read(self):
        """
        Read audio data from the file.
        
        Returns:
            numpy.ndarray: Audio data as a numpy array
        """
        if not self.is_running or self.file_data is None:
            return None
        
        # Get the next chunk of data
        end_pos = self.position + self.chunk_size
        
        if end_pos <= len(self.file_data):
            # Regular read
            data = self.file_data[self.position:end_pos]
            self.position = end_pos
        elif self.position < len(self.file_data):
            # Read the remaining data
            data = self.file_data[self.position:]
            missing = self.chunk_size - len(data)
            
            if self.loop:
                # Loop back to the beginning
                data = np.concatenate((data, self.file_data[:missing]))
                self.position = missing
            else:
                # Pad with zeros
                data = np.pad(data, (0, missing), 'constant')
                self.position = len(self.file_data)
        else:
            # End of file
            if self.loop:
                # Start over
                data = self.file_data[:self.chunk_size]
                self.position = self.chunk_size
            else:
                # No more data
                self.is_running = False
                return None
        
        # Add a small delay to simulate real-time processing
        time.sleep(self.chunk_size / self.sample_rate / 2)
        
        return data
    
    def close(self):
        """
        Close the audio file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        self.file_data = None
        self.position = 0
        return True

#--------------------------------------
#   DUMMY INPUT
#--------------------------------------

class DummyInput(AudioInput):
    """
    Dummy audio input for testing or when no audio input is available.
    
    This generates synthetic audio data with configurable patterns.
    """
    
    def __init__(self, config):
        """
        Initialize the dummy input.
        
        Args:
            config (dict): Configuration dictionary
        """
        super().__init__(config)
        self.pattern = config.get("pattern", "sine")
        self.frequency = config.get("frequency", 440)
        self.amplitude = config.get("amplitude", 0.5)
        self.time = 0
    
    def open(self):
        """
        Open the dummy input.
        
        Returns:
            bool: True if successful, False otherwise
        """
        self.time = 0
        return True
    
    def read(self):
        """
        Read audio data from the dummy input.
        
        Returns:
            numpy.ndarray: Audio data as a numpy array
        """
        if not self.is_running:
            return None
        
        # Generate a time array for this chunk
        t = np.arange(self.time, self.time + self.chunk_size) / self.sample_rate
        self.time += self.chunk_size
        
        # Generate the pattern
        if self.pattern == "sine":
            # Pure sine wave
            data = self.amplitude * np.sin(2 * np.pi * self.frequency * t)
        elif self.pattern == "sweep":
            # Frequency sweep
            f = np.linspace(100, 1000, len(t))
            data = self.amplitude * np.sin(2 * np.pi * f * t)
        elif self.pattern == "noise":
            # White noise
            data = self.amplitude * np.random.uniform(-1, 1, size=len(t))
        elif self.pattern == "beat":
            # Beat pattern
            beat_freq = 2  # 2 Hz = 120 BPM
            beat_env = 0.5 * (1 + np.sin(2 * np.pi * beat_freq * t - np.pi/2))
            data = self.amplitude * beat_env * np.sin(2 * np.pi * self.frequency * t)
        else:
            # Default to sine
            data = self.amplitude * np.sin(2 * np.pi * self.frequency * t)
        
        # Add a small delay to simulate real-time processing
        time.sleep(self.chunk_size / self.sample_rate / 2)
        
        return data
    
    def close(self):
        """
        Close the dummy input.
        
        Returns:
            bool: True if successful, False otherwise
        """
        self.time = 0
        return True

#--------------------------------------
#   AUDIO PROCESSOR
#--------------------------------------

class AudioProcessor:
    """
    Audio processor for the Audio LED Visualization System.
    
    This class handles audio input and processing, providing
    real-time analysis for visualization.
    """
    
    # Constants
    MAX_RETRY_COUNT = 3  # Maximum number of times to retry initialization
    
    def __init__(self, config):
        """
        Initialize the audio processor.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        
        # Get audio configuration
        self.audio_config = config.get("audio", {})
        self.input_method = AudioInputMethod(self.audio_config.get("input_method", "microphone"))
        
        # Get processing configuration
        self.processing_mode = ProcessingMode(self.audio_config.get("processing_mode", "frequency_bands"))
        self.frequency_bands = self.audio_config.get("frequency_bands", DEFAULT_FREQUENCY_BANDS)
        
        # Initialize variables
        self.input_device = None
        self.is_running = False
        self.data_queue = queue.Queue(maxsize=10)
        
        # Processing parameters
        self.sample_rate = self.audio_config.get("sample_rate", 44100)
        self.channels = self.audio_config.get("channels", 1)
        self.chunk_size = self.audio_config.get("chunk_size", 1024)
        self.overlap = self.audio_config.get("overlap", 0.5)
        
        # FFT parameters
        self.window = np.hanning(self.chunk_size)
        self.fft_size = self.chunk_size
        self.fft_freqs = np.fft.rfftfreq(self.fft_size, 1.0/self.sample_rate)
        
        # Frequency band indices
        self.band_indices = self._calculate_band_indices()
        
        # Beat detection parameters
        self.beat_threshold = self.audio_config.get("beat_threshold", 0.5)
        self.beat_history = []
        self.energy_history = []
        self.beat_history_size = 43  # About 1 second at 44.1kHz with 1024 chunk size
        
        # Initialize the processing thread
        self.processor_thread = None
    
    def start(self):
        """Start the audio processor thread"""
        if self.is_running:
            return
        
        # Set running state
        self.is_running = True
        
        # Initialize audio input
        max_retries = AudioProcessor.MAX_RETRY_COUNT
        retry_count = 0

        while retry_count < max_retries:
            try:
                # Initialize and open the audio input
                if self.input_device is None:
                    self.input_device = self.create_audio_input()
                
                # Start the processing thread
                self.processor_thread = threading.Thread(target=self._processor_thread, daemon=True)
                self.processor_thread.start()
                
                logger.info("Audio processor started")
                return True
            except Exception as e:
                retry_count += 1
                logger.error(f"Failed to start audio processor (attempt {retry_count}/{max_retries}): {e}")
                time.sleep(0.5)  # Wait before retrying
                
                # Clean up resources before retry
                self.close_audio_input()
                
        # If we get here, all retries failed
        logger.error(f"Could not start audio processor after {max_retries} attempts")
        self.is_running = False
        return False
    
    def stop(self):
        """
        Stop the audio processor.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_running:
            logger.warning("Audio processor already stopped")
            return True
        
        # Stop the processing thread
        self.is_running = False
        if self.processor_thread is not None and self.processor_thread.is_alive():
            self.processor_thread.join(timeout=1.0)
        
        # Stop the audio input
        if self.input_device is not None:
            self.input_device.stop()
        
        logger.info("Audio processor stopped")
        return True
    
    def get_processed_data(self, block=True, timeout=None):
        """
        Get processed audio data from the queue.
        
        Args:
            block (bool, optional): Whether to block until data is available
            timeout (float, optional): Timeout in seconds
            
        Returns:
            dict: Processed audio data
        """
        try:
            return self.data_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None
    
    def _processor_thread(self):
        """
        Thread function for processing audio data.
        """
        while self.is_running:
            try:
                # Get audio data from the input device
                data = self.input_device.get_audio_data(block=True, timeout=0.1)
                
                if data is not None:
                    # Process the audio data
                    processed_data = self._process_audio(data)
                    
                    # Put the processed data in the queue (non-blocking)
                    try:
                        self.data_queue.put(processed_data, block=False)
                    except queue.Full:
                        # Queue is full, remove the oldest item
                        try:
                            self.data_queue.get_nowait()
                            self.data_queue.put(processed_data, block=False)
                        except queue.Empty:
                            pass
            except Exception as e:
                logger.error(f"Error processing audio data: {e}")
                time.sleep(0.1)
    
    def _process_audio(self, data):
        """
        Process audio data.
        
        Args:
            data (numpy.ndarray): Audio data as a numpy array
            
        Returns:
            dict: Processed audio data
        """
        # Apply window function
        windowed_data = data * self.window
        
        # Calculate FFT
        fft_data = np.fft.rfft(windowed_data, n=self.fft_size)
        # Get magnitudes (absolute values)
        fft_magnitudes = np.abs(fft_data) / (self.fft_size / 2)
        
        # Ignore DC component (0 Hz)
        fft_magnitudes[0] = 0
        
        # Calculate RMS volume
        rms_volume = np.sqrt(np.mean(data**2))
        
        # Normalize volume (0 to 1)
        normalized_volume = min(1.0, rms_volume / 0.1)
        
        # Process according to the mode
        result = {
            "volume": normalized_volume,
            "timestamp": time.time()
        }
        
        if self.processing_mode == ProcessingMode.RAW:
            # Just return the raw audio data
            result["raw_data"] = data
            
        elif self.processing_mode == ProcessingMode.SPECTRUM:
            # Return the FFT spectrum
            result["spectrum"] = fft_magnitudes
            result["frequencies"] = self.fft_freqs
            
        elif self.processing_mode == ProcessingMode.FREQUENCY_BANDS:
            # Calculate frequency band energies
            band_energies = self._calculate_band_energies(fft_magnitudes)
            result["bands"] = band_energies
            result["band_freqs"] = self.frequency_bands
            
        elif self.processing_mode == ProcessingMode.BEAT_DETECTION:
            # Detect beats
            band_energies = self._calculate_band_energies(fft_magnitudes)
            beats = self._detect_beats(band_energies)
            result["bands"] = band_energies
            result["band_freqs"] = self.frequency_bands
            result["beats"] = beats
        
        return result
    
    def _calculate_band_indices(self):
        """
        Calculate the FFT indices for each frequency band.
        
        Returns:
            list: List of (start_index, end_index) tuples for each band
        """
        band_indices = []
        
        for band in self.frequency_bands:
            start_freq, end_freq = band
            
            # Find the FFT indices for this frequency range
            start_idx = np.argmax(self.fft_freqs >= start_freq)
            end_idx = np.argmax(self.fft_freqs >= end_freq) if end_freq < self.fft_freqs[-1] else len(self.fft_freqs) - 1
            
            band_indices.append((start_idx, end_idx))
        
        return band_indices
    
    def _calculate_band_energies(self, fft_magnitudes):
        """
        Calculate the energy in each frequency band.
        
        Args:
            fft_magnitudes (numpy.ndarray): FFT magnitudes
            
        Returns:
            list: List of band energies (normalized 0-1)
        """
        band_energies = []
        
        for start_idx, end_idx in self.band_indices:
            # Calculate the average energy in this band
            if start_idx < end_idx:
                band_energy = np.mean(fft_magnitudes[start_idx:end_idx])
                # Normalize (empirical values based on testing)
                band_energy = min(1.0, band_energy / 0.05)
                band_energies.append(band_energy)
            else:
                band_energies.append(0.0)
        
        return band_energies
    
    def _detect_beats(self, band_energies):
        """
        Detect beats in the audio.
        
        Args:
            band_energies (list): Frequency band energies
            
        Returns:
            list: List of beat detection results for each band (0 or 1)
        """
        # Add the current energies to the history
        self.energy_history.append(band_energies)
        
        # Keep the history at a reasonable size
        if len(self.energy_history) > self.beat_history_size:
            self.energy_history.pop(0)
        
        # If we don't have enough history, return no beats
        if len(self.energy_history) < self.beat_history_size / 2:
            return [0] * len(band_energies)
        
        # Calculate the average and variance for each band
        band_averages = np.mean(self.energy_history, axis=0)
        band_variances = np.var(self.energy_history, axis=0)
        
        # Detect beats
        beats = []
        for i, energy in enumerate(band_energies):
            avg = band_averages[i]
            var = band_variances[i]
            
            # A beat is detected if the energy is significantly above the average
            # The threshold is adjusted based on the variance
            threshold = avg + self.beat_threshold * (var + 0.01)**0.5
            
            beat = 1 if energy > threshold else 0
            beats.append(beat)
        
        return beats
    
    def create_audio_input(self):
        """Create and initialize the audio input device"""
        if self.input_method == AudioInputMethod.MICROPHONE:
            input_device = MicrophoneInput(self.audio_config)
        elif self.input_method == AudioInputMethod.FILE:
            input_device = FileInput(self.audio_config)
        elif self.input_method == AudioInputMethod.DUMMY:
            input_device = DummyInput(self.audio_config)
        else:
            raise ValueError(f"Unsupported audio input method: {self.input_method}")
        
        # Start the input device
        if not input_device.start():
            raise RuntimeError("Failed to start audio input device")
        
        return input_device
    
    def close_audio_input(self):
        """Close the audio input device"""
        if self.input_device is not None:
            self.input_device.stop()
            self.input_device = None

# Test the module if run directly
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test configuration
    test_config = {
        "audio": {
            "input_method": "dummy",
            "sample_rate": 44100,
            "chunk_size": 1024,
            "channels": 1,
            "pattern": "sweep",
            "processing_mode": "frequency_bands"
        }
    }
    
    # Create the audio processor
    processor = AudioProcessor(test_config)
    
    # Start the processor
    processor.start()
    
    try:
        # Process for 5 seconds
        for _ in range(50):
            # Get processed data
            data = processor.get_processed_data(timeout=0.1)
            
            if data:
                # Print the volume
                print(f"Volume: {data['volume']:.2f}")
                
                # Print the frequency bands
                if 'bands' in data:
                    print("Frequency Bands:")
                    for i, energy in enumerate(data['bands']):
                        if i < len(data['band_freqs']):
                            min_freq, max_freq = data['band_freqs'][i]
                            print(f"  {min_freq}-{max_freq} Hz: {energy:.2f}")
                
                # Print beat detection
                if 'beats' in data:
                    print("Beats:", data['beats'])
                
                print()
            
            time.sleep(0.1)
    
    finally:
        # Stop the processor
        processor.stop() 