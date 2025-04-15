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

# This module handles audio processing for the Audio LED Visualization System.
# It takes raw audio data from the input handler and performs various analyses:
# - Fast Fourier Transform (FFT) for frequency analysis
# - Amplitude detection for volume visualization
# - Beat detection for rhythm visualization
#
# The processed audio data is provided in a format suitable for conversion to RGB values.

#--------------------------------------
#       IMPORTS
#--------------------------------------

import os
import sys
import logging
import time
import struct
import numpy as np
from array import array
from collections import deque

# Import for audio input handling
from audio_led.audio import input_handler

# Configure module logger
logger = logging.getLogger(__name__)

#--------------------------------------
#       CONSTANTS
#--------------------------------------

# Window functions
WINDOW_BLACKMAN = "blackman"
WINDOW_HAMMING = "hamming"
WINDOW_HANN = "hann"
WINDOW_NONE = "none"

# Default frequency bands for visualization
DEFAULT_FREQ_BANDS = 64

# Default smoothing factor (0-1)
DEFAULT_SMOOTHING = 0.5

# Default frequency range (Hz)
DEFAULT_MIN_FREQ = 20
DEFAULT_MAX_FREQ = 20000

# Default parameters for beat detection
BEAT_SENSITIVITY = 1.3  # Higher values = less sensitive
BEAT_DECAY = 0.995      # How quickly the beat detection recovers

#--------------------------------------
#       CLASSES
#--------------------------------------

class AudioProcessor:
    """
    Process audio data for visualization.
    
    This class takes raw audio data and performs various analyses to extract
    features suitable for RGB conversion and visualization.
    """
    
    def __init__(self, env_config):
        """
        Initialize the audio processor.
        
        Args:
            env_config (dict): Environment configuration including processing settings
        """
        self.config = env_config.get('config', {}).get('processing', {})
        self.audio_config = env_config.get('config', {}).get('audio', {})
        
        # Initialize processing parameters
        self.fft_enabled = self.config.get('fft_enabled', True)
        self.window_type = self.config.get('window_type', WINDOW_BLACKMAN)
        self.freq_bands = self.config.get('bands', DEFAULT_FREQ_BANDS)
        self.smoothing = self.config.get('smoothing', DEFAULT_SMOOTHING)
        
        # Set frequency range
        freq_range = self.config.get('frequency_range', [DEFAULT_MIN_FREQ, DEFAULT_MAX_FREQ])
        self.min_freq = freq_range[0]
        self.max_freq = freq_range[1]
        
        # Get audio parameters from config
        self.sample_rate = self.audio_config.get('sample_rate', 44100)
        self.channels = self.audio_config.get('channels', 2)
        self.chunk_size = self.audio_config.get('chunk_size', 2048)
        self.sample_width = self.audio_config.get('format', 16) // 8  # Convert bits to bytes
        
        # Initialize state variables
        self.spectrum = np.zeros(self.freq_bands)
        self.prev_spectrum = np.zeros(self.freq_bands)
        self.peak_spectrum = np.zeros(self.freq_bands)
        self.energy_history = deque(maxlen=50)  # For beat detection
        self.last_beat_time = time.time()
        self.beat_detected = False
        self.volume = 0
        self.peak_volume = 0
        
        # Initialize audio input handler
        self.audio_input = None
        self.running = False
        
        # Initialize audio scaling parameters
        self.volume_scale = 1.0
        self.bass_scale = 1.0
        self.mid_scale = 1.0
        self.treble_scale = 1.0
        self.beat_sensitivity = 1.0
        
        # Create window function for FFT
        self._create_window()
        
        logger.info(f"Audio processor initialized (fft={self.fft_enabled}, bands={self.freq_bands}, window={self.window_type})")
    
    def _create_window(self):
        """
        Create the window function for FFT analysis.
        """
        if self.window_type == WINDOW_BLACKMAN:
            self.window = np.blackman(self.chunk_size)
        elif self.window_type == WINDOW_HAMMING:
            self.window = np.hamming(self.chunk_size)
        elif self.window_type == WINDOW_HANN:
            self.window = np.hanning(self.chunk_size)
        else:
            self.window = np.ones(self.chunk_size)
    
    def process(self, audio_data=None):
        """
        Process a chunk of audio data.
        
        Args:
            audio_data (bytes or numpy.ndarray, optional): Raw audio data.
                If None, gets data from the audio input handler.
            
        Returns:
            dict: Processed audio data including spectrum, volume, beat, etc.
        """
        try:
            # If audio_data is None and we have an audio input handler, get data from it
            if audio_data is None and self.audio_input is not None:
                audio_data = self.audio_input.get_audio_chunk()
            
            if audio_data is None:
                # Return a dummy object with zero values if no data is available
                return {
                    'spectrum': np.zeros(self.freq_bands),
                    'volume': 0,
                    'normalized_volume': 0,
                    'beat': False,
                    'peak_spectrum': np.zeros(self.freq_bands),
                    'raw_data': np.zeros(0)
                }
            
            # Convert audio data to numpy array if needed
            data_array = self._convert_to_array(audio_data)
            
            if data_array is None or len(data_array) == 0:
                # Return a dummy object with zero values if data conversion failed
                return {
                    'spectrum': np.zeros(self.freq_bands),
                    'volume': 0,
                    'normalized_volume': 0,
                    'beat': False,
                    'peak_spectrum': np.zeros(self.freq_bands),
                    'raw_data': np.zeros(0)
                }
            
            # Calculate volume (amplitude)
            volume = self._calculate_volume(data_array)
            
            # Get spectrum through FFT
            if self.fft_enabled:
                spectrum = self._calculate_spectrum(data_array)
            else:
                spectrum = np.zeros(self.freq_bands)
            
            # Safety check for NaN or infinite values
            if spectrum is None or np.isnan(spectrum).any() or np.isinf(spectrum).any():
                spectrum = np.zeros(self.freq_bands)
            
            # Apply smoothing to the spectrum
            smoothed_spectrum = self._apply_smoothing(spectrum)
            
            # Detect beats
            beat_detected = self._detect_beat(smoothed_spectrum)
            
            # Update state
            self.prev_spectrum = smoothed_spectrum
            self.volume = volume
            self.peak_volume = max(self.peak_volume, volume)
            self.beat_detected = beat_detected
            
            # Return processed data
            return {
                'spectrum': smoothed_spectrum,
                'volume': volume,
                'normalized_volume': volume / (self.peak_volume if self.peak_volume > 0 else 1),
                'beat': beat_detected,
                'peak_spectrum': self.peak_spectrum,
                'raw_data': data_array
            }
        except Exception as e:
            logger.error(f"Error processing audio data: {e}")
            # Return a dummy object in case of error
            return {
                'spectrum': np.zeros(self.freq_bands),
                'volume': 0,
                'normalized_volume': 0,
                'beat': False,
                'peak_spectrum': np.zeros(self.freq_bands),
                'raw_data': np.zeros(0)
            }
    
    def _convert_to_array(self, audio_data):
        """
        Convert audio data to a numpy array.
        
        Args:
            audio_data (bytes or numpy.ndarray): Raw audio data
            
        Returns:
            numpy.ndarray: Audio data as a numpy array
        """
        try:
            # If already a numpy array, just return it
            if isinstance(audio_data, np.ndarray):
                return audio_data
            
            # Convert bytes to numpy array based on sample width
            if self.sample_width == 1:  # 8-bit (unsigned)
                data_array = np.frombuffer(audio_data, dtype=np.uint8)
                # Convert to signed representation (centered at 128)
                data_array = data_array.astype(np.int16) - 128
            elif self.sample_width == 2:  # 16-bit
                data_array = np.frombuffer(audio_data, dtype=np.int16)
            elif self.sample_width == 3:  # 24-bit
                # Handle 24-bit audio (less common)
                data_array = np.zeros(len(audio_data) // 3, dtype=np.int32)
                for i in range(len(data_array)):
                    data_array[i] = (audio_data[i*3] | 
                                    (audio_data[i*3+1] << 8) | 
                                    (audio_data[i*3+2] << 16))
                    if data_array[i] & 0x800000:
                        data_array[i] |= ~0xffffff  # Sign extension
            elif self.sample_width == 4:  # 32-bit
                data_array = np.frombuffer(audio_data, dtype=np.int32)
            else:
                logger.error(f"Unsupported sample width: {self.sample_width}")
                return None
            
            # If stereo, average the channels
            if self.channels == 2:
                data_array = np.mean([data_array[::2], data_array[1::2]], axis=0)
            
            return data_array
            
        except Exception as e:
            logger.error(f"Error converting audio data: {e}")
            return None
    
    def _calculate_volume(self, data_array):
        """
        Calculate the volume (amplitude) of the audio data.
        
        Args:
            data_array (numpy.ndarray): Audio data as a numpy array
            
        Returns:
            float: Volume level (RMS amplitude)
        """
        try:
            # Calculate RMS amplitude
            if len(data_array) > 0:
                return np.sqrt(np.mean(np.square(data_array)))
            else:
                return 0
                
        except Exception as e:
            logger.error(f"Error calculating volume: {e}")
            return 0
    
    def _calculate_spectrum(self, data_array):
        """
        Calculate the frequency spectrum using FFT.
        
        Args:
            data_array (numpy.ndarray): Audio data as a numpy array
            
        Returns:
            numpy.ndarray: Frequency spectrum
        """
        try:
            # Resize data array or window if necessary
            data_len = len(data_array)
            if data_len != len(self.window):
                if data_len < len(self.window):
                    # Pad the data array
                    padded_data = np.zeros(len(self.window))
                    padded_data[:data_len] = data_array
                    data_array = padded_data
                else:
                    # Resize the window
                    if self.window_type == WINDOW_BLACKMAN:
                        self.window = np.blackman(data_len)
                    elif self.window_type == WINDOW_HAMMING:
                        self.window = np.hamming(data_len)
                    elif self.window_type == WINDOW_HANN:
                        self.window = np.hanning(data_len)
                    else:
                        self.window = np.ones(data_len)
            
            # Apply window function
            windowed_data = data_array * self.window
            
            # Perform FFT
            fft_data = np.abs(np.fft.rfft(windowed_data))
            
            # Square the FFT data to get power spectrum
            fft_data = fft_data ** 2
            
            # Convert to logarithmic scale
            fft_data = np.log10(fft_data + 1)
            
            # Calculate frequency bins
            freqs = np.fft.rfftfreq(len(windowed_data), 1.0 / self.sample_rate)
            
            # Map frequencies to bands (logarithmic scale)
            min_freq_idx = np.argmax(freqs >= self.min_freq)
            max_freq_idx = np.argmax(freqs >= self.max_freq) if np.any(freqs >= self.max_freq) else len(freqs) - 1
            
            # Convert to log scale for frequency bands
            log_min = np.log10(self.min_freq)
            log_max = np.log10(self.max_freq)
            
            # Create bands
            bands = np.zeros(self.freq_bands)
            
            for i in range(min_freq_idx, max_freq_idx):
                # Determine which band this frequency belongs to
                if freqs[i] <= 0:
                    continue
                    
                log_freq = np.log10(freqs[i])
                band_idx = int((log_freq - log_min) / (log_max - log_min) * self.freq_bands)
                
                if 0 <= band_idx < self.freq_bands:
                    bands[band_idx] += fft_data[i]
            
            # Normalize bands
            max_val = np.max(bands) if np.max(bands) > 0 else 1
            bands = bands / max_val
            
            # Update peak spectrum (with slow decay)
            self.peak_spectrum = np.maximum(self.peak_spectrum * 0.99, bands)
            
            return bands
            
        except Exception as e:
            logger.error(f"Error calculating spectrum: {e}")
            return np.zeros(self.freq_bands)
    
    def _apply_smoothing(self, spectrum):
        """
        Apply smoothing to the spectrum for more pleasing visualization.
        
        Args:
            spectrum (numpy.ndarray): Current frequency spectrum
            
        Returns:
            numpy.ndarray: Smoothed frequency spectrum
        """
        # If no previous spectrum or smoothing disabled, return as is
        if self.smoothing <= 0 or self.prev_spectrum is None:
            return spectrum
            
        # Apply smoothing factor
        return self.prev_spectrum * self.smoothing + spectrum * (1 - self.smoothing)
    
    def _detect_beat(self, spectrum):
        """
        Detect beats in the audio using the spectrum.
        
        Args:
            spectrum (numpy.ndarray): Frequency spectrum
            
        Returns:
            bool: True if a beat is detected, False otherwise
        """
        try:
            # Calculate energy in the lower frequency bands (where bass is)
            bass_range = int(self.freq_bands * 0.2)  # Lower 20% of frequencies
            energy = np.sum(spectrum[:bass_range])
            
            # Add to history
            self.energy_history.append(energy)
            
            # If we don't have enough history, no beat
            if len(self.energy_history) < 10:
                return False
            
            # Calculate average energy
            avg_energy = np.mean(self.energy_history)
            
            # Calculate variance (to avoid triggering on quiet parts)
            variance = np.std(self.energy_history)
            
            # Check if current energy exceeds average by threshold
            is_beat = energy > avg_energy * BEAT_SENSITIVITY and energy > 0.1
            
            # Only trigger if enough time has passed since last beat
            now = time.time()
            if is_beat and now - self.last_beat_time > 0.1:  # At most 10 beats per second
                self.last_beat_time = now
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting beat: {e}")
            return False
    
    def start(self):
        """Start the audio processor by initializing the audio input."""
        if self.running:
            return
            
        try:
            # Create audio input handler with proper environment config structure
            env_config = {
                'config': {
                    'audio': self.audio_config
                },
                'env_info': {}  # Add empty env_info to match expected structure
            }
            self.audio_input = input_handler.AudioInputHandler(env_config)
            self.running = True
            logger.info("Audio processor started")
        except Exception as e:
            logger.error(f"Failed to start audio processor: {e}")
            self.running = False
    
    def stop(self):
        """Stop the audio processor and clean up resources."""
        if not self.running:
            return
            
        try:
            if self.audio_input is not None:
                self.audio_input.close()
                self.audio_input = None
            self.running = False
            logger.info("Audio processor stopped")
        except Exception as e:
            logger.error(f"Error stopping audio processor: {e}")
            
    def reset(self):
        """Reset the audio processor state."""
        self.spectrum = np.zeros(self.freq_bands)
        self.prev_spectrum = np.zeros(self.freq_bands)
        self.peak_spectrum = np.zeros(self.freq_bands)
        self.energy_history.clear()
        self.last_beat_time = time.time()
        self.beat_detected = False
        self.volume = 0
        self.peak_volume = 0

# Test the module if run directly
if __name__ == "__main__":
    from input_handler import AudioInputHandler
    import matplotlib.pyplot as plt
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create a simple environment configuration
    env_config = {
        'config': {
            'audio': {
                'input_source': 'auto',
                'play_audio': True,
                'sample_rate': 44100,
                'channels': 2,
                'chunk_size': 2048,
                'format': 16
            },
            'processing': {
                'fft_enabled': True,
                'window_type': WINDOW_BLACKMAN,
                'bands': 64,
                'smoothing': 0.5,
                'frequency_range': [20, 20000]
            }
        },
        'env_info': {
            'capabilities': {
                'audio_input': True,
                'audio_output': True
            }
        }
    }
    
    # Initialize the audio input handler
    audio_input = AudioInputHandler(env_config)
    
    # Initialize the audio processor
    audio_proc = AudioProcessor(env_config)
    
    # Set up a simple matplotlib visualization
    plt.figure(figsize=(10, 6))
    plt.ion()  # Interactive mode
    
    # Process audio data in a loop
    try:
        while True:
            # Get audio data
            audio_data = audio_input.get_audio_chunk()
            if audio_data is None:
                break
            
            # Process audio data
            processed = audio_proc.process(audio_data)
            if processed is None:
                continue
            
            # Get spectrum and volume
            spectrum = processed['spectrum']
            volume = processed['normalized_volume']
            beat = processed['beat']
            
            # Clear the plot
            plt.clf()
            
            # Plot spectrum
            plt.bar(range(len(spectrum)), spectrum, color='b', alpha=0.5)
            
            # Plot peak spectrum
            plt.plot(range(len(spectrum)), processed['peak_spectrum'], 'r--')
            
            # Add volume and beat indicators
            plt.text(0.02, 0.95, f"Volume: {volume:.2f}", transform=plt.gca().transAxes)
            plt.text(0.02, 0.90, f"Beat: {'Yes' if beat else 'No'}", transform=plt.gca().transAxes)
            
            plt.ylim(0, 1.1)
            plt.title("Audio Spectrum")
            plt.xlabel("Frequency Band")
            plt.ylabel("Amplitude")
            
            # Update the plot
            plt.draw()
            plt.pause(0.01)
            
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Close the audio input
        audio_input.close()
        plt.close() 